from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import pipeline
import torch

import torch.nn.functional as F

app = FastAPI()
#app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
MODEL_PATH = "./clinical_model"

# -------- Load Biomedical NER Model --------
NER_MODEL_NAME = "d4data/biomedical-ner-all"

ner_pipeline = pipeline(
    "ner",
    model=NER_MODEL_NAME,
    aggregation_strategy="simple"
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()



LABELS = [
    "Dropout Risk",
    "Efficacy Concern",
    "Normal Trial",
    "Safety Risk"
]


class TrialInput(BaseModel):
    text: str

class NERInput(BaseModel):
    text: str

# -------- Risk Phrase Extraction --------
def extract_risk_phrases(text):
    keywords = [
        "severe", "adverse", "mortality", "toxicity",
        "discontinuation", "withdrawal", "failed",
        "no significant improvement", "hepatotoxicity",
        "cardiac", "renal failure", "grade 3", "grade 4"
    ]

    found = []
    for word in keywords:
        if word.lower() in text.lower():
            found.append(word)

    return list(set(found))


# -------- Confidence Level Logic --------
def confidence_level(conf):
    if conf >= 0.85:
        return "High"
    elif conf >= 0.65:
        return "Moderate"
    else:
        return "Low"


# -------- Enterprise NLG --------
def generate_summary(label, confidence, highlights):

    confidence_pct = round(confidence * 100, 1)
    level = confidence_level(confidence)

    report = f"""
Clinical Trial Risk Assessment Report
--------------------------------------

Prediction: {label}
Confidence: {confidence_pct}% ({level} certainty)

"""

    if label == "Safety Risk":
        report += f"""
Executive Assessment:
Significant safety concerns identified.

Clinical Interpretation:
Signals such as {', '.join(highlights) if highlights else 'serious adverse patterns'} suggest elevated patient risk.

Regulatory Impact:
May require enhanced monitoring or protocol reassessment.

Recommended Action:
Immediate expert safety review advised.
"""

    elif label == "Dropout Risk":
        report += f"""
Executive Assessment:
Elevated probability of participant discontinuation.

Clinical Interpretation:
Withdrawal indicators ({', '.join(highlights) if highlights else 'tolerability issues'}) may compromise adherence.

Operational Impact:
High dropout may reduce statistical power.

Recommended Action:
Consider adherence optimization strategies.
"""

    elif label == "Efficacy Concern":
        report += f"""
Executive Assessment:
Potential therapeutic performance limitations.

Clinical Interpretation:
Indicators ({', '.join(highlights) if highlights else 'endpoint shortfalls'}) suggest limited benefit.

Strategic Impact:
Primary endpoints may not support regulatory approval.

Recommended Action:
Reevaluate study design or patient selection.
"""

    else:
        report += f"""
Executive Assessment:
Trial appears clinically stable.

Clinical Interpretation:
No dominant safety or efficacy risks identified.

Regulatory Outlook:
Acceptable benefit-risk balance.

Recommended Action:
Continue monitoring per protocol.
"""

    return report


# -------- API Endpoint --------
@app.post("/analyze")
def analyze_trial(data: TrialInput):

    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)[0]

    confidence, predicted_class = torch.max(probabilities, dim=0)

    top_label = LABELS[predicted_class.item()]
    confidence = confidence.item()

    highlights = extract_risk_phrases(data.text)

    # Remove highlights for normal trial
    if top_label == "Normal Trial":
        highlights = []

    risk_score = round(confidence * 100, 1)

    return {
        "top_label": top_label,
        "confidence": confidence,
        "confidence_level": confidence_level(confidence),
        "risk_score": risk_score,
        "summary": generate_summary(top_label, confidence, highlights),
        "risk_phrases": highlights,
        "all_scores": {
            LABELS[i]: float(probabilities[i])
            for i in range(len(LABELS))
        }
    }

# -------- NER API Endpoint --------
@app.post("/extract-ner")
def extract_ner(data: NERInput):

    entities = ner_pipeline(data.text)

    formatted_entities = []

    for entity in entities:
        formatted_entities.append({
            "entity": str(entity["word"]),
            "label": str(entity["entity_group"]),
            "confidence": float(round(float(entity["score"]), 3))
        })

    return {
        "entities": formatted_entities,
        "total_entities": int(len(formatted_entities))
    }
# -------- Improved Safety Severity Scoring --------
# -------- Improved Safety Severity Scoring (Refined) --------
def compute_safety_score(prediction, confidence, entities):

    # -------- Base Model Score (0–60) --------
    if prediction == "Safety Risk":
        base_score = confidence * 60
    elif prediction == "Dropout Risk":
        base_score = confidence * 40
    elif prediction == "Efficacy Concern":
        base_score = confidence * 35
    else:
        base_score = confidence * 20

    # -------- Entity Influence (max 40 total) --------
    entity_score = 0
    entity_types = set([ent["label"].lower() for ent in entities])

    # ---- Severity Grading ----
    for ent in entities:
        if ent["label"].lower() == "severity":
            word = ent["entity"].lower()

            if "life-threatening" in word:
                entity_score += 25
            elif "severe" in word:
                entity_score += 20
            elif "moderate" in word:
                entity_score += 12
            elif "mild" in word:
                entity_score += 5
            else:
                entity_score += 10

    # ---- Other Entity Types ----
    if "sign_symptom" in entity_types:
        entity_score += 15

    if "dosage" in entity_types:
        entity_score += 5

    if "medication" in entity_types:
        entity_score += 5

    # Cap entity influence
    entity_score = min(entity_score, 40)

    final_score = min(base_score + entity_score, 100)

    # -------- Risk Level --------
    if final_score >= 85:
        level = "Critical"
    elif final_score >= 65:
        level = "High"
    elif final_score >= 45:
        level = "Moderate"
    else:
        level = "Low"

    return round(final_score, 2), level

# -------- Drug → Adverse Event Linking (With Dose + Causality) --------
def link_drug_adverse_events(entities, text):

    drugs = []
    doses = []
    adverse_events = []
    severities = []

    for ent in entities:
        label = ent["label"].lower()

        if label == "medication":
            drugs.append(ent["entity"])

        elif label == "dosage":
            doses.append(ent["entity"])

        elif label in ["sign_symptom", "disease_disorder"]:
            adverse_events.append(ent["entity"])

        elif label == "severity":
            severities.append(ent["entity"])

    # -------- Causality Keywords --------
    causality_keywords = [
        "after", "following", "due to",
        "associated with", "caused by",
        "induced by", "resulted in"
    ]

    causality_detected = any(
        keyword in text.lower() for keyword in causality_keywords
    )

    links = []

    # Only link if causality is detected
    if causality_detected:
        for drug in drugs:
            for event in adverse_events:
                link = {
                    "drug": drug,
                    "adverse_event": event,
                    "causality_detected": True
                }

                if doses:
                    link["dose"] = ", ".join(doses)

                if severities:
                    link["severity"] = ", ".join(severities)

                links.append(link)

    return links

# -------- Critical Safety Escalation Detector --------
def detect_critical_event(text, entities):

    critical_keywords = [
        "life-threatening",
        "acute liver failure",
        "multi-organ",
        "organ failure",
        "icu",
        "intensive care",
        "fatal",
        "death",
        "cardiotoxicity",
        "severe hypotension",
        "emergency discontinuation"
    ]

    text_lower = text.lower()

    keyword_flag = any(k in text_lower for k in critical_keywords)

    # Also escalate if severity entity contains life-threatening
    severity_flag = any(
        ent["label"].lower() == "severity" and
        "life" in ent["entity"].lower()
        for ent in entities
    )

    return keyword_flag or severity_flag

# -------- Unified Intelligence Endpoint --------
@app.post("/intelligence")
def clinical_intelligence(data: TrialInput):

    # -------- Risk Classification --------
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)[0]

    confidence_tensor, predicted_class = torch.max(probabilities, dim=0)

    top_label = LABELS[predicted_class.item()]
    confidence = float(confidence_tensor.item())



    highlights = extract_risk_phrases(data.text)

    if top_label == "Normal Trial":
        highlights = []

    # -------- Biomedical NER Extraction --------
    ner_results = ner_pipeline(data.text)

    # -------- Clean WordPiece Merge --------
    final_entities = []

    current_word = ""
    current_label = None
    current_scores = []

    for entity in ner_results:
        word = entity["word"]
        label = entity["entity_group"]
        score = float(entity["score"])

        # If token is continuation (##)
        if word.startswith("##"):
            current_word += word[2:]
            current_scores.append(score)
        else:
            # Save previous entity if exists
            if current_word:
                final_entities.append({
                    "entity": current_word,
                    "label": current_label,
                    "confidence": round(sum(current_scores) / len(current_scores), 3)
                })

            # Start new entity
            current_word = word
            current_label = label
            current_scores = [score]

    # Append last entity
    if current_word:
        final_entities.append({
            "entity": current_word,
            "label": current_label,
            "confidence": round(sum(current_scores) / len(current_scores), 3)
        })

    # -------- Safety Scoring (AFTER entities are built) --------
    safety_score, safety_level = compute_safety_score(
        top_label,
        confidence,
        final_entities
    )
    # -------- CRITICAL SAFETY OVERRIDE --------
    # -------- CRITICAL SAFETY OVERRIDE --------
    if detect_critical_event(data.text, final_entities):
        safety_score = max(safety_score, 90)
        safety_level = "Critical"
        top_label = "Safety Risk"

        # Boost confidence slightly for critical cases
        confidence = max(confidence, 0.85)
    # -------- Drug-Event Linking --------
    drug_event_links = link_drug_adverse_events(final_entities, data.text)

    return {
        "prediction": top_label,
        "confidence": confidence,
        "confidence_level": confidence_level(confidence),
        "risk_phrases": highlights,
        "entities": final_entities,
        "entity_count": len(final_entities),
        "safety_severity_score": safety_score,
        "safety_level": safety_level,
        "drug_event_links": drug_event_links,
        "summary": generate_summary(top_label, confidence, highlights)
    }