import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ============================================================
# GPU CHECK
# ============================================================

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("clinical_risk_dataset.csv")

# ============================================================
# ADD BALANCED AUGMENTATION
# ============================================================

augmented_samples = [

    # SAFETY DOMINANT
    {"text": "32% of patients discontinued therapy following severe hepatotoxicity and grade 3 cardiac adverse events.", "label": "Safety Risk"},
    {"text": "Multiple grade 4 toxicities and life-threatening complications resulted in early termination.", "label": "Safety Risk"},
    {"text": "Severe neutropenia and cardiotoxicity were reported with treatment-related mortality.", "label": "Safety Risk"},
    {"text": "High incidence of grade 3 immune-related adverse events required protocol modification.", "label": "Safety Risk"},
    {"text": "Organ toxicity and severe adverse reactions led to treatment discontinuation.", "label": "Safety Risk"},

    # PURE DROPOUT (MILD)
    {"text": "36% of participants discontinued treatment due to persistent nausea and mild fatigue. No grade 3 or 4 adverse events were observed.", "label": "Dropout Risk"},
    {"text": "High withdrawal rates were reported due to tolerability concerns despite absence of serious toxicity.", "label": "Dropout Risk"},
    {"text": "Patients discontinued therapy due to moderate gastrointestinal discomfort. No life-threatening complications occurred.", "label": "Dropout Risk"},
    {"text": "Adherence declined because of mild dizziness and headache, without severe adverse reactions.", "label": "Dropout Risk"},
    {"text": "Dropout increased due to compliance issues and mild side effects. Safety profile remained acceptable.", "label": "Dropout Risk"},
    {"text": "Treatment withdrawal was common; however, no grade 3 adverse events or mortality were documented.", "label": "Dropout Risk"},
    {"text": "Discontinuation was driven by moderate tolerability issues rather than serious toxicity.", "label": "Dropout Risk"},
    {"text": "Participants withdrew due to mild dermatologic reactions with no severe safety signals.", "label": "Dropout Risk"},
    {"text": "Nearly one-third discontinued treatment, though no organ toxicity or severe complications were reported.", "label": "Dropout Risk"},
    {"text": "Withdrawal rates were elevated due to mild side effects and inconvenience, not due to serious harm.", "label": "Dropout Risk"},
]

aug_df = pd.DataFrame(augmented_samples)
df = pd.concat([df, aug_df], ignore_index=True)

print("Augmented samples added:", len(aug_df))

# ============================================================
# ENCODE LABELS
# ============================================================

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

print("\nLabel Mapping:")
for label, idx in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{label} -> {idx}")

# ============================================================
# STRATIFIED SPLIT (IMPORTANT)
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ============================================================
# TOKENIZER
# ============================================================

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ============================================================
# CLASS WEIGHTS (IMPORTANT FIX)
# ============================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df["label"]),
    y=df["label"]
)

class_weights = torch.tensor(class_weights, dtype=torch.float)

if torch.cuda.is_available():
    class_weights = class_weights.cuda()

# ============================================================
# MODEL
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4
)

# ============================================================
# CUSTOM TRAINER WITH WEIGHTED LOSS
# ============================================================

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir="./clinical_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=1,
    fp16=True
)

# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# ============================================================
# TRAIN
# ============================================================

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# ============================================================
# SAVE MODEL
# ============================================================

trainer.save_model("./clinical_model")
tokenizer.save_pretrained("./clinical_model")

print("\n🔥 Improved model training complete. Saved in ./clinical_model")
