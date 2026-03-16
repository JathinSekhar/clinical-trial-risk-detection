import streamlit as st
import requests
import plotly.graph_objects as go
import json

API_URL = "http://127.0.0.1:8000/intelligence"

st.set_page_config(
    page_title="Clinical Trial Intelligence",
    layout="wide"
)

# ======================================================
# PREMIUM HACKATHON CSS
# ======================================================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.gradient-title {
    font-size: 44px;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6ec4, #7873f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #9ca3af;
    font-size: 18px;
    margin-bottom: 20px;
}

.main-card {
    padding: 45px;
    border-radius: 22px;
    background: linear-gradient(135deg, #1f2937, #111827);
    box-shadow: 0px 12px 35px rgba(0,0,0,0.45);
    text-align: center;
    margin-bottom: 30px;
}

.result-card {
    padding: 25px;
    border-radius: 16px;
    background: #1e293b;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

.badge {
    display: inline-block;
    padding: 8px 16px;
    margin: 6px;
    border-radius: 25px;
    background-color: #334155;
    color: white;
    font-size: 14px;
}

.section-title {
    margin-top: 30px;
    margin-bottom: 15px;
    font-size: 22px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown('<div class="gradient-title">🧠 Clinical Trial Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered early safety risk detection for clinical research</div>', unsafe_allow_html=True)
st.divider()

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.header("🔍 Analyze Trial")

    trial_text = st.text_area(
        "Clinical Trial Summary",
        height=200,
        placeholder="Paste clinical trial description..."
    )

    st.markdown("### ⚡ Quick Demo")
    if st.button("Load Safety Case"):
        trial_text = "A 58-year-old patient developed severe hepatotoxicity after receiving 150mg pembrolizumab."

    analyze = st.button("🚀 Analyze Trial", width="stretch")

# ======================================================
# MAIN VIEW
# ======================================================

if not analyze:
    st.markdown("""
    <div class="main-card">
        <h2>Welcome</h2>
        <p>
        Paste a clinical trial summary in the sidebar and click <b>Analyze Trial</b> 
        to detect safety risks, extract entities, identify drug-adverse links,
        and generate an executive risk brief.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    if not trial_text.strip():
        st.warning("Please enter trial text.")
        st.stop()

    with st.spinner("🧠 Running AI Risk Engine..."):
        try:
            response = requests.post(API_URL, json={"text": trial_text})
            result = response.json()
        except Exception as e:
            st.error(f"Backend connection failed: {e}")
            st.stop()

    # Extract data
    prediction = result.get("prediction", "Unknown")
    risk_score = result.get("safety_severity_score", 0)
    safety_level = result.get("safety_level", "Low")
    confidence = result.get("confidence", 0)
    entities = result.get("entities", [])
    links = result.get("drug_event_links", [])
    summary = result.get("summary", "")

    # ======================================================
    # HERO RISK CARD
    # ======================================================
    if risk_score < 40:
        color = "#22c55e"
        status_text = "LOW RISK"
    elif risk_score < 75:
        color = "#facc15"
        status_text = "MODERATE RISK"
    else:
        color = "#ef4444"
        status_text = "HIGH RISK"

    st.markdown(f"""
    <div class="main-card">
        <h2 style="color:{color};">{status_text}</h2>
        <h1 style="margin-top:10px;">{risk_score}/100</h1>
        <p>{prediction}</p>
        <p>Model Confidence: {round(confidence*100,1)}%</p>
    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # GAUGE
    # ======================================================
    st.markdown('<div class="section-title">📊 Risk Severity Index</div>', unsafe_allow_html=True)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={'suffix': "/100"},
        gauge={
            "axis": {"range": [0,100]},
            "steps": [
                {"range":[0,40], "color":"#22c55e"},
                {"range":[40,75], "color":"#facc15"},
                {"range":[75,100], "color":"#ef4444"}
            ]
        }
    ))
    gauge.update_layout(height=350)
    st.plotly_chart(gauge, width="stretch")

    # ======================================================
    # WHY THIS RISK? (Explainability Section)
    # ======================================================
    st.markdown('<div class="section-title">🧠 Why This Risk?</div>', unsafe_allow_html=True)

    explanations = []

    # Risk score logic
    if risk_score >= 75:
        explanations.append("High composite safety severity score detected.")
    elif risk_score >= 40:
        explanations.append("Moderate composite safety severity score detected.")
    else:
        explanations.append("Low overall severity score observed.")

    # Severe term detection
    for ent in entities:
        if ent.get("label") == "Severity" and ent.get("entity").lower() in ["severe", "critical", "life-threatening"]:
            explanations.append("High severity clinical terminology detected in trial description.")

    # Disease detection
    for ent in entities:
        if ent.get("label") == "Disease_disorder":
            explanations.append(f"Adverse clinical event identified: {ent.get('entity')}.")

    # Causality detection
    for link in links:
        if link.get("causality_detected"):
            explanations.append("Causal relationship detected between drug and adverse event.")

    # Display explanations
    if explanations:
        explanation_html = "<ul>"
        for item in explanations:
            explanation_html += f"<li>{item}</li>"
        explanation_html += "</ul>"

        st.markdown(f"""
        <div class="result-card">
        {explanation_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.write("No significant contributing factors detected.")

    # ======================================================
    # ENTITIES
    # ======================================================
    st.markdown('<div class="section-title">🔎 Extracted Clinical Entities</div>', unsafe_allow_html=True)

    if entities:
        badge_html = ""
        for ent in entities:
            text = ent.get("entity", "")
            label = ent.get("label", "")
            badge_html += f'<span class="badge">{label}: {text}</span>'
        st.markdown(badge_html, unsafe_allow_html=True)
    else:
        st.write("No entities detected.")

    # ======================================================
    # DRUG EVENT LINKS
    # ======================================================
    st.markdown('<div class="section-title">🔗 Drug–Adverse Event Links</div>', unsafe_allow_html=True)

    if links:
        for link in links:
            st.markdown(f"""
            <div class="result-card">
            <b>Drug:</b> {link.get("drug")} <br>
            <b>Adverse Event:</b> {link.get("adverse_event")} <br>
            <b>Severity:</b> {link.get("severity")} <br>
            <b>Causality Detected:</b> {link.get("causality_detected")} <br>
            <b>Dose:</b> {link.get("dose")}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No drug-event links identified.")

    # ======================================================
    # EXECUTIVE SUMMARY
    # ======================================================
    st.markdown('<div class="section-title">📑 Executive Risk Brief</div>', unsafe_allow_html=True)
    formatted_summary = summary.replace("\n", "<br>")

    st.markdown(f"""
    <div class="result-card">
    {formatted_summary}
    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # DOWNLOAD
    # ======================================================
    st.download_button(
        "📥 Download Risk Report (JSON)",
        data=json.dumps(result, indent=4),
        file_name="clinical_risk_report.json",
        mime="application/json"
    )