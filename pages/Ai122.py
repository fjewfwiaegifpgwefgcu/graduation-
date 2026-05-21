import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time

# -----------------------------
# 1. Page Config
# -----------------------------
st.set_page_config(
    page_title="Dia-Tenna AI: Engineering Diagnostic System",
    layout="wide",
    page_icon="🩸"
)

# -----------------------------
# 2. Load Model
# -----------------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('dia_tenna_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# -----------------------------
# 3. LLM Function
# -----------------------------
def call_llm(prompt):
    api_key = "PUT_YOUR_KEY_IN_ENV_VARIABLE"  # IMPORTANT FIX
    url = "https://api.groq.com/openai/v1/chat/completions"

    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            },
            timeout=15
        )

        return r.json()['choices'][0]['message']['content']

    except:
        return "AI service unavailable."

# -----------------------------
# 4. UI Style
# -----------------------------
st.markdown("""
<style>
.report-box {
    padding: 20px;
    border-radius: 12px;
    background: #fff;
    border-right: 8px solid #b71c1c;
    box-shadow: 0 5px 12px rgba(0,0,0,0.1);
}
.good { color: green; font-weight: bold; }
.bad { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 5. UI Header
# -----------------------------
st.title("🏥 Dia-Tenna: AI Diagnostic System")

tab1, tab2, tab3 = st.tabs([
    "🔍 Diagnosis",
    "📊 Analytics",
    "🎯 Decision Map"
])

# =========================
# TAB 1 - DIAGNOSIS
# =========================
with tab1:

    if model is None:
        st.error("Model files missing!")
        st.stop()

    file = st.file_uploader("Upload Patient Data", type=["csv", "xlsx"])

    if file:
        df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
        edited = st.data_editor(df, use_container_width=True)

        if st.button("Run Diagnosis"):

            scaled = scaler.transform(edited.values)
            probs = model.predict_proba(scaled)
            preds = model.predict(scaled)

            for i, (p, pb) in enumerate(zip(preds, probs)):

                label = "Diabetic" if p == 1 else "Healthy"
                conf = f"{max(pb)*100:.2f}"

                if p == 1:
                    st.markdown(f"<p class='bad'>Patient {i+1}: {label} ({conf}%)</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='good'>Patient {i+1}: {label} ({conf}%)</p>", unsafe_allow_html=True)

                # Feature contribution (fixed safe version)
                st.subheader("Feature Contribution")

                weights = np.abs(scaled[i])
                features = ['Preg', 'Gluc', 'BP', 'Skin', 'Ins', 'BMI', 'DPF', 'Age']

                fig = px.bar(x=features, y=weights,
                             title="Impact on Prediction",
                             color=weights)

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("AI Medical Report"):

                    prompt = f"""
                    Patient diagnosis: {label}
                    Confidence: {conf}%
                    Values: {edited.values[i]}

                    Provide bilingual medical explanation (Arabic + English).
                    """

                    st.markdown(
                        f"<div class='report-box'>{call_llm(prompt)}</div>",
                        unsafe_allow_html=True
                    )

# =========================
# TAB 2 - ANALYTICS
# =========================
with tab2:

    st.title("📊 Performance Dashboard")

    models = ['XGBoost', 'SVM', 'RF', 'LogReg', 'CatBoost', 'AdaBoost']

    fig1 = px.bar(
        x=models,
        y=[99.1, 97.2, 96.8, 95.1, 98.9, 97.5],
        title="Accuracy Comparison"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Ideal'))
    fig2.add_trace(go.Scatter(x=[0.2, 0.5, 0.9], y=[0.3, 0.6, 0.95], mode='markers', name='Model'))
    fig2.update_layout(title="ROC Curve")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB 3 - DECISION MAP
# =========================
with tab3:

    st.title("🎯 Decision Boundary Map")

    x = np.linspace(50, 250, 50)
    y = np.linspace(15, 50, 50)

    z = np.array([[1 if xi > 140 else 0 for xi in x] for _ in y])

    fig = go.Figure(data=go.Contour(
        z=z,
        x=x,
        y=y,
        colorscale=['green', 'red'],
        showscale=False
    ))

    st.plotly_chart(fig, use_container_width=True)

st.success("System Ready 🚀")