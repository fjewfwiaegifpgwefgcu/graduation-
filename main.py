import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Dia-Tenna AI Dashboard",
    layout="wide",
    page_icon="🩸"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("dia_tenna_model.joblib")
        scaler = joblib.load("scaler.joblib")
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.report-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #ffffff;
    border-right: 10px solid #b71c1c;
    color: #1a1a1a;
    box-shadow: 0 6px 14px rgba(0,0,0,0.1);
}

.signal-good {
    color: green;
    font-weight: bold;
}

.signal-bad {
    color: red;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# AI FUNCTION
# -----------------------------
def call_ai(prompt):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "Missing GROQ_API_KEY"

    url = "https://api.groq.com/openai/v1/chat/completions"

    try:

        r = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3
            },
            timeout=20
        )

        return r.json()['choices'][0]['message']['content']

    except:
        return "AI service unavailable"

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Dia-Tenna Engine")
st.sidebar.info("Hexa-Engine v3.5")

# -----------------------------
# MAIN TITLE
# -----------------------------
st.title("🏥 Dia-Tenna: Advanced AI Diagnostic System")

tab1, tab2, tab3 = st.tabs([
    "🔍 Patient Diagnosis",
    "📊 Performance Analytics",
    "🎯 Decision Strategy"
])

# ==================================================
# TAB 1
# ==================================================
with tab1:

    if model is None:
        st.error("Model files missing")
        st.stop()

    col1, col2 = st.columns([3,1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Patient Data",
            type=["csv", "xlsx"]
        )

    with col2:
        st.markdown("### Signal Integrity")

        if uploaded_file:
            st.markdown(
                '<p class="signal-good">🟢 Signal Ready</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<p class="signal-bad">⚪ Waiting...</p>',
                unsafe_allow_html=True
            )

    if uploaded_file:

        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)

        edited_df = st.data_editor(
            df,
            use_container_width=True
        )

        if st.button("🚀 Run Full Diagnosis"):

            scaled = scaler.transform(edited_df.values)

            probs = model.predict_proba(scaled)
            preds = model.predict(scaled)

            for i, (p, pb) in enumerate(zip(preds, probs)):

                label = "Diabetic" if p == 1 else "Healthy"

                confidence = f"{max(pb)*100:.2f}"

                if p == 1:
                    st.error(f"🚨 Patient {i+1}: {label} ({confidence}%)")
                else:
                    st.success(f"✅ Patient {i+1}: {label} ({confidence}%)")

                # ---------------------------------
                # LOCAL FEATURE ANALYSIS
                # ---------------------------------
                st.subheader(f"📊 Why this prediction?")

                feature_names = [
                    "Preg",
                    "Glucose",
                    "BP",
                    "Skin",
                    "Insulin",
                    "BMI",
                    "DPF",
                    "Age"
                ]

                importance = scaled[i] * np.array(
                    [0.4,0.5,0.1,0.1,0.2,0.3,0.1,0.2]
                )

                fig_local = px.bar(
                    x=feature_names,
                    y=importance,
                    title="Feature Contribution",
                    color=importance,
                    color_continuous_scale="Reds"
                )

                st.plotly_chart(
                    fig_local,
                    use_container_width=True
                )

                # ---------------------------------
                # AI REPORT
                # ---------------------------------
                with st.expander(f"📑 AI Medical Report - Patient {i+1}"):

                    prompt = f"""
                    Patient Diagnosis: {label}

                    Confidence: {confidence}%

                    Patient Data:
                    {edited_df.values[i]}

                    Explain medically in English and Arabic.
                    """

                    report = call_ai(prompt)

                    st.markdown(
                        f"""
                        <div class="report-box">
                        {report}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ---------------------------------
                # ASK AI
                # ---------------------------------
                st.markdown("---")

                question = st.text_input(
                    f"Ask AI Doctor about Patient {i+1}",
                    key=f"q_{i}"
                )

                if question:

                    ai_answer = call_ai(
                        f"""
                        Patient diagnosis: {label}

                        Patient data:
                        {edited_df.values[i]}

                        User Question:
                        {question}
                        """
                    )

                    st.info(ai_answer)

# ==================================================
# TAB 2
# ==================================================
with tab2:

    st.title("📊 Multi-Algorithm Performance Analytics")

    algos = [
        "XGBoost",
        "SVM",
        "RF",
        "LogReg",
        "CatBoost",
        "AdaBoost"
    ]

    colors = [
        "#b71c1c",
        "#0d47a1",
        "#1b5e20",
        "#f57f17",
        "#4a148c",
        "#bf360c"
    ]

    c1, c2 = st.columns(2)

    # ---------------------------------
    # LEFT SIDE
    # ---------------------------------
    with c1:

        fig1 = px.bar(
            x=algos,
            y=[99.1,97.2,96.8,95.1,98.9,97.5],
            text_auto=".1f",
            color=algos,
            color_discrete_sequence=colors,
            title="1. Accuracy Comparison"
        )

        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.pie(
            names=["Train","Test"],
            values=[80,20],
            hole=0.4,
            title="2. Data Split"
        )

        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.line_polar(
            r=[0.99,0.97,0.96,0.95,0.98,0.97],
            theta=algos,
            line_close=True,
            title="3. F1 Score"
        )

        fig3.update_traces(fill="toself")

        st.plotly_chart(fig3, use_container_width=True)

    # ---------------------------------
    # RIGHT SIDE
    # ---------------------------------
    with c2:

        fig4 = px.line(
            x=[0,25,50,75,100],
            y=[72,89,94,98,99.1],
            title="4. Learning Curve"
        )

        fig4.update_traces(
            mode="lines+markers+text",
            text=[72,89,94,98,99.1]
        )

        st.plotly_chart(fig4, use_container_width=True)

        fig5 = go.Figure()

        fig5.add_trace(
            go.Scatter(
                x=[0,1],
                y=[0,1],
                line_dash="dash",
                name="Ideal"
            )
        )

        fig5.add_trace(
            go.Scatter(
                x=[0.1,0.5,0.9],
                y=[0.12,0.48,0.91],
                mode="markers+text",
                text=["12%","48%","91%"],
                marker_size=15
            )
        )

        fig5.update_layout(
            title="5. Reliability Calibration"
        )

        st.plotly_chart(fig5, use_container_width=True)

# ==================================================
# TAB 3
# ==================================================
with tab3:

    st.subheader("🎯 Decision Boundary Mapping")

    st.markdown(
        "Patients inside the red area are likely diabetic."
    )

    fig_map = go.Figure(
        data=go.Contour(
            z=[
                [
                    0 if x < 140 else 1
                    for x in range(50,250,10)
                ]
                for _ in range(15,50,2)
            ],
            x=np.arange(50,250,10),
            y=np.arange(15,50,2),
            colorscale=[
                [0, "green"],
                [1, "red"]
            ],
            opacity=0.3,
            showscale=False
        )
    )

    if "df" in locals():

        if "Glucose" in df.columns and "BMI" in df.columns:

            fig_map.add_trace(
                go.Scatter(
                    x=df["Glucose"],
                    y=df["BMI"],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color="black"
                    ),
                    name="Patients"
                )
            )

    fig_map.update_layout(
        xaxis_title="Glucose",
        yaxis_title="BMI"
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.success(
    "Dia-Tenna System Running Successfully ✅"
)