import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("dia_tenna_model.joblib")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Dia-Tenna AI Dashboard", layout="wide")

st.title("🩸 Dia-Tenna AI Diabetes Prediction Dashboard")

# -----------------------------
# Input Section
# -----------------------------
st.sidebar.header("Patient Data Input")

def user_input():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 15, 1)
    glucose = st.sidebar.slider("Glucose", 0, 200, 100)
    bp = st.sidebar.slider("Blood Pressure", 0, 122, 70)
    skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 846, 80)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Age", 10, 100, 30)

    data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    return data

input_data = user_input()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High risk of diabetes detected")
    else:
        st.success("✅ Low risk of diabetes")

# -----------------------------
# Simple Visualization
# -----------------------------
st.subheader("📊 Input Overview")

df = pd.DataFrame(input_data, columns=[
    "Pregnancies","Glucose","BP","Skin","Insulin","BMI","DPF","Age"
])

fig = px.bar(df.T, labels={"index":"Features", "value":"Value"})
st.plotly_chart(fig, use_container_width=True)