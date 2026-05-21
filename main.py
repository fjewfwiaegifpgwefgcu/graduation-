import streamlit as st

st.set_page_config(page_title="Graduation Project", layout="wide")

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center; color:#b71c1c;'>🧠 Graduation Project</h1>
<h4 style='text-align:center; color:gray;'>AI Medical Diagnostic System</h4>
<hr>
""", unsafe_allow_html=True)

# ---------------- CARDS ----------------
col1, col2, col3, col4 = st.columns(4)

# 🩸 DIABETES
with col1:
    st.markdown("### 🩸 Diabetes Model")
    st.info("AI System for Diabetes Prediction")
    if st.button("Open Diabetes"):
        st.switch_page("pages/Ai12.py")

# 🧬 CANCER
with col2:
    st.markdown("### 🧬 Cancer Model")
    st.info("AI Breast Cancer Detection")
    if st.button("Open Cancer"):
        st.switch_page("pages/Ai.py")

# 🦴 BONE FRACTURE (COMING SOON)
with col3:
    st.markdown("### 🦴 Bone Fracture")
    st.warning("Coming Soon...")
    st.button("Not Available", disabled=True)

# 📘 BOOK
with col4:
    st.markdown("### 📘 Project Book")
    st.info("Documentation & Report")
    if st.button("Open Book"):
        st.warning("Still Under Development")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center;'>© 2026 Graduation AI Project</p>
""", unsafe_allow_html=True)