import streamlit as st

st.set_page_config(page_title="Graduation Project", layout="wide")

# ---------------- HEADER ----------------
st.markdown("""
    <h1 style='text-align:center; color:#b71c1c;'>
        🧠 Graduation Project Dashboard
    </h1>
    <h4 style='text-align:center; color:gray;'>
        AI Medical Diagnostic System
    </h4>
    <hr>
""", unsafe_allow_html=True)

# ---------------- CARDS ----------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 🩸 Diabetes Model")
    st.info("Predict diabetes using AI model")
    if st.button("Open Diabetes"):
        st.switch_page("pages/diabetes.py")

with col2:
    st.markdown("### 🧬 Cancer Model")
    st.info("AI Cancer detection system")
    if st.button("Open Cancer"):
        st.switch_page("pages/cancer.py")

with col3:
    st.markdown("### 🦴 Bone Fracture")
    st.info("X-ray fracture detection (Coming soon)")
    if st.button("Open Bone Model"):
        st.warning("Not available yet")

with col4:
    st.markdown("### 📘 Project Book")
    st.info("Full documentation & report")
    if st.button("Open Book"):
        st.switch_page("pages/book.py")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center;'>© 2026 Graduation Project - AI Medical System</p>
""", unsafe_allow_html=True)