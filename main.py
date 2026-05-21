import streamlit as st

st.set_page_config(
    page_title="Graduation Project",
    layout="wide",
    page_icon="🧠"
)

# ---------------- BACKGROUND STYLE ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    text-align: center;
    color: white;
    transition: 0.3s;
}

.card:hover {
    transform: scale(1.05);
}

.btn {
    padding: 10px 20px;
    border-radius: 10px;
    border: none;
    cursor: pointer;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🧠 Graduation Project</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI Medical Diagnostic System</div>", unsafe_allow_html=True)

st.write("")

# ---------------- CARDS ----------------
col1, col2, col3, col4 = st.columns(4)

# 🩸 DIABETES
with col1:
    st.markdown("""
    <div class='card' style='background: linear-gradient(135deg,#ef4444,#b91c1c);'>
        <h2>🩸 Diabetes AI</h2>
        <p>Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Open Diabetes"):
        st.switch_page("pages/Ai12.py")

# 🧬 CANCER
with col2:
    st.markdown("""
    <div class='card' style='background: linear-gradient(135deg,#a855f7,#6b21a8);'>
        <h2>🧬 Cancer AI</h2>
        <p>Breast Cancer Detection</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Open Cancer"):
        st.switch_page("pages/Ai.py")

# 🦴 BONE
with col3:
    st.markdown("""
    <div class='card' style='background: linear-gradient(135deg,#64748b,#334155);'>
        <h2>🦴 Bone AI</h2>
        <p>Coming Soon</p>
    </div>
    """, unsafe_allow_html=True)

    st.button("Locked", disabled=True)

# 📘 BOOK
with col4:
    st.markdown("""
    <div class='card' style='background: linear-gradient(135deg,#22c55e,#15803d);'>
        <h2>📘 Documentation</h2>
        <p>Project Report</p>
    </div>
    """, unsafe_allow_html=True)

    st.button("Open Book", disabled=True)