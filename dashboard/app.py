"""Starter Streamlit interface for NALTRA."""

import streamlit as st

MODEL_OPTIONS = [
    "Naive Bayes",
    "SVM",
    "BiLSTM",
    "Multilingual Transformer",
    "Jev",
    "Laya",
    "Ensemble",
]

st.set_page_config(page_title="NALTRA", page_icon="🧭", layout="wide")

st.title("NALTRA")
st.caption("Natural Language Analysis & Taxonomy Robust Architecture")

st.info("This is the initial interface scaffold. Live model prediction is not connected yet.")

left, right = st.columns([2, 1])
with left:
    text = st.text_area(
        "Text",
        placeholder="Enter English, Turkish, or code-switched text…",
        height=180,
    )
with right:
    selected_model = st.selectbox("Model", MODEL_OPTIONS)
    st.markdown("**Planned output**")
    st.write("Labels, hierarchy, confidence, OOD status, latency, and explanation")

if st.button("Analyze", type="primary", disabled=not text.strip()):
    st.warning(f"{selected_model} is not connected yet.")

with st.expander("Research scope"):
    st.markdown(
        "- English, Turkish, and English–Turkish code-switching\n"
        "- Hierarchical multi-label scientific classification on CORDIS H2020\n"
        "- Calibration, robustness, Near-OOD, explainability, and latency evaluation"
    )
