"""Streamlit shell for the customer feedback emotion application."""

from pathlib import Path

import pandas as pd
import streamlit as st

from goemotions_project.labels import EMOTION_LABELS

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Customer Feedback Emotion Intelligence", layout="wide")
st.title("Customer Feedback Emotion Intelligence")
st.caption(
    "A GoEmotions project comparing a word embedding neural network with fine tuned DistilBERT."
)

demo_tab, results_tab, data_tab = st.tabs(["Live demo", "Model results", "Dataset"])

with demo_tab:
    st.subheader("Analyze one message")
    text = st.text_area(
        "English text",
        value="I finally finished the project, but I am still nervous about presenting it.",
        height=120,
    )
    model_name = st.selectbox("Model", ["DistilBERT", "Word Embedding + MLP"])
    threshold = st.slider("Prediction threshold", 0.05, 0.95, 0.50, 0.05)

    if st.button("Analyze", type="primary"):
        st.warning(
            "The interface is ready, but no trained checkpoint is connected yet. "
            "Add the final inference pipeline after training the models."
        )
        st.code(
            f"predict(text={text!r}, model={model_name!r}, threshold={threshold:.2f})",
            language="python",
        )

with results_tab:
    st.subheader("Experiment comparison")
    st.info("Final macro F1, micro F1, timing, and per label results will appear here.")
    placeholder = pd.DataFrame(
        [
            {"Model": "Word Embedding + MLP", "Macro F1": None, "Micro F1": None},
            {"Model": "DistilBERT", "Macro F1": None, "Micro F1": None},
        ]
    )
    st.dataframe(placeholder, hide_index=True, use_container_width=True)

with data_tab:
    st.subheader("GoEmotions labels")
    st.write(", ".join(EMOTION_LABELS))
    st.caption(
        "The production prototype analyzes emotion expressed in text. It does not determine a "
        "person's mental state and should support, not replace, human review."
    )

