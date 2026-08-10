"""Streamlit application for the customer feedback emotion project."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goemotions_project.distilbert_inference import DistilBertPredictor  # noqa: E402
from goemotions_project.labels import EMOTION_LABELS  # noqa: E402

MODEL_DIR = ROOT / "models" / "distilbert_best"
MODEL_WEIGHTS = MODEL_DIR / "model.safetensors"
SUMMARY_PATH = ROOT / "results" / "distilbert" / "experiment_summary.csv"


@st.cache_resource
def load_distilbert(model_dir: str) -> DistilBertPredictor:
    return DistilBertPredictor(model_dir)


def read_default_threshold() -> float:
    metadata_path = MODEL_DIR / "model_metadata.json"
    if not metadata_path.exists():
        return 0.5
    return round(float(json.loads(metadata_path.read_text())["threshold"]), 2)


def model_weights_available() -> bool:
    """Reject missing files and unpulled Git LFS pointer files."""
    return MODEL_WEIGHTS.exists() and MODEL_WEIGHTS.stat().st_size > 1_000_000

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
    threshold = st.slider(
        "Prediction threshold",
        0.05,
        0.95,
        read_default_threshold(),
        0.05,
    )

    if st.button("Analyze", type="primary"):
        if model_name == "Word Embedding + MLP":
            st.warning("The baseline checkpoint has not been connected yet.")
        elif not model_weights_available():
            st.error(
                "The trained DistilBERT weights are missing or are only a Git LFS pointer. Run "
                "`git lfs install` and `git lfs pull`, or run the training command in the README."
            )
        elif not text.strip():
            st.error("Enter English text before running the analysis.")
        else:
            with st.spinner("Analyzing emotion..."):
                predictor = load_distilbert(str(MODEL_DIR))
                result = predictor.predict(text, threshold)
            if result["labels"]:
                st.success("Predicted emotions: " + ", ".join(result["labels"]))
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Selected emotion": label,
                                "Confidence": result["probabilities"][label],
                            }
                            for label in result["labels"]
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No emotion exceeded the selected threshold.")
            probabilities = pd.DataFrame(
                [
                    {"Emotion": label, "Probability": probability}
                    for label, probability in result["probabilities"].items()
                ]
            ).sort_values("Probability", ascending=False)
            st.bar_chart(
                probabilities.head(10).set_index("Emotion"),
                horizontal=True,
            )
            st.dataframe(
                probabilities.head(10),
                hide_index=True,
                use_container_width=True,
            )

with results_tab:
    st.subheader("Experiment comparison")
    if SUMMARY_PATH.exists():
        summary = pd.read_csv(SUMMARY_PATH)
        st.dataframe(summary, hide_index=True, use_container_width=True)
    else:
        st.info("Run the DistilBERT experiments to generate model results.")

with data_tab:
    st.subheader("GoEmotions labels")
    st.write(", ".join(EMOTION_LABELS))
    st.caption(
        "The production prototype analyzes emotion expressed in text. It does not determine a "
        "person's mental state and should support, not replace, human review."
    )
