"""Streamlit application for the customer feedback emotion project."""

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goemotions_project.baseline_inference import BaselinePredictor  # noqa: E402
from goemotions_project.distilbert_inference import DistilBertPredictor  # noqa: E402
from goemotions_project.labels import EMOTION_LABELS  # noqa: E402

DISTILBERT_NAME = "DistilBERT"
BASELINE_NAME = "GloVe + MLP"
MODEL_DIRS = {
    DISTILBERT_NAME: ROOT / "models" / "distilbert_best",
    BASELINE_NAME: ROOT / "models" / "baseline_best",
}
MODEL_WEIGHTS = {
    DISTILBERT_NAME: MODEL_DIRS[DISTILBERT_NAME] / "model.safetensors",
    BASELINE_NAME: MODEL_DIRS[BASELINE_NAME] / "model.pt",
}
SUMMARY_PATHS = {
    DISTILBERT_NAME: ROOT / "results" / "distilbert" / "experiment_summary.csv",
    BASELINE_NAME: ROOT / "results" / "baseline" / "experiment_summary.csv",
}


@st.cache_resource
def load_distilbert(model_dir: str) -> DistilBertPredictor:
    return DistilBertPredictor(model_dir)


@st.cache_resource
def load_baseline(model_dir: str) -> BaselinePredictor:
    return BaselinePredictor(model_dir)


def read_default_threshold(model_name: str) -> float:
    metadata_path = MODEL_DIRS[model_name] / "model_metadata.json"
    if not metadata_path.exists():
        return 0.5
    return round(float(json.loads(metadata_path.read_text())["threshold"]), 2)


def model_weights_available(model_name: str) -> bool:
    """Reject missing files and an unpulled DistilBERT Git LFS pointer."""
    weights_path = MODEL_WEIGHTS[model_name]
    return weights_path.exists() and weights_path.stat().st_size > 1_000_000


def load_predictor(model_name: str) -> DistilBertPredictor | BaselinePredictor:
    model_dir = str(MODEL_DIRS[model_name])
    if model_name == DISTILBERT_NAME:
        return load_distilbert(model_dir)
    return load_baseline(model_dir)


def missing_model_message(model_name: str) -> str:
    if model_name == DISTILBERT_NAME:
        return (
            "The trained DistilBERT weights are missing or are only a Git LFS pointer. Run "
            "`git lfs install` and `git lfs pull`, or run the training command in the README."
        )
    return (
        "The trained Baseline checkpoint is missing. Pull the latest repository changes or run "
        "the Baseline training command in the README."
    )


def prediction_table(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Emotion": label, "Probability": probability}
            for label, probability in result["probabilities"].items()
        ]
    ).sort_values("Probability", ascending=False)

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
    model_name = st.selectbox("Model", [DISTILBERT_NAME, BASELINE_NAME])
    threshold = st.slider(
        "Prediction threshold",
        0.05,
        0.95,
        read_default_threshold(model_name),
        0.05,
        key=f"threshold_{model_name}",
    )

    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.error("Enter English text before running the analysis.")
        elif not model_weights_available(model_name):
            st.error(missing_model_message(model_name))
        else:
            with st.spinner("Analyzing emotion..."):
                predictor = load_predictor(model_name)
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
                    width="stretch",
                )
            else:
                st.info("No emotion exceeded the selected threshold.")
            probabilities = prediction_table(result)
            st.bar_chart(
                probabilities.head(10).set_index("Emotion"),
                horizontal=True,
            )
            st.dataframe(
                probabilities.head(10),
                hide_index=True,
                width="stretch",
            )

with results_tab:
    st.subheader("Experiment comparison")
    summaries = []
    for model_name, summary_path in SUMMARY_PATHS.items():
        if summary_path.exists():
            model_summary = pd.read_csv(summary_path)
            model_summary.insert(0, "model", model_name)
            summaries.append(model_summary)
    if summaries:
        summary = pd.concat(summaries, ignore_index=True)
        st.dataframe(summary, hide_index=True, width="stretch")
    else:
        st.info("Run the model experiments to generate comparison results.")

with data_tab:
    st.subheader("GoEmotions labels")
    st.write(", ".join(EMOTION_LABELS))
    st.caption(
        "The production prototype analyzes emotion expressed in text. It does not determine a "
        "person's mental state and should support, not replace, human review."
    )
