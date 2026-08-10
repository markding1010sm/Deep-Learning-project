"""Load and run the trained DistilBERT emotion classifier."""

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from goemotions_project.inference import select_labels
from goemotions_project.labels import EMOTION_LABELS, NUM_LABELS


def choose_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_prediction(
    probabilities: Sequence[float],
    threshold: float,
) -> dict[str, object]:
    """Convert probabilities into the public prediction response."""
    selected = select_labels(probabilities, threshold)
    return {
        "labels": [label for label, _ in selected],
        "probabilities": {
            label: float(probability)
            for label, probability in zip(EMOTION_LABELS, probabilities, strict=True)
        },
        "threshold": float(threshold),
    }


class DistilBertPredictor:
    """Reusable predictor for CLI and Streamlit inference."""

    def __init__(self, model_dir: str | Path, device: torch.device | None = None) -> None:
        self.model_dir = Path(model_dir)
        metadata_path = self.model_dir / "model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing model metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text())
        self.default_threshold = float(self.metadata["threshold"])
        self.max_length = int(self.metadata["max_length"])
        self.device = device or choose_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            attn_implementation="eager",
        )
        if self.model.config.num_labels != NUM_LABELS:
            raise ValueError(f"Expected {NUM_LABELS} labels, found {self.model.config.num_labels}")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str, threshold: float | None = None) -> dict[str, object]:
        if not text.strip():
            raise ValueError("Text cannot be empty")
        selected_threshold = self.default_threshold if threshold is None else float(threshold)
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        logits = self.model(**inputs).logits
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        return format_prediction(probabilities, selected_threshold)
