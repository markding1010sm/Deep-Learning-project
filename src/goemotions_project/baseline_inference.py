"""Load and run the trained GloVe + MLP baseline classifier."""

import json
from pathlib import Path

import torch

from goemotions_project.baseline_data import PAD_ID, encode
from goemotions_project.distilbert_inference import choose_device, format_prediction
from goemotions_project.labels import NUM_LABELS
from goemotions_project.models.baseline import AverageEmbeddingClassifier


class BaselinePredictor:
    """Reusable predictor for CLI and Streamlit inference."""

    def __init__(self, model_dir: str | Path, device: torch.device | None = None) -> None:
        self.model_dir = Path(model_dir)
        metadata_path = self.model_dir / "model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing model metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text())
        self.vocab: dict[str, int] = json.loads((self.model_dir / "vocab.json").read_text())
        self.default_threshold = float(self.metadata["threshold"])
        self.max_length = int(self.metadata["max_length"])
        self.device = device or choose_device()

        self.model = AverageEmbeddingClassifier(
            vocab_size=int(self.metadata["vocab_size"]),
            embedding_dim=int(self.metadata["embedding_dim"]),
            hidden_dim=int(self.metadata["hidden_dim"]),
            dropout=float(self.metadata["dropout"]),
            padding_idx=PAD_ID,
        )
        if self.model.classifier[-1].out_features != NUM_LABELS:
            raise ValueError(f"Expected {NUM_LABELS} labels in the classifier head")
        state_dict = torch.load(
            self.model_dir / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str, threshold: float | None = None) -> dict[str, object]:
        if not text.strip():
            raise ValueError("Text cannot be empty")
        selected_threshold = self.default_threshold if threshold is None else float(threshold)
        input_ids = torch.tensor([encode(text, self.vocab, self.max_length)]).to(self.device)
        logits = self.model(input_ids)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        return format_prediction(probabilities, selected_threshold)
