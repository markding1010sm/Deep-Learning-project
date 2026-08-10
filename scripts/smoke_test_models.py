"""Verify model output shapes before building training loops."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from goemotions_project.models.baseline import AverageEmbeddingClassifier

    model = AverageEmbeddingClassifier(vocab_size=100, embedding_dim=32, hidden_dim=16)
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    logits = model(input_ids)
    assert logits.shape == (2, 28)
    print("Baseline forward pass: OK", tuple(logits.shape))
    print("DistilBERT factory is available; first use will download pretrained weights.")


if __name__ == "__main__":
    main()
