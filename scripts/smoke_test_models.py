"""Verify model output shapes before building training loops."""

import torch

from goemotions_project.models.baseline import AverageEmbeddingClassifier


def main() -> None:
    model = AverageEmbeddingClassifier(vocab_size=100, embedding_dim=32, hidden_dim=16)
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    logits = model(input_ids)
    assert logits.shape == (2, 28)
    print("Baseline forward pass: OK", tuple(logits.shape))
    print("DistilBERT factory is available; first use will download pretrained weights.")


if __name__ == "__main__":
    main()

