"""Word embedding plus MLP baseline."""

import torch
from torch import nn

from goemotions_project.labels import NUM_LABELS


class AverageEmbeddingClassifier(nn.Module):
    """Average non padding token embeddings and classify with an MLP."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_weights: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        if pretrained_weights is not None:
            if pretrained_weights.shape != self.embedding.weight.shape:
                raise ValueError("Pretrained embedding shape does not match the model")
            self.embedding.weight.data.copy_(pretrained_weights)
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_LABELS),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        mask = input_ids.ne(self.padding_idx).unsqueeze(-1)
        summed = (embeddings * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths
        return self.classifier(pooled)

