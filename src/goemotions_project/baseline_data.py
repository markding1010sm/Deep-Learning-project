"""Vocabulary, GloVe embeddings, and DataLoader helpers for the baseline model."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from goemotions_project.data import labels_to_multihot

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1

_TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphabetic-only tokenizer shared by vocab building and encoding."""
    return _TOKEN_RE.findall(text.lower())


def build_vocab(
    texts: Iterable[str],
    max_vocab_size: int,
    min_token_freq: int,
) -> dict[str, int]:
    """Build a word to id mapping from training text only, to avoid leakage."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))

    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    frequent = [word for word, count in counts.most_common() if count >= min_token_freq]
    for word in frequent[: max_vocab_size - len(vocab)]:
        vocab[word] = len(vocab)
    return vocab


def load_glove_embeddings(
    vocab: dict[str, int],
    embedding_dim: int,
    seed: int,
) -> tuple[torch.Tensor, float]:
    """Build an embedding matrix aligned to `vocab`, backed by pretrained GloVe vectors.

    Words missing from GloVe (including <pad>/<unk>) get a small random vector so
    the model can still learn something for them; <pad> is zeroed so it never
    contributes to the mean-pooled representation. Returns the matrix and the
    fraction of vocab words actually found in GloVe, for reporting.
    """
    import gensim.downloader as api

    glove = api.load(f"glove-wiki-gigaword-{embedding_dim}")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
    matrix[PAD_ID] = 0.0

    hits = 0
    for word, index in vocab.items():
        if word in glove:
            matrix[index] = glove[word]
            hits += 1
    coverage = hits / len(vocab)
    print(f"GloVe coverage: {hits:,} / {len(vocab):,} vocab words found ({coverage:.1%})")
    return torch.tensor(matrix), coverage


def encode(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    """Tokenize, numericalize, and pad/truncate a single text to a fixed length."""
    ids = [vocab.get(token, UNK_ID) for token in tokenize(text)][:max_length]
    ids.extend([PAD_ID] * (max_length - len(ids)))
    return ids


class BaselineDataset(Dataset):
    """Pre-encodes text/labels once so training epochs only index tensors."""

    def __init__(
        self,
        texts: Sequence[str],
        label_rows: Sequence[Sequence[int]],
        vocab: dict[str, int],
        max_length: int,
    ) -> None:
        self.input_ids = torch.tensor(
            [encode(text, vocab, max_length) for text in texts],
            dtype=torch.long,
        )
        self.labels = torch.stack([labels_to_multihot_tensor(row) for row in label_rows])

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.labels[index]


def labels_to_multihot_tensor(label_ids: Iterable[int]) -> torch.Tensor:
    return torch.tensor(labels_to_multihot(label_ids), dtype=torch.float32)


def build_dataloader(
    dataset: BaselineDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a deterministic DataLoader, matching the DistilBERT track's convention."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )
