"""Tokenization and DataLoader helpers for DistilBERT training."""

from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from goemotions_project.data import labels_to_multihot
from goemotions_project.labels import NUM_LABELS


class MultilabelDataCollator:
    """Dynamically pad tokens and keep labels as float multi-hot tensors."""

    def __init__(self, tokenizer) -> None:
        self.base_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.float32)
        model_features = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = self.base_collator(model_features)
        batch["labels"] = labels
        return batch


def tokenize_dataset(dataset, tokenizer, max_length: int):
    """Tokenize every split while preserving 28-dimensional float labels."""
    from datasets import List, Value

    def preprocess(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=max_length)
        encoded["labels"] = [labels_to_multihot(label_ids) for label_ids in batch["labels"]]
        return encoded

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing GoEmotions",
    )
    label_feature = List(Value("float32"), length=NUM_LABELS)
    for split_name in tokenized:
        tokenized[split_name] = tokenized[split_name].cast_column("labels", label_feature)
    return tokenized


def limit_splits(dataset, train: int | None, validation: int | None, test: int | None):
    """Optionally select deterministic prefixes for smoke tests."""
    from datasets import DatasetDict

    limits = {"train": train, "validation": validation, "test": test}
    output = DatasetDict(dataset)
    for split_name, limit in limits.items():
        if limit is not None:
            output[split_name] = output[split_name].select(
                range(min(limit, len(output[split_name])))
            )
    return output


def build_dataloader(
    split,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a deterministic DataLoader with dynamic padding."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        split,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=MultilabelDataCollator(tokenizer),
        generator=generator,
        num_workers=0,
    )


def compute_pos_weight(
    label_rows: Iterable[Iterable[int]],
    cap: float = 10.0,
) -> torch.Tensor:
    """Compute capped square-root positive weights from training labels only."""
    positives = np.zeros(NUM_LABELS, dtype=np.float64)
    total = 0
    for label_ids in label_rows:
        total += 1
        for label_id in label_ids:
            positives[label_id] += 1
    if total == 0:
        raise ValueError("Cannot compute class weights from an empty training set")
    negatives = total - positives
    safe_positives = np.maximum(positives, 1.0)
    weights = np.sqrt(negatives / safe_positives)
    weights = np.clip(weights, 1.0, cap)
    return torch.tensor(weights, dtype=torch.float32)
