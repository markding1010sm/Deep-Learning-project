"""Dataset loading and label encoding helpers."""

from collections.abc import Iterable

from goemotions_project.labels import NUM_LABELS

DATASET_NAME = "google-research-datasets/go_emotions"
DATASET_CONFIG = "simplified"


def load_goemotions():
    """Load the official simplified GoEmotions splits from Hugging Face."""
    from datasets import load_dataset

    return load_dataset(DATASET_NAME, DATASET_CONFIG)


def labels_to_multihot(label_ids: Iterable[int]) -> list[float]:
    """Convert a collection of label ids into a 28 element multi hot vector."""
    target = [0.0] * NUM_LABELS
    for label_id in label_ids:
        if not 0 <= label_id < NUM_LABELS:
            raise ValueError(f"Label id {label_id} is outside [0, {NUM_LABELS - 1}]")
        target[label_id] = 1.0
    return target


def add_multihot_targets(dataset):
    """Add a `target` field that is ready for BCE based training."""

    def encode(example):
        return {"target": labels_to_multihot(example["labels"])}

    return dataset.map(encode)

