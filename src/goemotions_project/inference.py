"""Model independent inference helpers."""

from collections.abc import Sequence

from goemotions_project.labels import EMOTION_LABELS, NUM_LABELS


def select_labels(
    probabilities: Sequence[float], threshold: float = 0.5
) -> list[tuple[str, float]]:
    """Return labels at or above the threshold, ordered by probability."""
    if len(probabilities) != NUM_LABELS:
        raise ValueError(f"Expected {NUM_LABELS} probabilities, received {len(probabilities)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1")

    selected = [
        (label, float(probability))
        for label, probability in zip(EMOTION_LABELS, probabilities, strict=True)
        if probability >= threshold
    ]
    return sorted(selected, key=lambda item: item[1], reverse=True)

