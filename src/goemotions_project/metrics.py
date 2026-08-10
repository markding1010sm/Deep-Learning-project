"""Metrics for multi label classification."""

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from goemotions_project.labels import EMOTION_LABELS


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for saved model logits."""
    positive = logits >= 0
    negative = ~positive
    output = np.empty_like(logits, dtype=float)
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_values = np.exp(logits[negative])
    output[negative] = exp_values / (1.0 + exp_values)
    return output


def compute_multilabel_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute summary and per label metrics at a fixed threshold."""
    y_pred = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    per_label = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(EMOTION_LABELS)
    }

    return {
        "threshold": threshold,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_label": per_label,
    }

