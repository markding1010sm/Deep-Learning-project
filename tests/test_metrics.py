import numpy as np
import pytest

from goemotions_project.distilbert_training import select_best_experiment
from goemotions_project.metrics import find_best_threshold


def test_find_best_threshold_uses_macro_f1() -> None:
    y_true = np.zeros((2, 28), dtype=int)
    y_true[0, 0] = 1
    y_true[1, 1] = 1
    probabilities = np.zeros((2, 28), dtype=float)
    probabilities[0, 0] = 0.45
    probabilities[1, 1] = 0.45
    probabilities[0, 1] = 0.35

    best, curve = find_best_threshold(y_true, probabilities, [0.3, 0.4, 0.5])

    assert best["threshold"] == pytest.approx(0.4)
    assert len(curve) == 3


def test_find_best_threshold_requires_candidates() -> None:
    with pytest.raises(ValueError):
        find_best_threshold(np.zeros((1, 28)), np.zeros((1, 28)), [])


def test_experiment_selection_ignores_test_metrics() -> None:
    rows = [
        {
            "experiment": "better_validation",
            "validation_macro_f1": 0.5,
            "validation_micro_f1": 0.6,
            "threshold": 0.4,
            "test_macro_f1": 0.0,
        },
        {
            "experiment": "better_test",
            "validation_macro_f1": 0.4,
            "validation_micro_f1": 0.7,
            "threshold": 0.5,
            "test_macro_f1": 1.0,
        },
    ]

    assert select_best_experiment(rows)["experiment"] == "better_validation"
