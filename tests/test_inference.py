import pytest

from goemotions_project.inference import select_labels


def test_select_labels_orders_predictions() -> None:
    probabilities = [0.0] * 28
    probabilities[14] = 0.72
    probabilities[23] = 0.81

    assert select_labels(probabilities, threshold=0.5) == [
        ("relief", 0.81),
        ("fear", 0.72),
    ]


def test_select_labels_validates_shape() -> None:
    with pytest.raises(ValueError):
        select_labels([0.5])

