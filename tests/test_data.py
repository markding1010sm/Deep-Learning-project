import pytest

from goemotions_project.data import labels_to_multihot


def test_labels_to_multihot() -> None:
    target = labels_to_multihot([2, 14, 27])
    assert len(target) == 28
    assert sum(target) == 3.0
    assert target[2] == 1.0
    assert target[14] == 1.0
    assert target[27] == 1.0


def test_labels_to_multihot_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        labels_to_multihot([28])

