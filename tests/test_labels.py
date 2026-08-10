from goemotions_project.labels import EMOTION_LABELS, NUM_LABELS


def test_label_inventory() -> None:
    assert NUM_LABELS == 28
    assert len(set(EMOTION_LABELS)) == NUM_LABELS
    assert EMOTION_LABELS[-1] == "neutral"

