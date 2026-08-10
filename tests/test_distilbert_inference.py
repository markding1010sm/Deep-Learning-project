import pytest

from goemotions_project.distilbert_inference import format_prediction


def test_format_prediction_uses_public_schema() -> None:
    probabilities = [0.0] * 28
    probabilities[13] = 0.8
    probabilities[19] = 0.6
    result = format_prediction(probabilities, threshold=0.5)
    assert result["labels"] == ["excitement", "nervousness"]
    assert result["probabilities"]["excitement"] == pytest.approx(0.8)
    assert result["threshold"] == pytest.approx(0.5)
