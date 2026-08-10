import json

import torch

from goemotions_project.baseline_inference import BaselinePredictor
from goemotions_project.labels import EMOTION_LABELS
from goemotions_project.models.baseline import AverageEmbeddingClassifier


def test_baseline_predictor_loads_checkpoint_and_uses_default_threshold(tmp_path) -> None:
    model = AverageEmbeddingClassifier(
        vocab_size=3,
        embedding_dim=4,
        hidden_dim=3,
        dropout=0.0,
    )
    torch.save(model.state_dict(), tmp_path / "model.pt")
    (tmp_path / "vocab.json").write_text(json.dumps({"<pad>": 0, "<unk>": 1, "hello": 2}))
    (tmp_path / "model_metadata.json").write_text(
        json.dumps(
            {
                "threshold": 0.4,
                "max_length": 5,
                "vocab_size": 3,
                "embedding_dim": 4,
                "hidden_dim": 3,
                "dropout": 0.0,
            }
        )
    )

    result = BaselinePredictor(tmp_path, device=torch.device("cpu")).predict("hello")

    assert result["threshold"] == 0.4
    assert list(result["probabilities"]) == list(EMOTION_LABELS)
    assert all(0.0 <= probability <= 1.0 for probability in result["probabilities"].values())
