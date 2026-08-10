"""Model definitions."""

from goemotions_project.models.baseline import AverageEmbeddingClassifier
from goemotions_project.models.distilbert import build_distilbert_classifier

__all__ = ["AverageEmbeddingClassifier", "build_distilbert_classifier"]

