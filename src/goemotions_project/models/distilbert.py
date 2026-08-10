"""DistilBERT model factory."""

from goemotions_project.labels import ID_TO_LABEL, LABEL_TO_ID, NUM_LABELS

DEFAULT_CHECKPOINT = "distilbert/distilbert-base-uncased"


def build_distilbert_classifier(checkpoint: str = DEFAULT_CHECKPOINT):
    """Create a DistilBERT classifier configured for 28 independent labels."""
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        attn_implementation="eager",
    )
