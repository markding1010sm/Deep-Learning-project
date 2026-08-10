"""Canonical label order for the simplified GoEmotions dataset."""

EMOTION_LABELS: tuple[str, ...] = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)

NUM_LABELS = len(EMOTION_LABELS)
LABEL_TO_ID = {label: index for index, label in enumerate(EMOTION_LABELS)}
ID_TO_LABEL = dict(enumerate(EMOTION_LABELS))

