import torch
from datasets import Dataset, DatasetDict

from goemotions_project.distilbert_data import compute_pos_weight, tokenize_dataset


class TinyTokenizer:
    def __call__(self, texts, truncation, max_length):
        assert truncation is True
        return {
            "input_ids": [
                [index + 1] * min(len(text), max_length)
                for index, text in enumerate(texts)
            ],
            "attention_mask": [[1] * min(len(text), max_length) for text in texts],
        }


def test_tokenized_labels_are_28_dimension_float_tensors() -> None:
    split = Dataset.from_dict({"text": ["hi", "hello"], "labels": [[0, 27], [4]]})
    tokenized = tokenize_dataset(
        DatasetDict(train=split, validation=split, test=split),
        TinyTokenizer(),
        max_length=64,
    )

    tokenized.set_format("torch")
    labels = tokenized["train"][:]["labels"]
    assert labels.shape == (2, 28)
    assert labels.dtype == torch.float32


def test_pos_weight_is_capped_and_uses_all_labels() -> None:
    rows = [[0], [0], [1], [27], [0, 1]]
    weights = compute_pos_weight(rows, cap=3.0)
    assert weights.shape == (28,)
    assert weights.dtype == torch.float32
    assert float(weights.min()) >= 1.0
    assert float(weights.max()) <= 3.0
