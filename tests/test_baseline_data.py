import torch

from goemotions_project.baseline_data import (
    PAD_ID,
    UNK_ID,
    BaselineDataset,
    build_vocab,
    encode,
    tokenize,
)


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Wait, why?! I'm confused.") == ["wait", "why", "i'm", "confused"]


def test_build_vocab_drops_rare_words_and_reserves_special_tokens() -> None:
    texts = ["cat cat cat", "dog dog", "rare"]
    vocab = build_vocab(texts, max_vocab_size=10, min_token_freq=2)

    assert vocab["<pad>"] == PAD_ID
    assert vocab["<unk>"] == UNK_ID
    assert "cat" in vocab
    assert "dog" in vocab
    assert "rare" not in vocab


def test_build_vocab_respects_max_size() -> None:
    texts = ["a b c d e"] * 5
    vocab = build_vocab(texts, max_vocab_size=4, min_token_freq=1)
    assert len(vocab) == 4


def test_encode_pads_and_truncates_to_fixed_length() -> None:
    vocab = {"<pad>": 0, "<unk>": 1, "hello": 2, "world": 3}

    short = encode("hello world", vocab, max_length=5)
    assert short == [2, 3, 0, 0, 0]

    long = encode("hello world hello world hello", vocab, max_length=3)
    assert long == [2, 3, 2]


def test_encode_maps_unknown_words_to_unk() -> None:
    vocab = {"<pad>": 0, "<unk>": 1, "hello": 2}
    assert encode("hello mystery", vocab, max_length=2) == [2, UNK_ID]


def test_baseline_dataset_returns_input_ids_and_multihot_labels() -> None:
    vocab = {"<pad>": 0, "<unk>": 1, "hello": 2}
    dataset = BaselineDataset(
        texts=["hello", "mystery"],
        label_rows=[[0, 27], [4]],
        vocab=vocab,
        max_length=3,
    )
    assert len(dataset) == 2
    input_ids, labels = dataset[0]
    assert torch.equal(input_ids, torch.tensor([2, 0, 0]))
    assert labels.shape == (28,)
    assert labels.sum().item() == 2
