"""Print a fast first pass summary of the GoEmotions dataset."""

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from goemotions_project.data import load_goemotions
    from goemotions_project.labels import EMOTION_LABELS

    dataset = load_goemotions()
    print("Split sizes")
    for split_name, split in dataset.items():
        print(f"  {split_name}: {len(split):,}")

    counts: Counter[int] = Counter()
    for label_ids in dataset["train"]["labels"]:
        counts.update(label_ids)

    print("\nTraining label counts")
    for label_id, label in enumerate(EMOTION_LABELS):
        print(f"  {label:16s} {counts[label_id]:6d}")

    most_common_id, most_common_n = counts.most_common(1)[0]
    least_common_id, least_common_n = min(counts.items(), key=lambda kv: kv[1])
    print(
        f"\nClass imbalance: '{EMOTION_LABELS[most_common_id]}' ({most_common_n:,}) is "
        f"{most_common_n / least_common_n:.0f}x more frequent than "
        f"'{EMOTION_LABELS[least_common_id]}' ({least_common_n:,})"
    )

    train_labels = dataset["train"]["labels"]
    multi_label_n = sum(1 for label_ids in train_labels if len(label_ids) > 1)
    print(
        f"\nExamples with more than one label: {multi_label_n:,} / {len(train_labels):,} "
        f"({multi_label_n / len(train_labels):.1%})"
    )

    pair_counts: Counter[tuple[str, str]] = Counter()
    for label_ids in train_labels:
        if len(label_ids) < 2:
            continue
        names = sorted(EMOTION_LABELS[i] for i in label_ids)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                pair_counts[(first, second)] += 1

    print("\nTop 10 co-occurring emotion pairs")
    for (first, second), count in pair_counts.most_common(10):
        print(f"  {first} + {second}: {count}")

    lengths = sorted(len(text.split()) for text in dataset["train"]["text"])
    n = len(lengths)

    def percentile(p: float) -> int:
        return lengths[int(p * (n - 1))]

    print("\nComment length in words (train split)")
    print(
        f"  min={lengths[0]}  p50={percentile(0.5)}  p90={percentile(0.9)}  "
        f"p99={percentile(0.99)}  max={lengths[-1]}  mean={sum(lengths) / n:.1f}"
    )


if __name__ == "__main__":
    main()
