"""Print a fast first pass summary of the GoEmotions dataset and save EDA plots.

Saves three artifacts to `--output-dir` (default `results/eda/`):
- `label_counts.png`: per-class bar chart (class imbalance, at a glance)
- `length_histogram.png`: comment length distribution, in words
- `cooccurrence_heatmap.png` + `cooccurrence_top_pairs.csv`: which emotions
  get labeled together on the same comment
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/eda")
    return parser.parse_args()


def save_label_counts_plot(counts: Counter, emotion_labels, output_dir: Path) -> None:
    ordered = sorted(
        ((emotion_labels[i], counts[i]) for i in range(len(emotion_labels))),
        key=lambda pair: pair[1],
    )
    labels, values = zip(*ordered, strict=True)
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(labels, values, color="#4C72B0")
    ax.set_xlabel("Training examples")
    ax.set_title("GoEmotions: training label counts (class imbalance)")
    fig.tight_layout()
    fig.savefig(output_dir / "label_counts.png", dpi=150)
    plt.close(fig)


def save_length_histogram(lengths: list[int], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=30, color="#55A868")
    ax.set_xlabel("Comment length (words)")
    ax.set_ylabel("Training examples")
    ax.set_title("GoEmotions: comment length distribution (train split)")
    fig.tight_layout()
    fig.savefig(output_dir / "length_histogram.png", dpi=150)
    plt.close(fig)


def save_cooccurrence(
    train_labels,
    emotion_labels,
    output_dir: Path,
) -> Counter[tuple[str, str]]:
    num_labels = len(emotion_labels)
    matrix = [[0] * num_labels for _ in range(num_labels)]
    pair_counts: Counter[tuple[str, str]] = Counter()
    for label_ids in train_labels:
        if len(label_ids) < 2:
            continue
        for i, first_id in enumerate(label_ids):
            for second_id in label_ids[i + 1 :]:
                matrix[first_id][second_id] += 1
                matrix[second_id][first_id] += 1
        names = sorted(emotion_labels[i] for i in label_ids)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                pair_counts[(first, second)] += 1

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(num_labels))
    ax.set_xticklabels(emotion_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(num_labels))
    ax.set_yticklabels(emotion_labels, fontsize=6)
    ax.set_title("GoEmotions: label co-occurrence counts (train split)")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "cooccurrence_heatmap.png", dpi=150)
    plt.close(fig)

    pairs_df = pd.DataFrame(
        [
            {"label_a": first, "label_b": second, "count": count}
            for (first, second), count in pair_counts.most_common()
        ]
    )
    pairs_df.to_csv(output_dir / "cooccurrence_top_pairs.csv", index=False)
    return pair_counts


def main() -> None:
    from goemotions_project.data import load_goemotions
    from goemotions_project.labels import EMOTION_LABELS

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    save_label_counts_plot(counts, EMOTION_LABELS, output_dir)

    train_labels = dataset["train"]["labels"]
    multi_label_n = sum(1 for label_ids in train_labels if len(label_ids) > 1)
    print(
        f"\nExamples with more than one label: {multi_label_n:,} / {len(train_labels):,} "
        f"({multi_label_n / len(train_labels):.1%})"
    )

    pair_counts = save_cooccurrence(train_labels, EMOTION_LABELS, output_dir)
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
    save_length_histogram(lengths, output_dir)

    print(f"\nSaved plots and co-occurrence table to {output_dir}/")


if __name__ == "__main__":
    main()
