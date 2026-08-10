"""Print a fast first pass summary of the GoEmotions dataset."""

from collections import Counter

from goemotions_project.data import load_goemotions
from goemotions_project.labels import EMOTION_LABELS


def main() -> None:
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


if __name__ == "__main__":
    main()

