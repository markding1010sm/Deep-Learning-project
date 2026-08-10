"""Summarize common misclassification patterns for a trained baseline experiment.

Reads `results/baseline/<experiment>/test_predictions.csv` (produced by
`train_baseline.py`) and reports which labels are most often missed or
over-predicted, which false-negative/false-positive labels tend to
co-occur on the same wrong example, and a few concrete examples, to
support the "common errors" part of the baseline evaluation.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_labels(value: str) -> set[str]:
    if not isinstance(value, str) or not value:
        return set()
    return set(value.split("|"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/baseline")
    parser.add_argument("--experiment")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    experiment = args.experiment
    if experiment is None:
        summary = json.loads((results_dir / "run_summary.json").read_text())
        experiment = summary["best_experiment"]

    predictions = pd.read_csv(results_dir / experiment / "test_predictions.csv")
    true_sets = predictions["true_labels"].apply(parse_labels)
    pred_sets = predictions["predicted_labels"].apply(parse_labels)

    exact_match = sum(t == p for t, p in zip(true_sets, pred_sets, strict=True))
    print(f"Experiment: {experiment}")
    print(
        f"Exact label-set match: {exact_match:,} / {len(predictions):,} "
        f"({exact_match / len(predictions):.1%})"
    )

    missed: Counter[str] = Counter()  # true label the model failed to predict
    spurious: Counter[str] = Counter()  # predicted label that was not true
    # Cartesian product of an example's missed and spurious labels: this is NOT
    # a one-to-one "true label X got confused for predicted label Y" count (that
    # would require single-label examples). It just counts how often a missed
    # label and a spurious label show up together on the same wrong example.
    error_cooccurrence: Counter[tuple[str, str]] = Counter()
    for true_labels, pred_labels in zip(true_sets, pred_sets, strict=True):
        example_missed = true_labels - pred_labels
        example_spurious = pred_labels - true_labels
        missed.update(example_missed)
        spurious.update(example_spurious)
        for missed_label in example_missed:
            for spurious_label in example_spurious:
                error_cooccurrence[(missed_label, spurious_label)] += 1

    print("\nMost frequently missed labels (false negatives)")
    for label, count in missed.most_common(10):
        print(f"  {label:16s} {count:5d}")

    print("\nMost frequently over-predicted labels (false positives)")
    for label, count in spurious.most_common(10):
        print(f"  {label:16s} {count:5d}")

    print("\nTop 10 co-occurring false-negative/false-positive label pairs")
    print("(missed label + spurious label on the same wrong example, not a 1:1 confusion)")
    for (missed_label, spurious_label), count in error_cooccurrence.most_common(10):
        print(f"  missed {missed_label!r} with spurious {spurious_label!r}: {count}")

    print(f"\n{args.examples} example misclassifications")
    wrong = predictions[[t != p for t, p in zip(true_sets, pred_sets, strict=True)]]
    for _, row in wrong.head(args.examples).iterrows():
        print(f"  text: {row['text']!r}")
        print(f"    true:      {row['true_labels'] or '(none)'}")
        print(f"    predicted: {row['predicted_labels'] or '(none)'}")


if __name__ == "__main__":
    main()
