"""Evaluate both trained models on the hand-labeled out-of-domain test set.

Applies each model's own locked-in default threshold (never tuned on this
set - it's a final generalization check, matching the project rule that
the OOD set must not be used for tuning). Reports an aggregate macro/micro
F1 and exact-match rate per model, next to each model's in-domain test
score, so the generalization gap is visible.

Per-label OOD numbers are also saved for reference, but are explicitly
NOT the headline result: with 50 examples across 28 labels, most labels
have only one or two OOD examples, so per-label precision/recall here is
noisy and should be read qualitatively, not as a reliable estimate.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood-csv", default="data/ood/customer_feedback_template.csv")
    parser.add_argument("--baseline-dir", default="models/baseline_best")
    parser.add_argument("--distilbert-dir", default="models/distilbert_best")
    parser.add_argument("--output-dir", default="results/ood")
    return parser.parse_args()


def parse_label_set(value: str) -> set[str]:
    if not isinstance(value, str) or not value:
        return set()
    return set(value.split("|"))


def main() -> None:
    from goemotions_project.baseline_inference import BaselinePredictor
    from goemotions_project.distilbert_inference import DistilBertPredictor
    from goemotions_project.labels import EMOTION_LABELS, LABEL_TO_ID, NUM_LABELS
    from goemotions_project.metrics import compute_multilabel_metrics

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ood = pd.read_csv(args.ood_csv)
    true_sets = ood["labels"].apply(parse_label_set)
    y_true = np.zeros((len(ood), NUM_LABELS), dtype=int)
    for row_index, labels in enumerate(true_sets):
        for label in labels:
            y_true[row_index, LABEL_TO_ID[label]] = 1

    predictors = {
        "baseline": BaselinePredictor(args.baseline_dir),
        "distilbert": DistilBertPredictor(args.distilbert_dir),
    }
    in_domain_scores = {
        "baseline": json.loads(Path(args.baseline_dir, "model_metadata.json").read_text()),
        "distilbert": json.loads(Path(args.distilbert_dir, "model_metadata.json").read_text()),
    }

    predictions = pd.DataFrame({"id": ood["id"], "text": ood["text"], "true_labels": ood["labels"]})
    summary_rows = []

    for model_name, predictor in predictors.items():
        probabilities = np.zeros((len(ood), NUM_LABELS))
        predicted_label_strings = []
        threshold = None
        for row_index, text in enumerate(ood["text"]):
            result = predictor.predict(text)  # default threshold, not tuned on OOD data
            threshold = result["threshold"]
            for label, probability in result["probabilities"].items():
                probabilities[row_index, LABEL_TO_ID[label]] = probability
            predicted_label_strings.append("|".join(result["labels"]))

        predictions[f"{model_name}_predicted_labels"] = predicted_label_strings
        predictions[f"{model_name}_probabilities"] = [
            json.dumps(
                {label: round(float(p), 6) for label, p in zip(EMOTION_LABELS, row, strict=True)},
                sort_keys=True,
            )
            for row in probabilities
        ]

        metrics = compute_multilabel_metrics(y_true, probabilities, threshold)
        pred_sets = [parse_label_set(s) for s in predicted_label_strings]
        exact_match = sum(t == p for t, p in zip(true_sets, pred_sets, strict=True))

        per_label_path = output_dir / f"{model_name}_ood_per_label.csv"
        pd.DataFrame(
            [{"label": label, **stats} for label, stats in metrics["per_label"].items()]
        ).to_csv(per_label_path, index=False)

        in_domain = in_domain_scores[model_name]
        summary_rows.append(
            {
                "model": model_name,
                "ood_threshold": threshold,
                "ood_macro_f1": metrics["macro_f1"],
                "ood_micro_f1": metrics["micro_f1"],
                "ood_exact_match": exact_match / len(ood),
                "in_domain_test_macro_f1": in_domain["test_macro_f1"],
                "in_domain_test_micro_f1": in_domain["test_micro_f1"],
                "macro_f1_drop": in_domain["test_macro_f1"] - metrics["macro_f1"],
                "micro_f1_drop": in_domain["test_micro_f1"] - metrics["micro_f1"],
            }
        )
        print(
            f"[{model_name}] OOD macro_f1={metrics['macro_f1']:.4f} "
            f"micro_f1={metrics['micro_f1']:.4f} exact_match={exact_match}/{len(ood)} "
            f"(in-domain test macro_f1={in_domain['test_macro_f1']:.4f})"
        )

    predictions.to_csv(output_dir / "ood_predictions.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "ood_summary.csv", index=False)
    print(f"\nSaved OOD comparison to {output_dir}/")


if __name__ == "__main__":
    main()
