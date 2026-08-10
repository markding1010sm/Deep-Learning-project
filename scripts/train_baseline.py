"""Train and evaluate the configured GloVe + MLP baseline experiments."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.toml")
    parser.add_argument("--run-root", default=".training_runs/baseline")
    parser.add_argument("--results-dir", default="results/baseline")
    parser.add_argument("--final-model-dir", default="models/baseline_best")
    parser.add_argument("--experiments", nargs="+")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


def main() -> None:
    from goemotions_project.baseline_training import run_all_experiments

    args = parse_args()
    result = run_all_experiments(
        config_path=args.config,
        run_root=args.run_root,
        results_dir=args.results_dir,
        final_model_dir=args.final_model_dir,
        selected_experiments=args.experiments,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_test_samples=args.max_test_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
