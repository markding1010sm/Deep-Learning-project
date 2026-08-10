"""Run one prediction from the exported DistilBERT checkpoint."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from goemotions_project.distilbert_inference import DistilBertPredictor

    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--model-dir", default="models/distilbert_best")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    predictor = DistilBertPredictor(args.model_dir)
    print(json.dumps(predictor.predict(args.text, args.threshold), indent=2))


if __name__ == "__main__":
    main()
