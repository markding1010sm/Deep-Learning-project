"""End-to-end GloVe + MLP baseline experiment runner.

Mirrors the structure and output layout of `distilbert_training.py` (same
config shape, same results/ file names, same validation-only checkpoint
selection and threshold search) so the two tracks are directly comparable.
"""

from __future__ import annotations

import json
import platform
import shutil
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from goemotions_project.baseline_data import (
    PAD_ID,
    BaselineDataset,
    build_dataloader,
    build_vocab,
    load_glove_embeddings,
)
from goemotions_project.data import load_goemotions
from goemotions_project.distilbert_data import compute_pos_weight
from goemotions_project.distilbert_inference import choose_device
from goemotions_project.distilbert_training import (
    save_test_predictions,
    select_best_experiment,
    validation_selection_key,
)
from goemotions_project.labels import EMOTION_LABELS
from goemotions_project.metrics import compute_multilabel_metrics, find_best_threshold, sigmoid
from goemotions_project.models.baseline import AverageEmbeddingClassifier
from goemotions_project.seed import set_seed


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    epochs: int
    learning_rate: float
    freeze_embeddings: bool
    class_weighted: bool
    pos_weight_cap: float = 10.0


@dataclass(frozen=True)
class RunConfig:
    dataset_name: str
    dataset_config: str
    max_length: int
    max_vocab_size: int
    min_token_freq: int
    embedding_dim: int
    hidden_dim: int
    dropout: float
    batch_size: int
    weight_decay: float
    seed: int
    thresholds: tuple[float, ...]
    experiments: tuple[ExperimentSpec, ...]


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate the TOML experiment configuration."""
    with Path(path).open("rb") as config_file:
        raw = tomllib.load(config_file)
    threshold_config = raw["threshold"]
    thresholds = tuple(
        float(value)
        for value in np.arange(
            threshold_config["minimum"],
            threshold_config["maximum"] + threshold_config["step"] / 2,
            threshold_config["step"],
        )
    )
    experiments = tuple(
        ExperimentSpec(name=name, **settings) for name, settings in raw["experiments"].items()
    )
    return RunConfig(
        dataset_name=raw["data"]["dataset_name"],
        dataset_config=raw["data"]["dataset_config"],
        max_length=int(raw["data"]["max_length"]),
        max_vocab_size=int(raw["data"]["max_vocab_size"]),
        min_token_freq=int(raw["data"]["min_token_freq"]),
        embedding_dim=int(raw["model"]["embedding_dim"]),
        hidden_dim=int(raw["model"]["hidden_dim"]),
        dropout=float(raw["model"]["dropout"]),
        batch_size=int(raw["training"]["batch_size"]),
        weight_decay=float(raw["training"]["weight_decay"]),
        seed=int(raw["training"]["seed"]),
        thresholds=thresholds,
        experiments=experiments,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _save_per_label(path: Path, metrics: dict[str, Any]) -> None:
    rows = [
        {"label": label, **label_metrics} for label, label_metrics in metrics["per_label"].items()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return labels, probabilities, mean unweighted loss, and elapsed seconds."""
    model.eval()
    all_labels: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0
    _synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            batch_loss = _loss(logits, labels, pos_weight=None)
            batch_size = labels.shape[0]
            total_loss += float(batch_loss.item()) * batch_size
            total_examples += batch_size
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    _synchronize(device)
    elapsed = time.perf_counter() - start
    labels_array = np.concatenate(all_labels)
    logits_array = np.concatenate(all_logits)
    return labels_array, sigmoid(logits_array), total_loss / total_examples, elapsed


def publish_training_artifacts(
    run_dir: Path,
    result_dir: Path,
    config: RunConfig,
    spec: ExperimentSpec,
    vocab_size: int,
) -> None:
    """Copy reportable training artifacts, matching the DistilBERT track's layout."""
    result_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "best_validation.json",
        "experiment_result.json",
        "threshold_curve.csv",
        "training_history.csv",
        "validation_per_label.csv",
    ):
        shutil.copy2(run_dir / filename, result_dir / filename)
    _write_json(
        result_dir / "config.json",
        {
            "data": {
                "dataset_name": config.dataset_name,
                "dataset_config": config.dataset_config,
                "max_length": config.max_length,
                "vocab_size": vocab_size,
            },
            "model": {
                "embedding_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "dropout": config.dropout,
            },
            "training": {
                "batch_size": config.batch_size,
                "weight_decay": config.weight_decay,
                "seed": config.seed,
                "epochs_run": spec.epochs,
            },
            "thresholds": list(config.thresholds),
            "experiment": asdict(spec),
        },
    )


def train_experiment(
    spec: ExperimentSpec,
    config: RunConfig,
    train_dataset: BaselineDataset,
    validation_loader,
    train_label_rows,
    vocab_size: int,
    embedding_weights: torch.Tensor,
    run_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Train one experiment and save its best validation checkpoint."""
    set_seed(config.seed)
    run_dir = run_root / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = run_dir / "best_model.pt"

    model = AverageEmbeddingClassifier(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        padding_idx=PAD_ID,
        pretrained_weights=embedding_weights,
        freeze_embeddings=spec.freeze_embeddings,
    ).to(device)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=spec.learning_rate,
        weight_decay=config.weight_decay,
    )
    pos_weight = None
    if spec.class_weighted:
        pos_weight = compute_pos_weight(train_label_rows, spec.pos_weight_cap).to(device)

    history: list[dict[str, float | int]] = []
    best_validation: dict[str, Any] | None = None
    training_seconds = 0.0

    for epoch in range(1, spec.epochs + 1):
        train_loader = build_dataloader(
            train_dataset,
            config.batch_size,
            shuffle=True,
            seed=config.seed + epoch,
        )
        model.train()
        epoch_loss = 0.0
        seen_examples = 0
        _synchronize(device)
        epoch_start = time.perf_counter()
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = _loss(logits, labels, pos_weight)
            loss.backward()
            optimizer.step()
            batch_size = labels.shape[0]
            epoch_loss += float(loss.item()) * batch_size
            seen_examples += batch_size
        _synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
        training_seconds += epoch_seconds

        y_true, probabilities, validation_loss, validation_seconds = evaluate_model(
            model, validation_loader, device
        )
        validation_metrics, threshold_curve = find_best_threshold(
            y_true, probabilities, config.thresholds
        )
        history_row = {
            "epoch": epoch,
            "train_loss": epoch_loss / seen_examples,
            "validation_loss": validation_loss,
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_micro_f1": validation_metrics["micro_f1"],
            "threshold": validation_metrics["threshold"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "training_seconds": epoch_seconds,
            "validation_seconds": validation_seconds,
        }
        history.append(history_row)
        print(
            f"[{spec.name}] epoch={epoch}/{spec.epochs} "
            f"train_loss={history_row['train_loss']:.4f} "
            f"val_macro_f1={history_row['validation_macro_f1']:.4f} "
            f"val_micro_f1={history_row['validation_micro_f1']:.4f} "
            f"threshold={history_row['threshold']:.2f}",
            flush=True,
        )

        is_best = best_validation is None or validation_selection_key(
            validation_metrics
        ) > validation_selection_key(best_validation["metrics"])
        if is_best:
            torch.save(model.state_dict(), best_model_path)
            best_validation = {
                "epoch": epoch,
                "metrics": validation_metrics,
                "threshold_curve": threshold_curve,
                "validation_loss": validation_loss,
            }
            _write_json(run_dir / "best_validation.json", best_validation)
            _save_per_label(run_dir / "validation_per_label.csv", validation_metrics)
            pd.DataFrame(threshold_curve).to_csv(run_dir / "threshold_curve.csv", index=False)

        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    if best_validation is None:
        raise RuntimeError(f"Experiment {spec.name} did not produce a checkpoint")
    result = {
        "name": spec.name,
        "spec": asdict(spec),
        "epochs_run": spec.epochs,
        "training_seconds": training_seconds,
        "trainable_parameters": sum(p.numel() for p in trainable_parameters),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "best_validation": best_validation,
        "best_model_path": str(best_model_path),
    }
    _write_json(run_dir / "experiment_result.json", result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_all_experiments(
    config_path: str | Path,
    run_root: str | Path,
    results_dir: str | Path,
    final_model_dir: str | Path,
    selected_experiments: list[str] | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, Any]:
    """Train, select, test, and export all requested baseline experiments."""
    config = load_run_config(config_path)
    set_seed(config.seed)
    device = choose_device()
    print(f"Using device: {device}", flush=True)

    raw_dataset = load_goemotions()
    train_split = raw_dataset["train"]
    if max_train_samples is not None:
        train_split = train_split.select(range(min(max_train_samples, len(train_split))))
    validation_split = raw_dataset["validation"]
    if max_eval_samples is not None:
        validation_split = validation_split.select(
            range(min(max_eval_samples, len(validation_split)))
        )
    test_split = raw_dataset["test"]
    if max_test_samples is not None:
        test_split = test_split.select(range(min(max_test_samples, len(test_split))))

    vocab = build_vocab(train_split["text"], config.max_vocab_size, config.min_token_freq)
    embedding_weights, glove_coverage = load_glove_embeddings(
        vocab, config.embedding_dim, config.seed
    )

    train_dataset = BaselineDataset(
        train_split["text"], train_split["labels"], vocab, config.max_length
    )
    validation_dataset = BaselineDataset(
        validation_split["text"], validation_split["labels"], vocab, config.max_length
    )
    test_dataset = BaselineDataset(
        test_split["text"], test_split["labels"], vocab, config.max_length
    )
    validation_loader = build_dataloader(
        validation_dataset, config.batch_size, shuffle=False, seed=config.seed
    )
    test_loader = build_dataloader(test_dataset, config.batch_size, shuffle=False, seed=config.seed)

    run_root = Path(run_root)
    results_dir = Path(results_dir)
    final_model_dir = Path(final_model_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    available = {spec.name: spec for spec in config.experiments}
    names = selected_experiments or list(available)
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Unknown experiments: {', '.join(unknown)}")

    experiment_results = []
    for name in names:
        experiment_results.append(
            train_experiment(
                available[name],
                config,
                train_dataset,
                validation_loader,
                train_split["labels"],
                len(vocab),
                embedding_weights,
                run_root,
                device,
            )
        )

    summary_rows = []
    for result in experiment_results:
        spec = available[result["name"]]
        experiment_dir = results_dir / result["name"]
        publish_training_artifacts(
            run_root / result["name"], experiment_dir, config, spec, len(vocab)
        )
        model = AverageEmbeddingClassifier(
            vocab_size=len(vocab),
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            padding_idx=PAD_ID,
        ).to(device)
        model.load_state_dict(torch.load(result["best_model_path"], map_location=device))
        y_true, probabilities, test_loss, test_seconds = evaluate_model(model, test_loader, device)
        threshold = float(result["best_validation"]["metrics"]["threshold"])
        test_metrics = compute_multilabel_metrics(y_true, probabilities, threshold)
        _write_json(experiment_dir / "test_metrics.json", test_metrics)
        _save_per_label(experiment_dir / "test_per_label.csv", test_metrics)
        save_test_predictions(
            experiment_dir / "test_predictions.csv", test_split, y_true, probabilities, threshold
        )
        summary_rows.append(
            {
                "experiment": result["name"],
                "best_epoch": result["best_validation"]["epoch"],
                "threshold": threshold,
                "validation_macro_f1": result["best_validation"]["metrics"]["macro_f1"],
                "validation_micro_f1": result["best_validation"]["metrics"]["micro_f1"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_micro_f1": test_metrics["micro_f1"],
                "test_loss": test_loss,
                "training_seconds": result["training_seconds"],
                "inference_seconds_per_example": test_seconds / len(test_split),
                "trainable_parameters": result["trainable_parameters"],
                "total_parameters": result["total_parameters"],
            }
        )
        result["test_metrics"] = test_metrics
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)
    summary["threshold_distance_from_0.5"] = (summary["threshold"] - 0.5).abs()
    summary = summary.sort_values(
        ["validation_macro_f1", "validation_micro_f1", "threshold_distance_from_0.5"],
        ascending=[False, False, True],
    ).drop(columns="threshold_distance_from_0.5")
    summary.to_csv(results_dir / "experiment_summary.csv", index=False)

    best_name = str(select_best_experiment(summary_rows)["experiment"])
    best_result = next(r for r in experiment_results if r["name"] == best_name)
    shutil.copy2(best_result["best_model_path"], final_model_dir / "model.pt")
    vocab_path = final_model_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, indent=2, sort_keys=True) + "\n")

    best_summary = summary.iloc[0].to_dict()
    metadata = {
        "dataset_name": config.dataset_name,
        "dataset_config": config.dataset_config,
        "experiment": best_name,
        "threshold": float(best_summary["threshold"]),
        "max_length": config.max_length,
        "vocab_size": len(vocab),
        "glove_coverage": glove_coverage,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "labels": list(EMOTION_LABELS),
        "validation_macro_f1": float(best_summary["validation_macro_f1"]),
        "validation_micro_f1": float(best_summary["validation_micro_f1"]),
        "test_macro_f1": float(best_summary["test_macro_f1"]),
        "test_micro_f1": float(best_summary["test_micro_f1"]),
        "seed": config.seed,
        "device": str(device),
        "hardware": platform.platform(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_json(final_model_dir / "model_metadata.json", metadata)
    _write_json(results_dir / "best_model_metrics.json", metadata)
    output = {
        "best_experiment": best_name,
        "best_model_dir": str(final_model_dir),
        "results_dir": str(results_dir),
        "metadata": metadata,
    }
    _write_json(results_dir / "run_summary.json", output)
    return output
