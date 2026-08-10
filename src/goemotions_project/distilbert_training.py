"""End-to-end DistilBERT experiment runner."""

from __future__ import annotations

import json
import math
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from goemotions_project.data import load_goemotions
from goemotions_project.distilbert_data import (
    build_dataloader,
    compute_pos_weight,
    limit_splits,
    tokenize_dataset,
)
from goemotions_project.distilbert_inference import choose_device
from goemotions_project.labels import EMOTION_LABELS
from goemotions_project.metrics import compute_multilabel_metrics, find_best_threshold, sigmoid
from goemotions_project.models.distilbert import build_distilbert_classifier
from goemotions_project.seed import set_seed


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    epochs: int
    learning_rate: float
    freeze_encoder: bool
    class_weighted: bool
    pos_weight_cap: float = 10.0


@dataclass(frozen=True)
class RunConfig:
    dataset_name: str
    dataset_config: str
    max_length: int
    checkpoint: str
    batch_size: int
    weight_decay: float
    warmup_ratio: float
    gradient_clip: float
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
        ExperimentSpec(name=name, **settings)
        for name, settings in raw["experiments"].items()
    )
    return RunConfig(
        dataset_name=raw["data"]["dataset_name"],
        dataset_config=raw["data"]["dataset_config"],
        max_length=int(raw["data"]["max_length"]),
        checkpoint=raw["model"]["checkpoint"],
        batch_size=int(raw["training"]["batch_size"]),
        weight_decay=float(raw["training"]["weight_decay"]),
        warmup_ratio=float(raw["training"]["warmup_ratio"]),
        gradient_clip=float(raw["training"]["gradient_clip"]),
        seed=int(raw["training"]["seed"]),
        thresholds=thresholds,
        experiments=experiments,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - step, 0)
        decay_steps = max(total_steps - warmup_steps, 1)
        return float(remaining) / float(decay_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
    )


def evaluate_model(
    model,
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
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _save_per_label(path: Path, metrics: dict[str, Any]) -> None:
    rows = [
        {"label": label, **label_metrics}
        for label, label_metrics in metrics["per_label"].items()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def validation_selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    """Return the validation-only key used for checkpoints and experiments."""
    return (
        float(metrics["macro_f1"]),
        float(metrics["micro_f1"]),
        -abs(float(metrics["threshold"]) - 0.5),
    )


def select_best_experiment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select an experiment without consulting any test-set metric."""
    if not rows:
        raise ValueError("At least one experiment result is required")
    return max(
        rows,
        key=lambda row: validation_selection_key(
            {
                "macro_f1": row["validation_macro_f1"],
                "micro_f1": row["validation_micro_f1"],
                "threshold": row["threshold"],
            }
        ),
    )


def publish_training_artifacts(
    run_dir: Path,
    result_dir: Path,
    config: RunConfig,
    spec: ExperimentSpec,
    epochs_run: int,
) -> None:
    """Copy reportable training artifacts while excluding resumable optimizer state."""
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
            },
            "model": {"checkpoint": config.checkpoint},
            "training": {
                "batch_size": config.batch_size,
                "weight_decay": config.weight_decay,
                "warmup_ratio": config.warmup_ratio,
                "gradient_clip": config.gradient_clip,
                "seed": config.seed,
                "epochs_run": epochs_run,
            },
            "thresholds": list(config.thresholds),
            "experiment": asdict(spec),
        },
    )


def _experiment_model(spec: ExperimentSpec, checkpoint: str, device: torch.device):
    model = build_distilbert_classifier(checkpoint)
    if spec.freeze_encoder:
        for parameter in model.distilbert.parameters():
            parameter.requires_grad = False
    model.to(device)
    return model


def train_experiment(
    spec: ExperimentSpec,
    config: RunConfig,
    tokenized_train,
    validation_loader,
    tokenizer,
    train_label_rows,
    run_root: Path,
    device: torch.device,
    resume: bool,
    epochs_override: int | None,
) -> dict[str, Any]:
    """Train one experiment and save its best validation checkpoint."""
    set_seed(config.seed)
    run_dir = run_root / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = run_dir / "best_model"
    state_path = run_dir / "training_state.pt"
    epochs = epochs_override or spec.epochs
    steps_per_epoch = math.ceil(len(tokenized_train) / config.batch_size)
    total_steps = steps_per_epoch * epochs

    model = _experiment_model(spec, config.checkpoint, device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=spec.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, total_steps, config.warmup_ratio)
    pos_weight = None
    if spec.class_weighted:
        pos_weight = compute_pos_weight(train_label_rows, spec.pos_weight_cap).to(device)

    start_epoch = 1
    history: list[dict[str, float | int]] = []
    best_validation: dict[str, Any] | None = None
    global_step = 0
    training_seconds = 0.0
    if resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        _move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        history = state["history"]
        best_validation = state["best_validation"]
        global_step = int(state["global_step"])
        training_seconds = float(state["training_seconds"])

    for epoch in range(start_epoch, epochs + 1):
        train_loader = build_dataloader(
            tokenized_train,
            tokenizer,
            config.batch_size,
            shuffle=True,
            seed=config.seed + epoch,
        )
        model.train()
        epoch_loss = 0.0
        seen_examples = 0
        _synchronize(device)
        epoch_start = time.perf_counter()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = _loss(logits, labels, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, config.gradient_clip)
            optimizer.step()
            scheduler.step()
            batch_size = labels.shape[0]
            epoch_loss += float(loss.item()) * batch_size
            seen_examples += batch_size
            global_step += 1
        _synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
        training_seconds += epoch_seconds

        y_true, probabilities, validation_loss, validation_seconds = evaluate_model(
            model,
            validation_loader,
            device,
        )
        validation_metrics, threshold_curve = find_best_threshold(
            y_true,
            probabilities,
            config.thresholds,
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
            f"[{spec.name}] epoch={epoch}/{epochs} "
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
            model.save_pretrained(best_model_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_model_dir)
            best_validation = {
                "epoch": epoch,
                "metrics": validation_metrics,
                "threshold_curve": threshold_curve,
                "validation_loss": validation_loss,
            }
            _write_json(run_dir / "best_validation.json", best_validation)
            _save_per_label(
                run_dir / "validation_per_label.csv",
                validation_metrics,
            )
            pd.DataFrame(threshold_curve).to_csv(run_dir / "threshold_curve.csv", index=False)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best_validation": best_validation,
                "global_step": global_step,
                "training_seconds": training_seconds,
            },
            state_path,
        )
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    if best_validation is None:
        raise RuntimeError(f"Experiment {spec.name} did not produce a checkpoint")
    result = {
        "name": spec.name,
        "spec": asdict(spec),
        "epochs_run": epochs,
        "training_seconds": training_seconds,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation": best_validation,
        "best_model_dir": str(best_model_dir),
    }
    _write_json(run_dir / "experiment_result.json", result)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _label_names(row: np.ndarray) -> str:
    return "|".join(
        label for label, active in zip(EMOTION_LABELS, row, strict=True) if active
    )


def save_test_predictions(
    path: Path,
    raw_test,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    predictions = (probabilities >= threshold).astype(int)
    rows = []
    for index in range(len(raw_test)):
        rows.append(
            {
                "id": raw_test[index].get("id", str(index)),
                "text": raw_test[index]["text"],
                "true_labels": _label_names(y_true[index]),
                "predicted_labels": _label_names(predictions[index]),
                "probabilities": json.dumps(
                    {
                        label: round(float(probability), 6)
                        for label, probability in zip(
                            EMOTION_LABELS,
                            probabilities[index],
                            strict=True,
                        )
                    },
                    sort_keys=True,
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _copy_best_model(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def run_all_experiments(
    config_path: str | Path,
    run_root: str | Path,
    results_dir: str | Path,
    final_model_dir: str | Path,
    selected_experiments: list[str] | None = None,
    resume: bool = False,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_test_samples: int | None = None,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train, select, test, and export all requested DistilBERT experiments."""
    config = load_run_config(config_path)
    set_seed(config.seed)
    device = choose_device()
    print(f"Using device: {device}", flush=True)
    raw_dataset = load_goemotions()
    raw_dataset = limit_splits(
        raw_dataset,
        max_train_samples,
        max_eval_samples,
        max_test_samples,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    tokenized = tokenize_dataset(raw_dataset, tokenizer, config.max_length)
    validation_loader = build_dataloader(
        tokenized["validation"],
        tokenizer,
        config.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    test_loader = build_dataloader(
        tokenized["test"],
        tokenizer,
        config.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    run_root = Path(run_root)
    results_dir = Path(results_dir)
    final_model_dir = Path(final_model_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    available = {spec.name: spec for spec in config.experiments}
    names = selected_experiments or list(available)
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Unknown experiments: {', '.join(unknown)}")

    experiment_results = []
    train_label_rows = raw_dataset["train"]["labels"]
    for name in names:
        experiment_results.append(
            train_experiment(
                available[name],
                config,
                tokenized["train"],
                validation_loader,
                tokenizer,
                train_label_rows,
                run_root,
                device,
                resume,
                epochs_override,
            )
        )

    summary_rows = []
    for result in experiment_results:
        experiment_dir = results_dir / result["name"]
        publish_training_artifacts(
            run_root / result["name"],
            experiment_dir,
            config,
            available[result["name"]],
            int(result["epochs_run"]),
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            result["best_model_dir"],
            attn_implementation="eager",
        )
        model.to(device)
        y_true, probabilities, test_loss, test_seconds = evaluate_model(
            model,
            test_loader,
            device,
        )
        threshold = float(result["best_validation"]["metrics"]["threshold"])
        test_metrics = compute_multilabel_metrics(y_true, probabilities, threshold)
        _write_json(experiment_dir / "test_metrics.json", test_metrics)
        _save_per_label(experiment_dir / "test_per_label.csv", test_metrics)
        save_test_predictions(
            experiment_dir / "test_predictions.csv",
            raw_dataset["test"],
            y_true,
            probabilities,
            threshold,
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
                "inference_seconds_per_example": test_seconds / len(raw_dataset["test"]),
                "trainable_parameters": result["trainable_parameters"],
                "total_parameters": result["total_parameters"],
            }
        )
        result["test_metrics"] = test_metrics
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)
    summary["threshold_distance_from_0.5"] = (summary["threshold"] - 0.5).abs()
    summary = summary.sort_values(
        [
            "validation_macro_f1",
            "validation_micro_f1",
            "threshold_distance_from_0.5",
        ],
        ascending=[False, False, True],
    ).drop(columns="threshold_distance_from_0.5")
    summary.to_csv(results_dir / "experiment_summary.csv", index=False)
    best_name = str(select_best_experiment(summary_rows)["experiment"])
    best_result = next(result for result in experiment_results if result["name"] == best_name)
    _copy_best_model(Path(best_result["best_model_dir"]), final_model_dir)
    best_summary = summary.iloc[0].to_dict()
    metadata = {
        "base_checkpoint": config.checkpoint,
        "dataset_name": config.dataset_name,
        "dataset_config": config.dataset_config,
        "experiment": best_name,
        "threshold": float(best_summary["threshold"]),
        "max_length": config.max_length,
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
