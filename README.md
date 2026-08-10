# GoEmotions Customer Feedback Project

This repository contains the project framework for a fine grained, multi label emotion
classification system. The study compares a neural baseline based on word embeddings with a
fine tuned DistilBERT model. A Streamlit application will present predictions and experiment
results.

## Research question

How much does a fine tuned DistilBERT model improve fine grained emotion classification over a
simpler word embedding neural network, and how well do both models generalize from Reddit
comments to customer feedback style text?

## Planned workflow

1. Explore GoEmotions and verify its official train, validation, and test splits.
2. Train a Word2Vec or GloVe plus MLP baseline in PyTorch.
3. Create a small human labeled customer feedback test set.
4. Fine tune DistilBERT and run controlled experiments.
5. Compare in domain and out of domain results.
6. Connect the best checkpoints to a Streamlit web application.
7. Prepare figures, error analysis, presentation slides, and the final report.

## Repository structure

```text
.
├── app/                         Streamlit application
├── configs/                     Training configurations
├── data/ood/                    Human labeled out of domain test set
├── models/                      Published best model and local checkpoints
├── notebooks/                   Exploration notebooks
├── results/                     Metrics and generated figures
├── scripts/                     EDA and model smoke test commands
├── src/goemotions_project/      Reusable project code
└── tests/                       Fast unit tests
```

## Setup

The course recommends `uv` for package management.

```bash
uv sync --dev
```

The lock file selects CUDA 12.8 PyTorch wheels on Linux and Windows. On macOS, uv falls back to
the native PyPI wheel so Apple Silicon machines can use the MPS backend.

Run the tests and code checks:

```bash
uv run pytest
uv run ruff check .
```

Inspect the dataset:

```bash
uv run python scripts/eda.py
```

Check that both model architectures can perform a forward pass:

```bash
uv run python scripts/smoke_test_models.py
```

Run a small end-to-end DistilBERT smoke experiment:

```bash
uv run python scripts/train_distilbert.py \
  --experiments frozen_encoder \
  --max-train-samples 64 \
  --max-eval-samples 32 \
  --max-test-samples 32 \
  --epochs-override 1 \
  --run-root .training_runs/smoke \
  --results-dir .training_runs/smoke_results \
  --final-model-dir .training_runs/smoke_model
```

Run all final DistilBERT experiments:

```bash
uv run python scripts/train_distilbert.py
```

Resume an interrupted run:

```bash
uv run python scripts/train_distilbert.py --resume
```

Run one command-line prediction after training:

```bash
uv run python scripts/predict_distilbert.py \
  "I finally finished the project, but I am nervous about presenting it."
```

Start the web application:

```bash
uv run streamlit run app/streamlit_app.py
```

## Data

The project uses the simplified configuration of the
[GoEmotions dataset](https://huggingface.co/datasets/google-research-datasets/go_emotions).
It contains 27 emotion categories plus neutral and supports multi label classification.

Do not change the official test set or use it for model selection. Use the validation set to
select hyperparameters and classification thresholds. The manually labeled customer feedback
set is a final out of domain test set and must not be used for tuning.

## Minimum experiment plan

| Experiment | Purpose |
| --- | --- |
| Word embedding plus MLP | Establish a neural baseline |
| DistilBERT default run | Establish the Transformer result |
| Threshold tuning | Improve multi label prediction decisions |
| Class weighted loss | Test whether rare emotion recall improves |
| Frozen versus unfrozen encoder | Compare speed and performance |

Primary metrics are macro F1 and micro F1. Per label precision, recall, and F1 should be included
in the final error analysis.

## Current status

The repository provides the project structure, data loader, model definitions, metric helpers,
DistilBERT experiment runner, resumable checkpoints, inference interface, Streamlit integration,
and tests. The committed experiment tables and best checkpoint document the latest completed run.

## DistilBERT experiment design

Three controlled experiments use the same official dataset splits, label order, validation
threshold search, and test evaluation code:

1. A frozen encoder experiment trains only the DistilBERT classification head.
2. Full fine tuning updates all DistilBERT parameters.
3. Class weighted fine tuning uses capped square-root positive weights computed from the training
   split only.

The global prediction threshold is selected on the validation split from 0.10 through 0.70 in
steps of 0.05. The final model is selected by validation macro F1. Model weights are stored with
Git LFS, so collaborators should run `git lfs pull` after cloning or pulling the repository.

## DistilBERT results

The final experiments were trained with seed 42 on Apple M5 using the MPS backend. Validation
metrics selected both the epoch and global threshold. The official test split was evaluated only
after those choices were locked.

| Experiment | Best epoch | Threshold | Validation Macro F1 | Validation Micro F1 | Test Macro F1 | Test Micro F1 | Training time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen encoder | 5 | 0.15 | 0.3694 | 0.4989 | 0.3578 | 0.4933 | 5.8 min |
| Full fine tuning | 3 | 0.20 | 0.4737 | 0.5979 | 0.4624 | 0.5975 | 24.8 min |
| Class weighted fine tuning | 2 | 0.60 | **0.5128** | 0.5814 | **0.5057** | 0.5767 | 24.7 min |

Class weighted fine tuning is the published model because it achieved the highest validation
Macro F1. Its default inference threshold is 0.60. Detailed metrics, per-label scores, test
predictions, and error-analysis fields are available in `results/distilbert/`.
