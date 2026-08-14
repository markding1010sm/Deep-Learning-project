# GoEmotions Customer Feedback Project

This repository contains a fine grained, multi label emotion classification system. The study
compares a GloVe plus MLP baseline with a fine tuned DistilBERT model. A Streamlit application
runs both trained checkpoints and presents their predictions and experiment results.

## Research question

How much does a fine tuned DistilBERT model improve fine grained emotion classification over a
simpler word embedding neural network, and how well do both models generalize from Reddit
comments to customer feedback style text?

## Project workflow

1. Load the official GoEmotions train, validation, and test splits.
2. Explore label frequencies, text lengths, and label co-occurrence patterns.
3. Train five controlled GloVe plus MLP baseline experiments.
4. Train three controlled DistilBERT experiments.
5. Select checkpoints and global thresholds using validation Macro F1 only.
6. Evaluate the locked models on the official test split.
7. Evaluate the same locked models on 50 human reviewed customer feedback messages.
8. Publish the best checkpoints and connect both models to Streamlit.

## Repository structure

```text
.
├── app/                         Streamlit application
├── configs/                     Training configurations
├── data/ood/                    Human labeled out of domain test set
├── models/                      Published best model and local checkpoints
├── notebooks/                   Exploration notebooks
├── results/                     Metrics and generated figures
├── scripts/                     EDA, training, inference, and analysis commands
├── src/goemotions_project/      Reusable project code
└── tests/                       Fast unit tests
```

## Setup

The course recommends `uv` for package management.

```bash
git lfs install
git lfs pull
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

Run all final GloVe plus MLP experiments:

```bash
uv run python scripts/train_baseline.py
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

Inspect common Baseline errors after training:

```bash
uv run python scripts/analyze_baseline_errors.py
```

Evaluate both locked models on the out of domain customer feedback set:

```bash
uv run python scripts/evaluate_ood.py
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

## Controlled experiments

| Model | Experiment | Purpose |
| --- | --- | --- |
| GloVe + MLP | Fine tune embeddings | Train the classifier and pretrained embeddings together |
| GloVe + MLP | Frozen embeddings | Train only the MLP classifier |
| GloVe + MLP | Class weighted | Improve performance on less frequent emotions |
| GloVe + MLP | Low learning rate | Evaluate a learning rate of 0.0003 |
| GloVe + MLP | High learning rate | Evaluate a learning rate of 0.003 |
| DistilBERT | Frozen encoder | Train only the classification head |
| DistilBERT | Full fine tuning | Update all DistilBERT parameters |
| DistilBERT | Class weighted | Apply capped positive class weights during fine tuning |

All experiments use the same official splits, canonical 28-label order, validation-only model
selection, and threshold search from 0.10 through 0.70 in steps of 0.05. Primary metrics are
Macro F1 and Micro F1. Per-label precision, recall, and F1 are saved for error analysis.

## Exploratory data analysis

Run `uv run python scripts/eda.py` to regenerate the committed EDA artifacts:

| Artifact | Description |
| --- | --- |
| `results/eda/label_counts.png` | Label-frequency imbalance |
| `results/eda/length_histogram.png` | Comment-length distribution |
| `results/eda/cooccurrence_heatmap.png` | Label co-occurrence heatmap |
| `results/eda/cooccurrence_top_pairs.csv` | Most frequent label pairs |

## Current status

The repository contains completed GloVe plus MLP and DistilBERT training pipelines, published best
checkpoints, in domain and out of domain test predictions, per-label metrics, EDA artifacts,
error-analysis utilities, and a Streamlit interface for both models. The out of domain evaluation
uses 50 human reviewed customer feedback messages with each model's locked validation threshold.
The codebase currently has 19 automated tests covering data, metrics, inference, checkpoint
loading, and the Baseline Streamlit interaction.

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

## GloVe plus MLP experiment design

The Baseline represents each comment with the average of its non-padding GloVe word embeddings,
then applies a two-layer MLP classifier. Its vocabulary is built from the training split only.
Five controlled experiments compare frozen versus trainable embeddings, class weighting, and
three learning rates. Checkpoints and thresholds follow the same validation-only selection rules
as the DistilBERT experiments.

The published vocabulary contains 13,161 tokens, and pretrained 300-dimensional GloVe vectors
cover 95.46% of those tokens. The final Baseline was trained with seed 42 using a CUDA device on
Windows.

## GloVe plus MLP results

| Experiment | Best epoch | Threshold | Validation Macro F1 | Validation Micro F1 | Test Macro F1 | Test Micro F1 | Training time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Class weighted | 7 | 0.50 | **0.4481** | 0.5152 | 0.4303 | 0.5071 | 14.9 s |
| Fine tune embeddings | 7 | 0.15 | 0.4360 | 0.5140 | 0.4306 | 0.5067 | 25.6 s |
| High learning rate | 3 | 0.20 | 0.4296 | **0.5313** | 0.4197 | **0.5272** | 17.4 s |
| Low learning rate | 10 | 0.15 | 0.4011 | 0.5209 | 0.3774 | 0.5100 | 17.1 s |
| Frozen embeddings | 10 | 0.15 | 0.3455 | 0.4725 | 0.3460 | 0.4709 | 10.5 s |

Class weighted training is the published Baseline because it achieved the highest validation
Macro F1. Its default inference threshold is 0.50. Detailed results are available in
`results/baseline/`, and `scripts/analyze_baseline_errors.py` reports common false-negative and
false-positive patterns from its saved test predictions.

## Published model comparison

| Model | Parameters | Validation Macro F1 | Validation Micro F1 | Test Macro F1 | Test Micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GloVe + MLP | 4.03M | 0.4481 | 0.5152 | 0.4303 | 0.5071 |
| DistilBERT | 66.98M | **0.5128** | **0.5814** | **0.5057** | **0.5767** |

DistilBERT improves Test Macro F1 by 0.0754 and Test Micro F1 by 0.0696 over the published
Baseline. Training-time measurements should not be treated as a controlled hardware comparison:
the DistilBERT experiments ran on Apple M5 MPS, while the Baseline experiments ran on Windows
CUDA.

## Out of domain customer feedback results

The final generalization check applies the models' locked thresholds to 50 human reviewed customer
feedback messages. These messages were not used for training, model selection, or threshold tuning.

| Model | Locked threshold | OOD Macro F1 | OOD Micro F1 | Exact match |
| --- | ---: | ---: | ---: | ---: |
| GloVe + MLP | 0.50 | 0.5728 | 0.5185 | 18% |
| DistilBERT | 0.60 | **0.6470** | **0.6538** | **34%** |

On this same 50-message set, DistilBERT improves Macro F1 by 0.0742, Micro F1 by 0.1353, and
exact match by 16 percentage points. Treat these numbers as exploratory: `grief` has no examples,
and 15 of the 28 labels have support of two or fewer. Detailed predictions and per-label results
are available in `results/ood/`.

## Streamlit application

The live demo loads both published checkpoints and supports:

- switching between DistilBERT and GloVe plus MLP;
- model-specific default thresholds of 0.60 and 0.50;
- manual threshold adjustment;
- selected emotion labels with confidence values;
- a Top 10 probability chart; and
- a combined view of both experiment tables.

Start the application with `uv run streamlit run app/streamlit_app.py`. If the DistilBERT weights
are still a Git LFS pointer, run `git lfs pull` before using the live demo.
