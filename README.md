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
├── models/                      Local model checkpoints, ignored by Git
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

The repository currently provides the project structure, data loader, model definitions, metric
helpers, a web interface shell, and tests. Training loops, saved checkpoints, and final experiment
results still need to be completed by the team.

