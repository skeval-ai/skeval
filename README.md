# skeval

> *Not just what the model says — but what it means.*

**skeval** is a lightweight, production-ready Python library for semantic sentence classification and LLM output evaluation. It fills the gap that standard benchmarks leave open: distinguishing *what kind of language* a model uses — facts, emotions, opinions, and instructions — rather than just measuring fluency or accuracy.

[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://pypi.org/project/skeval/)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://skeval.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Release](https://img.shields.io/pypi/v/skeval?label=release)](https://pypi.org/project/skeval/)
[![Repo Status](https://img.shields.io/badge/repo%20status-Active-brightgreen)](https://github.com/skeval-ai/skeval)
[![Tests](https://github.com/skeval-ai/skeval/actions/workflows/tests.yml/badge.svg)](https://github.com/skeval-ai/skeval/actions/workflows/tests.yml)
[![Security](https://github.com/skeval-ai/skeval/actions/workflows/security.yml/badge.svg)](https://github.com/skeval-ai/skeval/actions/workflows/security.yml)

---

## Table of Contents

- [Overview](#overview)
- [Technical Stack](#technical-stack)
- [Getting Started](#getting-started)
- [Core Features](#core-features)
- [CLI Reference](#cli-reference)
- [API Reference](#api-reference)
- [Development Workflow](#development-workflow)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Maintenance & Support](#maintenance--support)
- [License](#license)

---

## Overview

Most LLM evaluation focuses on accuracy, BLEU/ROUGE scores, or reasoning benchmarks. Real-world language understanding also requires distinguishing **facts from opinions**, detecting **emotions**, and identifying **intent**.

skeval provides a modular semantic classification and evaluation layer that works directly with LLM outputs, custom datasets, and benchmark pipelines.

**Target users:** ML engineers, NLP researchers, and LLM application developers who need semantic-level evaluation beyond token-level metrics.

---

## Technical Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | `>=3.9` |
| Neural network | PyTorch | `>=2.6.0, <3.0.0` |
| ML interface | scikit-learn | `>=1.3.0, <2.0.0` |
| Data handling | pandas | `>=2.0.0, <3.0.0` |
| Numerics | NumPy | `>=1.24.0, <3.0.0` |
| Progress bars | tqdm | `>=4.66.0, <5.0.0` |
| Docs | Sphinx + RTD theme | `>=7.3.0` |
| Linting | black, flake8, isort | `>=24.0.0` |
| Security | bandit, pip-audit | `>=1.7.8` |

**Architecture:** `EmbeddingBag` (bag-of-words averaging) → `Linear` (class logits) — fast, CPU-friendly, sklearn-compatible.

---

## Getting Started

### Prerequisites

- Python `3.9`, `3.10`, `3.11`, or `3.12`
- pip `>=23.0`

### Installation

**From PyPI (recommended):**

```bash
pip install skeval
```

**From source:**

```bash
git clone https://github.com/skeval-ai/skeval.git
cd skeval
pip install -e .
```

**With optional transformer backend (v0.3.0+):**

```bash
pip install skeval[transformers]
```

### Quick Start

```python
from skeval.classifier import SentenceClassifier
from skeval.evaluator import Evaluator

# 1. Define training data
sentences = [
    "Water boils at 100 degrees Celsius",
    "I feel sad today",
    "I think this movie is amazing",
    "Please close the door",
]
labels = ["fact", "emotion", "opinion", "instruction"]

# 2. Train
classifier = SentenceClassifier(embed_dim=64, epochs=20)
classifier.fit(sentences, labels)

# 3. Predict
predictions = classifier.predict([
    "The sky is blue",
    "I am so happy",
    "I believe dogs are better than cats",
    "Turn off the lights",
])

# 4. Evaluate
evaluator = Evaluator()
results = evaluator.evaluate(predictions, ["fact", "emotion", "opinion", "instruction"])
print(results["accuracy"])
print(results["per_class"])
```

---

## Core Features

### Semantic Classification

Classify sentences into four built-in categories — or any custom taxonomy you define:

| Label | Example |
|---|---|
| `fact` | "Water boils at 100 degrees Celsius" |
| `emotion` | "I feel so happy today" |
| `opinion` | "I think this film is overrated" |
| `instruction` | "Please close the door" |

```python
classifier = SentenceClassifier(embed_dim=64, epochs=30, lr=0.01)
classifier.fit(sentences, labels)
predictions = classifier.predict(new_sentences)
```

### Probability Outputs

Get per-class confidence scores — compatible with LIME, SHAP, and ONNX:

```python
proba = classifier.predict_proba(["The sky is blue"])
# shape: (1, 4) — one probability per class
print(proba[0])  # e.g. [0.82, 0.05, 0.08, 0.05]
```

### Validation Split & Early Stopping

```python
classifier = SentenceClassifier(
    embed_dim=64,
    epochs=100,
    val_split=0.2,   # hold out 20% for validation
    patience=5,      # stop if no improvement for 5 epochs
    random_state=42,
)
classifier.fit(sentences, labels)
```

### sklearn Integration

Works directly with `GridSearchCV`, `Pipeline`, and `cross_val_score`:

```python
from sklearn.model_selection import GridSearchCV
from skeval.classifier import SentenceClassifier

param_grid = {"embed_dim": [32, 64, 128], "epochs": [10, 20]}
grid = GridSearchCV(SentenceClassifier(random_state=0), param_grid, cv=3)
grid.fit(sentences, labels)
print(grid.best_params_)
```

### Model Persistence

```python
# Save
classifier.save("saved_model/")
# Writes: saved_model/model.pt + saved_model/metadata.json

# Load in a new session
classifier = SentenceClassifier()
classifier.load("saved_model/")
predictions = classifier.predict(["Water is wet"])
```

### Dataset Utilities

```python
from skeval.dataset.loader import DatasetLoader

# CSV
sentences, labels = DatasetLoader.load_csv(
    "data/train.csv", text_col="text", label_col="label"
)

# JSON Lines
sentences, labels = DatasetLoader.load_json(
    "data/train.jsonl", text_key="text", label_key="label"
)
```

### Evaluation Metrics

```python
from skeval.evaluator import Evaluator

results = Evaluator().evaluate(predictions, ground_truth)
```

Returns:

| Key | Description |
|---|---|
| `accuracy` | Overall fraction of correct predictions |
| `per_class` | `{label: {precision, recall, f1-score, support}}` |
| `macro_avg` | Unweighted average across classes |
| `weighted_avg` | Support-weighted average across classes |
| `confusion_matrix` | 2-D list — rows = true, columns = predicted |
| `labels` | Sorted list of all class names |

---

## CLI Reference

After installation, the `skeval` command is available:

```bash
skeval --help
skeval --version
```

### Train

```bash
skeval train \
    --data data/train.csv \
    --text-col text \
    --label-col label \
    --save-dir saved_model/ \
    --embed-dim 64 \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.005
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data` | Yes | — | Path to `.csv` or `.jsonl` training file |
| `--text-col` | Yes | — | Column name for sentence text |
| `--label-col` | Yes | — | Column name for labels |
| `--save-dir` | Yes | — | Directory to write `model.pt` and `metadata.json` |
| `--embed-dim` | No | `64` | Embedding dimension |
| `--epochs` | No | `10` | Number of training epochs |
| `--batch-size` | No | `32` | Mini-batch size |
| `--lr` | No | `0.005` | Learning rate |

### Evaluate

```bash
skeval evaluate \
    --model-dir saved_model/ \
    --data data/test.csv \
    --text-col text \
    --label-col label \
    --output report.json
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model-dir` | Yes | — | Directory containing saved model |
| `--data` | Yes | — | Path to `.csv` or `.jsonl` test file |
| `--text-col` | Yes | — | Column name for text |
| `--label-col` | Yes | — | Column name for labels |
| `--output` | No | — | Optional path to save JSON results |

---

## API Reference

### `SentenceClassifier`

```python
from skeval.classifier import SentenceClassifier
```

#### Constructor

```python
SentenceClassifier(
    embed_dim: int = 64,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.005,
    random_state: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    val_split: float = 0.0,
    patience: int = 0,
)
```

#### Methods

| Method | Signature | Description |
|---|---|---|
| `fit` | `fit(X, y) -> self` | Build vocabulary and train on labelled sentences |
| `predict` | `predict(X) -> list[str]` | Predict class labels |
| `predict_proba` | `predict_proba(X) -> np.ndarray` | Softmax probabilities, shape `(n, n_classes)` |
| `score` | `score(X, y) -> float` | Mean accuracy |
| `save` | `save(save_dir)` | Persist model and vocabulary to disk |
| `load` | `load(save_dir)` | Restore model from disk |

### `Evaluator`

```python
from skeval.evaluator import Evaluator

results = Evaluator().evaluate(predictions, ground_truth)
```

Returns a `dict` with keys: `accuracy`, `per_class`, `macro_avg`, `weighted_avg`, `confusion_matrix`, `labels`.

### `DatasetLoader`

```python
from skeval.dataset.loader import DatasetLoader

sentences, labels = DatasetLoader.load_csv(path, text_col, label_col)
sentences, labels = DatasetLoader.load_json(path, text_key, label_key)
```

---

## Development Workflow

### Setup

```bash
git clone https://github.com/skeval-ai/skeval.git
cd skeval
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e ".[dev,docs]"
```

### Branch Naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feat/<description>` | `feat/transformer-backend` |
| Bug fix | `fix/<description>` | `fix/predict-empty-input` |
| Docs | `docs/<description>` | `docs/update-usage-rst` |
| CI/CD | `ci/<description>` | `ci/add-coverage-badge` |

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src/skeval --cov-report=term-missing

# Single file
pytest tests/test_sentence_classifier.py -v
```

### Code Quality

```bash
black src/ tests/          # format
isort src/ tests/          # sort imports
flake8 src/ tests/         # lint
mypy src/                  # type check
bandit -r src/             # security scan
```

### Pull Request Requirements

- All CI checks must pass: `test`, `lint`, `build`, `install-and-import`, `security-audit`
- At least one reviewer approval required
- Branch must be up to date with `main` before merge
- PRs from forks require maintainer approval before CI runs

### CI/CD Pipeline

| Check | Trigger | Description |
|---|---|---|
| Tests | PR + merge queue | pytest on Python 3.9, 3.10, 3.11, 3.12 |
| Lint | PR + merge queue | black, flake8, isort, mypy |
| Build | PR + merge queue | package build dry-run |
| Security | PR + merge queue | bandit + pip-audit |
| Install | PR + merge queue | pip install + import smoke test |
| Release | Tag push | publish to PyPI |

---

## Troubleshooting & FAQ

**`ModuleNotFoundError: No module named 'skeval'`**
Run `pip install -e .` from the repo root, or `pip install skeval` from PyPI.

**`RuntimeError: Model is not fitted. Call fit() or load() first.`**
You called `predict()` before `fit()` or `load()`. Always fit or load before predicting.

**`TypeError: fit() got an unexpected keyword argument 'epochs'`**
`epochs`, `lr`, and `batch_size` are constructor parameters, not `fit()` arguments:
```python
# Correct
classifier = SentenceClassifier(epochs=20, lr=0.01)
classifier.fit(sentences, labels)
```

**`ValueError: X and y must have the same length`**
Your sentences and labels lists are different lengths. Check your dataset for missing rows.

**`make html` fails in docs/**
Run `pip install sphinx sphinx-rtd-theme myst-parser` then `sphinx-build -b html . _build/html` from the `docs/` directory.

**Predictions are all the same class**
Your training data is likely class-imbalanced or too small. Use at least 10–20 examples per class and increase `epochs`.

---

## Maintenance & Support

### Release Cycle

| Version | Status | Python |
|---|---|---|
| `0.2.x` | Active | 3.9 – 3.12 |
| `0.1.x` | End of life | 3.9 – 3.11 |
| `0.3.0` | In development | 3.9 – 3.12 |

### Reporting Issues

- **Bug reports:** [Open an issue](https://github.com/skeval-ai/skeval/issues) with steps to reproduce, Python version, and full error traceback.
- **Feature requests:** Open an issue with the `enhancement` label and describe the use case.
- **Security vulnerabilities:** Do not open a public issue. Email the maintainers directly or use GitHub's private security advisory.

### Roadmap (v0.3.0)

- Transformer backend via `sentence-transformers`
- Multi-label classification support
- Sarcasm detection
- Benchmark dataset release
- Full CLI test coverage

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for the full text.

Copyright (c) 2024 skeval Contributors.

Third-party dependencies are governed by their respective licenses (PyTorch — BSD-3, scikit-learn — BSD-3, NumPy — BSD-3, pandas — BSD-3).

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a PR.

Full documentation: [skeval.readthedocs.io](https://skeval.readthedocs.io/en/latest/)
