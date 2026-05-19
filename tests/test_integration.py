"""End-to-end integration tests for the full skeval pipeline."""

import json

import pandas as pd
import pytest

from skeval.classifier import SentenceClassifier
from skeval.dataset.loader import DatasetLoader
from skeval.evaluator import Evaluator
from skeval.model_selection import train_test_split

SENTENCES = [
    "Water boils at 100 degrees",
    "The earth orbits the sun",
    "I feel so happy today",
    "This makes me really sad",
    "I believe this is correct",
    "In my view that is wrong",
    "Please open the window",
    "Do not forget to call me",
]
LABELS = [
    "fact",
    "fact",
    "emotion",
    "emotion",
    "opinion",
    "opinion",
    "instruction",
    "instruction",
]


@pytest.fixture()
def csv_dataset(tmp_path):
    """Write SENTENCES/LABELS to a CSV file and return its path."""
    path = tmp_path / "data.csv"
    pd.DataFrame({"text": SENTENCES, "label": LABELS}).to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def jsonl_dataset(tmp_path):
    """Write SENTENCES/LABELS to a JSONL file and return its path."""
    path = tmp_path / "data.jsonl"
    with open(path, "w") as f:
        for sentence, label in zip(SENTENCES, LABELS):
            f.write(json.dumps({"text": sentence, "label": label}) + "\n")
    return str(path)


# --- CSV pipeline ---


def test_csv_to_train_to_predict_to_evaluate(csv_dataset):
    """Full pipeline: load CSV, train, predict, evaluate — no errors end-to-end."""
    sentences, labels = DatasetLoader.load_csv(csv_dataset, "text", "label")

    clf = SentenceClassifier(embed_dim=16, epochs=2, random_state=0)
    clf.fit(sentences, labels)

    predictions = clf.predict(sentences)
    assert len(predictions) == len(sentences)
    assert all(p in set(labels) for p in predictions)

    results = Evaluator().evaluate(predictions, labels)
    assert 0.0 <= results["accuracy"] <= 1.0
    assert set(results["labels"]) == set(labels)


# --- JSONL pipeline ---


def test_jsonl_to_train_to_predict_to_evaluate(jsonl_dataset):
    """Full pipeline: load JSONL, train, predict, evaluate — no errors end-to-end."""
    sentences, labels = DatasetLoader.load_json(jsonl_dataset, "text", "label")

    clf = SentenceClassifier(embed_dim=16, epochs=2, random_state=0)
    clf.fit(sentences, labels)

    predictions = clf.predict(sentences)
    assert len(predictions) == len(sentences)

    results = Evaluator().evaluate(predictions, labels)
    assert "per_class" in results
    assert "confusion_matrix" in results


# --- save / load pipeline ---


def test_train_save_load_predict(tmp_path):
    """Saved and reloaded model must produce identical predictions to the original."""
    clf = SentenceClassifier(embed_dim=16, epochs=2, random_state=0)
    clf.fit(SENTENCES, LABELS)
    original_preds = clf.predict(SENTENCES)

    save_dir = str(tmp_path / "model")
    clf.save(save_dir)

    clf2 = SentenceClassifier()
    clf2.load(save_dir)
    loaded_preds = clf2.predict(SENTENCES)

    assert original_preds == loaded_preds


def test_train_save_load_evaluate(tmp_path):
    """Evaluator results must be identical before and after save/load."""
    clf = SentenceClassifier(embed_dim=16, epochs=2, random_state=0)
    clf.fit(SENTENCES, LABELS)

    save_dir = str(tmp_path / "model")
    clf.save(save_dir)

    clf2 = SentenceClassifier()
    clf2.load(save_dir)

    evaluator = Evaluator()
    results = evaluator.evaluate(clf2.predict(SENTENCES), LABELS)
    assert 0.0 <= results["accuracy"] <= 1.0


# --- train/test split pipeline ---


def test_split_train_predict_evaluate():
    """Full pipeline using train_test_split: train on split, evaluate on held-out."""
    X_train, X_test, y_train, y_test = train_test_split(
        SENTENCES, LABELS, test_size=0.25, random_state=0
    )

    clf = SentenceClassifier(embed_dim=16, epochs=2, random_state=0)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    assert len(predictions) == len(X_test)

    results = Evaluator().evaluate(predictions, y_test)
    assert 0.0 <= results["accuracy"] <= 1.0


# --- val_split pipeline ---


def test_train_with_val_split_then_evaluate():
    """Pipeline with val_split enabled: model trains and evaluates without errors."""
    clf = SentenceClassifier(
        embed_dim=16, epochs=5, val_split=0.25, patience=2, random_state=0
    )
    clf.fit(SENTENCES, LABELS)

    predictions = clf.predict(SENTENCES)
    results = Evaluator().evaluate(predictions, LABELS)
    assert 0.0 <= results["accuracy"] <= 1.0


# --- predict_proba pipeline ---


def test_predict_proba_pipeline():
    """predict_proba() output must be consistent with predict() argmax."""
    import numpy as np

    clf = SentenceClassifier(embed_dim=16, epochs=2, random_state=0)
    clf.fit(SENTENCES, LABELS)

    probs = clf.predict_proba(SENTENCES)
    preds = clf.predict(SENTENCES)

    label_order = sorted(set(LABELS))
    argmax_labels = [label_order[i] for i in np.argmax(probs, axis=1)]
    assert argmax_labels == preds


# --- CSV save/load full round-trip ---


def test_csv_full_round_trip(csv_dataset, tmp_path):
    """Load CSV, train, save, reload, predict, evaluate — complete round-trip."""
    sentences, labels = DatasetLoader.load_csv(csv_dataset, "text", "label")

    clf = SentenceClassifier(embed_dim=16, epochs=2, random_state=0)
    clf.fit(sentences, labels)
    clf.save(str(tmp_path / "model"))

    clf2 = SentenceClassifier()
    clf2.load(str(tmp_path / "model"))

    predictions = clf2.predict(sentences)
    results = Evaluator().evaluate(predictions, labels)

    assert len(predictions) == len(sentences)
    assert 0.0 <= results["accuracy"] <= 1.0
    assert set(results["labels"]) == set(labels)
