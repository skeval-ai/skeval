import os
import warnings

import numpy as np
import pytest

from skeval.classifier import SentenceClassifier

SENTENCES = ["Water boils at 100C", "I love you", "I think so", "Do this now"]
LABELS = ["fact", "emotion", "opinion", "instruction"]


@pytest.fixture()
def fitted_clf():
    """Return a SentenceClassifier already fitted on SENTENCES/LABELS."""
    clf = SentenceClassifier(embed_dim=16, epochs=1, random_state=0)
    clf.fit(SENTENCES, LABELS)
    return clf


# --- initialisation ---

def test_default_init():
    """All hyper-parameters should match their documented defaults on construction."""
    clf = SentenceClassifier()
    assert clf.embed_dim == 64
    assert clf.epochs == 5
    assert clf.batch_size == 32
    assert clf.lr == 0.005
    assert clf.random_state is None
    assert clf.num_workers == 0
    assert clf.pin_memory is False
    assert clf.val_split == 0.0
    assert clf.patience == 0
    assert clf.model is None


def test_custom_init():
    """Constructor should store every explicitly passed hyper-parameter."""
    clf = SentenceClassifier(embed_dim=32, epochs=3, lr=0.01, random_state=42)
    assert clf.embed_dim == 32
    assert clf.epochs == 3
    assert clf.lr == 0.01
    assert clf.random_state == 42


# --- get_params / set_params ---

def test_get_params_keys():
    """get_params() must return exactly the set of sklearn-recognised parameters."""
    clf = SentenceClassifier()
    params = clf.get_params()
    expected = {
        "embed_dim", "epochs", "batch_size", "lr", "random_state",
        "num_workers", "pin_memory", "val_split", "patience",
    }
    assert expected == set(params.keys())


def test_set_params_returns_self():
    """set_params() should update the attribute and return the same instance."""
    clf = SentenceClassifier()
    result = clf.set_params(embed_dim=128)
    assert result is clf
    assert clf.embed_dim == 128


def test_set_params_invalid_key():
    """set_params() must raise ValueError for any unrecognised parameter name."""
    clf = SentenceClassifier()
    with pytest.raises(ValueError, match="Invalid parameter"):
        clf.set_params(nonexistent=99)


# --- fit ---

def test_fit_builds_vocab_and_labels(fitted_clf):
    """fit() must populate the vocabulary, label encoder, and the neural model."""
    assert fitted_clf.vocab.is_built
    assert fitted_clf.label_encoder.is_built
    assert fitted_clf.model is not None


def test_fit_returns_self():
    """fit() should return the classifier instance to allow method chaining."""
    clf = SentenceClassifier(embed_dim=16, epochs=1)
    result = clf.fit(SENTENCES, LABELS)
    assert result is clf


def test_fit_empty_X_raises():
    """fit() must raise ValueError when X is an empty list."""
    clf = SentenceClassifier()
    with pytest.raises(ValueError, match="non-empty"):
        clf.fit([], ["fact"])


def test_fit_non_string_X_raises():
    """fit() must raise ValueError when X contains non-string elements."""
    clf = SentenceClassifier()
    with pytest.raises(ValueError, match="strings"):
        clf.fit([1, 2], ["fact", "emotion"])  # type: ignore[list-item]


def test_fit_mismatched_lengths_raises():
    """fit() must raise ValueError when X and y have different lengths."""
    clf = SentenceClassifier()
    with pytest.raises(ValueError, match="same length"):
        clf.fit(SENTENCES, LABELS[:2])


def test_fit_empty_y_raises():
    """fit() must raise ValueError when y is an empty list."""
    clf = SentenceClassifier()
    with pytest.raises(ValueError, match="non-empty"):
        clf.fit(["hello"], [])


# --- predict ---

def test_predict_returns_list(fitted_clf):
    """predict() should return a list of label strings, one per input sentence."""
    preds = fitted_clf.predict(["I am sad"])
    assert isinstance(preds, list)
    assert len(preds) == 1
    assert preds[0] in LABELS


def test_predict_batch(fitted_clf):
    """predict() should handle a full batch and return one label per sentence."""
    preds = fitted_clf.predict(SENTENCES)
    assert len(preds) == len(SENTENCES)
    assert all(p in LABELS for p in preds)


def test_predict_before_fit_raises():
    """predict() must raise RuntimeError when called before fit()."""
    clf = SentenceClassifier()
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(["hello"])


def test_predict_empty_input_raises(fitted_clf):
    """predict() must raise ValueError when given an empty input list."""
    with pytest.raises(ValueError):
        fitted_clf.predict([])


def test_predict_unknown_tokens(fitted_clf):
    """predict() should still return a valid label for sentences with OOV tokens."""
    preds = fitted_clf.predict(["zzzzunknownzzz xyzxyz"])
    assert len(preds) == 1
    assert preds[0] in LABELS


# --- predict_proba ---

def test_predict_proba_shape(fitted_clf):
    """predict_proba() must return an array of shape (n_samples, n_classes)."""
    probs = fitted_clf.predict_proba(SENTENCES)
    assert probs.shape == (len(SENTENCES), len(LABELS))


def test_predict_proba_sums_to_one(fitted_clf):
    """Each row of predict_proba() must sum to 1.0 (valid probability distribution)."""
    probs = fitted_clf.predict_proba(SENTENCES)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(SENTENCES)), atol=1e-5)


def test_predict_proba_before_fit_raises():
    """predict_proba() must raise RuntimeError when called before fit()."""
    clf = SentenceClassifier()
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_proba(["hello"])


# --- score ---

def test_score_returns_float(fitted_clf):
    """score() must return a float accuracy value in [0.0, 1.0]."""
    s = fitted_clf.score(SENTENCES, LABELS)
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


# --- val_split and early stopping ---

def test_val_split_runs():
    """fit() with val_split > 0 should complete without error and produce a model."""
    clf = SentenceClassifier(embed_dim=16, epochs=3, val_split=0.25, random_state=0)
    clf.fit(SENTENCES, LABELS)
    assert clf.model is not None


def test_patience_early_stop():
    """fit() with patience > 0 should stop early when val loss stops improving."""
    clf = SentenceClassifier(
        embed_dim=16, epochs=50, val_split=0.25, patience=2, random_state=0
    )
    clf.fit(SENTENCES, LABELS)
    assert clf.model is not None


# --- random_state reproducibility ---

def test_random_state_reproducibility():
    """Two classifiers with the same random_state must produce identical predictions."""
    clf1 = SentenceClassifier(embed_dim=16, epochs=2, random_state=42)
    clf2 = SentenceClassifier(embed_dim=16, epochs=2, random_state=42)
    clf1.fit(SENTENCES, LABELS)
    clf2.fit(SENTENCES, LABELS)
    assert clf1.predict(SENTENCES) == clf2.predict(SENTENCES)


# --- save / load ---

def test_save_creates_files(fitted_clf, tmp_path):
    """save() must write model.pt and metadata.json into the specified directory."""
    fitted_clf.save(str(tmp_path / "out"))
    assert os.path.exists(tmp_path / "out" / "model.pt")
    assert os.path.exists(tmp_path / "out" / "metadata.json")


def test_save_before_fit_raises():
    """save() must raise RuntimeError when called before the model is trained."""
    clf = SentenceClassifier()
    with pytest.raises(RuntimeError, match="No model"):
        clf.save("/tmp/nowhere")


def test_load_restores_model(fitted_clf, tmp_path):
    """load() must restore embed_dim, vocab, label encoder, and the model weights."""
    save_dir = str(tmp_path / "out")
    fitted_clf.save(save_dir)
    clf2 = SentenceClassifier()
    clf2.load(save_dir)
    assert clf2.model is not None
    assert clf2.vocab.is_built
    assert clf2.label_encoder.is_built
    assert clf2.embed_dim == fitted_clf.embed_dim


def test_load_predict_consistent(fitted_clf, tmp_path):
    """A loaded classifier must produce the same predictions as the original."""
    save_dir = str(tmp_path / "out")
    fitted_clf.save(save_dir)
    clf2 = SentenceClassifier()
    clf2.load(save_dir)
    assert clf2.predict(SENTENCES) == fitted_clf.predict(SENTENCES)


# --- deprecated train() ---

def test_train_deprecated():
    """train() must emit a DeprecationWarning directing users to use fit()."""
    clf = SentenceClassifier(embed_dim=16)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        clf.train(SENTENCES, LABELS, epochs=1)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_train_still_works():
    """train() should still fit the model correctly despite being deprecated."""
    clf = SentenceClassifier(embed_dim=16)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        clf.train(SENTENCES, LABELS, epochs=1)
    assert clf.model is not None
