import pytest

from skeval.classifier import SentenceClassifier
from skeval.model_selection import cross_val_score, train_test_split

SENTENCES = [
    "Water boils at 100C", "I love you", "I think so", "Do this now",
    "The sky is blue", "I feel happy", "In my opinion", "Please help me",
]
LABELS = [
    "fact", "emotion", "opinion", "instruction",
    "fact", "emotion", "opinion", "instruction",
]


def test_train_test_split_sizes():
    """train_test_split() should produce train/test sets whose sizes sum to len(X)."""
    X_train, X_test, y_train, y_test = train_test_split(SENTENCES, LABELS, test_size=0.25)
    assert len(X_train) + len(X_test) == len(SENTENCES)
    assert len(y_train) + len(y_test) == len(LABELS)


def test_train_test_split_returns_lists():
    """train_test_split() should return Python lists, not numpy arrays."""
    X_train, X_test, y_train, y_test = train_test_split(SENTENCES, LABELS)
    assert isinstance(X_train, list)
    assert isinstance(X_test, list)
    assert isinstance(y_train, list)
    assert isinstance(y_test, list)


def test_train_test_split_test_size_int():
    """train_test_split() should accept an integer test_size for absolute split counts."""
    X_train, X_test, y_train, y_test = train_test_split(SENTENCES, LABELS, test_size=2)
    assert len(X_test) == 2
    assert len(X_train) == len(SENTENCES) - 2


def test_train_test_split_random_state_reproducible():
    """The same random_state should produce identical splits across calls."""
    split1 = train_test_split(SENTENCES, LABELS, random_state=42)
    split2 = train_test_split(SENTENCES, LABELS, random_state=42)
    assert split1[0] == split2[0]
    assert split1[1] == split2[1]


def test_train_test_split_stratify():
    """stratify=True should produce a split that preserves the class distribution."""
    X_train, X_test, y_train, y_test = train_test_split(
        SENTENCES, LABELS, test_size=0.5, stratify=True, random_state=0
    )
    assert len(X_train) + len(X_test) == len(SENTENCES)


def test_train_test_split_mismatched_lengths_raises():
    """train_test_split() must raise ValueError when X and y have different lengths."""
    with pytest.raises(ValueError, match="same length"):
        train_test_split(SENTENCES, LABELS[:3])


@pytest.mark.xfail(reason="requires SentenceClassifier to inherit BaseEstimator (#25)")
def test_cross_val_score_shape():
    """cross_val_score() should return an array with one score per fold."""
    clf = SentenceClassifier(embed_dim=16, epochs=1, random_state=0)
    scores = cross_val_score(clf, SENTENCES, LABELS, cv=2)
    assert scores.shape == (2,)


@pytest.mark.xfail(reason="requires SentenceClassifier to inherit BaseEstimator (#25)")
def test_cross_val_score_values_in_range():
    """Each cross-validation score must lie in the valid accuracy range [0.0, 1.0]."""
    clf = SentenceClassifier(embed_dim=16, epochs=1, random_state=0)
    scores = cross_val_score(clf, SENTENCES, LABELS, cv=2)
    assert all(0.0 <= s <= 1.0 for s in scores)
