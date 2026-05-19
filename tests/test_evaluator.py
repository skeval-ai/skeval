import pytest

from skeval.evaluator import Evaluator


def test_evaluator_basic():
    """evaluate() should return accuracy and standard metric keys for valid inputs."""
    evaluator = Evaluator()
    results = evaluator.evaluate(["fact", "emotion"], ["fact", "opinion"])
    assert results["accuracy"] == 0.5
    assert "confusion_matrix" in results
    assert "per_class" in results


def test_evaluator_perfect_accuracy():
    """accuracy should be 1.0 when all predictions exactly match the ground truth."""
    evaluator = Evaluator()
    results = evaluator.evaluate(["fact", "emotion"], ["fact", "emotion"])
    assert results["accuracy"] == 1.0


def test_evaluator_all_wrong():
    """accuracy should be 0.0 when no prediction matches the ground truth."""
    evaluator = Evaluator()
    results = evaluator.evaluate(["fact", "fact"], ["emotion", "emotion"])
    assert results["accuracy"] == 0.0


def test_evaluator_mismatch_length():
    """evaluate() must raise ValueError when predictions and ground_truth differ in length."""
    evaluator = Evaluator()
    with pytest.raises(ValueError, match="Mismatch in lengths"):
        evaluator.evaluate(["fact"], ["fact", "emotion"])


def test_evaluator_empty_predictions():
    """evaluate() must raise ValueError when the predictions list is empty."""
    evaluator = Evaluator()
    with pytest.raises(ValueError, match="non-empty"):
        evaluator.evaluate([], ["fact"])


def test_evaluator_empty_ground_truth():
    """evaluate() must raise ValueError when the ground_truth list is empty."""
    evaluator = Evaluator()
    with pytest.raises(ValueError, match="non-empty"):
        evaluator.evaluate(["fact"], [])


def test_evaluator_returns_sorted_labels():
    """The labels list in the result should be in ascending alphabetical order."""
    evaluator = Evaluator()
    results = evaluator.evaluate(["fact", "emotion"], ["fact", "emotion"])
    assert "labels" in results
    assert sorted(results["labels"]) == results["labels"]
