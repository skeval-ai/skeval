from skeval.metrics import compute_metrics


def test_compute_metrics_accuracy():
    """accuracy should reflect the fraction of correctly classified samples."""
    y_true = ["fact", "emotion", "fact", "opinion"]
    y_pred = ["fact", "fact", "fact", "opinion"]
    results = compute_metrics(y_true, y_pred)
    assert results["accuracy"] == 0.75


def test_compute_metrics_perfect():
    """accuracy should be 1.0 when every prediction matches the ground truth."""
    y_true = ["fact", "emotion", "opinion"]
    y_pred = ["fact", "emotion", "opinion"]
    results = compute_metrics(y_true, y_pred)
    assert results["accuracy"] == 1.0


def test_compute_metrics_all_wrong():
    """accuracy should be 0.0 when no prediction matches the ground truth."""
    y_true = ["fact", "fact"]
    y_pred = ["emotion", "emotion"]
    results = compute_metrics(y_true, y_pred)
    assert results["accuracy"] == 0.0


def test_compute_metrics_zero_division():
    """Precision and recall for a never-predicted class should be 0.0, not an error."""
    y_true = ["fact", "emotion"]
    y_pred = ["fact", "fact"]
    results = compute_metrics(y_true, y_pred)
    assert results["accuracy"] == 0.5
    assert results["per_class"]["emotion"]["precision"] == 0.0
    assert results["per_class"]["emotion"]["recall"] == 0.0


def test_compute_metrics_per_class_keys():
    """Each per-class entry must contain precision, recall, f1-score, and support."""
    y_true = ["fact", "emotion"]
    y_pred = ["fact", "emotion"]
    results = compute_metrics(y_true, y_pred)
    for label in ["fact", "emotion"]:
        assert "precision" in results["per_class"][label]
        assert "recall" in results["per_class"][label]
        assert "f1-score" in results["per_class"][label]
        assert "support" in results["per_class"][label]


def test_compute_metrics_labels_sorted():
    """The labels list in the result should be in ascending alphabetical order."""
    y_true = ["opinion", "fact", "emotion"]
    y_pred = ["opinion", "fact", "emotion"]
    results = compute_metrics(y_true, y_pred)
    assert results["labels"] == sorted(results["labels"])


def test_compute_metrics_confusion_matrix_shape():
    """The confusion matrix must be square with side length equal to the label count."""
    y_true = ["fact", "emotion", "opinion"]
    y_pred = ["fact", "emotion", "opinion"]
    results = compute_metrics(y_true, y_pred)
    n = len(results["labels"])
    assert len(results["confusion_matrix"]) == n
    assert all(len(row) == n for row in results["confusion_matrix"])


def test_compute_metrics_macro_avg_present():
    """The result dict must include both macro_avg and weighted_avg aggregates."""
    results = compute_metrics(["fact", "emotion"], ["fact", "fact"])
    assert "macro_avg" in results
    assert "weighted_avg" in results


def test_compute_metrics_single_class():
    """compute_metrics should work correctly when only one class is present."""
    results = compute_metrics(["fact", "fact"], ["fact", "fact"])
    assert results["accuracy"] == 1.0
    assert "fact" in results["per_class"]
