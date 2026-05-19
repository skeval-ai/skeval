from typing import Any, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import cross_val_score as _cv_score
from sklearn.model_selection import train_test_split as _tts


def train_test_split(
    X: List[str],
    y: List[str],
    test_size: Union[float, int] = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: bool = False,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split sentences and labels into random train and test subsets.

    A thin, type-annotated wrapper around
    :func:`sklearn.model_selection.train_test_split` that preserves
    ``List[str]`` types for sentences and labels.

    Args:
        X: Sentence strings to split.
        y: Labels aligned with ``X``.
        test_size: If float, the proportion of the dataset to include in the
            test split. If int, the absolute number of test samples.
        random_state: Seed for the random number generator.
        shuffle: Whether to shuffle the data before splitting.
        stratify: If ``True``, the split is stratified by ``y`` so that each
            split has roughly the same class distribution.

    Returns:
        A 4-tuple ``(X_train, X_test, y_train, y_test)`` of string lists.

    Raises:
        ValueError: If ``X`` and ``y`` have different lengths.

    Example:
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.2, random_state=42
        ... )
    """
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length, got {len(X)} and {len(y)}."
        )
    stratify_arr = y if stratify else None
    X_train, X_test, y_train, y_test = _tts(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_arr,
    )
    return list(X_train), list(X_test), list(y_train), list(y_test)


def cross_val_score(
    estimator: Any,
    X: List[str],
    y: List[str],
    cv: int = 5,
    scoring: str = "accuracy",
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """Evaluate a classifier using k-fold cross-validation.

    A convenience wrapper around
    :func:`sklearn.model_selection.cross_val_score` with sensible defaults
    for ``SentenceClassifier``.

    Args:
        estimator: A fitted or unfitted sklearn-compatible estimator, e.g.
            :class:`~skeval.classifier.SentenceClassifier`.
        X: Sentence strings.
        y: Labels aligned with ``X``.
        cv: Number of cross-validation folds.
        scoring: Scoring metric passed to sklearn (default ``"accuracy"``).
        n_jobs: Number of jobs for parallel fold execution. ``None`` means 1;
            ``-1`` uses all available CPUs.

    Returns:
        Float array of shape ``(cv,)`` with the score for each fold.

    Example:
        >>> from skeval import SentenceClassifier
        >>> from skeval.model_selection import cross_val_score
        >>> scores = cross_val_score(SentenceClassifier(), X, y, cv=5)
        >>> print(scores.mean())
    """
    return _cv_score(  # type: ignore[no-any-return]
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs
    )
