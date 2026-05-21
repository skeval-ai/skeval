Changelog
=========

All notable changes to skeval are documented here.

The format follows `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

0.2.0 — 2026-05-19
-------------------

**Added**

* :mod:`~skeval.model_selection` — new module with :func:`~skeval.model_selection.train_test_split` and :func:`~skeval.model_selection.cross_val_score`
* :meth:`~skeval.classifier.SentenceClassifier.predict_proba` — probability outputs for LIME, SHAP, and ONNX compatibility
* Validation split and early stopping in :meth:`~skeval.classifier.SentenceClassifier.fit` (``val_split``, ``patience`` parameters)
* Batched prediction in :meth:`~skeval.classifier.SentenceClassifier.predict` and :meth:`~skeval.classifier.SentenceClassifier.predict_proba`
* ``num_workers`` and ``pin_memory`` parameters on the DataLoader for faster data loading
* ``random_state`` parameter on :class:`~skeval.classifier.SentenceClassifier` for reproducible training
* Input validation in :meth:`~skeval.classifier.SentenceClassifier.fit` and :meth:`~skeval.classifier.SentenceClassifier.predict`
* :class:`~skeval.classifier.SentenceClassifier` now inherits from ``sklearn.base.BaseEstimator`` — fully compatible with sklearn pipelines and ``GridSearchCV``
* ``check_estimator()`` compliance tests in CI
* Integration tests for full pipeline (CSV → train → save → load → predict → evaluate)
* Test suite grew from 11 to 79 tests
* Google-style docstrings on all public classes and functions
* Ecosystem compatibility documentation (LIME, SHAP, ONNX, skore, GridSearchCV)
* Read the Docs configuration and Sphinx docs build workflow

**Changed**

* :class:`~skeval.utils.helpers.VocabBuilder` now builds ``word2idx`` and ``idx2word`` in a single pass
* ``transformers`` and ``datasets`` moved to optional extras (``pip install skeval[transformers]``)
* Dependency upper bounds pinned to prevent silent breakage
* Full type annotations with ``mypy --strict`` enforced in CI

**Full Changelog**: https://github.com/skeval-ai/skeval/compare/v0.1.2...v0.2.0

----

0.1.2 — 2026-04-25
-------------------

**Changed**

* Renamed the library to ``skeval``

**Fixed**

* Miscellaneous fixes and stability improvements

**Full Changelog**: https://github.com/skeval-ai/skeval/compare/v0.1.1...v0.1.2

----

0.1.1 — 2026-04-25
-------------------

**Fixed**

* CI workflow: updated ``actions/checkout`` to ``v4`` and ``actions/setup-python`` to ``v5`` (``v6`` does not exist and caused CI failures)
* README: corrected ``predict()`` usage example — method takes a list of strings, not a single string
* README: corrected example output keys (``per_class_f1`` → ``per_class``)
* README: fixed install URL placeholder (``your-username`` → correct repo path)

----

0.1.0 — 2026-04-25
-------------------

First public release.

**Added**

* :class:`~skeval.classifier.SentenceClassifier` — train, predict, save, and load a PyTorch sentence classifier
* :class:`~skeval.classifier.BasicTextClassifier` — ``EmbeddingBag + Linear`` neural network architecture
* :class:`~skeval.evaluator.Evaluator` — evaluate predicted labels against ground truth
* :func:`~skeval.metrics.compute_metrics` — accuracy, per-class precision / recall / F1, confusion matrix via scikit-learn
* :class:`~skeval.dataset.loader.DatasetLoader` — load training data from CSV or JSON Lines files
* :class:`~skeval.dataset.loader.SentenceDataset` — PyTorch ``Dataset`` wrapper with variable-length collation
* :class:`~skeval.utils.helpers.VocabBuilder` — bag-of-words tokenizer with ``<PAD>`` / ``<UNK>`` support
* :class:`~skeval.utils.helpers.LabelEncoder` — string label ↔ integer index mapping
* ``scripts/train_model.py`` — CLI script for training from file
* ``scripts/evaluate_llm.py`` — CLI script for evaluation from file
* Sphinx documentation
* Full pytest test suite
