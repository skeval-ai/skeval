Changelog
=========

All notable changes to skeval are documented here.

The format follows `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

0.2.1 — 2026-05-21
-------------------

**Documentation**

* Updated docs for v0.2.0 — @direkkakkar319-ops
* Updated documentation for v0.2.0 — @direkkakkar319-ops
* Fixed README clone URL — @donglrd
* Fixed changelog repo URL — @donglrd
* Removed scikit-learn boilerplate from PR template — @donglrd

**Fixed**

* ``CONTRIBUTING.md`` links updated to correct repository URL — @Raviteja2299
* Replaced ``Sentinel AI`` label in CLI description — @taiman724
* Replaced ``Sentinel AI`` label in ``train_model`` script — @taiman724
* Replaced ``Sentinel AI`` label in ``evaluate_llm`` script — @taiman724
* Replaced ``Sentinel AI`` label in ``04_custom_labels`` example — @taiman724

**CI**

* CI Gate workflow fixed to correctly trigger checks on contributor PRs — @direkkakkar319-ops
* Fork PR approved workflow runs configured — @direkkakkar319-ops
* Merge queue support added (``merge_group`` event) — @direkkakkar319-ops
* CI checks run fixed — @direkkakkar319-ops

**New Contributors**

* @Raviteja2299 made their first contribution
* @taiman724 made their first contribution
* @donglrd made their first contribution

**Full Changelog**: https://github.com/skeval-ai/skeval/compare/v0.2.0...v0.2.1

----

0.2.0 — 2026-05-19
-------------------

**Added**

* :mod:`~skeval.model_selection` — new module with :func:`~skeval.model_selection.train_test_split` and :func:`~skeval.model_selection.cross_val_score` — @direkkakkar319-ops
* :meth:`~skeval.classifier.SentenceClassifier.predict_proba` — probability outputs for LIME, SHAP, and ONNX compatibility — @direkkakkar319-ops
* Validation split and early stopping in :meth:`~skeval.classifier.SentenceClassifier.fit` (``val_split``, ``patience`` parameters) — @direkkakkar319-ops
* Batched prediction in :meth:`~skeval.classifier.SentenceClassifier.predict` and :meth:`~skeval.classifier.SentenceClassifier.predict_proba` — @direkkakkar319-ops
* ``num_workers`` and ``pin_memory`` parameters on the DataLoader for faster data loading — @direkkakkar319-ops
* ``random_state`` parameter on :class:`~skeval.classifier.SentenceClassifier` for reproducible training — @direkkakkar319-ops
* Input validation in :meth:`~skeval.classifier.SentenceClassifier.fit` and :meth:`~skeval.classifier.SentenceClassifier.predict` — @direkkakkar319-ops
* :class:`~skeval.classifier.SentenceClassifier` now inherits from ``sklearn.base.BaseEstimator`` — fully compatible with sklearn pipelines and ``GridSearchCV`` — @direkkakkar319-ops
* ``check_estimator()`` compliance tests in CI — @direkkakkar319-ops
* Integration tests for full pipeline (CSV → train → save → load → predict → evaluate) — @direkkakkar319-ops
* Test suite grew from 11 to 79 tests — @direkkakkar319-ops
* Google-style docstrings on all public classes and functions — @direkkakkar319-ops
* Ecosystem compatibility documentation (LIME, SHAP, ONNX, skore, GridSearchCV) — @direkkakkar319-ops
* Read the Docs configuration and Sphinx docs build workflow — @direkkakkar319-ops

**Changed**

* :class:`~skeval.utils.helpers.VocabBuilder` now builds ``word2idx`` and ``idx2word`` in a single pass — @direkkakkar319-ops
* ``transformers`` and ``datasets`` moved to optional extras (``pip install skeval[transformers]``) — @direkkakkar319-ops
* Dependency upper bounds pinned to prevent silent breakage — @direkkakkar319-ops
* Full type annotations with ``mypy --strict`` enforced in CI — @direkkakkar319-ops

**Full Changelog**: https://github.com/skeval-ai/skeval/compare/v0.1.2...v0.2.0

----

0.1.2 — 2026-04-25
-------------------

**Changed**

* Renamed the library to ``skeval`` — @direkkakkar319-ops

**Fixed**

* Miscellaneous fixes and stability improvements — @direkkakkar319-ops

**Full Changelog**: https://github.com/skeval-ai/skeval/compare/v0.1.1...v0.1.2

----

0.1.1 — 2026-04-25
-------------------

**Fixed**

* CI workflow: updated ``actions/checkout`` to ``v4`` and ``actions/setup-python`` to ``v5`` (``v6`` does not exist and caused CI failures) — @direkkakkar319-ops
* README: corrected ``predict()`` usage example — method takes a list of strings, not a single string — @direkkakkar319-ops
* README: corrected example output keys (``per_class_f1`` → ``per_class``) — @direkkakkar319-ops
* README: fixed install URL placeholder (``your-username`` → correct repo path) — @direkkakkar319-ops

**Full Changelog**: https://github.com/skeval-ai/skeval/compare/v0.1.0...v0.1.1

----

0.1.0 — 2026-04-25
-------------------

First public release of skeval - a lightweight semantic evaluation layer for LLMs.

**Added**

* :class:`~skeval.classifier.SentenceClassifier` — train, predict, save, and load a PyTorch sentence classifier — @direkkakkar319-ops
* :class:`~skeval.classifier.BasicTextClassifier` — ``EmbeddingBag + Linear`` neural network architecture — @direkkakkar319-ops
* :class:`~skeval.evaluator.Evaluator` — evaluate predicted labels against ground truth — @direkkakkar319-ops
* :func:`~skeval.metrics.compute_metrics` — accuracy, per-class precision / recall / F1, confusion matrix via scikit-learn — @direkkakkar319-ops
* :class:`~skeval.dataset.loader.DatasetLoader` — load training data from CSV or JSON Lines files — @direkkakkar319-ops
* :class:`~skeval.dataset.loader.SentenceDataset` — PyTorch ``Dataset`` wrapper with variable-length collation — @direkkakkar319-ops
* :class:`~skeval.utils.helpers.VocabBuilder` — bag-of-words tokenizer with ``<PAD>`` / ``<UNK>`` support — @direkkakkar319-ops
* :class:`~skeval.utils.helpers.LabelEncoder` — string label ↔ integer index mapping — @direkkakkar319-ops
* ``scripts/train_model.py`` — CLI script for training from file — @direkkakkar319-ops
* ``scripts/evaluate_llm.py`` — CLI script for evaluation from file — @direkkakkar319-ops
* Sphinx documentation — @direkkakkar319-ops
* Full pytest test suite — @direkkakkar319-ops
* 5 example scripts in ``examples/`` — @direkkakkar319-ops

**CI**

* Bumped ``actions/setup-python`` from 5 to 6 — @dependabot[bot]
* Bumped ``github/codeql-action`` from 3 to 4 — @dependabot[bot]
* Bumped ``actions/checkout`` from 4 to 6 — @dependabot[bot]
* Bumped ``actions/setup-node`` from 4 to 6 — @dependabot[bot]
* Bumped ``actions/github-script`` from 7 to 9 — @dependabot[bot]
* Fixed action versions in release workflow — @direkkakkar319-ops

**Full Changelog**: https://github.com/skeval-ai/skeval/commits/v0.1.0
