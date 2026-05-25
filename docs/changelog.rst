Changelog
=========

All notable changes to skeval are documented here.

The format follows `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

.. _Unreleased: https://github.com/skeval-ai/skeval/compare/v0.2.1...HEAD

`Unreleased`_
-------------

Fixed
~~~~~

- Replace ``assert self.model is not None`` with an explicit ``RuntimeError`` in
  ``_batch_forward`` — assertions are silently stripped under ``python -O``,
  causing an unhelpful ``AttributeError`` from PyTorch when calling
  ``predict()`` or ``predict_proba()`` on an unfitted model.
  See :issue:`113` by :user:`direkkakkar319-ops`.

----

.. _0.2.1: https://github.com/skeval-ai/skeval/compare/v0.2.0...v0.2.1

`0.2.1`_ (2026-05-21)
----------------------

Fixed
~~~~~

- Updated ``CONTRIBUTING.md`` links to the correct repository URL.
  By :user:`Raviteja2299`.
- Replaced ``Sentinel AI`` label in CLI description, ``train_model`` script,
  ``evaluate_llm`` script, and ``04_custom_labels`` example.
  By :user:`taiman724`.
- Fixed README clone URL and changelog repository URL.
  By :user:`donglrd`.
- Removed scikit-learn boilerplate from PR template.
  By :user:`donglrd`.

New Contributors
~~~~~~~~~~~~~~~~

- :user:`Raviteja2299` made their first contribution.
- :user:`taiman724` made their first contribution.
- :user:`donglrd` made their first contribution.

----

.. _0.2.0: https://github.com/skeval-ai/skeval/compare/v0.1.2...v0.2.0

`0.2.0`_ (2026-05-19)
----------------------

Release highlights
~~~~~~~~~~~~~~~~~~

- New :mod:`~skeval.model_selection` module with :func:`~skeval.model_selection.train_test_split`
  and :func:`~skeval.model_selection.cross_val_score`.
- :meth:`~skeval.classifier.SentenceClassifier.predict_proba` — probability
  outputs for LIME, SHAP, and ONNX compatibility.
- :class:`~skeval.classifier.SentenceClassifier` now inherits from
  ``sklearn.base.BaseEstimator``, enabling full sklearn pipeline and
  ``GridSearchCV`` compatibility.

Added
~~~~~

- :mod:`~skeval.model_selection` — new module with
  :func:`~skeval.model_selection.train_test_split` and
  :func:`~skeval.model_selection.cross_val_score`.
  By :user:`direkkakkar319-ops`.
- :meth:`~skeval.classifier.SentenceClassifier.predict_proba` — probability
  outputs for LIME, SHAP, and ONNX compatibility.
  By :user:`direkkakkar319-ops`.
- Validation split and early stopping in
  :meth:`~skeval.classifier.SentenceClassifier.fit`
  (``val_split``, ``patience`` parameters).
  By :user:`direkkakkar319-ops`.
- Batched prediction in
  :meth:`~skeval.classifier.SentenceClassifier.predict` and
  :meth:`~skeval.classifier.SentenceClassifier.predict_proba`.
  By :user:`direkkakkar319-ops`.
- ``num_workers`` and ``pin_memory`` parameters on the DataLoader for faster
  data loading.
  By :user:`direkkakkar319-ops`.
- ``random_state`` parameter on :class:`~skeval.classifier.SentenceClassifier`
  for reproducible training.
  By :user:`direkkakkar319-ops`.
- Input validation in :meth:`~skeval.classifier.SentenceClassifier.fit` and
  :meth:`~skeval.classifier.SentenceClassifier.predict`.
  By :user:`direkkakkar319-ops`.
- :class:`~skeval.classifier.SentenceClassifier` now inherits from
  ``sklearn.base.BaseEstimator`` — fully compatible with sklearn pipelines and
  ``GridSearchCV``.
  By :user:`direkkakkar319-ops`.
- ``check_estimator()`` compliance tests in CI.
  By :user:`direkkakkar319-ops`.
- Integration tests for the full pipeline
  (CSV → train → save → load → predict → evaluate).
  By :user:`direkkakkar319-ops`.
- Test suite grew from 11 to 79 tests.
  By :user:`direkkakkar319-ops`.
- Google-style docstrings on all public classes and functions.
  By :user:`direkkakkar319-ops`.
- Ecosystem compatibility documentation (LIME, SHAP, ONNX, skore, GridSearchCV).
  By :user:`direkkakkar319-ops`.
- Read the Docs configuration and Sphinx docs build workflow.
  By :user:`direkkakkar319-ops`.

Changed
~~~~~~~

- :class:`~skeval.utils.helpers.VocabBuilder` now builds ``word2idx`` and
  ``idx2word`` in a single pass.
  By :user:`direkkakkar319-ops`.
- ``transformers`` and ``datasets`` moved to optional extras
  (``pip install skeval[transformers]``).
  By :user:`direkkakkar319-ops`.
- Dependency upper bounds pinned to prevent silent breakage.
  By :user:`direkkakkar319-ops`.
- Full type annotations with ``mypy --strict`` enforced in CI.
  By :user:`direkkakkar319-ops`.

----

.. _0.1.2: https://github.com/skeval-ai/skeval/compare/v0.1.1...v0.1.2

`0.1.2`_ (2026-04-25)
----------------------

Changed
~~~~~~~

- Renamed the library to ``skeval``.
  By :user:`direkkakkar319-ops`.

Fixed
~~~~~

- Miscellaneous fixes and stability improvements.
  By :user:`direkkakkar319-ops`.

----

.. _0.1.1: https://github.com/skeval-ai/skeval/compare/v0.1.0...v0.1.1

`0.1.1`_ (2026-04-25)
----------------------

Fixed
~~~~~

- CI workflow: updated ``actions/checkout`` to ``v4`` and
  ``actions/setup-python`` to ``v5`` (``v6`` does not exist and caused CI
  failures).
  By :user:`direkkakkar319-ops`.
- README: corrected ``predict()`` usage example — method takes a list of
  strings, not a single string.
  By :user:`direkkakkar319-ops`.
- README: corrected example output keys (``per_class_f1`` → ``per_class``).
  By :user:`direkkakkar319-ops`.
- README: fixed install URL placeholder
  (``your-username`` → correct repository path).
  By :user:`direkkakkar319-ops`.

----

.. _0.1.0: https://github.com/skeval-ai/skeval/commits/v0.1.0

`0.1.0`_ (2026-04-25)
----------------------

Release highlights
~~~~~~~~~~~~~~~~~~

First public release of skeval — a lightweight semantic evaluation layer for
LLMs.

Added
~~~~~

- :class:`~skeval.classifier.SentenceClassifier` — train, predict, save, and
  load a PyTorch sentence classifier.
  By :user:`direkkakkar319-ops`.
- :class:`~skeval.classifier.BasicTextClassifier` —
  ``EmbeddingBag + Linear`` neural network architecture.
  By :user:`direkkakkar319-ops`.
- :class:`~skeval.evaluator.Evaluator` — evaluate predicted labels against
  ground truth.
  By :user:`direkkakkar319-ops`.
- :func:`~skeval.metrics.compute_metrics` — accuracy, per-class precision /
  recall / F1, and confusion matrix via scikit-learn.
  By :user:`direkkakkar319-ops`.
- :class:`~skeval.dataset.loader.DatasetLoader` — load training data from CSV
  or JSON Lines files.
  By :user:`direkkakkar319-ops`.
- :class:`~skeval.dataset.loader.SentenceDataset` — PyTorch ``Dataset`` wrapper
  with variable-length collation.
  By :user:`direkkakkar319-ops`.
- :class:`~skeval.utils.helpers.VocabBuilder` — bag-of-words tokenizer with
  ``<PAD>`` / ``<UNK>`` support.
  By :user:`direkkakkar319-ops`.
- :class:`~skeval.utils.helpers.LabelEncoder` — string label ↔ integer index
  mapping.
  By :user:`direkkakkar319-ops`.
- ``scripts/train_model.py`` — CLI script for training from file.
  By :user:`direkkakkar319-ops`.
- ``scripts/evaluate_llm.py`` — CLI script for evaluation from file.
  By :user:`direkkakkar319-ops`.
- Sphinx documentation.
  By :user:`direkkakkar319-ops`.
- Full pytest test suite.
  By :user:`direkkakkar319-ops`.
- 5 example scripts in ``examples/``.
  By :user:`direkkakkar319-ops`.
