Welcome to skeval
======================

**skeval** is a lightweight PyTorch-powered library for classifying and evaluating the *semantic type* of sentences — facts, emotions, opinions, and instructions.

It fills the gap that standard LLM benchmarks leave open: not *how fluent* the output is, but *what kind of language* it uses.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   architecture
   api
   ecosystem
   changelog
   contributing

----

Quickstart
----------

Install the package:

.. code-block:: bash

   pip install skeval

Train a classifier and evaluate it in a few lines:

.. code-block:: python

   from skeval.classifier import SentenceClassifier
   from skeval.evaluator import Evaluator

   classifier = SentenceClassifier(embed_dim=64)

   sentences = [
       "Water boils at 100 degrees Celsius",
       "I feel happy today",
       "I think pizza is the best food",
       "Please close the door",
   ]
   labels = ["fact", "emotion", "opinion", "instruction"]

   classifier.fit(sentences, labels, epochs=20)

   predictions = classifier.predict(["The sky is blue", "I am sad"])
   print(predictions)  # e.g. ['fact', 'emotion']

   evaluator = Evaluator()
   results = evaluator.evaluate(predictions, ["fact", "emotion"])
   print(results["accuracy"])

----

Key Features
------------

* **Four semantic categories** out of the box: ``fact``, ``emotion``, ``opinion``, ``instruction``
* **Custom labels** — bring any label taxonomy you need
* **sklearn-compatible** — works with ``GridSearchCV``, ``Pipeline``, and ``cross_val_score``
* **Save / load** trained models to disk
* **Evaluation metrics** — accuracy, per-class precision / recall / F1, confusion matrix
* **Probability outputs** — ``predict_proba()`` for LIME, SHAP, and ONNX compatibility
* **Model selection** — ``train_test_split`` and ``cross_val_score`` in ``skeval.model_selection``
* **Dataset utilities** — load from CSV or JSON Lines files
* **Training scripts** — CLI scripts for training and evaluation without writing Python
