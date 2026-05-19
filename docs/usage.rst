Usage Guide
===========

Python API
----------

Training a Classifier
^^^^^^^^^^^^^^^^^^^^^

Create a :class:`~skeval.classifier.SentenceClassifier` and call :meth:`~skeval.classifier.SentenceClassifier.fit`:

.. code-block:: python

   from skeval.classifier import SentenceClassifier

   classifier = SentenceClassifier(embed_dim=64, random_state=42)

   sentences = [
       "Water boils at 100 degrees Celsius",
       "Paris is the capital of France",
       "I am feeling very sad today",
       "This is the worst day of my life",
       "I think this movie is amazing",
       "In my opinion, pizza is the best food",
       "Please close the door",
       "Open the window right now",
   ]
   labels = [
       "fact", "fact",
       "emotion", "emotion",
       "opinion", "opinion",
       "instruction", "instruction",
   ]

   classifier.fit(sentences, labels, epochs=20, lr=0.01)

The label vocabulary is inferred automatically from the labels you provide — you are not limited to the four default categories.

Training with Validation Split and Early Stopping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass ``val_split`` to hold out a fraction of training data for validation, and ``patience`` to stop early when validation loss stops improving:

.. code-block:: python

   classifier = SentenceClassifier(
       embed_dim=64,
       val_split=0.2,
       patience=5,
       random_state=42,
   )
   classifier.fit(sentences, labels, epochs=100, lr=0.01)

DataLoader Performance Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``num_workers`` and ``pin_memory`` to speed up data loading on multi-core machines or when training on GPU:

.. code-block:: python

   classifier = SentenceClassifier(
       embed_dim=64,
       num_workers=4,
       pin_memory=True,
   )

Making Predictions
^^^^^^^^^^^^^^^^^^

:meth:`~skeval.classifier.SentenceClassifier.predict` takes a list of strings and returns a list of predicted label strings:

.. code-block:: python

   predictions = classifier.predict([
       "The sky is blue",
       "I am so happy",
       "I believe dogs are better than cats",
       "Turn off the lights",
   ])
   print(predictions)
   # ['fact', 'emotion', 'opinion', 'instruction']

Probability Outputs
^^^^^^^^^^^^^^^^^^^

:meth:`~skeval.classifier.SentenceClassifier.predict_proba` returns a ``(n_samples, n_classes)`` NumPy array of softmax probabilities, compatible with LIME, SHAP, and ONNX:

.. code-block:: python

   proba = classifier.predict_proba([
       "The sky is blue",
       "I am so happy",
   ])
   print(proba.shape)   # (2, 4)
   print(proba[0])      # e.g. [0.82, 0.05, 0.08, 0.05]

Model Selection
^^^^^^^^^^^^^^^

Use :mod:`skeval.model_selection` for splitting data and cross-validating:

.. code-block:: python

   from skeval.model_selection import train_test_split, cross_val_score
   from skeval.classifier import SentenceClassifier

   X_train, X_test, y_train, y_test = train_test_split(
       sentences, labels, test_size=0.25, random_state=42, stratify=True
   )

   clf = SentenceClassifier(embed_dim=64, epochs=20, random_state=0)
   scores = cross_val_score(clf, sentences, labels, cv=4)
   print(scores)          # e.g. [0.75, 0.88, 0.62, 0.75]
   print(scores.mean())   # e.g. 0.75

sklearn Integration
^^^^^^^^^^^^^^^^^^^

:class:`~skeval.classifier.SentenceClassifier` inherits from ``sklearn.base.BaseEstimator``, making it compatible with sklearn pipelines and ``GridSearchCV``:

.. code-block:: python

   from sklearn.model_selection import GridSearchCV
   from skeval.classifier import SentenceClassifier

   param_grid = {"embed_dim": [32, 64, 128], "epochs": [10, 20]}
   grid = GridSearchCV(
       SentenceClassifier(random_state=0), param_grid, cv=3
   )
   grid.fit(sentences, labels)
   print(grid.best_params_)

Saving and Loading
^^^^^^^^^^^^^^^^^^

Save the trained model and vocabulary to a directory:

.. code-block:: python

   classifier.save("saved_model/")

Load it back in a new session:

.. code-block:: python

   from skeval.classifier import SentenceClassifier

   classifier = SentenceClassifier()
   classifier.load("saved_model/")

   predictions = classifier.predict(["Water is wet"])

Two files are written: ``model.pt`` (PyTorch weights) and ``metadata.json`` (vocabulary and label mappings).

Evaluating Predictions
^^^^^^^^^^^^^^^^^^^^^^

Pass predictions and ground-truth labels to :class:`~skeval.evaluator.Evaluator`:

.. code-block:: python

   from skeval.evaluator import Evaluator

   evaluator = Evaluator()
   results = evaluator.evaluate(predictions, ground_truth)

   print(results["accuracy"])
   print(results["per_class"])
   print(results["confusion_matrix"])

The returned dictionary contains:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Description
   * - ``accuracy``
     - Overall fraction of correct predictions
   * - ``per_class``
     - Dict of ``{label: {precision, recall, f1-score, support}}``
   * - ``macro_avg``
     - Unweighted average of per-class metrics
   * - ``weighted_avg``
     - Support-weighted average of per-class metrics
   * - ``confusion_matrix``
     - 2-D list; rows = true labels, columns = predicted labels
   * - ``labels``
     - Sorted list of all class names used to index the matrix

Loading Data from Files
^^^^^^^^^^^^^^^^^^^^^^^

Use :class:`~skeval.dataset.loader.DatasetLoader` to read CSV or JSON Lines files:

.. code-block:: python

   from skeval.dataset.loader import DatasetLoader

   # CSV
   sentences, labels = DatasetLoader.load_csv(
       "data/train.csv", text_col="text", label_col="label"
   )

   # JSON Lines
   sentences, labels = DatasetLoader.load_json(
       "data/train.jsonl", text_key="text", label_key="label"
   )

   classifier.fit(sentences, labels)

----

Command-Line Interface
----------------------

After installing the package, a ``skeval`` command is available:

.. code-block:: bash

   skeval --help
   skeval train --help
   skeval evaluate --help

Check the installed version:

.. code-block:: bash

   skeval --version

Training via CLI
^^^^^^^^^^^^^^^^

.. code-block:: bash

   skeval train \
       --data data/train.csv \
       --text-col text \
       --label-col label \
       --save-dir saved_model/ \
       --epochs 20 \
       --batch-size 32 \
       --embed-dim 64 \
       --lr 0.005

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``--data``
     - Path to ``.csv`` or ``.jsonl`` training file (required)
   * - ``--text-col``
     - Column / key name that holds the sentence text (required)
   * - ``--label-col``
     - Column / key name that holds the label (required)
   * - ``--save-dir``
     - Directory to write ``model.pt`` and ``metadata.json`` (required)
   * - ``--embed-dim``
     - Embedding dimension (default: ``64``)
   * - ``--epochs``
     - Number of training epochs (default: ``10``)
   * - ``--batch-size``
     - Mini-batch size (default: ``32``)
   * - ``--lr``
     - Learning rate (default: ``0.005``)

Evaluation via CLI
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   skeval evaluate \
       --model-dir saved_model/ \
       --data data/test.csv \
       --text-col text \
       --label-col label \
       --output report.json

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``--model-dir``
     - Directory containing ``model.pt`` and ``metadata.json`` (required)
   * - ``--data``
     - Path to test ``.csv`` or ``.jsonl`` file (required)
   * - ``--text-col``
     - Column / key name for the sentence text (required)
   * - ``--label-col``
     - Column / key name for the ground-truth label (required)
   * - ``--output``
     - Optional path to save the JSON results file

----

Legacy Scripts
--------------

skeval also ships standalone scripts in ``scripts/`` that work without installation.

Training
^^^^^^^^

Train a model from a CSV or JSONL file and save it to disk:

.. code-block:: bash

   python scripts/train_model.py \
       --data data/train.csv \
       --text-col text \
       --label-col label \
       --save-dir saved_model/ \
       --epochs 20 \
       --batch-size 32 \
       --embed-dim 64 \
       --lr 0.005

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``--data``
     - Path to ``.csv`` or ``.jsonl`` training file (required)
   * - ``--text-col``
     - Column / key name that holds the sentence text (required)
   * - ``--label-col``
     - Column / key name that holds the label (required)
   * - ``--save-dir``
     - Directory to write ``model.pt`` and ``metadata.json`` (required)
   * - ``--embed-dim``
     - Embedding dimension (default: ``64``)
   * - ``--epochs``
     - Number of training epochs (default: ``10``)
   * - ``--batch-size``
     - Mini-batch size (default: ``32``)
   * - ``--lr``
     - Learning rate (default: ``0.005``)

Evaluation
^^^^^^^^^^

Load a trained model and evaluate it on a held-out test set:

.. code-block:: bash

   python scripts/evaluate_llm.py \
       --model-dir saved_model/ \
       --data data/test.csv \
       --text-col text \
       --label-col label \
       --output report.json

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``--model-dir``
     - Directory containing ``model.pt`` and ``metadata.json`` (required)
   * - ``--data``
     - Path to test ``.csv`` or ``.jsonl`` file (required)
   * - ``--text-col``
     - Column / key name for the sentence text (required)
   * - ``--label-col``
     - Column / key name for the ground-truth label (required)
   * - ``--output``
     - Optional path to save the JSON results file

The script prints accuracy, per-class F1, and the confusion matrix, and optionally writes them to the path given by ``--output``.
