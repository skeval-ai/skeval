"""Tracking experiments with skore.

skore is an open-source ML experiment tracker that works with any
sklearn-compatible estimator.  This example shows how to use skore's
CrossValidationReport and ComparisonReport to inspect and compare
SentenceClassifier configurations.

Install skore first:
    pip install skore
"""

import warnings

import numpy as np
import skore
from sklearn.exceptions import UndefinedMetricWarning

from skeval.classifier import SentenceClassifier

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class SkoreClassifier(SentenceClassifier):
    """Thin wrapper that adds classes_ for skore tool compatibility."""

    def fit(self, X, y):
        super().fit(X, y)
        self.classes_ = np.array(sorted(set(y)))
        return self

sentences = [
    # fact (10)
    "Water boils at 100 degrees Celsius",
    "Paris is the capital of France",
    "The moon orbits the Earth",
    "Light travels at 300000 km per second",
    "The human body has 206 bones",
    "Oxygen has atomic number 8",
    "The Great Wall of China is over 21000 km long",
    "Mount Everest is the tallest mountain on Earth",
    "The Amazon is the largest river by discharge",
    "DNA carries genetic information",
    # emotion (10)
    "I am feeling very sad today",
    "This is the worst day of my life",
    "I feel so excited right now",
    "She was overwhelmed with joy",
    "I am so angry I cannot think straight",
    "This makes me incredibly happy",
    "I feel anxious about the meeting",
    "He was devastated by the news",
    "I love spending time with my family",
    "She felt proud of her achievement",
    # opinion (10)
    "I think this movie is amazing",
    "In my opinion pizza is the best food",
    "I believe coffee is better than tea",
    "Personally I prefer winter over summer",
    "I think remote work is more productive",
    "In my view this policy is misguided",
    "I feel that kindness matters most",
    "I believe education should be free",
    "Personally I find classical music relaxing",
    "I think dogs make better pets than cats",
    # instruction (10)
    "Please close the door",
    "Open the window right now",
    "Turn off the lights",
    "Send me the report by Monday",
    "Submit your assignment before Friday",
    "Click the button to continue",
    "Add the ingredients and stir well",
    "Save your work before closing",
    "Reply to this email within 24 hours",
    "Install the latest version to proceed",
]
labels = (
    ["fact"] * 10
    + ["emotion"] * 10
    + ["opinion"] * 10
    + ["instruction"] * 10
)

# --- cross-validated baseline ---
print("Running cross-validation baseline...")
baseline = skore.CrossValidationReport(
    SkoreClassifier(embed_dim=64, epochs=40, lr=0.01, random_state=42),
    X=sentences,
    y=labels,
    splitter=2,
)
print(f"Accuracy:  {baseline.metrics.accuracy()}")
print(f"Precision: {baseline.metrics.precision()}")
print(f"Recall:    {baseline.metrics.recall()}")

# --- compare hyperparameter configurations ---
print("\nComparing hyperparameter configurations...")
reports = []
for embed_dim in [32, 64]:
    for lr in [0.005, 0.01]:
        clf = SkoreClassifier(
            embed_dim=embed_dim,
            epochs=40,
            lr=lr,
            random_state=42,
        )
        report = skore.CrossValidationReport(clf, X=sentences, y=labels, splitter=2)
        reports.append(report)

comparison = skore.ComparisonReport(reports)
print(comparison.metrics.accuracy())
