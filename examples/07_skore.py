"""Tracking experiments with skore.

skore is an open-source ML experiment tracker that works with any
sklearn-compatible estimator.  This example shows how to use skore's
CrossValidationReport and ComparisonReport to inspect and compare
SentenceClassifier configurations.

Install skore first:
    pip install skore
"""

import skore

from skeval.classifier import SentenceClassifier

sentences = [
    "Water boils at 100 degrees Celsius",
    "Paris is the capital of France",
    "The moon orbits the Earth",
    "Light travels at 300,000 km per second",
    "I am feeling very sad today",
    "This is the worst day of my life",
    "I feel so excited right now",
    "She was overwhelmed with joy",
    "I think this movie is amazing",
    "In my opinion, pizza is the best food",
    "I believe coffee is better than tea",
    "Personally, I prefer winter over summer",
    "Please close the door",
    "Open the window right now",
    "Turn off the lights",
    "Send me the report by Monday",
]
labels = [
    "fact", "fact", "fact", "fact",
    "emotion", "emotion", "emotion", "emotion",
    "opinion", "opinion", "opinion", "opinion",
    "instruction", "instruction", "instruction", "instruction",
]

# --- cross-validated baseline ---
print("Running cross-validation baseline...")
baseline = skore.CrossValidationReport(
    SentenceClassifier(embed_dim=64, epochs=40, lr=0.01, random_state=42),
    X=sentences,
    y=labels,
    cv=2,
)
print(baseline.metrics.summarize())

# --- compare hyperparameter configurations ---
print("\nComparing hyperparameter configurations...")
reports = []
for embed_dim in [32, 64]:
    for lr in [0.005, 0.01]:
        clf = SentenceClassifier(
            embed_dim=embed_dim,
            epochs=40,
            lr=lr,
            random_state=42,
        )
        report = skore.CrossValidationReport(clf, X=sentences, y=labels, cv=2)
        reports.append(report)

comparison = skore.ComparisonReport(reports)
print(comparison.metrics.summarize())
