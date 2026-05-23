import importlib.util
import sys
from pathlib import Path


def load_train_model_module():
    """Loads `train_model.py` as an importable module."""
    repo_root = Path(__file__).parents[1]
    module_path = repo_root / "scripts" / "train_model.py"
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    spec = importlib.util.spec_from_file_location("train_model", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_model_main_uses_fit_with_constructor_hyperparams(monkeypatch, tmp_path):
    """Verify `main()` calls `fit()` with constructor hyperparams and never calls `train()`."""
    module = load_train_model_module()
    data_file = tmp_path / "train.csv"
    data_file.write_text("text,label\nSun rises from east,fact\nI am feeling happy,emotion\n", encoding="utf-8")
    save_dir = tmp_path / "model"
    calls = {}

    class FakeDatasetLoader:
        """Loads fake data for testing."""
        @staticmethod
        def load_csv(path, text_col, label_col):
            """Record the call arguments and return fake sentences and labels."""
            calls["load_csv"] = (path, text_col, label_col)
            return ["Sun rises from east", "I am feeling happy"], ["fact","emotion"]

    class FakeSentenceClassifier:
        """Replaces `SentenceClassifier` during the test"""
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def train(self, *args, **kwargs):
            """Main guard for the test, crashes if deprecated `train()` is called."""
            raise AssertionError("Deprecated train() should not be used")

        def fit(self, sentences, labels):
            """Records sentences and labels when the script calls `fit()`"""
            calls["fit"] = (sentences, labels)
            return self

        def save(self, save_path):
            """Records `.save()` was called and with what path"""        
            calls["save"] = save_path

    monkeypatch.setattr(module, "DatasetLoader", FakeDatasetLoader)
    monkeypatch.setattr(module, "SentenceClassifier", FakeSentenceClassifier)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_model.py",
            "--data",
            str(data_file),
            "--text-col",
            "text",
            "--label-col",
            "label",
            "--save-dir",
            str(save_dir),
            "--embed-dim",
            "12",
            "--epochs",
            "7",
            "--batch-size",
            "3",
            "--lr",
            "0.1",
        ],
    )

    module.main()

    assert calls["load_csv"] == (str(data_file), "text", "label")
    assert calls["init"] == {
        "embed_dim": 12,
        "epochs": 7,
        "batch_size": 3,
        "lr": 0.1,
    }
    assert calls["fit"] == (["hello", "bye"], ["fact", "greeting"])
    assert calls["save"] == str(save_dir)
