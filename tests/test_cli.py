import json
import subprocess
import sys

import pytest

from skeval.cli import _evaluate, _train


def _make_args(**kwargs):
    """Build a simple namespace for _train / _evaluate."""
    import argparse

    return argparse.Namespace(**kwargs)


def test_cli_version():
    """skeval --version must exit 0 and print the current version string."""
    result = subprocess.run(
        [sys.executable, "-m", "skeval.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "0.2.2" in result.stdout


def test_train_file_not_found(tmp_path, capsys):
    """_train must print an error and exit 1 when the data file is missing."""
    args = _make_args(
        data=str(tmp_path / "missing.csv"),
        text_col="text",
        label_col="label",
        save_dir=str(tmp_path / "model"),
        embed_dim=16,
        epochs=1,
        batch_size=2,
        lr=0.01,
    )
    with pytest.raises(SystemExit) as exc:
        _train(args)
    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().out


def test_train_unsupported_format(tmp_path, capsys):
    """_train must print an error and exit 1 for unsupported file extensions."""
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("hello")
    args = _make_args(
        data=str(bad_file),
        text_col="text",
        label_col="label",
        save_dir=str(tmp_path / "model"),
        embed_dim=16,
        epochs=1,
        batch_size=2,
        lr=0.01,
    )
    with pytest.raises(SystemExit) as exc:
        _train(args)
    assert exc.value.code == 1
    assert "unsupported format" in capsys.readouterr().out


def test_train_csv(tmp_path, monkeypatch, capsys):
    """_train must load CSV, call fit(), and save the model."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("text,label\nhello world,fact\nfeel great,emotion\n")
    save_dir = tmp_path / "model"

    calls = {}

    class FakeClassifier:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def fit(self, sentences, labels):
            calls["fit"] = (sentences, labels)
            return self

        def save(self, path):
            calls["save"] = path

    import skeval.cli as cli_module

    monkeypatch.setattr(cli_module, "SentenceClassifier", FakeClassifier, raising=False)

    args = _make_args(
        data=str(csv_file),
        text_col="text",
        label_col="label",
        save_dir=str(save_dir),
        embed_dim=16,
        epochs=1,
        batch_size=2,
        lr=0.01,
    )
    _train(args)

    assert "fit" in calls
    assert calls["save"] == str(save_dir)
    out = capsys.readouterr().out
    assert "samples" in out
    assert "saved" in out


def test_train_jsonl(tmp_path, monkeypatch):
    """_train must load JSONL files correctly."""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text(
        '{"text": "hello", "label": "fact"}\n'
        '{"text": "sad day", "label": "emotion"}\n'
    )
    save_dir = tmp_path / "model"

    calls = {}

    class FakeClassifier:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def fit(self, sentences, labels):
            calls["fit"] = (sentences, labels)
            return self

        def save(self, path):
            calls["save"] = path

    import skeval.cli as cli_module

    monkeypatch.setattr(cli_module, "SentenceClassifier", FakeClassifier, raising=False)

    args = _make_args(
        data=str(jsonl_file),
        text_col="text",
        label_col="label",
        save_dir=str(save_dir),
        embed_dim=16,
        epochs=1,
        batch_size=2,
        lr=0.01,
    )
    _train(args)

    assert "fit" in calls
    assert calls["save"] == str(save_dir)


def test_evaluate_model_not_found(tmp_path, capsys):
    """_evaluate must print an error and exit 1 when model.pt is missing."""
    args = _make_args(
        model_dir=str(tmp_path / "no_model"),
        data=str(tmp_path / "data.csv"),
        text_col="text",
        label_col="label",
        output=None,
    )
    with pytest.raises(SystemExit) as exc:
        _evaluate(args)
    assert exc.value.code == 1
    assert "no trained model" in capsys.readouterr().out


def test_evaluate_data_not_found(tmp_path, capsys):
    """_evaluate must print an error and exit 1 when the data file is missing."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_bytes(b"")

    args = _make_args(
        model_dir=str(model_dir),
        data=str(tmp_path / "missing.csv"),
        text_col="text",
        label_col="label",
        output=None,
    )
    with pytest.raises(SystemExit) as exc:
        _evaluate(args)
    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().out


def test_evaluate_unsupported_format(tmp_path, capsys):
    """_evaluate must print an error and exit 1 for unsupported file extensions."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_bytes(b"")
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("hello")

    args = _make_args(
        model_dir=str(model_dir),
        data=str(bad_file),
        text_col="text",
        label_col="label",
        output=None,
    )
    with pytest.raises(SystemExit) as exc:
        _evaluate(args)
    assert exc.value.code == 1
    assert "unsupported format" in capsys.readouterr().out


def _setup_evaluate(tmp_path, monkeypatch, data_file, output=None):
    """Shared setup for _evaluate success tests."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_bytes(b"")

    fake_results = {"accuracy": 1.0, "per_class": {}}

    class FakeClassifier:
        def load(self, _path):
            pass

        def predict(self, sentences):
            return ["fact"] * len(sentences)

    class FakeEvaluator:
        def evaluate(self, _predictions, _ground_truth):
            return fake_results

    import skeval.cli as cli_module

    monkeypatch.setattr(cli_module, "SentenceClassifier", FakeClassifier, raising=False)
    monkeypatch.setattr(cli_module, "Evaluator", FakeEvaluator, raising=False)

    return _make_args(
        model_dir=str(model_dir),
        data=str(data_file),
        text_col="text",
        label_col="label",
        output=output,
    )


def test_evaluate_csv(tmp_path, monkeypatch, capsys):
    """_evaluate must load CSV, predict, and print JSON results."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("text,label\nhello world,fact\n")
    args = _setup_evaluate(tmp_path, monkeypatch, csv_file)
    _evaluate(args)
    out = capsys.readouterr().out
    assert "accuracy" in out


def test_evaluate_jsonl(tmp_path, monkeypatch, capsys):
    """_evaluate must load JSONL files correctly."""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"text": "hello", "label": "fact"}\n')
    args = _setup_evaluate(tmp_path, monkeypatch, jsonl_file)
    _evaluate(args)
    out = capsys.readouterr().out
    assert "accuracy" in out


def test_evaluate_output_flag(tmp_path, monkeypatch, capsys):
    """_evaluate must save results to a JSON file when --output is given."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("text,label\nhello world,fact\n")
    output_file = tmp_path / "results.json"
    args = _setup_evaluate(tmp_path, monkeypatch, csv_file, output=str(output_file))
    _evaluate(args)

    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert "accuracy" in data
    assert "saved" in capsys.readouterr().out
