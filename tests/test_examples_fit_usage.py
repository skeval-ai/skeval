from pathlib import Path

ISSUE_105_EXAMPLE = Path("examples/01_quickstart.py")


def test_quickstart_uses_fit_instead_of_deprecated_train():
    repo_root = Path(__file__).parents[1]
    text = (repo_root / ISSUE_105_EXAMPLE).read_text(encoding="utf-8")

    assert "classifier.train(" not in text
    assert "classifier.fit(" in text
