import ast
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"


def _example_files():
    """Return all Python example files sorted by name."""
    return sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.name)
def test_example_uses_fit_not_train(path):
    """Each example file must call .fit() and must not call deprecated .train()."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "train"
        ):
            raise AssertionError(
                f"{path.name} line {node.lineno}: deprecated .train() call found. "
                "Use .fit() instead."
            )
