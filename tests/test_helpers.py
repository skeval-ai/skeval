from skeval.utils.helpers import normalize_text


def test_normalize_text_lowercases():
    """normalize_text() should convert all characters to lowercase."""
    assert normalize_text("Hello World") == "hello world"


def test_normalize_text_removes_punctuation():
    """normalize_text() should strip punctuation characters from the string."""
    assert normalize_text("Hello, World!") == "hello world"


def test_normalize_text_strips_whitespace():
    """normalize_text() should remove leading and trailing whitespace."""
    assert normalize_text("  hello  ") == "hello"


def test_normalize_text_empty_string():
    """normalize_text() should return an empty string when given one."""
    assert normalize_text("") == ""


def test_normalize_text_numbers_kept():
    """normalize_text() should preserve numeric characters."""
    assert "100" in normalize_text("Water boils at 100C.")
