import pytest

from skeval.utils.helpers import VocabBuilder, normalize_text


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


# --- VocabBuilder: reserved token layout ---


@pytest.fixture()
def built_vocab():
    v = VocabBuilder()
    v.build(["Water boils at 100C", "I love you"])
    return v


def test_cls_token_reserved_at_index_2():
    """<CLS> must be present at index 2 before build() is called."""
    v = VocabBuilder()
    assert v.word2idx["<CLS>"] == 2
    assert v.idx2word[2] == "<CLS>"


def test_word_indices_start_at_3(built_vocab):
    """Vocabulary words must be assigned indices starting at 3."""
    word_indices = [
        idx
        for token, idx in built_vocab.word2idx.items()
        if token not in ("<PAD>", "<UNK>", "<CLS>")
    ]
    assert all(idx >= 3 for idx in word_indices)
    assert min(word_indices) == 3


# --- VocabBuilder: encode_padded ---


def test_encode_padded_exact_length(built_vocab):
    """encode_padded() must return exactly max_len elements."""
    assert len(built_vocab.encode_padded("Water boils", max_len=6)) == 6


def test_encode_padded_cls_at_position_0(built_vocab):
    """encode_padded() must place <CLS> (index 2) at position 0."""
    result = built_vocab.encode_padded("Water boils", max_len=6)
    assert result[0] == 2


def test_encode_padded_right_padded_with_zeros(built_vocab):
    """encode_padded() must right-pad short sequences with <PAD> (index 0)."""
    result = built_vocab.encode_padded("Water", max_len=5)
    assert result[-1] == 0
    assert result.count(0) == 3  # max_len - 1 (CLS) - 1 (water) = 3 pads


def test_encode_padded_truncates_long_sentences(built_vocab):
    """encode_padded() must truncate sentences longer than max_len - 1 words."""
    result = built_vocab.encode_padded("Water boils at 100C I love you", max_len=4)
    assert len(result) == 4


def test_encode_padded_before_build_raises():
    """encode_padded() must raise ValueError if build() has not been called."""
    v = VocabBuilder()
    with pytest.raises(ValueError, match="not been built"):
        v.encode_padded("hello", max_len=5)


def test_encode_padded_max_len_too_small_raises(built_vocab):
    """encode_padded() must raise ValueError when max_len < 2."""
    with pytest.raises(ValueError, match="max_len"):
        built_vocab.encode_padded("hello", max_len=1)
