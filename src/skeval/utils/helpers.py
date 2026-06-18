import re
from collections import Counter
from typing import Dict, List


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation from a string.

    Args:
        text: Raw input string.

    Returns:
        Normalized string with punctuation removed and leading/trailing
        whitespace stripped.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


class VocabBuilder:
    """Builds a vocabulary from a text corpus and encodes sentences as indices.

    Three indices are permanently reserved:

    ======  =======  =====================================================
    Index   Token    Purpose
    ======  =======  =====================================================
    0       <PAD>    Padding — fills unused positions in padded sequences
    1       <UNK>    Unknown — replaces out-of-vocabulary tokens
    2       <CLS>    Classification — prepended by ``encode_padded()``
    ======  =======  =====================================================

    All vocabulary words are assigned indices starting at 3.

    Attributes:
        min_freq: Minimum number of occurrences for a token to be included.
        word2idx: Mapping from token string to integer index.
        idx2word: Reverse mapping from integer index to token string.
        is_built: Whether ``build()`` has been called.
    """

    def __init__(self, min_freq: int = 1) -> None:
        """Initialise an empty vocabulary with the three reserved tokens.

        Args:
            min_freq: Tokens appearing fewer than this many times across the
                corpus are excluded from the vocabulary.
        """
        self.min_freq = min_freq
        # 0=<PAD>, 1=<UNK>, 2=<CLS> — all reserved; words start at index 3
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>", 2: "<CLS>"}
        self.is_built: bool = False

    def build(self, sentences: List[str]) -> None:
        """Populate ``word2idx`` and ``idx2word`` from a list of sentences.

        Args:
            sentences: Training corpus. Each element is a single sentence string.
        """
        counter = Counter(
            word for sentence in sentences for word in normalize_text(sentence).split()
        )
        for idx, word in enumerate(
            (w for w, c in counter.items() if c >= self.min_freq),
            start=len(self.word2idx),
        ):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.is_built = True

    def encode(self, sentence: str) -> List[int]:
        """Convert a sentence into a list of vocabulary indices.

        Unknown tokens are mapped to the ``<UNK>`` index (1).

        Args:
            sentence: Raw input sentence.

        Returns:
            List of integer indices, one per token after normalisation.

        Raises:
            ValueError: If ``build()`` has not been called yet.
        """
        if not self.is_built:
            raise ValueError("Vocabulary has not been built yet. Call build() first.")
        words = normalize_text(sentence).split()
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]

    def encode_padded(self, sentence: str, max_len: int) -> List[int]:
        """Encode a sentence as a ``[CLS]``-prefixed, fixed-length token list.

        Prepends ``<CLS>`` (index 2), encodes up to ``max_len - 1`` words,
        then right-pads with ``<PAD>`` (index 0) to exactly ``max_len``
        tokens. Sentences longer than ``max_len - 1`` words are truncated.

        Args:
            sentence: Raw input sentence.
            max_len: Total sequence length including the ``<CLS>`` token.
                Must be at least 2 (room for ``<CLS>`` + one token).

        Returns:
            List of exactly ``max_len`` integer indices.

        Raises:
            ValueError: If ``build()`` has not been called yet.
            ValueError: If ``max_len`` is less than 2.
        """
        if not self.is_built:
            raise ValueError("Vocabulary has not been built yet. Call build() first.")
        if max_len < 2:
            raise ValueError(f"max_len must be at least 2 (got {max_len}).")
        cls_idx = self.word2idx["<CLS>"]
        pad_idx = self.word2idx["<PAD>"]
        word_indices = self.encode(sentence)[: max_len - 1]
        padding = [pad_idx] * (max_len - 1 - len(word_indices))
        return [cls_idx] + word_indices + padding

    def __len__(self) -> int:
        return len(self.word2idx)


class LabelEncoder:
    """Encodes string labels to integers and decodes them back.

    Labels are sorted alphabetically before assignment so that the mapping
    is deterministic across runs.

    Attributes:
        label2idx: Mapping from label string to integer class index.
        idx2label: Reverse mapping from integer class index to label string.
        is_built: Whether ``build()`` has been called.
    """

    def __init__(self) -> None:
        self.label2idx: Dict[str, int] = {}
        self.idx2label: Dict[int, str] = {}
        self.is_built: bool = False

    def build(self, labels: List[str]) -> None:
        """Assign a unique integer index to each unique label.

        Args:
            labels: Full list of training labels (duplicates allowed).
        """
        unique_labels = sorted(set(labels))
        for idx, label in enumerate(unique_labels):
            self.label2idx[label] = idx
            self.idx2label[idx] = label
        self.is_built = True

    def encode(self, label: str) -> int:
        """Convert a label string to its integer index.

        Args:
            label: Label string seen during ``build()``.

        Returns:
            Integer class index.

        Raises:
            ValueError: If ``build()`` has not been called or the label is unknown.
        """
        if not self.is_built:
            raise ValueError("LabelEncoder not built.")
        if label not in self.label2idx:
            raise ValueError(f"Unknown label: {label}")
        return self.label2idx[label]

    def decode(self, idx: int) -> str:
        """Convert an integer class index back to its label string.

        Args:
            idx: Integer index returned by the model.

        Returns:
            Corresponding label string.

        Raises:
            ValueError: If ``build()`` has not been called.
        """
        if not self.is_built:
            raise ValueError("LabelEncoder not built.")
        return self.idx2label[idx]

    @property
    def num_classes(self) -> int:
        """Total number of unique classes seen during ``build()``."""
        return len(self.label2idx)
