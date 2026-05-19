import json

import pandas as pd
import pytest
import torch

from skeval.dataset.loader import DatasetLoader, SentenceDataset, collate_fn
from skeval.utils.helpers import LabelEncoder, VocabBuilder

# --- VocabBuilder ---


def test_vocab_builder_basic():
    """build() should populate word2idx with all tokens and set is_built to True."""
    vocab = VocabBuilder(min_freq=1)
    vocab.build(["Hello world", "Hello there"])
    assert vocab.is_built
    assert len(vocab) == 5  # <PAD>, <UNK>, hello, world, there


def test_vocab_builder_min_freq():
    """Tokens below min_freq should be excluded from the vocabulary."""
    vocab = VocabBuilder(min_freq=2)
    vocab.build(["hello world", "hello there"])
    assert "hello" in vocab.word2idx
    assert "world" not in vocab.word2idx
    assert "there" not in vocab.word2idx


def test_vocab_builder_pad_unk_reserved():
    """Index 0 must always be <PAD> and index 1 must always be <UNK>."""
    vocab = VocabBuilder()
    vocab.build(["hello"])
    assert vocab.word2idx["<PAD>"] == 0
    assert vocab.word2idx["<UNK>"] == 1


def test_vocab_builder_encode_known():
    """Known tokens should be encoded to indices >= 2 (after <PAD> and <UNK>)."""
    vocab = VocabBuilder()
    vocab.build(["hello world"])
    encoded = vocab.encode("hello world")
    assert len(encoded) == 2
    assert all(idx >= 2 for idx in encoded)


def test_vocab_builder_encode_unknown_token():
    """Tokens not seen during build() should be mapped to the <UNK> index (1)."""
    vocab = VocabBuilder()
    vocab.build(["hello world"])
    encoded = vocab.encode("hello unknown")
    assert encoded[1] == 1


def test_vocab_builder_encode_before_build_raises():
    """encode() must raise ValueError if called before build()."""
    vocab = VocabBuilder()
    with pytest.raises(ValueError, match="not been built"):
        vocab.encode("hello")


def test_vocab_builder_normalises_case():
    """build() should normalise tokens to lowercase before indexing."""
    vocab = VocabBuilder()
    vocab.build(["Hello World"])
    assert "hello" in vocab.word2idx
    assert "world" in vocab.word2idx


def test_vocab_builder_len():
    """A freshly created VocabBuilder should have length 2 (<PAD> and <UNK>)."""
    vocab = VocabBuilder()
    assert len(vocab) == 2


# --- LabelEncoder ---


def test_label_encoder_basic():
    """build() should deduplicate labels and set is_built to True."""
    enc = LabelEncoder()
    enc.build(["fact", "emotion", "fact"])
    assert enc.num_classes == 2
    assert enc.is_built


def test_label_encoder_sorted_mapping():
    """Labels should be assigned indices in alphabetical order for determinism."""
    enc = LabelEncoder()
    enc.build(["opinion", "fact", "emotion"])
    assert enc.label2idx["emotion"] < enc.label2idx["fact"] < enc.label2idx["opinion"]


def test_label_encoder_encode_decode_roundtrip():
    """encode() followed by decode() should return the original label string."""
    enc = LabelEncoder()
    enc.build(["fact", "emotion"])
    assert enc.decode(enc.encode("fact")) == "fact"
    assert enc.decode(enc.encode("emotion")) == "emotion"


def test_label_encoder_encode_before_build_raises():
    """encode() must raise ValueError if called before build()."""
    enc = LabelEncoder()
    with pytest.raises(ValueError, match="not built"):
        enc.encode("fact")


def test_label_encoder_unknown_label_raises():
    """encode() must raise ValueError for a label not seen during build()."""
    enc = LabelEncoder()
    enc.build(["fact"])
    with pytest.raises(ValueError, match="Unknown label"):
        enc.encode("emotion")


def test_label_encoder_decode_before_build_raises():
    """decode() must raise ValueError if called before build()."""
    enc = LabelEncoder()
    with pytest.raises(ValueError, match="not built"):
        enc.decode(0)


# --- SentenceDataset ---


def test_sentence_dataset_len():
    """__len__ should return the number of sentences in the dataset."""
    vocab = VocabBuilder()
    vocab.build(["hello world", "bye world"])
    enc = LabelEncoder()
    enc.build(["pos", "neg"])
    ds = SentenceDataset(["hello world", "bye world"], ["pos", "neg"], vocab, enc)
    assert len(ds) == 2


def test_sentence_dataset_getitem():
    """__getitem__ should return a (LongTensor, LongTensor) pair for a given index."""
    vocab = VocabBuilder()
    vocab.build(["hello world"])
    enc = LabelEncoder()
    enc.build(["pos"])
    ds = SentenceDataset(["hello world"], ["pos"], vocab, enc)
    text, label = ds[0]
    assert isinstance(text, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert text.dtype == torch.long
    assert label.dtype == torch.long


# --- collate_fn ---


def test_collate_fn_output_shapes():
    """collate_fn should produce flat sentence tensor, 1-D label tensor, and offsets."""
    vocab = VocabBuilder()
    vocab.build(["hello world", "bye"])
    enc = LabelEncoder()
    enc.build(["pos", "neg"])
    ds = SentenceDataset(["hello world", "bye"], ["pos", "neg"], vocab, enc)
    sentences, labels, offsets = collate_fn([ds[0], ds[1]])
    assert sentences.ndim == 1
    assert labels.shape == (2,)
    assert offsets.shape == (2,)


# --- DatasetLoader ---


def test_dataset_loader_csv(tmp_path):
    """load_csv() should parse text and label columns from a CSV file."""
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame({"text": ["s1", "s2"], "label": ["fact", "emotion"]})
    df.to_csv(csv_file, index=False)
    texts, labels = DatasetLoader.load_csv(str(csv_file), "text", "label")
    assert texts == ["s1", "s2"]
    assert labels == ["fact", "emotion"]


def test_dataset_loader_json(tmp_path):
    """load_json() should parse text and label keys from a JSONL file."""
    json_file = tmp_path / "data.jsonl"
    with open(json_file, "w") as f:
        f.write(json.dumps({"text": "s1", "label": "fact"}) + "\n")
        f.write(json.dumps({"text": "s2", "label": "emotion"}) + "\n")
    texts, labels = DatasetLoader.load_json(str(json_file), "text", "label")
    assert texts == ["s1", "s2"]
    assert labels == ["fact", "emotion"]


def test_create_dataloader_batches():
    """create_dataloader() should yield batches with correctly shaped label tensors."""
    vocab = VocabBuilder()
    vocab.build(["hello world", "bye world"])
    enc = LabelEncoder()
    enc.build(["pos", "neg"])
    loader = DatasetLoader.create_dataloader(
        ["hello world", "bye world"], ["pos", "neg"], vocab, enc, batch_size=2
    )
    _, labels, _ = next(iter(loader))
    assert labels.shape == (2,)


def test_create_dataloader_num_workers():
    """create_dataloader() should accept num_workers and pin_memory without error."""
    vocab = VocabBuilder()
    vocab.build(["hello"])
    enc = LabelEncoder()
    enc.build(["pos"])
    loader = DatasetLoader.create_dataloader(
        ["hello"], ["pos"], vocab, enc, num_workers=0, pin_memory=False
    )
    assert loader is not None
