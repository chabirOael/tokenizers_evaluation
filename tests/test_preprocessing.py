"""``normalize_arabic`` / ``preprocess_dataset``: the lone-و join (added 2026-09-16).

A و written as a word of its own (``و القمر``) is glued to the Arabic word
after it; words that merely end in و (هو, أبو, نحو), a و before a digit /
Latin / punctuation and a و at the end of the text are untouched.
"""
from __future__ import annotations

import pytest

from arabic_eval.config import DataConfig
from arabic_eval.data.preprocessing import join_lone_waw, normalize_arabic, preprocess_dataset


@pytest.mark.parametrize("text, expected", [
    ("و القمر", "والقمر"),
    ("الكتاب و القلم و الدفتر", "الكتاب والقلم والدفتر"),
    ("،و القمر", "،والقمر"),                 # punctuation before the و: still a lone و
    ("(و القمر)", "(والقمر)"),
    ("وَ القمر", "وَالقمر"),                 # its own tashkeel is kept
    ("و\nالقمر", "والقمر"),                  # any whitespace run
    ("و   القمر", "والقمر"),
    ("و و القمر", "ووالقمر"),                # every match, left to right
    ("هو القمر", "هو القمر"),                 # word-final و is not a lone و
    ("هُوَ القمر", "هُوَ القمر"),             # ... even under tashkeel
    ("أبو القاسم", "أبو القاسم"),
    ("نحو الشمال", "نحو الشمال"),
    ("و 2024", "و 2024"),                     # only an Arabic letter may follow
    ("و HD 209458", "و HD 209458"),
    ("و ، القمر", "و ، القمر"),
    ("القمر و", "القمر و"),                   # nothing follows
    ("و", "و"),
    ("", ""),
    ("لا و لكن", "لا ولكن"),
])
def test_join_lone_waw(text, expected):
    assert join_lone_waw(text) == expected


def test_normalize_arabic_flag_default_off_and_order():
    # off by default in the function (the config turns it on), and applied before whitespace collapsing
    assert normalize_arabic("و  القمر") == "و القمر"
    assert normalize_arabic("و  القمر", join_lone_waw=True) == "والقمر"
    # alef folding is OFF by default (2026-09-16); when on it runs first, so a
    # hamza on the following word never blocks the join either way
    assert normalize_arabic("و أحمد", join_lone_waw=True) == "وأحمد"
    assert normalize_arabic("و أحمد", join_lone_waw=True, normalize_alef=True) == "واحمد"
    # tashkeel removal first: "وَ القمر" → "و القمر" → "والقمر"
    assert normalize_arabic("وَ القمر", join_lone_waw=True, remove_diacritics=True) == "والقمر"


def test_config_default_and_dataset_plumbing():
    assert DataConfig().preprocessing["join_lone_waw"] is True
    datasets = pytest.importorskip("datasets")
    ds = datasets.DatasetDict({"train": datasets.Dataset.from_dict({"text": ["الكتاب و القلم على الطاولة", "هو القمر و 2024 و النجم"]})})
    on = preprocess_dataset(ds, text_column="text", min_text_length=0, join_lone_waw=True)
    assert on["train"]["text"] == ["الكتاب والقلم على الطاولة", "هو القمر و 2024 والنجم"]
    off = preprocess_dataset(ds, text_column="text", min_text_length=0)
    assert off["train"]["text"] == ["الكتاب و القلم على الطاولة", "هو القمر و 2024 و النجم"]
    # the base config's preprocessing dict is accepted as-is by preprocess_dataset
    cfg = preprocess_dataset(ds, text_column="text", **DataConfig().preprocessing)
    assert cfg["train"]["text"] == ["الكتاب والقلم على الطاولة", "هو القمر و 2024 والنجم"]


def test_alef_normalization_is_off_and_plumbed():
    """2026-09-16: hamza-alef is kept by default, and the YAML key actually reaches normalize_arabic.

    Before, ``preprocess_dataset`` swallowed ``normalize_alef`` in ``**kwargs``
    and the function default (True) applied regardless of the YAML.
    """
    assert DataConfig().preprocessing["normalize_alef"] is False
    assert normalize_arabic("إلى أن آمن ٱلله") == "إلى أن آمن ٱلله"
    assert normalize_arabic("إلى أن آمن ٱلله", normalize_alef=True) == "الى ان امن الله"
    datasets = pytest.importorskip("datasets")
    ds = datasets.DatasetDict({"train": datasets.Dataset.from_dict({"text": ["ذهب إلى المدرسة أمس"]})})
    assert preprocess_dataset(ds, text_column="text", min_text_length=0)["train"]["text"] == ["ذهب إلى المدرسة أمس"]
    assert preprocess_dataset(ds, text_column="text", min_text_length=0, normalize_alef=True)["train"]["text"] == ["ذهب الى المدرسة امس"]
    assert preprocess_dataset(ds, text_column="text", **DataConfig().preprocessing)["train"]["text"] == ["ذهب إلى المدرسة أمس"]
    # tatweel is plumbed too
    assert preprocess_dataset(ds.map(lambda e: {"text": "ذهـــب"}), text_column="text", min_text_length=0)["train"]["text"] == ["ذهب"]
    assert preprocess_dataset(ds.map(lambda e: {"text": "ذهـــب"}), text_column="text", min_text_length=0, remove_tatweel=False)["train"]["text"] == ["ذهـــب"]


def test_unknown_preprocessing_key_is_rejected():
    datasets = pytest.importorskip("datasets")
    ds = datasets.DatasetDict({"train": datasets.Dataset.from_dict({"text": ["نص"]})})
    with pytest.raises(ValueError, match="unknown preprocessing key"):
        preprocess_dataset(ds, text_column="text", normalise_alef=True)


def test_training_provenance_sidecar(tmp_path):
    from arabic_eval.tokenizers.provenance import (
        PROVENANCE_FILENAME, read_training_provenance, write_training_provenance,
    )
    assert read_training_provenance(tmp_path) is None
    out = write_training_provenance(tmp_path, dataset_name="ds", preprocessing=DataConfig().preprocessing,
                                    num_texts=3, entry_point="test", tokenizer_type="bpe")
    assert out == tmp_path / PROVENANCE_FILENAME
    rec = read_training_provenance(tmp_path)
    assert rec["preprocessing"]["normalize_alef"] is False and rec["preprocessing_applied"] is True
    assert rec["num_texts"] == 3 and rec["tokenizer_type"] == "bpe"
    raw = write_training_provenance(tmp_path / "raw", dataset_name="ds", preprocessing=None,
                                    num_texts=1, entry_point="test")
    assert read_training_provenance(raw.parent)["preprocessing_applied"] is False
