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
    # alef normalization runs first, so a hamza on the following word does not block the join
    assert normalize_arabic("و أحمد", join_lone_waw=True) == "واحمد"
    assert normalize_arabic("و أحمد", join_lone_waw=True, normalize_alef=False) == "وأحمد"
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
    assert cfg["train"]["text"][0] == "الكتاب والقلم على الطاولة"
