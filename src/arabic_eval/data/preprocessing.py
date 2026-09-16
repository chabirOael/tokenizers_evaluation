"""Arabic text preprocessing and normalization."""
from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from datasets import DatasetDict


# Arabic diacritics (tashkeel) Unicode range
_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")

# Alef variants (آ أ إ ٱ) → bare alef. OFF by default since 2026-09-16: the
# fold rewrote 13.4 % of ArabicText-Large's words (إلى 65,533 vs الى 102 in
# the raw corpus) into spellings that the Phase 1–3 corpora, the pretraining
# mix and every eval benchmark never use — a train/eval skew for every
# from-scratch tokenizer while the native Llama/Qwen tokenizers saw raw text.
_ALEF_VARIANTS = re.compile(r"[\u0622\u0623\u0625\u0671]")

# Normalize teh marbuta to heh
_TEH_MARBUTA = "\u0629"
_HEH = "\u0647"

# Tatweel (kashida) stretching character
_TATWEEL = "\u0640"

# A و written as a word of its own, followed by whitespace and an Arabic word:
# "و القمر" → "والقمر". Arabic letters only (no digits, tashkeel or punctuation)
# on both sides of the test, so words that merely *end* in و (هو, أبو, نحو)
# are never glued to what follows, and "و 2024" / "و HD" stay as they are.
_AR_LETTER = "\u0621-\u063A\u0641-\u064A\u0671-\u06D3\u06FA-\u06FC"
_TASHKEEL = "\u064B-\u0652\u0670"
_LONE_WAW = re.compile(
    rf"(?<![{_AR_LETTER}{_TASHKEEL}])(\u0648[{_TASHKEEL}]?)\s+(?=[{_AR_LETTER}])"
)


def join_lone_waw(text: str) -> str:
    """Delete the whitespace after a lone conjunction و: ``و القمر`` → ``والقمر``.

    Only a و that is a word by itself (nothing Arabic before it) and that is
    followed by an Arabic letter is joined; its own tashkeel is kept. One
    left-to-right pass over every match, so ``و و القمر`` becomes ``ووالقمر``.
    """
    return _LONE_WAW.sub(r"\1", text)


def normalize_arabic(
    text: str,
    normalize_unicode: bool = True,
    remove_diacritics: bool = False,
    normalize_alef: bool = False,
    remove_tatweel: bool = True,
    join_lone_waw: bool = False,
) -> str:
    """Apply Arabic-specific text normalization."""
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)

    if remove_diacritics:
        text = _DIACRITICS.sub("", text)

    if normalize_alef:
        text = _ALEF_VARIANTS.sub("\u0627", text)

    if remove_tatweel:
        text = text.replace(_TATWEEL, "")

    if join_lone_waw:
        text = _LONE_WAW.sub(r"\1", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataset(
    dataset,  # DatasetDict
    text_column: str = "text",
    normalize_unicode: bool = True,
    remove_diacritics: bool = False,
    normalize_alef: bool = False,
    remove_tatweel: bool = True,
    min_text_length: int = 10,
    join_lone_waw: bool = False,
    **kwargs,
) -> DatasetDict:
    """Apply preprocessing to all splits in a DatasetDict.

    Every ``normalize_arabic`` knob is plumbed explicitly (the YAML
    ``data.preprocessing`` block maps 1:1 onto these arguments). An unknown
    key is an error rather than a silent no-op: before 2026-09-16
    ``normalize_alef`` was swallowed by ``**kwargs`` and the function-level
    default (then ``True``) silently applied whatever the YAML said.
    """
    if kwargs:
        raise ValueError(
            f"preprocess_dataset: unknown preprocessing key(s) {sorted(kwargs)}; "
            "known keys: normalize_unicode, remove_diacritics, normalize_alef, "
            "remove_tatweel, min_text_length, join_lone_waw"
        )

    def _process(example):
        text = example[text_column]
        if text is None:
            text = ""
        text = normalize_arabic(
            text,
            normalize_unicode=normalize_unicode,
            remove_diacritics=remove_diacritics,
            normalize_alef=normalize_alef,
            remove_tatweel=remove_tatweel,
            join_lone_waw=join_lone_waw,
        )
        example[text_column] = text
        return example

    dataset = dataset.map(_process, desc="Preprocessing")

    # Filter short texts
    if min_text_length > 0:
        dataset = dataset.filter(
            lambda ex: len(ex[text_column]) >= min_text_length,
            desc="Filtering short texts",
        )

    return dataset
