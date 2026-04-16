"""Arabic text preprocessing and normalization."""
from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from datasets import DatasetDict


# Arabic diacritics (tashkeel) Unicode range
_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")

# Normalize various alef forms to bare alef
_ALEF_VARIANTS = re.compile(r"[\u0622\u0623\u0625\u0671]")

# Normalize teh marbuta to heh
_TEH_MARBUTA = "\u0629"
_HEH = "\u0647"

# Tatweel (kashida) stretching character
_TATWEEL = "\u0640"


def normalize_arabic(
    text: str,
    normalize_unicode: bool = True,
    remove_diacritics: bool = False,
    normalize_alef: bool = True,
    remove_tatweel: bool = True,
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

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataset(
    dataset,  # DatasetDict
    text_column: str = "text",
    normalize_unicode: bool = True,
    remove_diacritics: bool = False,
    min_text_length: int = 10,
    **kwargs,
) -> DatasetDict:
    """Apply preprocessing to all splits in a DatasetDict."""

    def _process(example):
        text = example[text_column]
        if text is None:
            text = ""
        text = normalize_arabic(
            text,
            normalize_unicode=normalize_unicode,
            remove_diacritics=remove_diacritics,
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
