"""Arabic text primitives and token-string normalization.

Shared utilities used by both tokenizer implementations (e.g. araroopat,
which uses the constants and ``strip_diacritics``) and the intrinsic
metric module (which uses ``clean_token_string`` to normalize tokens
across all tokenizer families before computing morphological metrics).

Notable contents:
    - Arabic character classes (diacritics, long vowels, letters).
    - ``strip_diacritics``: drop tashkeel.
    - ``clean_token_string``: reverse HF ByteLevel BPE byte→char encoding,
      strip subword prefix markers, drop diacritics, keep only Arabic
      letters. Without this, every byte-level BPE token cleans to an
      empty string and ``root_bearing_token_pct`` silently becomes
      ``None`` for any byte-level tokenizer.
"""
from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger("arabic_eval.tokenizers.utils.arabic_text")

# ---------------------------------------------------------------------------
# Arabic character classes
# ---------------------------------------------------------------------------

# Tashkeel / harakat (short vowels and other diacritics)
ARABIC_DIACRITICS = set(
    "ًٌٍَُِّْٰٕٓٔ"
)

# Long-vowel letters (matres lectionis). When approximating a root these are
# usually *not* root consonants, so we strip them in the consonant-skeleton
# fallback. Real root extraction (qalsadi) does not need this list.
ARABIC_LONG_VOWELS = set("اوي")

# All Arabic letters considered by the heuristic (excludes punctuation/digits).
ARABIC_LETTERS = set(
    "ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآؤئة"
)

# Subword prefix markers we may need to strip from token strings.
# - "##" : WordPiece continuation
# - "▁" : SentencePiece word-start ("▁")
# - "Ġ" : ByteLevel BPE word-start ("Ġ")
_TOKEN_PREFIX_MARKERS: Tuple[str, ...] = ("##", "▁", "Ġ")


# ---------------------------------------------------------------------------
# Byte-level BPE decoder
# ---------------------------------------------------------------------------

def _build_bytelevel_decoder() -> dict:
    """Inverse of the GPT-2 / ByteLevel BPE bytes_to_unicode mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTELEVEL_INV = _build_bytelevel_decoder()


def _try_decode_bytelevel(s: str) -> str:
    """Reverse ByteLevel BPE byte→char encoding to recover the UTF-8 string.

    Returns ``s`` unchanged if it doesn't look ByteLevel-encoded.
    """
    if not s:
        return s
    if all(c in _BYTELEVEL_INV for c in s):
        try:
            return bytes(_BYTELEVEL_INV[c] for c in s).decode("utf-8")
        except (UnicodeDecodeError, KeyError):
            return s
    return s


# ---------------------------------------------------------------------------
# String normalization
# ---------------------------------------------------------------------------

def strip_diacritics(text: str) -> str:
    """Remove tashkeel from an Arabic string."""
    return "".join(c for c in text if c not in ARABIC_DIACRITICS)


def clean_token_string(token: str) -> str:
    """Strip subword markers and diacritics from a raw token string.

    Reverses ByteLevel BPE byte→char encoding when applicable, then strips
    leading WordPiece/SentencePiece markers, then keeps only Arabic letters.
    """
    if not token:
        return token
    s = _try_decode_bytelevel(token)
    for marker in _TOKEN_PREFIX_MARKERS:
        if s.startswith(marker):
            s = s[len(marker):]
    s = strip_diacritics(s)
    s = "".join(c for c in s if c in ARABIC_LETTERS or c in ARABIC_LONG_VOWELS)
    return s
