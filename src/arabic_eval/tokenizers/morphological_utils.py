"""Arabic morphological analysis helpers for tokenizer evaluation.

Provides root extraction, pattern (wazn) derivation, morpheme segmentation,
and token-string normalization needed by the morphological metrics in
``intrinsic_metrics.py``.

External libraries are optional and degrade gracefully:
    - ``qalsadi``: proper trilateral/quadrilateral root extraction.
    - ``pyarabic``: utility helpers (we only use it if present).
    - ``farasapy``: Farasa morpheme segmentation (already a project dep).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional, Tuple

logger = logging.getLogger("arabic_eval.tokenizers.morphological_utils")

# ---------------------------------------------------------------------------
# Arabic character classes
# ---------------------------------------------------------------------------

# Tashkeel / harakat (short vowels and other diacritics)
ARABIC_DIACRITICS = set(
    "ًٌٍَُِّْٰٕٓٔ"
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

# Special token strings we should ignore.
SPECIAL_TOKEN_STRINGS = {
    "<pad>", "<s>", "</s>", "<unk>", "<mask>", "<bow>", "<eow>",
}

# Pattern letters used for the canonical Arabic wazn skeleton (فعل / فعلل).
_PATTERN_LETTERS_TRI = ("ف", "ع", "ل")
_PATTERN_LETTERS_QUAD = ("ف", "ع", "ل", "ل")


# ---------------------------------------------------------------------------
# String normalization helpers
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


def consonant_skeleton(word: str) -> str:
    """Approximate a root by stripping diacritics and long vowels."""
    no_diac = strip_diacritics(word)
    return "".join(c for c in no_diac if c in ARABIC_LETTERS and c not in ARABIC_LONG_VOWELS)


# ---------------------------------------------------------------------------
# Root extraction
# ---------------------------------------------------------------------------

class RootExtractor:
    """Best-effort Arabic root extractor.

    Tries qalsadi's ``Analex`` (proper morphological analysis) first, then
    Tashaphyne (light stemmer) as a secondary, then a consonant-skeleton
    heuristic. Roots outside 3–4 consonants are rejected since Arabic
    roots are almost exclusively trilateral or quadrilateral.
    """

    def __init__(self) -> None:
        self._analex = None
        self._tashaphyne = None
        self._tried = False

    def _ensure_backends(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            from qalsadi.analex import Analex  # type: ignore
            self._analex = Analex()
            logger.info("qalsadi.Analex available — using it for root extraction.")
        except Exception as e:
            logger.warning("qalsadi.Analex unavailable (%s).", e)
            self._analex = None
        try:
            from tashaphyne.stemming import ArabicLightStemmer  # type: ignore
            self._tashaphyne = ArabicLightStemmer()
            logger.info("Tashaphyne available — using it as a secondary root source.")
        except Exception as e:
            logger.warning("Tashaphyne unavailable (%s).", e)
            self._tashaphyne = None

    @lru_cache(maxsize=20_000)
    def extract(self, word: str) -> Optional[str]:
        """Return the root of ``word`` as a consonant string, or ``None``."""
        word = strip_diacritics(word).strip()
        if not word:
            return None
        self._ensure_backends()

        if self._analex is not None:
            try:
                results = self._analex.check_word(word)
                for r in results or []:
                    root = getattr(r, "root", None)
                    if callable(root):
                        root = root()
                    if isinstance(root, str):
                        root = strip_diacritics(root)
                        if 3 <= len(root) <= 4:
                            return root
            except Exception:
                pass

        if self._tashaphyne is not None:
            try:
                self._tashaphyne.light_stem(word)
                root = strip_diacritics(self._tashaphyne.get_root() or "")
                if 3 <= len(root) <= 4:
                    return root
            except Exception:
                pass

        skeleton = consonant_skeleton(word)
        if 3 <= len(skeleton) <= 4:
            return skeleton
        if len(skeleton) > 4:
            return skeleton[:3]
        return None


# ---------------------------------------------------------------------------
# Pattern (wazn) derivation
# ---------------------------------------------------------------------------

def _root_letters_for(root: str) -> Tuple[str, ...]:
    if len(root) == 4:
        return _PATTERN_LETTERS_QUAD
    return _PATTERN_LETTERS_TRI


def derive_pattern(text: str, root: str) -> Optional[str]:
    """Derive the morphological pattern of ``text`` w.r.t. ``root``.

    Each root consonant in ``text`` (matched greedily, in order) is replaced
    with the corresponding canonical pattern letter (ف/ع/ل). Non-root
    characters are kept verbatim. Returns ``None`` if not all root
    consonants are present in order.

    Example:  derive_pattern("كاتب", "كتب") -> "فاعل"
    """
    if not text or not root:
        return None
    text = strip_diacritics(text)
    pattern_letters = _root_letters_for(root)

    out: List[str] = []
    root_idx = 0
    for ch in text:
        if root_idx < len(root) and ch == root[root_idx]:
            out.append(pattern_letters[root_idx])
            root_idx += 1
        else:
            out.append(ch)

    if root_idx != len(root):
        return None
    return "".join(out)


def stem_pattern_span(text: str, root: str) -> Optional[str]:
    """Pattern of the *minimal span* of ``text`` covering all root letters.

    This trims any clitic-like prefix/suffix characters that lie outside
    the first→last root-letter range. Useful for measuring pattern
    conservation without being fooled by attached و/ال/etc.

    Example:  stem_pattern_span("والكتاب", "كتب") -> "فعال"
              (span is "كتاب", trimmed to root letters; ك→ف, ت→ع, ا kept, ب→ل)
    """
    if not text or not root:
        return None
    text = strip_diacritics(text)
    first = -1
    root_idx = 0
    for i, ch in enumerate(text):
        if ch == root[root_idx]:
            if first == -1:
                first = i
            root_idx += 1
            if root_idx == len(root):
                return derive_pattern(text[first : i + 1], root)
    return None


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------

def contains_subsequence(text: str, root: str) -> bool:
    """True if every root letter appears in ``text`` in order."""
    i = 0
    for c in text:
        if i < len(root) and c == root[i]:
            i += 1
    return i == len(root)


def filter_content_tokens(tokens: List[str]) -> List[str]:
    """Drop special tokens and empty cleaned strings."""
    out: List[str] = []
    for t in tokens:
        if t in SPECIAL_TOKEN_STRINGS:
            continue
        cleaned = clean_token_string(t)
        if cleaned:
            out.append(cleaned)
    return out


def aligned_token_offsets(tokens: List[str], original_word: str) -> Optional[List[Tuple[int, int]]]:
    """Return ``[(start, end), ...]`` char offsets into ``original_word`` for each token.

    Tokens are concatenated (after cleaning) and matched as a contiguous
    cover of the diacritic-stripped word. Returns ``None`` if the
    reconstruction does not match the word — usually because the tokenizer
    introduced/dropped characters (e.g. ByteLevel BPE artifacts that
    cleaning couldn't recover).
    """
    target = strip_diacritics(original_word)
    target_no_long_vowels_kept = target  # we keep long vowels in cleaning
    cleaned = [clean_token_string(t) for t in tokens]

    pos = 0
    offsets: List[Tuple[int, int]] = []
    for piece in cleaned:
        if not piece:
            offsets.append((pos, pos))
            continue
        # Greedy match starting at `pos`.
        if target.startswith(piece, pos):
            offsets.append((pos, pos + len(piece)))
            pos += len(piece)
        else:
            return None
    if pos != len(target):
        return None
    _ = target_no_long_vowels_kept  # silence linter
    return offsets


# ---------------------------------------------------------------------------
# Morpheme segmentation (Farasa)
# ---------------------------------------------------------------------------

class MorphemeSegmenter:
    """Wrap Farasa segmenter; degrade to ``None`` segments on failure."""

    _FARASA_SEP = "+"

    def __init__(self) -> None:
        self._segmenter = None
        self._tried = False

    def _ensure(self):
        if self._tried:
            return self._segmenter
        self._tried = True
        try:
            from farasa.segmenter import FarasaSegmenter  # type: ignore
            self._segmenter = FarasaSegmenter(interactive=True)
            logger.info("Farasa segmenter loaded for morpheme metrics.")
        except Exception as e:
            logger.warning(
                "Farasa segmenter unavailable (%s) — morpheme_integrity_rate "
                "will be skipped.",
                e,
            )
            self._segmenter = None
        return self._segmenter

    def segment_word(self, word: str) -> Optional[List[str]]:
        """Return Farasa morpheme segments for one word, e.g. ['و','ال','كتاب']."""
        seg = self._ensure()
        if seg is None:
            return None
        try:
            out = seg.segment(word)
        except Exception:
            return None
        # Farasa joins morphemes with '+' inside a word.
        # Collapse any whitespace it may insert and split.
        parts: List[str] = []
        for w in out.split():
            parts.extend(p for p in w.split(self._FARASA_SEP) if p)
        return parts or None
