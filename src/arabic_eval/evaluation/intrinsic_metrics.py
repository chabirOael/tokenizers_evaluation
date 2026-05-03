"""Intrinsic tokenizer evaluation metrics.

Two public entry points:
    - ``compute_intrinsic_metrics``: size/coverage metrics (fertility,
      compression ratio, UNK rate, vocab coverage) plus, optionally, the
      Arabic morphological metrics below.
    - ``compute_morphological_metrics``: seven Arabic morphological
      metrics (root_conservation_rate, pattern_conservation_rate,
      morpheme_integrity_rate, clitic_separation_accuracy,
      semantic_fragmentation_ratio, root_bearing_token_pct,
      pattern_bearing_token_pct).

The morphological metrics rely on:
    - ``RootExtractor``: qalsadi → tashaphyne → consonant-skeleton fallback
      chain for Arabic root extraction.
    - ``MorphemeSegmenter``: Farasa subprocess wrapper for morpheme
      segmentation.
Both backends are defined at the bottom of this module — they are pure
implementation details of the morphological metrics.
"""
from __future__ import annotations

import logging
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from arabic_eval.tokenizers.araroopat_backend import (
    ENCLITIC_SURFACES,
    PROCLITIC_SURFACES,
)
from arabic_eval.tokenizers.base import BaseTokenizer
from arabic_eval.tokenizers.utils.arabic_text import (
    ARABIC_LETTERS,
    ARABIC_LONG_VOWELS,
    clean_token_string,
    strip_diacritics,
)

logger = logging.getLogger("arabic_eval.evaluation.intrinsic_metrics")

DEFAULT_MORPH_SAMPLE_SIZE = 500

# Special token strings we should ignore when computing per-token metrics.
SPECIAL_TOKEN_STRINGS = {
    "<pad>", "<s>", "</s>", "<unk>", "<mask>", "<bow>", "<eow>",
}

# Pattern letters used for the canonical Arabic wazn skeleton (فعل / فعلل).
_PATTERN_LETTERS_TRI = ("ف", "ع", "ل")
_PATTERN_LETTERS_QUAD = ("ف", "ع", "ل", "ل")


# ===========================================================================
# Public API
# ===========================================================================

def compute_intrinsic_metrics(
    tokenizer: BaseTokenizer,
    texts: List[str],
    morphological_metrics: bool = True,
    morph_sample_size: int = DEFAULT_MORPH_SAMPLE_SIZE,
    morph_seed: int = 42,
) -> Dict[str, float]:
    """Compute intrinsic tokenizer quality metrics on a corpus.

    Returns the standard size/coverage metrics plus, if
    ``morphological_metrics`` is True, the five Arabic morphological metrics
    (see :func:`compute_morphological_metrics`).
    """
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_unk = 0
    unique_words: set = set()
    unique_words_with_unk: set = set()
    token_counts: List[int] = []

    unk_id = tokenizer.special_tokens.get("unk_token")

    for text in tqdm(texts, desc="Intrinsic metrics", unit="text"):
        words = text.split()
        total_words += len(words)
        total_chars += len(text)

        encoded = tokenizer.encode(text)
        n_tokens = len(encoded.input_ids)
        total_tokens += n_tokens
        token_counts.append(n_tokens)

        if unk_id is not None:
            total_unk += encoded.input_ids.count(unk_id)

        for word in words:
            unique_words.add(word)
            word_enc = tokenizer.encode(word)
            if unk_id is not None and unk_id in word_enc.input_ids:
                unique_words_with_unk.add(word)

    n_texts = len(texts)
    fertility = total_tokens / max(total_words, 1)
    compression_ratio = total_chars / max(total_tokens, 1)
    unk_rate = total_unk / max(total_tokens, 1)
    vocab_coverage = 1.0 - len(unique_words_with_unk) / max(len(unique_words), 1)
    avg_token_count = total_tokens / max(n_texts, 1)

    metrics: Dict[str, float] = {
        "fertility": round(fertility, 4),
        "compression_ratio": round(compression_ratio, 4),
        "unk_rate": round(unk_rate, 6),
        "vocab_coverage": round(vocab_coverage, 4),
        "avg_token_count": round(avg_token_count, 2),
        "vocab_size": tokenizer.vocab_size,
    }

    if morphological_metrics:
        morph = compute_morphological_metrics(
            tokenizer,
            texts,
            sample_size=morph_sample_size,
            seed=morph_seed,
        )
        metrics.update(morph)

    logger.info("Intrinsic metrics: %s", metrics)
    return metrics


def compute_morphological_metrics(
    tokenizer: BaseTokenizer,
    texts: List[str],
    sample_size: int = DEFAULT_MORPH_SAMPLE_SIZE,
    seed: int = 42,
    use_farasa: bool = True,
) -> Dict[str, float]:
    """Seven Arabic morphological metrics.

    Per-word metrics (averaged over the word sample):
        - root_conservation_rate: fraction of words whose full root appears
          as a subsequence inside a single token.
        - pattern_conservation_rate: fraction of words whose stem-span
          pattern (root letters + their immediate context, clitics trimmed)
          is recoverable from a single token.
        - morpheme_integrity_rate: fraction of Farasa internal morpheme
          boundaries that align with token boundaries (averaged over words
          that have at least one internal boundary).
        - clitic_separation_accuracy: fraction of clitic↔stem boundaries
          (proclitic-end and enclitic-start positions, identified via the
          CAMeL Tools surface inventory shared with AraRooPat) that align
          with token boundaries. Pooled across the sample (not averaged
          per-word) so words with more clitics weigh proportionally more.
          Accuracy ceiling: Farasa is sometimes inconsistent on the ``أ``
          question proclitic; CSA may slightly under-credit on
          interrogatives.

    Per-sample metrics:
        - semantic_fragmentation_ratio: total *raw* non-special tokens
          divided by total Farasa morphemes across the sample. SFR is
          computed independently of token↔word alignment success — token
          and morpheme counts are alignment-free, so tokenizers that
          occasionally fail to round-trip (ByteLevel BPE artifacts, etc.)
          still contribute fairly. Uses raw (pre-clean) token count so
          byte-level tokenizers like Charformer are not penalized to 0.

    Per-token metrics (averaged over all tokens in tokenized sample):
        - root_bearing_token_pct: % of tokens that contain at least one
          full Arabic root from the sample's root set.
        - pattern_bearing_token_pct: % of tokens that match a known stem
          pattern from the sample's pattern set.

    All metrics share the same word sample (deterministic, seed=42 by
    default) so they are directly comparable across tokenizers. Words for
    which root extraction fails are skipped uniformly across all metrics
    to keep the sampled population identical.
    """
    sample = _sample_words(texts, sample_size, seed)
    if not sample:
        logger.warning("Morphological metrics: empty word sample, skipping.")
        return _empty_morph_metrics()

    root_extractor = RootExtractor()
    segmenter = MorphemeSegmenter() if use_farasa else None

    root_conserved = 0
    root_total = 0
    pattern_conserved = 0
    pattern_total = 0
    integrity_sum = 0.0
    integrity_count = 0
    csa_resp_sum = 0
    csa_total_sum = 0
    sfr_tok_sum = 0
    sfr_morph_sum = 0

    sample_roots: set = set()
    sample_patterns: set = set()
    all_token_strings: List[str] = []
    # Track *raw* (pre-cleaning) non-special token count separately so that
    # byte-level tokenizers like Charformer — whose tokens are single bytes
    # that all clean to empty Arabic-letter strings — produce a mechanical
    # 0.0 instead of None on the *_bearing_token_pct metrics. With only
    # ``len(all_token_strings)`` we cannot distinguish "no tokens generated"
    # (truly not measurable -> None) from "tokens generated but none carry
    # Arabic letters" (a real, mechanical 0%).
    raw_token_count = 0

    for word in tqdm(sample, desc="Morphological metrics", unit="word"):
        root = root_extractor.extract(word)
        if root is None:
            continue
        sample_roots.add(root)

        word_pattern = derive_pattern(word, root)
        stem_pattern = stem_pattern_span(word, root)
        if stem_pattern:
            sample_patterns.add(stem_pattern)

        tokens = _word_tokens(tokenizer, word)
        content = filter_content_tokens(tokens)
        all_token_strings.extend(content)
        raw_token_count += sum(1 for t in tokens if t not in SPECIAL_TOKEN_STRINGS)

        # --- root_conservation_rate ---
        root_total += 1
        if any(contains_subsequence(t, root) for t in content):
            root_conserved += 1

        # --- pattern_conservation_rate ---
        if word_pattern and stem_pattern:
            pattern_total += 1
            if any(stem_pattern_span(t, root) == stem_pattern for t in content):
                pattern_conserved += 1

        # --- morpheme_integrity_rate, clitic_separation_accuracy, SFR ---
        if segmenter is not None:
            wm = _morpheme_metrics_for_word(word, tokens, segmenter)
            if wm is not None:
                if wm["integrity"] is not None:
                    integrity_sum += wm["integrity"]
                    integrity_count += 1
                if wm["csa_total"]:
                    csa_resp_sum += wm["csa_respected"]
                    csa_total_sum += wm["csa_total"]
                # SFR accumulates regardless of alignment / boundary
                # presence — token and morpheme counts are alignment-free.
                sfr_tok_sum += wm["raw_token_count"]
                sfr_morph_sum += wm["morpheme_count"]

    # --- per-token aggregates over the sampled tokens ---
    if all_token_strings and sample_roots:
        root_bearing = sum(
            1 for t in all_token_strings
            if any(contains_subsequence(t, r) for r in sample_roots)
        )
        root_bearing_pct = 100.0 * root_bearing / len(all_token_strings)
    elif raw_token_count > 0 and sample_roots:
        # Tokens were emitted but every one cleaned to an empty Arabic-letter
        # string — i.e. byte/sub-character tokenizers like Charformer where a
        # single byte cannot carry an Arabic letter. The metric is then
        # mechanically 0% (no token can contain a 3-letter root), not "not
        # measured."
        root_bearing_pct = 0.0
    else:
        root_bearing_pct = None

    if all_token_strings and sample_patterns:
        pattern_bearing = 0
        for t in all_token_strings:
            for r in sample_roots:
                p = stem_pattern_span(t, r)
                if p and p in sample_patterns:
                    pattern_bearing += 1
                    break
        pattern_bearing_pct = 100.0 * pattern_bearing / len(all_token_strings)
    elif raw_token_count > 0 and sample_patterns:
        # Same mechanical-zero rationale as root_bearing_token_pct above.
        pattern_bearing_pct = 0.0
    else:
        pattern_bearing_pct = None

    return {
        "root_conservation_rate": _safe_rate(root_conserved, root_total),
        "pattern_conservation_rate": _safe_rate(pattern_conserved, pattern_total),
        "morpheme_integrity_rate": (
            round(integrity_sum / integrity_count, 4) if integrity_count else None
        ),
        "clitic_separation_accuracy": _safe_rate(csa_resp_sum, csa_total_sum),
        "semantic_fragmentation_ratio": (
            round(sfr_tok_sum / sfr_morph_sum, 4) if sfr_morph_sum else None
        ),
        "root_bearing_token_pct": (
            round(root_bearing_pct, 2) if root_bearing_pct is not None else None
        ),
        "pattern_bearing_token_pct": (
            round(pattern_bearing_pct, 2) if pattern_bearing_pct is not None else None
        ),
        "morph_sample_size": root_total,
    }


# ===========================================================================
# Internal helpers (pure functions used by the metrics above)
# ===========================================================================

def _sample_words(texts: List[str], n: int, seed: int) -> List[str]:
    """Sample up to ``n`` distinct content words from ``texts``."""
    rng = random.Random(seed)
    seen: set = set()
    pool: List[str] = []
    for text in texts:
        for w in text.split():
            w = strip_diacritics(w)
            if len(w) < 3 or w in seen:
                continue
            seen.add(w)
            pool.append(w)
    rng.shuffle(pool)
    return pool[:n]


def _word_tokens(tokenizer: BaseTokenizer, word: str) -> List[str]:
    """Encode a single word and return its non-special token strings."""
    out = tokenizer.encode(word)
    if not out.tokens:
        # Reconstruct from IDs as a fallback: decode each id individually.
        try:
            return [tokenizer.decode([i]) for i in out.input_ids]
        except Exception:
            return []
    return out.tokens


def _empty_morph_metrics() -> Dict[str, Optional[float]]:
    return {
        "root_conservation_rate": None,
        "pattern_conservation_rate": None,
        "morpheme_integrity_rate": None,
        "clitic_separation_accuracy": None,
        "semantic_fragmentation_ratio": None,
        "root_bearing_token_pct": None,
        "pattern_bearing_token_pct": None,
        "morph_sample_size": 0,
    }


def _safe_rate(num: int, denom: int) -> Optional[float]:
    if denom == 0:
        return None
    return round(num / denom, 4)


def _clitic_boundaries(morphemes: List[str]) -> set:
    """Char offsets within a word that flank clitics (proclitic-end /
    enclitic-start positions).

    Walks proclitics from the left and enclitics from the right, stopping
    at the first morpheme that is not a known clitic surface (the stem).
    Membership tests are performed on the *diacritic-stripped* surface so
    that diacritized Farasa output is handled correctly. ``ك`` (proclitic
    "like" vs. enclitic 2ms pronoun) is disambiguated by position — left
    walks only test PROCLITIC_SURFACES, right walks only ENCLITIC_SURFACES.

    The boundary index is the cumulative *diacritic-stripped* length up to
    the relevant split, matching the convention used by
    ``aligned_token_offsets`` (which also operates on the stripped word).
    """
    if len(morphemes) < 2:
        return set()
    bounds: set = set()

    # Proclitics: left → first non-clitic. Never treat the final morpheme
    # as a proclitic (a word with no stem is not a thing).
    cum = 0
    for m in morphemes[:-1]:
        if strip_diacritics(m) in PROCLITIC_SURFACES:
            cum += len(strip_diacritics(m))
            bounds.add(cum)
        else:
            break

    # Enclitics: right → first non-clitic. Find the leftmost enclitic by
    # walking from the right; the boundary *before* that morpheme is a
    # clitic boundary, plus boundaries between consecutive enclitics.
    suffix_idx = len(morphemes)
    for i in range(len(morphemes) - 1, 0, -1):
        if strip_diacritics(morphemes[i]) in ENCLITIC_SURFACES:
            suffix_idx = i
        else:
            break
    if suffix_idx < len(morphemes):
        cum = sum(len(strip_diacritics(m)) for m in morphemes[:suffix_idx])
        bounds.add(cum)
        for j in range(suffix_idx, len(morphemes) - 1):
            cum += len(strip_diacritics(morphemes[j]))
            bounds.add(cum)

    return bounds


def _morpheme_metrics_for_word(
    word: str,
    tokens: List[str],
    segmenter: "MorphemeSegmenter",
) -> Optional[Dict[str, Optional[int]]]:
    """Per-word morpheme-aligned metric inputs in a single Farasa pass.

    Returns a dict with five fields, or ``None`` if Farasa fails entirely
    on this word (no morphemes returned). The fields are:

    ``integrity``
        Fraction of internal morpheme boundaries respected by the
        tokenizer. ``None`` for single-morpheme words or if the cleaned
        tokens cannot be aligned to the word.
    ``csa_respected`` / ``csa_total``
        Numerator and denominator for ``clitic_separation_accuracy``.
        Both ``None`` when there are no clitic boundaries in this word
        OR alignment failed.
    ``morpheme_count``
        Farasa morpheme count for this word — ≥ 1 always.
    ``raw_token_count``
        Non-special token count *before* cleaning. This is the SFR
        numerator: it counts byte-level tokens (Charformer) that would
        clean to empty Arabic-letter strings, so byte tokenizers report
        a real (high) fragmentation rather than 0.

    The morpheme/token counts are returned even when alignment fails, so
    SFR can still accumulate fairly for tokenizers that occasionally
    fail to round-trip (e.g. ByteLevel BPE on certain edge cases).
    """
    morphemes = segmenter.segment_word(word)
    if not morphemes:
        return None

    morpheme_count = len(morphemes)
    raw_token_count = sum(1 for t in tokens if t not in SPECIAL_TOKEN_STRINGS)

    # Internal morpheme boundaries (diacritic-stripped offsets).
    morph_boundaries: List[int] = []
    cum = 0
    for m in morphemes[:-1]:
        cum += len(strip_diacritics(m))
        morph_boundaries.append(cum)

    clitic_bounds = _clitic_boundaries(morphemes)

    content_tokens = [t for t in tokens if t not in SPECIAL_TOKEN_STRINGS]
    offsets = aligned_token_offsets(content_tokens, word)
    if offsets is None:
        token_boundaries: Optional[set] = None
    else:
        token_boundaries = {end for _, end in offsets[:-1]}

    integrity: Optional[float] = None
    if morph_boundaries and token_boundaries is not None:
        respected = sum(1 for b in morph_boundaries if b in token_boundaries)
        integrity = respected / len(morph_boundaries)

    csa_respected: Optional[int] = None
    csa_total: Optional[int] = None
    if clitic_bounds and token_boundaries is not None:
        csa_respected = sum(1 for b in clitic_bounds if b in token_boundaries)
        csa_total = len(clitic_bounds)

    return {
        "integrity": integrity,
        "csa_respected": csa_respected,
        "csa_total": csa_total,
        "morpheme_count": morpheme_count,
        "raw_token_count": raw_token_count,
    }


def consonant_skeleton(word: str) -> str:
    """Approximate a root by stripping diacritics and long vowels."""
    no_diac = strip_diacritics(word)
    return "".join(c for c in no_diac if c in ARABIC_LETTERS and c not in ARABIC_LONG_VOWELS)


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
    return offsets


# ===========================================================================
# Heavyweight backends (external linguistic tools)
# ===========================================================================

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
