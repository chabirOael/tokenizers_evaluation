"""Intrinsic tokenizer evaluation metrics."""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

from tqdm import tqdm

from arabic_eval.tokenizers.base import BaseTokenizer
from arabic_eval.tokenizers.morphological_utils import (
    MorphemeSegmenter,
    RootExtractor,
    SPECIAL_TOKEN_STRINGS,
    aligned_token_offsets,
    clean_token_string,
    contains_subsequence,
    derive_pattern,
    filter_content_tokens,
    stem_pattern_span,
    strip_diacritics,
)

logger = logging.getLogger("arabic_eval.tokenizers.intrinsic")

DEFAULT_MORPH_SAMPLE_SIZE = 500


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


# ---------------------------------------------------------------------------
# Morphological metrics
# ---------------------------------------------------------------------------

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


def compute_morphological_metrics(
    tokenizer: BaseTokenizer,
    texts: List[str],
    sample_size: int = DEFAULT_MORPH_SAMPLE_SIZE,
    seed: int = 42,
    use_farasa: bool = True,
) -> Dict[str, float]:
    """Five Arabic morphological conservation metrics.

    Per-word metrics (averaged over the word sample):
        - root_conservation_rate: fraction of words whose full root appears
          as a subsequence inside a single token.
        - pattern_conservation_rate: fraction of words whose stem-span
          pattern (root letters + their immediate context, clitics trimmed)
          is recoverable from a single token.
        - morpheme_integrity_rate: fraction of Farasa internal morpheme
          boundaries that align with token boundaries (averaged over words
          that have at least one internal boundary).

    Per-token metrics (averaged over all tokens in tokenized sample):
        - root_bearing_token_pct: % of tokens that contain at least one
          full Arabic root from the sample's root set.
        - pattern_bearing_token_pct: % of tokens that match a known stem
          pattern from the sample's pattern set.
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

        # --- morpheme_integrity_rate ---
        if segmenter is not None:
            integrity = _morpheme_integrity_for_word(word, tokens, segmenter)
            if integrity is not None:
                integrity_sum += integrity
                integrity_count += 1

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
        "root_bearing_token_pct": (
            round(root_bearing_pct, 2) if root_bearing_pct is not None else None
        ),
        "pattern_bearing_token_pct": (
            round(pattern_bearing_pct, 2) if pattern_bearing_pct is not None else None
        ),
        "morph_sample_size": root_total,
    }


def _empty_morph_metrics() -> Dict[str, Optional[float]]:
    return {
        "root_conservation_rate": None,
        "pattern_conservation_rate": None,
        "morpheme_integrity_rate": None,
        "root_bearing_token_pct": None,
        "pattern_bearing_token_pct": None,
        "morph_sample_size": 0,
    }


def _safe_rate(num: int, denom: int) -> Optional[float]:
    if denom == 0:
        return None
    return round(num / denom, 4)


def _morpheme_integrity_for_word(
    word: str,
    tokens: List[str],
    segmenter: MorphemeSegmenter,
) -> Optional[float]:
    """Fraction of internal morpheme boundaries respected by tokenization.

    Returns ``None`` if Farasa is unavailable, the word has no internal
    morpheme boundaries (single-morpheme word), or the token offsets
    cannot be aligned to the word.
    """
    morphemes = segmenter.segment_word(word)
    if not morphemes or len(morphemes) < 2:
        return None

    morph_boundaries: List[int] = []
    cum = 0
    for m in morphemes[:-1]:
        cum += len(strip_diacritics(m))
        morph_boundaries.append(cum)
    if not morph_boundaries:
        return None

    content_tokens = [t for t in tokens if t not in SPECIAL_TOKEN_STRINGS]
    offsets = aligned_token_offsets(content_tokens, word)
    if offsets is None:
        # Tokens didn't reconstruct cleanly — skip this word.
        return None
    token_boundaries = {end for _, end in offsets[:-1]}

    respected = sum(1 for b in morph_boundaries if b in token_boundaries)
    return respected / len(morph_boundaries)
