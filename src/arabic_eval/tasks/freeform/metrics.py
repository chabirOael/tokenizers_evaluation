"""Reference-based and reference-free metrics for free-form generations.

Everything here scores *decoded text*, so it is tokenizer-agnostic by
construction: chrF (character n-gram F-score, sacrebleu) and BERTScore are
reference-based; the degeneration detector, the Arabic-letter ratio, the
empty rate and the length statistics are reference-free and catch the
failure modes an accuracy-style number hides (repetition loops, language
drift, silence). ``reference_roundtrip_chrf`` is the decode-fidelity ceiling
of a tokenizer: chrF of ``decode(encode(reference))`` against the reference —
a lossy decoder (alef folding, ``##`` spacing, AraRooPat tier-3 fills) caps
every reference-based score below 100 before the model generates a word.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from sacrebleu.metrics import CHRF

_CHRF = CHRF()                                     # char order 6, word order 0, beta 2 — sacrebleu defaults
_RUN_RE = re.compile(r"(.)\1{19,}")                # one character 20+ times in a row


def chrf_sentence(hypothesis: str, reference: str) -> float:
    if not hypothesis.strip():
        return 0.0
    return float(_CHRF.sentence_score(hypothesis, [reference]).score)


def chrf_corpus(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    if not hypotheses:
        return 0.0
    return float(_CHRF.corpus_score(list(hypotheses), [list(references)]).score)


def ngram_duplicate_ratio(items: Sequence[str], n: int) -> float:
    grams = [" ".join(items[i:i + n]) if isinstance(items[0], str) and len(items[0]) > 1 else "".join(items[i:i + n])
             for i in range(len(items) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def is_degenerate(text: str) -> bool:
    """A repetition loop: a word 4-gram repeated three times or duplicate word
    4-grams making up half the text, one word five times in a row, one
    character twenty times in a row, or (for space-less loops) duplicate
    character 8-grams making up 60 % of a text of 40+ characters."""
    t = text.strip()
    if not t:
        return False
    if _RUN_RE.search(t):
        return True
    words = t.split()
    if len(words) >= 8:
        grams = Counter(" ".join(words[i:i + 4]) for i in range(len(words) - 3))
        if max(grams.values()) >= 3 or 1.0 - len(grams) / sum(grams.values()) >= 0.5:
            return True
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run >= 5:
            return True
    if len(t) >= 40:
        cg = Counter(t[i:i + 8] for i in range(len(t) - 7))
        if 1.0 - len(cg) / sum(cg.values()) >= 0.6:
            return True
    return False


def _is_arabic_letter(ch: str) -> bool:
    o = ord(ch)
    return ch.isalpha() and (0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF
                             or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF)


def arabic_letter_ratio(text: str) -> Optional[float]:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    return sum(1 for c in letters if _is_arabic_letter(c)) / len(letters)


def roundtrip(tokenizer: Any, text: str) -> str:
    return tokenizer.decode(list(tokenizer.encode(text).input_ids))


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    v = [x for x in xs if x is not None]
    return round(statistics.fmean(v), 6) if v else None


def _rate(flags: Sequence[bool]) -> Optional[float]:
    return round(sum(1 for f in flags if f) / len(flags), 6) if flags else None


def summarize(rows: Sequence[Dict[str, Any]], gen_wall_sec: float) -> Dict[str, Any]:
    """Aggregate the per-row records of ``FreeformCidarTask`` into the metric dict."""
    n = len(rows)
    gen_chars = sum(r["gen_chars"] for r in rows)
    gen_tokens = sum(r["gen_tokens"] for r in rows)
    return {
        "num_samples": n,
        "chrf": _mean([r["chrf"] for r in rows]),
        "chrf_corpus": round(chrf_corpus([r["generation"] for r in rows], [r["reference"] for r in rows]), 6) if n else None,
        "bertscore_f1": _mean([r.get("bertscore_f1") for r in rows]),
        "bertscore_p": _mean([r.get("bertscore_p") for r in rows]),
        "bertscore_r": _mean([r.get("bertscore_r") for r in rows]),
        "empty_rate": _rate([r["empty"] for r in rows]),
        "degenerate_rate": _rate([r["degenerate"] for r in rows]),
        "latin_rate": _rate([r["latin"] for r in rows]),
        "arabic_letter_ratio": _mean([r["arabic_letter_ratio"] for r in rows]),
        "hit_cap_rate": _rate([r["hit_cap"] for r in rows]),
        "marker_stop_rate": _rate([r["stop_reason"] == "marker" for r in rows]),
        "eos_rate": _rate([r["stop_reason"] == "eos" for r in rows]),
        "char_truncated_rate": _rate([r["char_truncated"] for r in rows]),
        "mean_gen_chars": round(gen_chars / n, 2) if n else None,
        "mean_gen_tokens": round(gen_tokens / n, 2) if n else None,
        "mean_ref_chars": round(sum(r["ref_chars"] for r in rows) / n, 2) if n else None,
        "gen_chars_per_sec": round(gen_chars / gen_wall_sec, 2) if gen_wall_sec > 0 else None,
        "gen_tokens_per_sec": round(gen_tokens / gen_wall_sec, 2) if gen_wall_sec > 0 else None,
        "generation_wall_sec": round(gen_wall_sec, 3),
        "reference_roundtrip_chrf": _mean([r["reference_roundtrip_chrf"] for r in rows]),
    }
