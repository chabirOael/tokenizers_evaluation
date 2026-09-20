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
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

from sacrebleu.metrics import CHRF

_CHRF = CHRF()                                     # char order 6, word order 0, beta 2 — sacrebleu defaults
_RUN_RE = re.compile(r"(.)\1{19,}")                # one character 20+ times in a row
_WORD_RE = re.compile(r"\S+")

# Loop thresholds — shared by the degeneration metric and the generation-time
# loop stop, so a stopped generation is exactly one the metric would flag.
LOOP_NGRAM_ORDER = 4          # a word 4-gram …
LOOP_NGRAM_REPEATS = 3        # … occurring this many times
LOOP_WORD_RUN = 5             # one word this many times in a row
LOOP_CHAR_RUN = 20            # one character this many times in a row


class Loop(NamedTuple):
    """A repetition loop found in a text: ``cut`` is the character offset of
    the *second* copy of the repeated unit (everything before it keeps the
    first full copy), ``kind`` names the rule, ``unit`` the repeated text."""
    cut: int
    kind: str                 # "ngram" | "word" | "char"
    unit: str


def detect_loop(text: str) -> Optional[Loop]:
    """The three loop rules of ``is_degenerate`` with a cut position: a word
    ``LOOP_NGRAM_ORDER``-gram occurring ``LOOP_NGRAM_REPEATS`` times (the unit
    is the span from its first to its second occurrence — the whole cycle, not
    just the 4-gram), one word ``LOOP_WORD_RUN`` times in a row, one character
    ``LOOP_CHAR_RUN`` times in a row. Tokenizer-agnostic: words and characters
    of the decoded text. When several rules fire the earliest cut wins.
    Returns ``None`` for a text without a loop."""
    t = text
    if not t.strip():
        return None
    found: List[Loop] = []
    m = _RUN_RE.search(t)
    if m:
        found.append(Loop(m.start() + 1, "char", m.group(1)))
    spans = [(w.start(), w.end()) for w in _WORD_RE.finditer(t)]
    words = [t[a:b] for a, b in spans]
    n = LOOP_NGRAM_ORDER
    if len(words) >= n * 2:
        first: Dict[tuple, int] = {}
        second: Dict[tuple, int] = {}
        count: Counter = Counter()
        for i in range(len(words) - n + 1):
            g = tuple(words[i:i + n])
            count[g] += 1
            if g not in first:
                first[g] = i
            elif g not in second:
                second[g] = i
        hits = [g for g, c in count.items() if c >= LOOP_NGRAM_REPEATS]
        if hits:
            g = min(hits, key=lambda g: first[g])       # the cycle that starts earliest
            a, b = first[g], second[g]
            found.append(Loop(spans[b][0], "ngram", t[spans[a][0]:spans[b][0]].rstrip()))
    run = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        if run >= LOOP_WORD_RUN:
            found.append(Loop(spans[i - run + 2][0], "word", words[i]))
            break
    if not found:
        return None
    return min(found, key=lambda l: l.cut)


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
    """A repetition loop: the three ``detect_loop`` rules (a word 4-gram
    repeated three times, one word five times in a row, one character twenty
    times in a row) or, ratio-based, duplicate word 4-grams making up half the
    text and (for space-less loops) duplicate character 8-grams making up 60 %
    of a text of 40+ characters."""
    t = text.strip()
    if not t:
        return False
    if detect_loop(t) is not None:
        return True
    words = t.split()
    if len(words) >= 8:
        grams = Counter(" ".join(words[i:i + 4]) for i in range(len(words) - 3))
        if 1.0 - len(grams) / sum(grams.values()) >= 0.5:
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
        "loop_stop_rate": _rate([bool(r.get("hit_loop")) for r in rows]),
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
