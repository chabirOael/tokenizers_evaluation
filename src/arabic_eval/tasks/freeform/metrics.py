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

import math
import re
import statistics
from collections import Counter
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

from sacrebleu.metrics import CHRF

_CHRF = CHRF()                                     # char order 6, word order 0, beta 2 — sacrebleu defaults
_RUN_RE = re.compile(r"([^\W_\u0640])\1{19,}")      # one letter or digit 20+ times in a row: a markdown table's
                                                   # padding (spaces), its separator row (dashes), a horizontal
                                                   # rule (---, ===, ___) or a tatweel stretch are formatting
_WORD_RE = re.compile(r"\S+")

# Loop stop — a *periodic tail*: the decoded text ends in k contiguous copies of
# the same unit of p words. The old rule ("a word 4-gram occurs 3 times anywhere",
# cut before its second occurrence) fired on numbered lists whose headings share
# the question's terms, on markdown tables and on answers that restate the
# question, and cut hundreds of characters before a real loop began (measured
# 2026-09-20: 53 of 68 stops of the untrained control were not loop-like).
LOOP_MAX_PERIOD = 60          # longest unit considered, in words (the longest loops measured were ~59 tokens)
LOOP_CHAR_RUN = 20            # one letter or digit this many times in a row


def loop_min_copies(period: int) -> int:
    """Contiguous copies of a ``period``-word unit that make a loop: 5 for a
    single word (the old word-run rule), 4 up to three words, 3 beyond."""
    if period <= 1:
        return 5
    if period <= 3:
        return 4
    return 3


class Loop(NamedTuple):
    """A repetition loop at the end of a text. ``cut`` is the character offset
    right after the *first* copy of the unit (everything before the loop is
    kept, plus one copy); ``rule`` is ``"period"`` (a periodic tail of
    ``period`` words) or ``"char"`` (one character ``LOOP_CHAR_RUN`` times in a
    row); ``unit`` the repeated text; ``copies`` the contiguous full copies at
    the tail; ``partial`` whether a trailing partial copy was counted too."""
    cut: int
    rule: str                 # "period" | "char"
    unit: str
    period: int
    copies: int
    partial: bool = False


def _periodic_tail(words: List[str], spans: List[tuple]) -> Optional[Loop]:
    """The periodic-tail rule. For every period ``p`` up to ``LOOP_MAX_PERIOD``
    take the longest suffix with ``words[i] == words[i + p]`` (the last word —
    possibly unfinished when the check runs — only has to be a *prefix* of the
    word one period back). Its length splits into ``copies`` full copies of
    the unit ``words[s:s + p]`` and a remainder that is a partial copy by
    construction (an unfinished last word counts toward the partial copy, never
    toward a full one). Fires when ``copies >= loop_min_copies(p)``, or when
    ``copies == loop_min_copies(p) - 1`` and the partial copy covers at least
    half the unit (``remainder >= ceil(p / 2)``). The cut is the end of the
    first copy; among the periods that fire the earliest cut wins (a loop of
    period p also satisfies every multiple of p, whose first copy ends later)."""
    n = len(words)
    best: Optional[Loop] = None
    for p in range(1, min(LOOP_MAX_PERIOD, n - 1) + 1):
        i = n - 1 - p
        if not words[i].startswith(words[n - 1]):
            continue
        unfinished = words[i] != words[n - 1]          # a strict prefix: the last word is still being written
        i -= 1
        while i >= 0 and words[i] == words[i + p]:
            i -= 1
        s = i + 1
        copies, remainder = divmod(n - s - unfinished, p)
        remainder += unfinished                        # the unfinished word belongs to the partial copy
        need = loop_min_copies(p)
        if copies >= need or (copies == need - 1 and remainder >= math.ceil(p / 2)):
            cand = Loop(spans[s + p - 1][1], "period", " ".join(words[s:s + p]), p, copies, copies < need)
            if best is None or cand.cut < best.cut:
                best = cand
    return best


def detect_loop(text: str) -> Optional[Loop]:
    """The generation-time loop rule, on decoded text (words and characters,
    so tokenizer-agnostic): a periodic tail (``_periodic_tail``) or one
    letter or digit ``LOOP_CHAR_RUN`` times in a row. When both fire the
    earlier cut wins. ``None`` for a text without a loop. Every text this
    returns a loop for is ``is_degenerate`` (a test pins the invariant)."""
    if not text.strip():
        return None
    found: List[Loop] = []
    m = _RUN_RE.search(text)
    if m:
        found.append(Loop(m.start() + 1, "char", m.group(1), 1, len(m.group(0)), False))
    spans = [(w.start(), w.end()) for w in _WORD_RE.finditer(text)]
    words = [text[a:b] for a, b in spans]
    loop = _periodic_tail(words, spans) if words else None
    if loop is not None:
        found.append(loop)
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
    """A repetition loop: the generation-time rules of ``detect_loop`` (a
    periodic tail, one letter or digit twenty times in a row) or, density-based on
    the whole text, duplicate word 4-grams making up half the text and (for
    space-less loops) duplicate character 8-grams making up 60 % of a text of
    40+ characters. Every text ``detect_loop`` stops is degenerate; the density
    rules add loops that ended before the tail (a loop-stopped generation's raw
    text ends in the loop, so the tail rule covers those)."""
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
