"""Normalization, quality rules and the dialect-marker gate.

Every function here is pure and tokenizer-independent; the pool builder
(``pool.py``) applies them in cheap-first order. A document is a string
whose *lines* are its paragraphs (``normalize_document`` guarantees no
empty lines and no leading/trailing whitespace per line).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ...config import MixQualityConfig

# --------------------------------------------------------------------------
# Character classes
# --------------------------------------------------------------------------

_TATWEEL = "ـ"
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_ALEF_VARIANTS = re.compile(r"[آأإٱ]")
_HORIZONTAL_WS = re.compile(r"[ \t   -​  　]+")
# Arabic-script letters (Arabic, Supplement, Extended-A, presentation forms) —
# Persian/Urdu letters count as Arabic *script*, which is what the ratio
# rule is about; dialect vs MSA is a separate gate.
_ARABIC_LETTER = re.compile(
    r"[ؠ-يٮ-ۓۺ-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]"
)
_LATIN_LETTER = re.compile(r"[A-Za-zÀ-ɏ]")
# Gopher-style symbol inventory: bullets, box drawing, arrows, hash/pipe,
# ellipsis, guillemets — the furniture of nav bars and listings.
_SYMBOLS = re.compile(r"[#|•●■□▪◦►▶→←↑↓★☆♦♣♠♥…«»~^*_=+<>{}\[\]\\/]")
_PUNCT_STRIP = "".join(chr(c) for c in range(0x21, 0x30)) + "".join(chr(c) for c in range(0x3A, 0x41)) \
    + "".join(chr(c) for c in range(0x5B, 0x61)) + "".join(chr(c) for c in range(0x7B, 0x7F)) \
    + "،؛؟٪٫٬٭۔«»…“”‘’"
_SENTENCE_SPLIT = re.compile(r"[.!?؟۔\n]+")
# "(بالإنجليزية: Damarla Chennapa Nayakadu)" and friends: a parenthesised
# span that contains a Latin letter. Wikipedia leads carry one per
# biography / place; removing the gloss keeps the article Arabic.
_LATIN_PARENTHETICAL = re.compile(r"\s*[(（][^()（）]*[A-Za-zÀ-ɏ][^()（）]*[)）]")


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def normalize_document(
    text: str,
    *,
    nfkc: bool = True,
    remove_tatweel: bool = True,
    normalize_alef: bool = False,
    remove_diacritics: bool = False,
) -> str:
    """Normalize a raw document while keeping its paragraph structure.

    Unlike ``preprocessing.normalize_arabic`` (which collapses *all*
    whitespace — fine for the tokenizer corpus, fatal for paragraph dedup
    and line-based quality rules), this keeps one paragraph per line:
    newlines unified, horizontal whitespace collapsed, empty lines dropped.
    """
    if not text:
        return ""
    if nfkc:
        text = unicodedata.normalize("NFKC", text)
    if remove_tatweel:
        text = text.replace(_TATWEEL, "")
    if remove_diacritics:
        text = _DIACRITICS.sub("", text)
    if normalize_alef:
        text = _ALEF_VARIANTS.sub("ا", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", "\n").replace(" ", "\n")
    lines = []
    for line in text.split("\n"):
        line = _HORIZONTAL_WS.sub(" ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def strip_latin_parentheticals(text: str) -> Tuple[str, int]:
    """Remove parenthesised spans containing Latin letters. Returns
    ``(text, n_removed)``; whitespace is re-collapsed per line."""
    out_lines: List[str] = []
    n = 0
    for line in text.split("\n"):
        new, k = _LATIN_PARENTHETICAL.subn("", line)
        n += k
        new = _HORIZONTAL_WS.sub(" ", new).strip()
        if new:
            out_lines.append(new)
    return "\n".join(out_lines), n


def count_words(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------
# Wikipedia tail sections
# --------------------------------------------------------------------------

def _header_key(line: str) -> str:
    """Comparison key for a section header: diacritics stripped, alef
    variants folded, trailing punctuation removed."""
    line = _DIACRITICS.sub("", line)
    line = _ALEF_VARIANTS.sub("ا", line)
    return line.strip().rstrip(":؛.").strip()


def strip_wikipedia_sections(text: str, headers: Sequence[str]) -> Tuple[str, bool]:
    """Cut the document at the first line that *is* one of ``headers``
    (references, external links, see-also, …) — those sections are link
    and citation listings, not prose. The first line is never a cut point.
    Returns ``(text, was_cut)``.
    """
    if not headers:
        return text, False
    keys = {_header_key(h) for h in headers}
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if i == 0:
            continue
        if _header_key(line) in keys:
            return "\n".join(lines[:i]), True
    return text, False


# --------------------------------------------------------------------------
# Quality rules
# --------------------------------------------------------------------------

@dataclass
class QualityStats:
    n_words: int
    n_lines: int
    latin_ratio: float
    arabic_ratio: float
    short_line_ratio: float
    dup_line_ratio: float
    symbol_ratio: float


def quality_stats(text: str, short_line_words: int = 5) -> QualityStats:
    words = text.split()
    lines = text.split("\n") if text else []
    n_words = len(words)
    n_lines = len(lines)
    n_alpha = sum(1 for ch in text if ch.isalpha())
    n_arabic = len(_ARABIC_LETTER.findall(text))
    n_latin = len(_LATIN_LETTER.findall(text))
    n_short = sum(1 for l in lines if len(l.split()) < short_line_words)
    n_dup = n_lines - len(set(lines))
    n_sym = len(_SYMBOLS.findall(text))
    return QualityStats(
        n_words=n_words,
        n_lines=n_lines,
        latin_ratio=(n_latin / n_alpha) if n_alpha else 0.0,
        arabic_ratio=(n_arabic / n_alpha) if n_alpha else 0.0,
        short_line_ratio=(n_short / n_lines) if n_lines else 1.0,
        dup_line_ratio=(n_dup / n_lines) if n_lines else 0.0,
        symbol_ratio=(n_sym / n_words) if n_words else 0.0,
    )


def quality_reason(text: str, cfg: MixQualityConfig) -> Optional[str]:
    """Return a drop reason, or ``None`` when the document passes every rule.

    Reasons (single source of truth for the manifest): ``too_short``,
    ``no_letters``, ``latin_heavy``, ``non_arabic_script``, ``short_lines``,
    ``dup_lines``, ``symbol_heavy``.
    """
    st = quality_stats(text, cfg.short_line_words)
    if st.n_words < cfg.min_words:
        return "too_short"
    if st.arabic_ratio == 0.0 and st.latin_ratio == 0.0:
        return "no_letters"
    if st.latin_ratio > cfg.max_latin_letter_ratio:
        return "latin_heavy"
    if st.arabic_ratio < cfg.min_arabic_letter_ratio:
        return "non_arabic_script"
    if st.short_line_ratio > cfg.max_short_line_ratio:
        return "short_lines"
    if st.dup_line_ratio > cfg.max_dup_line_ratio:
        return "dup_lines"
    if st.symbol_ratio > cfg.max_symbol_ratio:
        return "symbol_heavy"
    return None


def truncate_words(text: str, max_words: int) -> Tuple[str, bool]:
    """Truncate to at most ``max_words`` at a paragraph (line) boundary.
    If the first paragraph alone exceeds the cap it is hard-cut on words.
    Returns ``(text, was_truncated)``."""
    if max_words <= 0 or count_words(text) <= max_words:
        return text, False
    kept: List[str] = []
    budget = max_words
    for line in text.split("\n"):
        n = len(line.split())
        if n <= budget:
            kept.append(line)
            budget -= n
            if budget == 0:
                break
        else:
            if not kept:  # single huge paragraph — hard cut
                kept.append(" ".join(line.split()[:budget]))
            break
    return "\n".join(kept), True


# --------------------------------------------------------------------------
# Dialect-marker gate (closed list, high precision by design)
# --------------------------------------------------------------------------
#
# Scope, honestly stated: this catches documents written *in* dialect —
# the ones dense in dialect-only function words / particles. It does not
# catch dialect that avoids every listed marker, and it deliberately
# excludes tokens that are also MSA words or common names *after the
# folding below*: بس, ما, ليه, عم "uncle", زين, ياسر, هادي, راه/رآه, الحين,
# خلاص, ماشي, and the fold collisions إلى→الي, آية→ايه, بدو (Bedouins),
# هول (horror), لكان, كيما, تبع (followed), توًّا→توا, هلّا, هون. Region tags are
# for reporting only; the gate uses the union. Entries are stored in the
# same normalized form ``_marker_key`` produces (no diacritics, alef
# folded, ة→ه, ى→ي) so spelling variants match.

DIALECT_MARKERS: Dict[str, Tuple[str, ...]] = {
    "egyptian": (
        "مش", "عايز", "عايزه", "عايزين", "عاوز", "عاوزه", "دلوقتي", "دلوقت", "ازاي",
        "كده", "كدا", "بتاع", "بتاعه", "بتاعت", "بتاعك", "بتوع", "فين",
        "احنا", "انتو", "برضو", "برضه", "عشان", "علشان", "لسه", "معلش", "كمان",
        "امتى", "ازيك", "هتكون", "هيكون", "هنروح", "مفيش", "مافيش",
    ),
    "levantine": (
        "بدي", "بدك", "بده", "بدها", "بدنا", "بدكم", "بدهم", "هيك", "هلق",
        "شو", "ليش", "كتير", "منيح", "مناح", "رح", "تبعك",
        "هاد", "هاي", "هدول", "هيدا", "هيدي", "مو", "شوي", "لهون",
        "بكرا", "مبارح", "عنجد", "بلكي", "قديش", "اديش", "وينك", "وين",
    ),
    "gulf_iraqi": (
        "شنو", "وش", "وشو", "شلون", "شلونك", "هسه", "هسا", "اكو", "ماكو",
        "ابغى", "ابغا", "وايد", "چان", "اشلون", "شفيك", "هالحين", "شسمه",
        "يمعود", "وياك", "وياه", "هذولا", "هاذول",
    ),
    "maghrebi": (
        "واش", "بزاف", "كيفاش", "علاش", "ديال", "ديالي", "ديالك", "ديالو", "ديالها",
        "ديالنا", "ديالكم", "ديالهم", "غادي", "غاديه", "نتا", "نتي", "نتوما", "شكون",
        "وقتاش", "دابا", "زعما", "هاذ", "هاذي", "هاذو", "برشا",
        "لاباس", "بلاصه", "فاش", "منين", "علاه", "لاش", "كاين", "كاينه",
        "ماكاينش", "مكاينش", "بصح",
    ),
    "pan_dialect": (
        "اللي",
    ),
}


def _marker_key(token: str) -> str:
    token = _DIACRITICS.sub("", token)
    token = _ALEF_VARIANTS.sub("ا", token)
    token = token.replace("ة", "ه").replace("ى", "ي")
    return token.strip(_PUNCT_STRIP)


_MARKER_TO_REGION: Dict[str, str] = {}
for _region, _entries in DIALECT_MARKERS.items():
    for _e in _entries:
        _MARKER_TO_REGION.setdefault(_marker_key(_e), _region)
_MARKER_SET = frozenset(_MARKER_TO_REGION)


@dataclass
class DialectScore:
    n_words: int
    n_markers: int
    per_1k_words: float
    matched: Counter = field(default_factory=Counter)  # marker -> count

    def top(self, n: int = 5) -> List[Tuple[str, int]]:
        return self.matched.most_common(n)


def dialect_marker_score(text: str) -> DialectScore:
    """Count dialect-marker tokens (whole tokens after ``_marker_key``,
    with an optional leading و/ف proclitic peeled) per 1 000 words."""
    words = text.split()
    matched: Counter = Counter()
    for w in words:
        key = _marker_key(w)
        if not key:
            continue
        if key in _MARKER_SET:
            matched[key] += 1
            continue
        if len(key) >= 3 and key[0] in "وف" and key[1:] in _MARKER_SET:
            matched[key[1:]] += 1
    n_words = len(words)
    n_markers = sum(matched.values())
    per_1k = (1000.0 * n_markers / n_words) if n_words else 0.0
    return DialectScore(n_words=n_words, n_markers=n_markers, per_1k_words=per_1k, matched=matched)


def marker_region(marker: str) -> str:
    return _MARKER_TO_REGION.get(marker, "?")


# --------------------------------------------------------------------------
# Sentence sampling (for the opt-in CAMeL dialect-ID signal)
# --------------------------------------------------------------------------

def sample_sentences(text: str, k: int, min_words: int) -> List[str]:
    """Deterministically pick up to ``k`` evenly spaced sentences with at
    least ``min_words`` words. No RNG so the pool fingerprint stays honest."""
    candidates = [s.strip() for s in _SENTENCE_SPLIT.split(text)]
    candidates = [s for s in candidates if len(s.split()) >= min_words]
    if not candidates or k <= 0:
        return []
    if len(candidates) <= k:
        return candidates
    step = len(candidates) / k
    return [candidates[int(i * step)] for i in range(k)]
