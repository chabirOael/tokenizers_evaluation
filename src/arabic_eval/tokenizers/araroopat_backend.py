"""Client-side backend for the AraRooPat tokenizer.

`MorphAnalyzer` wraps the `CamelBridge` subprocess client (in
`araroopat_bridge.py`) plus pure-Python post-processing on the trimmed
analysis dicts the server returns:

* clitic feature tag → Arabic surface translation (`CAMEL_CLITIC_SURFACE`)
* pattern normalization (strip clitic surface chars from CAMeL's
  raw pattern → bare-stem template)
* analysis dict → `Analysis` dataclass

The actual `camel_tools` import only happens inside `.venv-camel` (the
isolated env used by the server). This module runs in the main `.venv`
and never touches camel-tools directly — letting the main env stay
lighteval-compatible.

Fail-loud policy: if the bridge can't reach the camel subprocess, calls
raise `CamelBridgeError`. There's no `is_available` sentinel and no
silent degradation — using araroopat without camel makes no sense (every
word would route to `[LIT_*]`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from arabic_eval.tokenizers.araroopat_bridge import CamelBridge, get_shared_bridge

logger = logging.getLogger("arabic_eval.tokenizers.araroopat.backend")


@dataclass(frozen=True)
class Analysis:
    """One CAMeL analysis flattened to the fields we care about.

    ``root`` is the un-separated consonant string (e.g. ``"كتب"``).

    ``pattern`` is CAMeL's ``pattern`` field with **clitic surface chars
    stripped**: e.g. CAMeL's ``"وَال1ُ2ّا3ِ"`` becomes our ``"1ُ2ّا3ِ"`` once
    we strip the wa_conj/Al_det clitic surfaces from the start. We do this
    so the pattern token is a *bare-stem* template; clitics live in their
    own ``[CLITIC_*]`` tokens and don't double-count at decode time.

    ``stem`` is CAMeL's ``stem`` field — the clitic-free, diacritized
    canonical stem for the (lemma, features) realized by this word.

    ``surface`` is the full diacritized form (CAMeL's ``diac``) including
    any clitics — kept for provenance and metric-time display.

    Clitic fields are translated to Arabic surface strings (via
    ``clitic_surface``) at construction; empty/``"0"`` values are None.
    """
    root: str
    pattern: str       # clitic-stripped (bare-stem) template
    pattern_raw: str   # CAMeL's pattern field with clitics still baked in
    stem: str          # clitic-free diacritized stem (CAMeL's `stem`)
    surface: str       # full diacritized surface (CAMeL's `diac`)
    lemma: str
    pos: str
    prc3: Optional[str] = None
    prc2: Optional[str] = None
    prc1: Optional[str] = None
    prc0: Optional[str] = None
    enc0: Optional[str] = None


def _norm_clitic(value: Optional[str]) -> Optional[str]:
    """CAMeL marks 'no clitic' with ``"0"``; normalize to None."""
    if value in (None, "", "0", "na"):
        return None
    return value


# ---------------------------------------------------------------------------
# CAMeL clitic feature tag → Arabic surface form
# ---------------------------------------------------------------------------
#
# CAMeL stores clitics as feature tags (e.g. "wa_conj", "Al_det"), not
# Arabic surface strings. We translate at analysis time so the rest of
# the tokenizer pipeline only deals with surface forms. Tags that are
# distinct linguistically but identical orthographically (e.g. "wa_conj"
# and "wa_part" both → "و") collapse to the same vocabulary entry — we
# trade some morphological information for a smaller, surface-based
# clitic vocab. The original tags are still recorded in
# ``vocab_metadata.json`` for provenance.
#
# This table is duplicated in `arabic_eval.tools.araroopat_camel_server` (which can't
# import this module since it runs in a different venv). Keep them in
# sync if you add tags.
# ---------------------------------------------------------------------------

CAMEL_CLITIC_SURFACE: Dict[str, str] = {
    # prc3 — question proclitic
    "AAA_quest": "أ", "Aa_quest": "أ",
    # prc2 — conjunctions and subordinators
    "wa_conj": "و", "wa_part": "و", "wa_prep": "و", "wa_sub": "و",
    "fa_conj": "ف", "fa_rc": "ف", "fa_conn": "ف", "fa_sub": "ف",
    "fa_part": "ف",
    # prc1 — prepositions / future / connective
    "bi_prep": "ب", "bi_part": "ب",
    "ka_prep": "ك",
    "li_prep": "ل", "li_jus": "ل", "li_sub": "ل",
    "sa_fut": "س",
    "ta_prep": "ت",
    # prc0 — definite article / negation
    "Al_det": "ال",
    "lA_neg": "لا",
    "mA_neg": "ما", "mA_part": "ما", "mA_rel": "ما", "ma_rel": "ما",
    # enc0 — pronominal enclitics (object / possessive / pronoun)
    "1s_dobj": "ي", "1s_poss": "ي", "1s_pron": "ي",
    "2ms_dobj": "ك", "2ms_poss": "ك", "2ms_pron": "ك",
    "2fs_dobj": "ك", "2fs_poss": "ك", "2fs_pron": "ك",
    "3ms_dobj": "ه", "3ms_poss": "ه", "3ms_pron": "ه",
    "3fs_dobj": "ها", "3fs_poss": "ها", "3fs_pron": "ها",
    "1p_dobj": "نا", "1p_poss": "نا", "1p_pron": "نا",
    "2mp_dobj": "كم", "2mp_poss": "كم", "2mp_pron": "كم",
    "2fp_dobj": "كن", "2fp_poss": "كن", "2fp_pron": "كن",
    "3mp_dobj": "هم", "3mp_poss": "هم", "3mp_pron": "هم",
    "3fp_dobj": "هن", "3fp_poss": "هن", "3fp_pron": "هن",
    "2d_dobj": "كما", "2d_poss": "كما", "2d_pron": "كما",
    "3d_dobj": "هما", "3d_poss": "هما", "3d_pron": "هما",
}


# ---------------------------------------------------------------------------
# Tag → bucket assignment (proclitic vs enclitic).
#
# CAMeL stacks clitics outer-to-inner as prc3 > prc2 > prc1 > prc0 on the
# prefix side and enc0 on the suffix side. The bucketing here mirrors the
# comments in CAMEL_CLITIC_SURFACE above — promoted to code so it can be
# consumed by the morphological metrics (`clitic_separation_accuracy`).
#
# Keep these in sync with CAMEL_CLITIC_SURFACE: every key in the table
# must appear in exactly one of the two sets.
# ---------------------------------------------------------------------------

_PROCLITIC_TAGS: frozenset = frozenset({
    # prc3 — question proclitic
    "AAA_quest", "Aa_quest",
    # prc2 — conjunctions and subordinators
    "wa_conj", "wa_part", "wa_prep", "wa_sub",
    "fa_conj", "fa_rc", "fa_conn", "fa_sub", "fa_part",
    # prc1 — prepositions / future / connective
    "bi_prep", "bi_part", "ka_prep",
    "li_prep", "li_jus", "li_sub", "sa_fut", "ta_prep",
    # prc0 — definite article / negation
    "Al_det", "lA_neg", "mA_neg", "mA_part", "mA_rel", "ma_rel",
})

_ENCLITIC_TAGS: frozenset = frozenset({
    "1s_dobj", "1s_poss", "1s_pron",
    "2ms_dobj", "2ms_poss", "2ms_pron",
    "2fs_dobj", "2fs_poss", "2fs_pron",
    "3ms_dobj", "3ms_poss", "3ms_pron",
    "3fs_dobj", "3fs_poss", "3fs_pron",
    "1p_dobj", "1p_poss", "1p_pron",
    "2mp_dobj", "2mp_poss", "2mp_pron",
    "2fp_dobj", "2fp_poss", "2fp_pron",
    "3mp_dobj", "3mp_poss", "3mp_pron",
    "3fp_dobj", "3fp_poss", "3fp_pron",
    "2d_dobj", "2d_poss", "2d_pron",
    "3d_dobj", "3d_poss", "3d_pron",
})

# Surface-form sets derived from the table. `ك` and `ي` deliberately appear
# in both because the same Arabic surface can be a proclitic (e.g. ka_prep
# "like") or an enclitic (e.g. 2ms_pron "your"); position disambiguates
# at metric time.
PROCLITIC_SURFACES: frozenset = frozenset(
    CAMEL_CLITIC_SURFACE[t] for t in _PROCLITIC_TAGS
)
ENCLITIC_SURFACES: frozenset = frozenset(
    CAMEL_CLITIC_SURFACE[t] for t in _ENCLITIC_TAGS
)


def clitic_surface(tag: Optional[str]) -> Optional[str]:
    """Translate a CAMeL clitic feature tag to its Arabic surface form.

    Returns None for empty/no-clitic markers. Returns the tag verbatim
    when not in the table — better to surface a rare unknown tag than
    silently drop it; vocab building will record it as low-frequency
    and the tokenizer can decide whether to keep it.
    """
    if not tag:
        return None
    if tag in CAMEL_CLITIC_SURFACE:
        return CAMEL_CLITIC_SURFACE[tag]
    logger.debug("Unknown CAMeL clitic tag: %r — keeping verbatim.", tag)
    return tag


# Arabic diacritic codepoints used by CAMeL pattern strings. Duplicated
# here (rather than imported) so this module has no dependency on
# morphological_utils — keeps the backend slim.
_PATTERN_DIACRITICS = set("ًٌٍَُِّْٰٕٓٔ")


def _strip_clitic_from_start(pat: str, clitic: str) -> str:
    """Drop ``clitic`` from the front of ``pat``, skipping interleaved diacritics.

    Returns ``pat`` unchanged if the clitic isn't present at the start.
    Matches against the *non-diacritic* characters of ``pat`` so that
    e.g. ``"وَال1ِ2ا3ِ"`` strips ``"و"`` to ``"ال1ِ2ا3ِ"``.
    """
    if not clitic:
        return pat
    consumed = 0
    i = 0
    while i < len(pat) and consumed < len(clitic):
        ch = pat[i]
        if ch in _PATTERN_DIACRITICS:
            i += 1
            continue
        if ch == clitic[consumed]:
            consumed += 1
            i += 1
        else:
            return pat
    if consumed != len(clitic):
        return pat
    while i < len(pat) and pat[i] in _PATTERN_DIACRITICS:
        i += 1
    return pat[i:]


def _strip_clitic_from_end(pat: str, clitic: str) -> str:
    """Mirror of ``_strip_clitic_from_start`` for trailing enclitics."""
    if not clitic:
        return pat
    consumed = 0
    j = len(pat)
    target = clitic[::-1]
    while j > 0 and consumed < len(target):
        ch = pat[j - 1]
        if ch in _PATTERN_DIACRITICS:
            j -= 1
            continue
        if ch == target[consumed]:
            consumed += 1
            j -= 1
        else:
            return pat
    if consumed != len(target):
        return pat
    while j > 0 and pat[j - 1] in _PATTERN_DIACRITICS:
        j -= 1
    return pat[:j]


def normalize_pattern(
    pattern_raw: str,
    prc3: Optional[str],
    prc2: Optional[str],
    prc1: Optional[str],
    prc0: Optional[str],
    enc0: Optional[str],
) -> str:
    """Strip clitic surface chars from ``pattern_raw`` to give a bare-stem pattern.

    Order matters — CAMeL stacks clitics outer-to-inner as prc3 > prc2 >
    prc1 > prc0 on the prefix side. We strip outermost-first there, then
    enc0 from the suffix.
    """
    pat = pattern_raw
    for clitic in (prc3, prc2, prc1, prc0):
        if clitic:
            pat = _strip_clitic_from_start(pat, clitic)
    if enc0:
        pat = _strip_clitic_from_end(pat, enc0)
    return pat


# ---------------------------------------------------------------------------
# Analysis dict → Analysis dataclass
# ---------------------------------------------------------------------------

def _dict_to_analysis(d: Dict[str, str]) -> Optional[Analysis]:
    """Apply post-processing to a trimmed analysis dict from the bridge.

    Returns None for analyses we reject (NTWS loanword markers, missing
    root/pattern, defective roots that shrink to <3 chars).
    """
    root = d.get("root") or ""
    pattern_raw = d.get("pattern") or ""
    if not root or not pattern_raw:
        return None
    # CAMeL uses '_' or '.' to separate root letters and '#' as a
    # placeholder for missing/weak letters in defective roots. Normalize
    # all of them away — '#' would break downstream Arabic-letter checks.
    root = root.replace("_", "").replace(".", "").replace("#", "")
    if not root or len(root) < 3:
        return None
    # CAMeL marks loanwords / non-Arabic-source words with root='NTWS'
    # ("Non-Triliteral Word Source"). These have no real morphological
    # decomposition — route them to the tokenizer's [LIT_*] fallback.
    if root == "NTWS" or "NTWS" in pattern_raw:
        return None

    prc3 = clitic_surface(_norm_clitic(d.get("prc3")))
    prc2 = clitic_surface(_norm_clitic(d.get("prc2")))
    prc1 = clitic_surface(_norm_clitic(d.get("prc1")))
    prc0 = clitic_surface(_norm_clitic(d.get("prc0")))
    enc0 = clitic_surface(_norm_clitic(d.get("enc0")))

    pattern_bare = normalize_pattern(pattern_raw, prc3, prc2, prc1, prc0, enc0)

    return Analysis(
        root=root,
        pattern=pattern_bare,
        pattern_raw=pattern_raw,
        stem=d.get("stem", "") or naive_pattern_fill(root, pattern_bare),
        surface=d.get("diac") or "",
        lemma=d.get("lex", ""),
        pos=d.get("pos", ""),
        prc3=prc3, prc2=prc2, prc1=prc1, prc0=prc0, enc0=enc0,
    )


# ---------------------------------------------------------------------------
# MorphAnalyzer — bridge wrapper with dict caches and batched analyze
# ---------------------------------------------------------------------------

class MorphAnalyzer:
    """Bridge-backed CAMeL Tools wrapper with batched analysis and result caching.

    Single-threaded use only (the bridge has one outstanding request at
    a time). The shared module-level bridge is reused across instances
    by default — subprocess startup is ~2-5s, not worth paying twice.
    """

    def __init__(
        self,
        generator_timeout_ms: int = 50,  # kept for backwards-compat; unused now
        bridge: Optional[CamelBridge] = None,
    ) -> None:
        # `generator_timeout_ms` used to drive a SIGALRM-based timeout
        # around per-call CAMeL generation. The bridge now bounds calls
        # via `select()` instead — kwarg accepted for config compat,
        # but not wired to anything (per-request timeout is bridge-level).
        self.generator_timeout_ms = generator_timeout_ms
        self._bridge = bridge if bridge is not None else get_shared_bridge()
        self._analyze_cache: Dict[str, Optional[Analysis]] = {}
        self._generate_cache: Dict[Tuple[str, str], Optional[str]] = {}

    # ------------------------------------------------------------------
    # Analysis: surface → (root, pattern, clitics, features)
    # ------------------------------------------------------------------

    def analyze(self, word: str) -> Optional[Analysis]:
        """Return the best disambiguated analysis or None if CAMeL can't analyze."""
        if not word:
            return None
        if word in self._analyze_cache:
            return self._analyze_cache[word]
        results = self._bridge.analyze([word])
        analysis = self._first_valid(results[0]) if results else None
        self._analyze_cache[word] = analysis
        return analysis

    def analyze_many(
        self,
        words: List[str],
        batch_size: int = 256,
    ) -> List[Optional[Analysis]]:
        """Batched analyze — one bridge round-trip per ``batch_size`` uncached words.

        Order-preserving. Words already in the cache don't go over the
        wire. Significant speedup vs per-word `analyze()` on the corpus
        pre-pass (saves one IPC round-trip per cache miss).
        """
        # First pass: identify uncached words preserving original positions.
        out: List[Optional[Analysis]] = [None] * len(words)
        uncached_positions: List[int] = []
        uncached_words: List[str] = []
        for i, w in enumerate(words):
            if not w:
                continue
            if w in self._analyze_cache:
                out[i] = self._analyze_cache[w]
            else:
                uncached_positions.append(i)
                uncached_words.append(w)

        # Second pass: batch the uncached words.
        for start in range(0, len(uncached_words), batch_size):
            batch = uncached_words[start:start + batch_size]
            results = self._bridge.analyze(batch)
            for offset, raw_candidates in enumerate(results):
                pos = uncached_positions[start + offset]
                word = uncached_words[start + offset]
                analysis = self._first_valid(raw_candidates)
                self._analyze_cache[word] = analysis
                out[pos] = analysis

        return out

    @staticmethod
    def _first_valid(candidates: List[Dict[str, str]]) -> Optional[Analysis]:
        """Walk top-scored candidates and return the first that survives validation.

        Most words yield a valid analysis at index 0. Falling through to
        index 1+ matters for words where the top MLE pick is e.g. an
        NTWS loanword analysis but a lower-scored "real" one exists.
        """
        for cand in candidates:
            a = _dict_to_analysis(cand)
            if a is not None:
                return a
        return None

    # ------------------------------------------------------------------
    # Generation: (root, pattern) → surface (rule-based, tier 2)
    # ------------------------------------------------------------------

    def generate(self, root: str, pattern: str) -> Optional[str]:
        """Tier-2 reconstruction (returns bare stem). Falls back to naive on failure."""
        key = (root, pattern)
        if key in self._generate_cache:
            return self._generate_cache[key]
        result = self._bridge.generate(root, pattern)
        self._generate_cache[key] = result
        return result


# ---------------------------------------------------------------------------
# Tier-3: naive slot substitution (always available, no CAMeL needed)
# ---------------------------------------------------------------------------

def naive_pattern_fill(root: str, pattern: str) -> str:
    """Substitute root letters into CAMeL pattern slots ('1', '2', '3', '4').

    CAMeL pattern notation: digits are slot positions, other characters are
    template letters/diacritics carried verbatim. Wrong on weak roots
    (و/ي/ا root letters) — produces e.g. ``قَوَلَ`` instead of ``قَالَ`` —
    but used only as the ultimate fallback when both lookup and the CAMeL
    generator have failed.
    """
    if not root or not pattern:
        return ""
    out: List[str] = []
    for ch in pattern:
        if ch in "1234":
            idx = int(ch) - 1
            if idx < len(root):
                out.append(root[idx])
        else:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# Pre-pass cache record format (used by the tokenizer's train())
# ---------------------------------------------------------------------------

@dataclass
class CorpusEntry:
    """One word's analysis, suitable for JSON serialization.

    ``pattern`` is the *bare-stem* (clitic-stripped) pattern.
    ``stem`` is the clitic-free diacritized form (CAMeL's ``stem`` field).
    ``surface`` is the full diacritized form (CAMeL's ``diac``) — kept for
    metadata only; reconstruction uses ``stem``.
    """
    word: str
    analyzed: bool
    root: Optional[str] = None
    pattern: Optional[str] = None        # clitic-stripped (bare-stem) pattern
    pattern_raw: Optional[str] = None    # original CAMeL pattern, for provenance
    stem: Optional[str] = None
    surface: Optional[str] = None
    proclitics: Tuple[str, ...] = ()
    enclitics: Tuple[str, ...] = ()

    @classmethod
    def from_analysis(cls, word: str, a: Optional[Analysis]) -> "CorpusEntry":
        if a is None:
            return cls(word=word, analyzed=False)
        proclitics = tuple(c for c in (a.prc3, a.prc2, a.prc1, a.prc0) if c)
        enclitics = tuple(c for c in (a.enc0,) if c)
        return cls(
            word=word,
            analyzed=True,
            root=a.root,
            pattern=a.pattern,
            pattern_raw=a.pattern_raw,
            stem=a.stem,
            surface=a.surface,
            proclitics=proclitics,
            enclitics=enclitics,
        )

    def to_dict(self) -> Dict:
        return {
            "word": self.word,
            "analyzed": self.analyzed,
            "root": self.root,
            "pattern": self.pattern,
            "pattern_raw": self.pattern_raw,
            "stem": self.stem,
            "surface": self.surface,
            "proclitics": list(self.proclitics),
            "enclitics": list(self.enclitics),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CorpusEntry":
        return cls(
            word=d["word"],
            analyzed=d["analyzed"],
            root=d.get("root"),
            pattern=d.get("pattern"),
            pattern_raw=d.get("pattern_raw"),
            stem=d.get("stem"),
            surface=d.get("surface"),
            proclitics=tuple(d.get("proclitics") or ()),
            enclitics=tuple(d.get("enclitics") or ()),
        )
