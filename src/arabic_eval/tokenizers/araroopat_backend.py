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

    ``particle`` is set (and ``root`` / ``pattern`` are empty) when the
    word is a closed-class preposition/particle from
    ``PREPOSITION_INVENTORY`` — those get one ``[PREP_*]`` token instead
    of a root+pattern decomposition. See ``_particle_analysis``.

    ``fem`` is ``"ة"`` when the word carries the tāʾ marbūṭa suffix —
    written ة word-finally, ت before a pronoun (مدرسة / مدرسته). It is
    stripped from ``pattern`` and emitted as ``[CLITICE_ة]``; see
    ``strip_fem_from_pattern``.
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
    particle: Optional[str] = None  # bare preposition surface, e.g. "من"
    fem: Optional[str] = None       # TAA_MARBUTA when the ة suffix was factored out

    @property
    def enclitics(self) -> Tuple[str, ...]:
        """Enclitic surfaces in emission order, innermost first: (ة, pronoun)."""
        return tuple(c for c in (self.fem, self.enc0) if c)


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
    # prc3 — question proclitic. ">a_ques" is Buckwalter-encoded (">" = أ).
    "AAA_quest": "أ", "Aa_quest": "أ", ">a_ques": "أ",
    # prc2 — conjunctions and subordinators
    "wa_conj": "و", "wa_part": "و", "wa_prep": "و", "wa_sub": "و",
    "fa_conj": "ف", "fa_rc": "ف", "fa_conn": "ف", "fa_sub": "ف",
    "fa_part": "ف",
    # prc1 — prepositions / future / connective
    "bi_prep": "ب", "bi_part": "ب",
    "ka_prep": "ك",
    "li_prep": "ل", "li_jus": "ل", "li_sub": "ل",
    "la_emph": "ل", "la_rc": "ل",
    "sa_fut": "س",
    "ta_prep": "ت",
    # prc0 — definite article / negation
    "Al_det": "ال",
    "lA_neg": "لا",
    "mA_neg": "ما", "mA_part": "ما", "mA_rel": "ما", "ma_rel": "ما",
    # enc0 — subordinating mA rides the suffix slot (عند + ما)
    "mA_sub": "ما",
    # enc0 — relative "man" fused onto a preposition: مِمَّن = مِن + مَن,
    # عَمَّن = عَن + مَن. CAMeL also puts the relative mA there for مِمّا /
    # عَمّا (tag mA_rel, mapped above). The assimilated spelling is
    # restored at decode by ``join_particle_enclitic``.
    "man_rel": "من",
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
    "AAA_quest", "Aa_quest", ">a_ques",
    # prc2 — conjunctions and subordinators
    "wa_conj", "wa_part", "wa_prep", "wa_sub",
    "fa_conj", "fa_rc", "fa_conn", "fa_sub", "fa_part",
    # prc1 — prepositions / future / connective
    "bi_prep", "bi_part", "ka_prep",
    "li_prep", "li_jus", "li_sub", "la_emph", "la_rc", "sa_fut", "ta_prep",
    # prc0 — definite article / negation
    "Al_det", "lA_neg", "mA_neg", "mA_part", "mA_rel", "ma_rel",
})

_ENCLITIC_TAGS: frozenset = frozenset({
    "mA_sub", "man_rel",
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
    logger.warning(
        "Unknown CAMeL clitic tag: %r — keeping verbatim. It will enter the vocab "
        "as non-Arabic text; add it to CAMEL_CLITIC_SURFACE.", tag,
    )
    return tag


# Arabic diacritic codepoints used by CAMeL pattern strings. Duplicated
# here (rather than imported) so this module has no dependency on
# morphological_utils — keeps the backend slim.
_PATTERN_DIACRITICS = set("ًٌٍَُِّْٰٕٓٔ")

# The tāʾ marbūṭa suffix. Always word-final (ت before a pronoun), never a
# root letter, and CAMeL's own ``stem`` field already excludes it — so the
# tokenizer treats it as an enclitic: stripped from the pattern, emitted as
# [CLITICE_ة], never as a [CHAR_*]. Not a CAMeL clitic *tag*, so it is
# deliberately absent from CAMEL_CLITIC_SURFACE / ENCLITIC_SURFACES (those
# feed the cross-tokenizer CSA metric).
TAA_MARBUTA = "ة"
_TAA = "ت"
# POS prefixes under which a stem-final ت before a pronoun is the ة suffix
# (مدرسته, حياته). On verbs the same ت is a subject suffix (كتبته).
_NOMINAL_POS_PREFIXES = ("noun", "adj")


def _last_letter(s: str) -> Tuple[int, str]:
    """(index, char) of the last non-diacritic char of ``s``; (-1, '') if none."""
    for i in range(len(s) - 1, -1, -1):
        if s[i] not in _PATTERN_DIACRITICS:
            return i, s[i]
    return -1, ""


def strip_fem_from_pattern(
    pattern_bare: str,
    proclitics: Tuple[Optional[str], ...],
    enc0: Optional[str],
    stem: str,
    pos: str,
    diac: str,
) -> Tuple[str, Optional[str]]:
    """Factor the ة suffix out of a clitic-stripped pattern.

    Returns ``(pattern, fem)`` where ``fem`` is ``TAA_MARBUTA`` or None.

    * Pattern ends in ة (plus an optional case vowel): always the suffix —
      ة is orthographically unambiguous. Applies to every ة-final word,
      broken plurals and numerals included (قضاة, ثلاثة); CAMeL's ``stem``
      excludes the ة there too.
    * Pattern ends in ت, a pronoun enclitic follows, the POS is nominal,
      and ``surface − proclitics − pronoun == stem + ت``: the ت is a ة
      realized before the pronoun (مدرسته, حياته, ومدرستها). The stem
      check rejects a radical ت (بيته: stem بَيْت) and the POS check a
      verbal subject ت (كتبته).
    """
    idx, last = _last_letter(pattern_bare)
    if idx < 0:
        return pattern_bare, None
    if last == TAA_MARBUTA:
        return pattern_bare[:idx], TAA_MARBUTA
    if (
        last == _TAA and enc0 and pos.startswith(_NOMINAL_POS_PREFIXES)
        and _strip_diac(stem)
    ):
        core = strip_proclitics_from_start(diac, proclitics)
        core = _strip_diac(_strip_clitic_from_end(core, enc0))
        if core == _strip_diac(stem) + _TAA:
            return pattern_bare[:idx], TAA_MARBUTA
    return pattern_bare, None


def strip_enclitics_from_end(text: str, enclitics: Tuple[str, ...]) -> str:
    """Strip an enclitic stack from the end, outermost first.

    ``enclitics`` is in emission order (innermost first: ``(ة, ه)``), the
    inverse of the order they peel off. The ة suffix surfaces as ت before
    a pronoun, so it strips as either.
    """
    for clitic in reversed(enclitics):
        if not clitic:
            continue
        if clitic == TAA_MARBUTA:
            out = _strip_clitic_from_end(text, TAA_MARBUTA)
            if out == text:
                out = _strip_clitic_from_end(text, _TAA)
            text = out
        else:
            text = _strip_clitic_from_end(text, clitic)
    return text


# Arabic letter block used to validate roots. CAMeL occasionally emits
# database markers ("FOREIGN") or Buckwalter/ASCII fragments ("Uٌٍ" for
# هيكتور) in the root field; anything outside this range is not a root.
_ARABIC_LETTER_RANGE = (("\u0621", "\u064a"), ("\u0671", "\u0671"))


# CAMeL's masked-radical placeholder. It is a legitimate part of a root
# token ('ق#ل'), so it is allowed through the Arabic-letter guard below —
# unlike the ASCII fragments that guard exists to reject.
WEAK_RADICAL_MARK = "#"


def _is_arabic_root(root: str) -> bool:
    return bool(root) and all(
        ch == WEAK_RADICAL_MARK
        or any(lo <= ch <= hi for lo, hi in _ARABIC_LETTER_RANGE)
        for ch in root
    )


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


# The لِ + الـ contraction: when the preposition li is followed by the definite
# article, Arabic orthography writes one lam, not two — لِ + الوَلَد → لِلوَلَد.
# The article's surface is therefore "ل", not "ال", and a literal strip of "ال"
# silently fails, leaving a stray lam in both the bare pattern and the
# reconstructed stem ('ل1ِ2ا3ِ' instead of '1ِ2ا3ِ'; 'لكتاب' instead of 'كتاب').
_LI_PREP, _AL_DET = "ل", "ال"


def strip_proclitics_from_start(text: str, proclitics: Tuple[Optional[str], ...]) -> str:
    """Strip a proclitic stack from the front, outermost first.

    Handles the لِ+الـ contraction: if the article fails to strip literally and
    the clitic just removed was the li preposition, strip a single lam instead.
    """
    prev: Optional[str] = None
    for clitic in proclitics:
        if not clitic:
            continue
        out = _strip_clitic_from_start(text, clitic)
        if out == text and clitic == _AL_DET and prev == _LI_PREP:
            out = _strip_clitic_from_start(text, _LI_PREP)
        prev = clitic
        text = out
    return text


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
    pat = strip_proclitics_from_start(pattern_raw, (prc3, prc2, prc1, prc0))
    if enc0:
        pat = _strip_clitic_from_end(pat, enc0)
    return pat


# ---------------------------------------------------------------------------
# Closed-class prepositions / particles → one [PREP_*] token
# ---------------------------------------------------------------------------
#
# Before this intercept these words were handled three inconsistent ways:
# the 2-radical ones (من عن في مذ كي لولا) failed the ≥3-radical gate and
# spilled into [LIT_BEGIN] [CHAR_*]... [LIT_END] — four tokens for من, the
# most frequent word in Arabic; منذ is NTWS; and the rest got a bogus
# root+pattern decomposition (إلى → [ROOT_#ل#][PAT_إِ2َى], حتى → [ROOT_حتت],
# لعل → [CLITICP_ل][ROOT_علل][PAT_1َ2َّ] with CAMeL misreading its ل as the
# emphatic lam). A function word has no root/wazn to preserve, so each one
# is a single fixed token; clitics still ride outside it ([CLITICP_و]
# [PREP_إلى] [CLITICE_ه] for وإليه).
#
# Match key: CAMeL's lemma (``lex``) with diacritics stripped — it stays
# stable across clitics and the ى/ي alternation (عليه → lex عَلَى). The
# clitic-stripped surface is checked too (حاشا lemmatizes to حاش).
# ---------------------------------------------------------------------------

PREPOSITION_INVENTORY: Tuple[str, ...] = (
    "من", "إلى", "عن", "على", "في", "حتى", "منذ", "مذ",
    "خلا", "عدا", "حاشا", "متى", "لعل", "كي", "لولا",
)

# Preposition + enclitic spellings that are not plain concatenation.
# Decode-side inverse of the enc0 split CAMeL makes at analysis time.
_PARTICLE_ASSIMILATION: Dict[Tuple[str, str], str] = {
    ("من", "ما"): "مما", ("من", "من"): "ممن",
    ("عن", "ما"): "عما", ("عن", "من"): "عمن",
}

_ALEF_MAKSURA, _YEH = "ى", "ي"


def _strip_diac(s: str) -> str:
    return "".join(c for c in s if c not in _PATTERN_DIACRITICS)


def join_particle_enclitic(particle: str, enclitic: str) -> str:
    """Attach an enclitic to a preposition with the orthographic adjustments.

    * ى-final prepositions turn the ى into ي before a pronoun
      (إلى + ه → إليه, على + هم → عليهم).
    * من / عن assimilate with the relative ما / من (مما, ممن, عما, عمن).
    * The 1sg ي merges with a ي-final base (إلى + ي → إلي, في + ي → في):
      the doubled letter is written with shadda, which undiacritized text
      does not carry.

    Must stay the inverse of the enc0 split CAMeL makes at encode time.
    """
    if (particle, enclitic) in _PARTICLE_ASSIMILATION:
        return _PARTICLE_ASSIMILATION[(particle, enclitic)]
    base = particle[:-1] + _YEH if particle.endswith(_ALEF_MAKSURA) else particle
    if enclitic == _YEH and base.endswith(_YEH):
        return base
    return base + enclitic


_PARTICLE_ASSIMILATION_INVERSE: Dict[str, str] = {
    fused: base for (base, _enc), fused in _PARTICLE_ASSIMILATION.items()
}


def _particle_analysis(
    d: Dict[str, str],
    particles: frozenset,
    word: Optional[str] = None,
) -> Optional[Analysis]:
    """Return a particle ``Analysis`` if ``d`` is a listed preposition, else None.

    Acceptance is the exact inverse of the decoder: the (diacritic-free)
    input must equal ``proclitics + join_particle_enclitic(p, enc0)`` for
    some listed ``p``. Matching on the *input* rather than on CAMeL's
    normalized surface keeps ي-spelled bare forms (علي — also the name
    Ali; إلي) out of the particle path, so they decode back verbatim.

    Guards: a definite article or a *possessive* enclitic marks a noun
    reading (CAMeL tags pronouns on prepositions as ``*_pron``), so those
    are left to the normal root+pattern path.
    """
    surface = _strip_diac(word if word else (d.get("diac") or ""))
    if not surface:
        return None
    prc = tuple(clitic_surface(_norm_clitic(d.get(k))) for k in ("prc3", "prc2", "prc1", "prc0"))
    enc_tag = _norm_clitic(d.get("enc0"))
    enc0 = clitic_surface(enc_tag)
    if prc[3] == _AL_DET or (enc_tag and enc_tag.endswith("_poss")):
        return None

    # Candidate particles: the lemma first (حاشا lemmatizes to حاش, and the
    # fused مِمَّن / عَمَّن carry the fused form as lemma), then the whole
    # inventory as a surface fallback.
    lemma_bare = _strip_diac(d.get("lex") or "")
    ordered: List[str] = []
    for cand in (lemma_bare, _PARTICLE_ASSIMILATION_INVERSE.get(lemma_bare), *particles):
        if cand and cand in particles and cand not in ordered:
            ordered.append(cand)

    # CAMeL sometimes reports a proclitic that is really the first letter of
    # the particle (لعل → prc1=la_emph + lemma لَعَلَّ), so try the claimed
    # stack first and drop the innermost proclitic until the surface agrees.
    prc_list = [c for c in prc if c]
    for particle in ordered:
        expected_tail = join_particle_enclitic(particle, enc0) if enc0 else particle
        for keep in range(len(prc_list), -1, -1):
            if "".join(prc_list[:keep]) + expected_tail == surface:
                kept = set(prc_list[:keep])
                return Analysis(
                    root="",
                    pattern="",
                    pattern_raw=d.get("pattern") or "",
                    stem="",
                    surface=word or d.get("diac") or "",
                    lemma=d.get("lex", ""),
                    pos=d.get("pos", ""),
                    prc3=prc[0] if prc[0] in kept else None,
                    prc2=prc[1] if prc[1] in kept else None,
                    prc1=prc[2] if prc[2] in kept else None,
                    prc0=prc[3] if prc[3] in kept else None,
                    enc0=enc0,
                    particle=particle,
                )
    return None


# ---------------------------------------------------------------------------
# Analysis dict → Analysis dataclass
# ---------------------------------------------------------------------------

def _dict_to_analysis(
    d: Dict[str, str],
    particles: frozenset = frozenset(PREPOSITION_INVENTORY),
    word: Optional[str] = None,
) -> Optional[Analysis]:
    """Apply post-processing to a trimmed analysis dict from the bridge.

    Listed prepositions short-circuit to a particle ``Analysis`` before
    any gate (they have no root to validate). Otherwise returns None for
    analyses we reject: NTWS/FOREIGN database markers, missing
    root/pattern, roots with fewer than 3 radicals, and roots carrying
    characters that are neither Arabic letters nor CAMeL's masked-radical
    placeholder.
    """
    particle = _particle_analysis(d, particles, word)
    if particle is not None:
        return particle

    root = d.get("root") or ""
    pattern_raw = d.get("pattern") or ""
    if not root or not pattern_raw:
        return None
    # CAMeL separates root letters with '.' (sometimes '_') and marks a
    # radical whose surface realization is not stable across the paradigm
    # with '#' — the weak letters و/ي/ا and the hamza family. That mark is
    # a RADICAL, not a missing field: قال/يقول/قول/أقوال all analyse as
    # 'ق.#.ل', and the letter that actually surfaces sits in the pattern as
    # literal template material ('1ا3َ', 'يَ1ُو3', '1َوْ3ِ').
    #
    # Deleting '#' is doubly destructive: it drops the root below the
    # 3-radical bar, and — worse — it renumbers the remaining radicals so
    # the pattern's slot digits no longer index the right letters. So count
    # radicals structurally and keep the placeholder in the token string.
    # Measured on the ArabicText-Large pre-pass, deletion routed 49 % of
    # word occurrences in real eval text to the character fallback.
    radicals = [r for r in root.replace("_", ".").split(".") if r]
    if len(radicals) == 1 and len(radicals[0]) >= 3:
        # Defensive: an unseparated root string ("كتب" rather than "ك.ت.ب").
        # Every entry observed in the corpus is dot-separated, but treating
        # the whole string as one radical would silently reject those.
        radicals = list(radicals[0])
    root = "".join(radicals)
    if len(radicals) < 3:
        return None
    # CAMeL marks loanwords / non-Arabic-source words with root='NTWS'
    # ("Non-Triliteral Word Source") and some database entries with
    # root='FOREIGN'. Neither has a real morphological decomposition —
    # route them to the tokenizer's [LIT_*] fallback.
    if root in ("NTWS", "FOREIGN") or "NTWS" in pattern_raw or "FOREIGN" in pattern_raw:
        return None
    # Catch-all: a root must be Arabic letters only. Without this, ASCII
    # fragments leak into the vocabulary as [ROOT_*] tokens (observed:
    # [ROOT_FOREIGN] freq 61, [ROOT_Uٌٍ] freq 3).
    if not _is_arabic_root(root):
        logger.debug("Rejecting non-Arabic root %r", root)
        return None

    prc3 = clitic_surface(_norm_clitic(d.get("prc3")))
    prc2 = clitic_surface(_norm_clitic(d.get("prc2")))
    prc1 = clitic_surface(_norm_clitic(d.get("prc1")))
    prc0 = clitic_surface(_norm_clitic(d.get("prc0")))
    enc0 = clitic_surface(_norm_clitic(d.get("enc0")))

    pattern_bare = normalize_pattern(pattern_raw, prc3, prc2, prc1, prc0, enc0)
    pattern_bare, fem = strip_fem_from_pattern(
        pattern_bare, (prc3, prc2, prc1, prc0), enc0,
        d.get("stem") or "", d.get("pos") or "", d.get("diac") or "",
    )

    return Analysis(
        root=root,
        pattern=pattern_bare,
        pattern_raw=pattern_raw,
        stem=d.get("stem", "") or naive_pattern_fill(root, pattern_bare),
        surface=d.get("diac") or "",
        lemma=d.get("lex", ""),
        pos=d.get("pos", ""),
        prc3=prc3, prc2=prc2, prc1=prc1, prc0=prc0, enc0=enc0,
        fem=fem,
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
        particles: Optional[frozenset] = None,
    ) -> None:
        # `generator_timeout_ms` used to drive a SIGALRM-based timeout
        # around per-call CAMeL generation. The bridge now bounds calls
        # via `select()` instead — kwarg accepted for config compat,
        # but not wired to anything (per-request timeout is bridge-level).
        self.generator_timeout_ms = generator_timeout_ms
        self._bridge = bridge if bridge is not None else get_shared_bridge()
        # Closed-class words that become one [PREP_*] token (see
        # PREPOSITION_INVENTORY). Configurable per tokenizer instance.
        self.particles: frozenset = (
            frozenset(particles) if particles is not None
            else frozenset(PREPOSITION_INVENTORY)
        )
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
        analysis = self._first_valid(results[0] if results else [], word, self.particles)
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
                analysis = self._first_valid(raw_candidates, word, self.particles)
                self._analyze_cache[word] = analysis
                out[pos] = analysis

        return out

    @staticmethod
    def _first_valid(
        candidates: List[Dict[str, str]],
        word: Optional[str] = None,
        particles: frozenset = frozenset(PREPOSITION_INVENTORY),
    ) -> Optional[Analysis]:
        """Walk top-scored candidates and return the first that survives validation.

        Most words yield a valid analysis at index 0. Falling through to
        index 1+ matters for words where the top MLE pick is e.g. an
        NTWS loanword analysis but a lower-scored "real" one exists.

        If no candidate survives but the bare surface itself is a listed
        preposition, return a particle analysis anyway — the token must
        not depend on CAMeL's database having an entry for it.
        """
        for cand in candidates:
            a = _dict_to_analysis(cand, particles, word)
            if a is not None:
                return a
        if word:
            bare = _strip_diac(word)
            if bare in particles:
                return Analysis(
                    root="", pattern="", pattern_raw="", stem="",
                    surface=word, lemma=word, pos="", particle=bare,
                )
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
            # A masked radical never has a slot in its own pattern (verified
            # on 457/457 masked radicals in the corpus: the digit is always
            # absent, the realized letter being literal template material).
            # Guard anyway so '#' can never reach output text.
            if idx < len(root) and root[idx] != WEAK_RADICAL_MARK:
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
    particle: Optional[str] = None       # set for [PREP_*] words; root/pattern None

    @classmethod
    def from_analysis(cls, word: str, a: Optional[Analysis]) -> "CorpusEntry":
        if a is None:
            return cls(word=word, analyzed=False)
        proclitics = tuple(c for c in (a.prc3, a.prc2, a.prc1, a.prc0) if c)
        enclitics = a.enclitics
        if a.particle:
            return cls(
                word=word, analyzed=True, surface=a.surface,
                proclitics=proclitics, enclitics=enclitics, particle=a.particle,
            )
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
            "particle": self.particle,
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
            particle=d.get("particle"),
        )
