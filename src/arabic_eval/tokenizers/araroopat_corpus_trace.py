"""Corpus-scale replay of ``AraRooPatTokenizer.train()`` for the explorer's 03 tab.

The small-text tracer (``araroopat_trace.py``) shows every intermediate of
``train()`` on a few sentences. This module produces the *same 16 steps*
for a real corpus (``Jr23xd23/ArabicText-Large`` by default, loaded exactly
as the pipeline loads it) where per-word exhaustiveness is impossible:
every step carries corpus-wide totals plus a bounded, seeded **sample**,
and says so in a ``_scale`` field the page renders as a banner.

Three ways in, mirroring what exists on disk:

* ``mode="train"`` — the real pre-pass rule: the on-disk CAMeL cache
  (``corpus_analysis.pkl``) is reused when its key matches and it covers
  every chunk of the corpus, otherwise every unique chunk goes through
  CAMeL (hours; the job reports progress and can be cancelled). Then the
  real ``_build_vocab`` / ``_build_reconstruction`` / ``_build_metadata``
  run on the entries. Where the real run did no CAMeL work (cache hit),
  the IPC / gates / peeler cards **replay a sample of chunks live** through
  the bridge and are labelled as a replay.
* ``mode="saved"`` — load a tokenizer directory written by ``save()``. The
  pre-pass cards cannot be reconstructed from artifacts and say so; vocab,
  reconstruction, metadata and the probe round-trip are real.

Design rule (inherited from the small tracer): nothing here re-implements
tokenizer logic. The real helpers run; where a step is *explained*, the
explanation is asserted against the real result and mismatches are
reported in the trace, never hidden.

Also here: the human-readable pattern views (``describe_pattern``: wazn
with ف/ع/ل, an approximate gloss from a closed list, a filled example)
and the token search index the page queries (``TokenIndex``).

Consumed by ``debugger/serve_araroopat_explorer.py``; rendered by the
03 tab of ``docs/araroopat_train_explorer.html``.
"""
from __future__ import annotations

import json
import logging
import pickle
import random
import re
import threading
import time
import unicodedata
import uuid
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from arabic_eval.tokenizers import araroopat as A
from arabic_eval.tokenizers import araroopat_backend as B
from arabic_eval.tokenizers.araroopat import (
    CHAR_INVENTORY,
    DIGIT_INVENTORY,
    PFX_PAT,
    PFX_ROOT,
    PUNCT_INVENTORY,
    SFX,
    SPECIAL_TOKENS_ORDERED,
    TOK_LIT_BEGIN,
    TOK_LIT_END,
    TOK_PROP_BEGIN,
    TOK_PROP_END,
    AraRooPatTokenizer,
    _classify_char,
    _extract_alpha_chunks,
    _split_runs,
    _strip_clitic_surfaces,
    entry_realization,
    join_word,
    pick_realization,
)
from arabic_eval.tokenizers.araroopat_backend import (
    CorpusEntry,
    MorphAnalyzer,
    _dict_to_analysis,
    naive_pattern_fill,
)
from arabic_eval.tokenizers.araroopat_bridge import CamelBridge, _resolve_camel_python
from arabic_eval.tokenizers.araroopat_trace import (
    _BATCH_SIZE,
    REAL_DEFAULTS,
    _cp,
    _cp_name,
    _inner,
    _split_key,
    _Trace,
    _trace_enclitic_stack,
    _trace_proclitic_stack,
    _WireTap,
    ref,
    trace_peel_words,
    trace_probe_roundtrip,
    trace_validate_words,
)
from arabic_eval.tokenizers.utils.arabic_text import strip_diacritics

logger = logging.getLogger("arabic_eval.tokenizers.araroopat_corpus_trace")

REPO_ROOT = Path(__file__).resolve().parents[3]
TOKENIZERS_DIR = REPO_ROOT / "outputs" / "tokenizers"
DEFAULT_CACHE_DIR = TOKENIZERS_DIR / "araroopat_cache"
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
DEFAULT_DATASET = "Jr23xd23/ArabicText-Large"

# Bounded sizes shipped to the page (the full tables live server-side and
# are reachable through the search endpoint).
SAMPLE_TEXTS = 6
SAMPLE_WORDS_PER_TEXT = 25
TOP_COUNTS = 200
TOP_FREQ = 150
BUDGET_HEAD = 60
BUDGET_WINDOW = 30
PASS3_CAP = 200
RECO_TABLE_SAMPLE = 200
META_TOP = 100
WIRE_CAP = 80
BATCH_MS_CAP = 6000

# ---------------------------------------------------------------------------
# Human-readable patterns: wazn, gloss, filled example
# ---------------------------------------------------------------------------

# CAMeL slot digits → the classical radical letters of a wazn. Quadriliteral
# roots use a second ل by convention (فَعْلَلَ).
WAZN_SLOT_LETTERS: Dict[str, str] = {"1": "ف", "2": "ع", "3": "ل", "4": "ل"}

# Approximate glosses, keyed on the *fully diacritized* wazn (precise) …
WAZN_GLOSSES_EXACT: Dict[str, str] = {
    # Form I verb + basic nouns
    "فَعَلَ": "Form I perfect verb, a-stem (كَتَبَ)",
    "فَعِلَ": "Form I perfect verb, i-stem (عَلِمَ)",
    "فَعُلَ": "Form I perfect verb, u-stem (كَبُرَ)",
    "فَعَلَت": "Form I perfect, 3rd fem. sg. (كَتَبَت)",
    "فَعْل": "Form I verbal noun / simple noun (فَضْل, ضَبْع)",
    "فِعْل": "simple noun, i-vowel (عِلْم)",
    "فُعْل": "simple noun, u-vowel (حُكْم) or broken plural (كُتْب)",
    "فَعَل": "simple noun, a-a (كَرَم, سَمَك)",
    "فِعَل": "broken plural (قِطَع) or noun",
    "فُعَل": "broken plural (دُوَل) or noun",
    "فُعُل": "broken plural (كُتُب)",
    "فَعال": "verbal noun / noun (سَلام, جَمال)",
    "فُعال": "noun (غُراب) or plural",
    "فَعّال": "intensive / occupational adjective (كَتّاب, فَعّال) — often with ة",
    "فَعِيل": "adjective (كَبِير) or noun",
    "فَعُول": "intensive adjective (صَبُور)",
    "فُعُول": "verbal noun (دُخُول) or broken plural (بُيُوت)",
    "فَعْلان": "adjective (عَطْشان) or verbal noun",
    "فُعْلان": "broken plural (قُمْصان)",
    "فَعَلان": "verbal noun of motion (طَيَران)",
    "فِعْلان": "broken plural (غِلْمان)",
    "فَعالَة": "verbal noun (زِراعة-type) / feminine noun",
    "فِعالَة": "verbal noun, often crafts and offices (تِجارة, وِزارة)",
    "فَعِيلَة": "feminine of فَعِيل / noun (حَدِيقة)",
    "فُعْلى": "feminine elative (كُبْرى)",
    "فَعْلى": "feminine adjective (سَكْرى) or plural",
    "فَعائِل": "broken plural of فَعِيلة / فَعالة (رَسائِل)",
    "فَواعِل": "broken plural of فاعِل / فاعِلة (عَوامِل)",
    "فَعالِل": "broken plural of quadriliterals (دَراهِم)",
    "فَعالِيل": "broken plural (قَنادِيل)",
    "فِعْلِيّ": "nisba adjective (عِلْمِيّ)",
    "فَعْلِيّ": "nisba adjective (وَطْنِيّ)",
    "فُعْلِيّ": "nisba adjective (كُلِّيّ)",
    # participles
    "فاعِل": "Form I active participle — doer (كاتِب)",
    "مَفْعُول": "Form I passive participle (مَكْتُوب)",
    "مُفَعِّل": "Form II active participle (مُعَلِّم)",
    "مُفَعَّل": "Form II passive participle (مُعَلَّم)",
    "مُفاعِل": "Form III active participle (مُشارِك)",
    "مُفاعَل": "Form III passive participle",
    "مُفْعِل": "Form IV active participle (مُرْسِل)",
    "مُفْعَل": "Form IV passive participle (مُرْسَل)",
    "مُتَفَعِّل": "Form V active participle (مُتَعَلِّم)",
    "مُتَفاعِل": "Form VI active participle (مُتَعاوِن)",
    "مُنْفَعِل": "Form VII active participle (مُنْكَسِر)",
    "مُفْتَعِل": "Form VIII active participle (مُجْتَمِع)",
    "مُفْتَعَل": "Form VIII passive participle (مُجْتَمَع)",
    "مُسْتَفْعِل": "Form X active participle (مُسْتَخْدِم)",
    "مُسْتَفْعَل": "Form X passive participle (مُسْتَخْدَم)",
    # nouns of place / instrument / elative
    "مَفْعَل": "noun of place or time (مَكْتَب) / Form I verbal noun",
    "مَفْعِل": "noun of place (مَسْجِد)",
    "مِفْعَل": "instrument noun (مِبْرَد)",
    "مِفْعال": "instrument noun (مِفْتاح)",
    "مِفْعَلة": "instrument noun (مِكْنَسة)",
    "مَفاعِل": "broken plural of مَفْعَل / مَفْعِل (مَكاتِب)",
    "مَفاعِيل": "broken plural of مَفْعُول / مِفْعال (مَفاتِيح)",
    "أَفْعَل": "elative / comparative (أَكْبَر) or Form IV perfect (أَرْسَل)",
    "أَفْعال": "broken plural (أَقْلام)",
    "أَفْعِلة": "broken plural (أَسْئِلة)",
    "أَفْعُل": "broken plural (أَشْهُر)",
    # derived-form verbs and verbal nouns
    "فَعَّلَ": "Form II perfect verb (عَلَّمَ)",
    "فَعَّل": "Form II stem (عَلَّم)",
    "تَفْعِيل": "Form II verbal noun (تَعْلِيم)",
    "تَفْعِلة": "Form II verbal noun, weak-final (تَسْمِية)",
    "فاعَلَ": "Form III perfect verb (شارَكَ)",
    "فاعَل": "Form III stem (شارَك)",
    "مُفاعَلة": "Form III verbal noun (مُشارَكة)",
    "فِعال": "Form III verbal noun (قِتال) or broken plural (رِجال)",
    "أَفْعَلَ": "Form IV perfect verb (أَرْسَلَ)",
    "إِفْعال": "Form IV verbal noun (إِرْسال)",
    "تَفَعَّلَ": "Form V perfect verb (تَعَلَّمَ)",
    "تَفَعَّل": "Form V stem (تَعَلَّم)",
    "تَفَعُّل": "Form V verbal noun (تَعَلُّم)",
    "تَفاعَلَ": "Form VI perfect verb (تَعاوَنَ)",
    "تَفاعَل": "Form VI stem (تَعاوَن)",
    "تَفاعُل": "Form VI verbal noun (تَعاوُن)",
    "ٱِنْفَعَلَ": "Form VII perfect verb (ٱِنْكَسَرَ)",
    "ٱِنْفَعَل": "Form VII stem (ٱِنْكَسَر)",
    "ٱِنْفِعال": "Form VII verbal noun (ٱِنْكِسار)",
    "ٱِفْتَعَلَ": "Form VIII perfect verb (ٱِجْتَمَعَ)",
    "ٱِفْتَعَل": "Form VIII stem (ٱِجْتَمَع)",
    "ٱِفْتِعال": "Form VIII verbal noun (ٱِجْتِماع)",
    "ٱِفْعَلَّ": "Form IX perfect verb — colours and defects (ٱِحْمَرَّ)",
    "ٱِفْعِلال": "Form IX verbal noun (ٱِحْمِرار)",
    "ٱِسْتَفْعَلَ": "Form X perfect verb (ٱِسْتَخْدَمَ)",
    "ٱِسْتَفْعَل": "Form X stem (ٱِسْتَخْدَم)",
    "ٱِسْتِفْعال": "Form X verbal noun (ٱِسْتِخْدام)",
    # imperfect (present) stems
    "يَفْعَل": "Form I imperfect, a-stem (يَذْهَب)",
    "يَفْعُل": "Form I imperfect, u-stem (يَكْتُب)",
    "يَفْعِل": "Form I imperfect, i-stem (يَجْلِس)",
    "تَفْعَل": "Form I imperfect, 2nd sg. / 3rd fem. (تَذْهَب)",
    "تَفْعُل": "Form I imperfect, 2nd sg. / 3rd fem. (تَكْتُب)",
    "تَفْعِل": "Form I imperfect, 2nd sg. / 3rd fem. (تَجْلِس)",
    "نَفْعَل": "Form I imperfect, 1st pl. (نَذْهَب)",
    "نَفْعُل": "Form I imperfect, 1st pl. (نَكْتُب)",
    "نَفْعِل": "Form I imperfect, 1st pl. (نَجْلِس)",
    "أَفْعُل": "Form I imperfect, 1st sg. (أَكْتُب) — or broken plural (أَشْهُر)",
    "أَفْعِل": "Form I imperfect, 1st sg. (أَجْلِس)",
    "يُفَعِّل": "Form II imperfect (يُعَلِّم)",
    "تُفَعِّل": "Form II imperfect, 2nd sg. / 3rd fem.",
    "يُفاعِل": "Form III imperfect (يُشارِك)",
    "يُفْعِل": "Form IV imperfect (يُرْسِل)",
    "تُفْعِل": "Form IV imperfect, 2nd sg. / 3rd fem.",
    "يَتَفَعَّل": "Form V imperfect (يَتَعَلَّم)",
    "يَتَفاعَل": "Form VI imperfect (يَتَعاوَن)",
    "يَنْفَعِل": "Form VII imperfect (يَنْكَسِر)",
    "يَفْتَعِل": "Form VIII imperfect (يَجْتَمِع)",
    "يَسْتَفْعِل": "Form X imperfect (يَسْتَخْدِم)",
    "يُفْعَل": "Form I/IV passive imperfect (يُكْتَب)",
    "يُفَعَّل": "Form II passive imperfect (يُعَلَّم)",
    "فُعِلَ": "Form I passive perfect (كُتِبَ)",
    "فُعِّلَ": "Form II passive perfect (عُلِّمَ)",
    "أُفْعِلَ": "Form IV passive perfect (أُرْسِلَ)",
    "ٱُفْتُعِلَ": "Form VIII passive perfect",
    "ٱُسْتُفْعِلَ": "Form X passive perfect",
    "ٱِفْعَل": "Form I imperative (ٱِذْهَب)",
    "ٱُفْعُل": "Form I imperative, u-stem (ٱُكْتُب)",
}

# … and on the diacritic-stripped skeleton (loose — lists the alternatives).
# Hamza-alef forms (أ إ ٱ) are folded to ا in the skeleton.
WAZN_GLOSSES_SKELETON: Dict[str, str] = {
    "فعل": "triliteral base: Form I verb (فَعَلَ / يَفْعَل) or a simple noun (فَعْل, فِعْل, فُعْل) — the vowels decide",
    "فاعل": "active participle Form I (فاعِل) or Form III stem (فاعَل)",
    "مفعول": "Form I passive participle (مَفْعُول)",
    "مفعل": "noun of place / instrument (مَفْعَل, مَفْعِل, مِفْعَل) or Form II / IV participle (مُفَعِّل, مُفْعِل, مُفَعَّل)",
    "مفعال": "instrument noun (مِفْعال)",
    "مفاعل": "Form III participle (مُفاعِل) or broken plural of مَفْعَل (مَفاعِل)",
    "مفاعيل": "broken plural (مَفاعِيل)",
    "فعال": "verbal noun / plural (فِعال, فَعال) or intensive adjective (فَعّال)",
    "فعيل": "adjective (فَعِيل)",
    "فعول": "intensive adjective (فَعُول) or verbal noun / plural (فُعُول)",
    "فعلان": "adjective (فَعْلان), verbal noun (فَعَلان) or plural (فُعْلان)",
    "فعائل": "broken plural (فَعائِل)",
    "فواعل": "broken plural (فَواعِل)",
    "فعالل": "broken plural of quadriliterals (فَعالِل)",
    "فعاليل": "broken plural (فَعالِيل)",
    "فعلي": "nisba adjective (فَعْلِيّ)",
    "افعل": "elative (أَفْعَل), Form IV perfect (أَفْعَلَ), 1st sg. imperfect (أَفْعُل) or imperative (ٱِفْعَل)",
    "افعال": "broken plural (أَفْعال) or Form IV verbal noun (إِفْعال)",
    "تفعيل": "Form II verbal noun (تَفْعِيل)",
    "تفعل": "Form V verb / verbal noun (تَفَعَّل, تَفَعُّل) or Form I imperfect 2nd sg. / 3rd fem. (تَفْعَل)",
    "تفاعل": "Form VI verb / verbal noun (تَفاعَل, تَفاعُل)",
    "مفاعلة": "Form III verbal noun (مُفاعَلة)",
    "انفعل": "Form VII verb (ٱِنْفَعَل)",
    "انفعال": "Form VII verbal noun (ٱِنْفِعال)",
    "افتعل": "Form VIII verb (ٱِفْتَعَل)",
    "افتعال": "Form VIII verbal noun (ٱِفْتِعال)",
    "افعلال": "Form IX verbal noun (ٱِفْعِلال)",
    "استفعل": "Form X verb (ٱِسْتَفْعَل)",
    "استفعال": "Form X verbal noun (ٱِسْتِفْعال)",
    "يفعل": "Form I imperfect (يَفْعَل / يَفْعُل / يَفْعِل) or passive (يُفْعَل)",
    "نفعل": "Form I imperfect, 1st pl. (نَفْعَل)",
    "يفعل ": "",
    "يفاعل": "Form III imperfect (يُفاعِل)",
    "يتفعل": "Form V imperfect (يَتَفَعَّل)",
    "يتفاعل": "Form VI imperfect (يَتَفاعَل)",
    "ينفعل": "Form VII imperfect (يَنْفَعِل)",
    "يفتعل": "Form VIII imperfect (يَفْتَعِل)",
    "يستفعل": "Form X imperfect (يَسْتَفْعِل)",
    "متفعل": "Form V active participle (مُتَفَعِّل)",
    "متفاعل": "Form VI active participle (مُتَفاعِل)",
    "منفعل": "Form VII active participle (مُنْفَعِل)",
    "مفتعل": "Form VIII participle (مُفْتَعِل / مُفْتَعَل)",
    "مستفعل": "Form X participle (مُسْتَفْعِل / مُسْتَفْعَل)",
    "فعلة": "feminine / unit noun (فَعْلة, فِعْلة) or verbal noun",
    "فعالة": "verbal noun (فِعالة, فَعالة) or feminine of فَعّال",
    "فعيلة": "feminine of فَعِيل (فَعِيلة)",
    "فعلى": "feminine elative (فُعْلى) or adjective (فَعْلى)",
    "افعلة": "broken plural (أَفْعِلة)",
    "مفعلة": "noun of place / instrument, feminine (مَفْعَلة, مِفْعَلة)",
}
WAZN_GLOSSES_SKELETON = {k: v for k, v in WAZN_GLOSSES_SKELETON.items() if v}

_HAMZA_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})


def wazn_of(pattern: str) -> str:
    """CAMeL slot template → classical wazn (``مَ1ْ2َ3َ`` → ``مَفْعَلَ``)."""
    return "".join(WAZN_SLOT_LETTERS.get(ch, ch) for ch in pattern)


def wazn_skeleton(pattern: str) -> str:
    """Diacritic-stripped, hamza-folded wazn — the loose gloss key."""
    return strip_diacritics(wazn_of(pattern)).translate(_HAMZA_ALEF)


def gloss_for(pattern: str) -> Tuple[Optional[str], Optional[str]]:
    """``(gloss, tier)`` — tier ``"exact"`` (diacritized wazn) or ``"skeleton"``."""
    w = wazn_of(pattern)
    if w in WAZN_GLOSSES_EXACT:
        return WAZN_GLOSSES_EXACT[w], "exact"
    sk = wazn_skeleton(pattern)
    if sk in WAZN_GLOSSES_SKELETON:
        return WAZN_GLOSSES_SKELETON[sk], "skeleton"
    return None, None


def describe_pattern(pattern: str, examples: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Everything the page needs to show a ``[PAT_*]`` token to a human.

    ``examples`` is the metadata list ``[[root, surface], ...]``; the first
    one is filled into the template with ``naive_pattern_fill`` (exact for
    sound roots; for a ``#`` root the realized weak letter is already
    literal template material, so the fill is still the inflected stem).
    """
    digits = [int(ch) for ch in pattern if ch in "1234"]
    n_slots = max(3, max(digits)) if digits else 3
    weak = [k for k in range(1, n_slots + 1) if k not in digits]
    gloss, tier = gloss_for(pattern)
    out: Dict[str, Any] = {
        "raw": pattern,
        "wazn": wazn_of(pattern),
        "skeleton": wazn_skeleton(pattern),
        "gloss": gloss,
        "gloss_tier": tier,
        "slots_present": sorted(set(digits)),
        "weak_slots": weak,
        "filled": None,
    }
    if examples:
        first = examples[0]
        root = first[0] if isinstance(first, (list, tuple)) else None
        surface = first[1] if isinstance(first, (list, tuple)) and len(first) > 1 else None
        if root:
            marks = []
            for ch in pattern:
                if ch in "1234":
                    idx = int(ch) - 1
                    if idx < len(root) and root[idx] != B.WEAK_RADICAL_MARK:
                        marks.append({"ch": root[idx], "slot": int(ch)})
                else:
                    marks.append({"ch": ch, "slot": 0})
            out["filled"] = {
                "root": root, "surface": surface,
                "stem": naive_pattern_fill(root, pattern), "marks": marks,
            }
    return out


# ---------------------------------------------------------------------------
# What is on disk
# ---------------------------------------------------------------------------

def _dataset_cache_present(dataset_name: str, cache_dir: Path) -> bool:
    """HF `datasets` caches ``owner/name`` as ``owner___name`` (lower-cased on some versions)."""
    if not cache_dir.exists():
        return False
    norm = lambda x: re.sub(r"[_\-]", "", x.lower())  # noqa: E731  ArabicText-Large ↔ arabic_text-large
    want = norm(dataset_name.replace("/", "___"))
    return any(p.is_dir() and norm(p.name) == want for p in cache_dir.iterdir())


def _data_config() -> Dict[str, Any]:
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["data"]


def list_sources(tokenizers_dir: Path = TOKENIZERS_DIR) -> Dict[str, Any]:
    """Saved araroopat tokenizers, pre-pass caches, dataset cache — for the page's selects."""
    saved: List[Dict[str, Any]] = []
    caches: List[Dict[str, Any]] = []
    if tokenizers_dir.exists():
        for d in sorted(tokenizers_dir.iterdir()):
            if not d.is_dir():
                continue
            cfg_path = d / "config.json"
            if cfg_path.exists() and (d / "vocab.json").exists():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    cfg = {}
                if cfg.get("tokenizer_class") == "AraRooPatTokenizer":
                    try:
                        vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        vocab = {}
                    fam = Counter(_split_key(t) for t in vocab)
                    saved.append({
                        "name": d.name, "path": str(d.relative_to(REPO_ROOT)),
                        "mtime": (d / "vocab.json").stat().st_mtime,
                        "vocab_size": len(vocab),
                        "roots": fam.get("root", 0), "patterns": fam.get("pat", 0),
                        "has_metadata": (d / "vocab_metadata.json").exists(),
                        "has_reconstruction": (d / "reconstruction.pkl").exists()
                                              or (d / "reconstruction.json").exists(),
                        "reconstruction_bytes": (d / "reconstruction.pkl").stat().st_size
                                                if (d / "reconstruction.pkl").exists() else None,
                        "has_prepositions": bool(cfg.get("prepositions")),
                        "has_func_words": bool(cfg.get("func_words")),
                        "has_prop_markers": TOK_PROP_BEGIN in vocab,
                        "config": {k: cfg.get(k) for k in (
                            "max_roots", "max_patterns", "min_root_freq", "min_pattern_freq",
                            "use_diacritized_surface", "clitic_peeler", "peel_bare_alef", "proper_nouns")},
                    })
            pkl = d / "corpus_analysis.pkl"
            if pkl.exists():
                caches.append({
                    "name": d.name, "path": str(d.relative_to(REPO_ROOT)),
                    "bytes": pkl.stat().st_size, "mtime": pkl.stat().st_mtime,
                    "json_view": (d / "corpus_analysis.json").exists(),
                    "is_default": d.resolve() == DEFAULT_CACHE_DIR.resolve(),
                })
    data_cfg = _data_config()
    ds_cache = REPO_ROOT / data_cfg.get("cache_dir", "outputs/data_cache")
    return {
        "saved": saved,
        "caches": caches,
        "default_cache": str(DEFAULT_CACHE_DIR.relative_to(REPO_ROOT)),
        "dataset": {
            "name": data_cfg.get("dataset_name", DEFAULT_DATASET),
            "cache_dir": str(ds_cache.relative_to(REPO_ROOT)),
            "cached_locally": _dataset_cache_present(data_cfg.get("dataset_name", DEFAULT_DATASET), ds_cache),
            "preprocessing": data_cfg.get("preprocessing", {}),
            "max_train_samples": data_cfg.get("max_train_samples"),
        },
        "tiers": {
            "compact": {"max_roots": 10000, "max_patterns": 1000},
            "balanced": {"max_roots": 10000, "max_patterns": 4000},
            "max": {"max_roots": 10000, "max_patterns": 6076},
        },
        "real_defaults": REAL_DEFAULTS,
    }


# ---------------------------------------------------------------------------
# Request validation (shared by the server)
# ---------------------------------------------------------------------------

_INT_PARAMS = ("max_roots", "max_patterns", "min_root_freq", "min_pattern_freq")
_BOOL_PARAMS = ("use_diacritized_surface",)
_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,80}$")


def validate_request(req: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce the POST body of ``/api/corpus/train`` into a clean request dict.

    Raises ``ValueError`` with a user-facing message.
    """
    mode = req.get("mode") or "train"
    if mode not in ("train", "saved"):
        raise ValueError("mode must be 'train' or 'saved'")
    params: Dict[str, Any] = {}
    raw = req.get("params") or {}
    for k in _INT_PARAMS:
        if k in raw and raw[k] not in (None, ""):
            try:
                params[k] = max(0, int(raw[k]))
            except (TypeError, ValueError):
                raise ValueError(f"{k} must be an integer")
    for k in _BOOL_PARAMS:
        if k in raw:
            params[k] = bool(raw[k])
    out: Dict[str, Any] = {
        "mode": mode,
        "params": params,
        "probe_text": str(req.get("probe_text") or "").strip(),
        "sample_size": max(5, min(200, int(req.get("sample_size") or 40))),
        "seed": int(req.get("seed") or 42),
        "verify": bool(req.get("verify", True)),
    }
    if not out["probe_text"]:
        raise ValueError("probe_text is empty — the encode → decode proof needs a text")
    if len(out["probe_text"]) > 20_000:
        raise ValueError("probe_text too long (> 20000 chars)")
    if mode == "saved":
        name = str(req.get("saved_dir") or "")
        if not _NAME_RE.match(name):
            raise ValueError("saved_dir must be a directory name under outputs/tokenizers")
        d = TOKENIZERS_DIR / name
        if not (d / "config.json").exists() or not (d / "vocab.json").exists():
            raise ValueError(f"{name} is not a saved araroopat tokenizer")
        out["saved_dir"] = name
    else:
        out["dataset_name"] = str(req.get("dataset_name") or DEFAULT_DATASET)
        mts = req.get("max_train_samples")
        out["max_train_samples"] = int(mts) if mts not in (None, "", 0, "0") else None
        if out["max_train_samples"] is not None and out["max_train_samples"] < 1:
            raise ValueError("max_train_samples must be ≥ 1 or blank")
        cache = str(req.get("cache_dir") or DEFAULT_CACHE_DIR.name)
        if not _NAME_RE.match(cache):
            raise ValueError("cache_dir must be a directory name under outputs/tokenizers")
        out["cache_dir"] = cache
        policy = req.get("cache_policy") or "auto"
        if policy not in ("auto", "ignore"):
            raise ValueError("cache_policy must be 'auto' or 'ignore'")
        out["cache_policy"] = policy
        out["write_cache"] = bool(req.get("write_cache", True))
    return out


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------

class JobCancelled(Exception):
    pass


def _pick(rng: random.Random, seq: List[Any], k: int) -> List[Any]:
    if len(seq) <= k:
        return list(seq)
    return rng.sample(seq, k)


def _entry_summary(e: CorpusEntry) -> Dict[str, Any]:
    return {"analyzed": e.analyzed, "root": e.root, "pattern": e.pattern,
            "proclitics": list(e.proclitics or ()), "enclitics": list(e.enclitics or ()),
            "particle": e.particle, "particle_kind": getattr(e, "particle_kind", "prep"),
            "clitic_only": bool(getattr(e, "clitic_only", False)),
            "proper": bool(getattr(e, "proper", False)),
            "peeled": bool(getattr(e, "peeled", False))}


def _analysis_summary(a: Optional[B.Analysis]) -> Dict[str, Any]:
    if a is None:
        return {"analyzed": False}
    return {"analyzed": True, "root": a.root or None, "pattern": a.pattern or None,
            "proclitics": list(a.proclitics), "enclitics": list(a.enclitics),
            "particle": a.particle, "particle_kind": a.particle_kind, "clitic_only": bool(a.clitic_only),
            "proper": bool(a.proper)}


def _agrees(entry: CorpusEntry, a: Optional[B.Analysis]) -> bool:
    es, as_ = _entry_summary(entry), _analysis_summary(a)
    if not es["analyzed"] and not as_["analyzed"]:
        return True
    if es["analyzed"] != as_["analyzed"]:
        return False
    return all(es[k] == as_[k] for k in ("root", "pattern", "proclitics", "enclitics", "particle", "clitic_only", "proper")) and \
        (not es["particle"] or es["particle_kind"] == as_["particle_kind"])


class CorpusTraceJob:
    """Background thread: load / train, build the 16-step trace, keep the tokenizer.

    Poll ``snapshot()`` for progress; when ``status == "done"`` the job holds
    ``trace``, ``tokenizer`` and ``index`` (a ``TokenIndex``). The job owns
    its own CAMeL subprocess so the small-text tab stays responsive while a
    corpus pre-pass runs. ``close()`` shuts that subprocess down.
    """

    def __init__(self, request: Dict[str, Any]) -> None:
        self.id = uuid.uuid4().hex
        self.request = request
        self.status = "queued"          # queued | running | done | error | cancelled
        self.stage = ""
        self.stage_detail = ""
        self.done = 0
        self.total = 0
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.error: Optional[str] = None
        self.log: List[str] = []
        self.trace: Optional[Dict[str, Any]] = None
        self.tokenizer: Optional[AraRooPatTokenizer] = None
        self.index: Optional["TokenIndex"] = None
        # Train mode only: every unique chunk with its encode-time category (see WordCategoryIndex).
        self.words: Optional["WordCategoryIndex"] = None
        # Both modes: the complete frequency tables of the freq step, searchable and paged.
        self.freq: Optional["FreqIndex"] = None
        self.bridge: Optional[CamelBridge] = None
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"araroopat-corpus-{self.id[:8]}", daemon=True)

    # ---- control ------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        if self.bridge is not None:
            try:
                self.bridge._shutdown_quietly()
            except Exception:  # noqa: BLE001
                pass

    def snapshot(self) -> Dict[str, Any]:
        now = self.finished_at or time.time()
        return {
            "job_id": self.id, "status": self.status, "stage": self.stage,
            "stage_detail": self.stage_detail, "done": self.done, "total": self.total,
            "elapsed_s": round(now - self.started_at, 1), "error": self.error,
            "log": self.log[-40:], "mode": self.request["mode"],
            "has_trace": self.trace is not None,
        }

    # ---- progress -----------------------------------------------------
    def _log(self, msg: str) -> None:
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        logger.info("corpus job %s: %s", self.id[:8], msg)

    def _stage(self, name: str, done: int = 0, total: int = 0, detail: str = "") -> None:
        self.stage, self.done, self.total, self.stage_detail = name, done, total, detail
        self._check_cancel()

    def _tick(self, done: int, total: Optional[int] = None) -> None:
        self.done = done
        if total is not None:
            self.total = total
        self._check_cancel()

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled()

    # ---- run ----------------------------------------------------------
    def _run(self) -> None:
        self.status = "running"
        try:
            if self.request["mode"] == "saved":
                self._trace_saved()
            else:
                self._trace_train()
            self._stage("index")
            assert self.tokenizer is not None and self.trace is not None
            self.index = TokenIndex(self.tokenizer)
            self.trace["corpus_id"] = self.id
            self.status = "done"
            self._log("done")
        except JobCancelled:
            self.status = "cancelled"
            self._log("cancelled")
            self.close()
        except Exception as e:  # noqa: BLE001
            logger.exception("corpus job failed")
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"
            self._log(self.error)
            self.close()
        finally:
            self.finished_at = time.time()

    def _make_tokenizer(self, params: Dict[str, Any], cache_flag: bool) -> AraRooPatTokenizer:
        tok = AraRooPatTokenizer(**{**params, "cache_corpus_analysis": cache_flag})
        # Same construction as tok._ensure_backend(), on the job's own bridge.
        if self.bridge is None:
            self.bridge = CamelBridge()
        tok._backend = MorphAnalyzer(
            generator_timeout_ms=tok.generator_timeout_ms, bridge=self.bridge,
            particles=frozenset(tok.prepositions), enable_peeler=tok.clitic_peeler,
            peel_bare_alef=tok.peel_bare_alef, func_words=frozenset(tok.func_words),
        )
        return tok

    @staticmethod
    def _effective(tok: AraRooPatTokenizer) -> Dict[str, Any]:
        return {
            "max_roots": tok.max_roots, "max_patterns": tok.max_patterns,
            "min_root_freq": tok.min_root_freq, "min_pattern_freq": tok.min_pattern_freq,
            "use_diacritized_surface": tok.use_diacritized_surface,
            "cache_corpus_analysis": tok.cache_corpus_analysis, "add_bos_eos": tok.add_bos_eos,
            "proper_nouns": tok.proper_nouns,
        }

    @staticmethod
    def _scale(shown: Any, total: Any, note: str, **extra: Any) -> Dict[str, Any]:
        return {"shown": shown, "total": total, "note": note, **extra}

    # ------------------------------------------------------------------
    # mode = train
    # ------------------------------------------------------------------
    def _trace_train(self) -> None:
        req = self.request
        rng = random.Random(req["seed"])
        tr = _Trace()
        tok = self._make_tokenizer(req["params"], cache_flag=req["write_cache"])
        backend: MorphAnalyzer = tok._backend  # type: ignore[assignment]
        bridge = self.bridge
        assert bridge is not None
        self._stage("camel", detail="starting the job's own CAMeL subprocess")
        bridge._ensure_started()
        bridge.analyze(["إحماء"])

        # ---- corpus ---------------------------------------------------
        self._stage("load_corpus", detail=req["dataset_name"])
        t_load = time.perf_counter()
        data_cfg = _data_config()
        if req.get("texts") is not None:
            # Injected corpus (tests / smoke runs) — bypasses the HF loader only.
            texts: List[str] = [str(t) for t in req["texts"]]
            eval_rows = 0
            loader_ref = "injected texts"
        else:
            from arabic_eval.data.loader import extract_texts, load_arabic_dataset  # heavy import, local
            ds = load_arabic_dataset(
                dataset_name=req["dataset_name"], cache_dir=data_cfg.get("cache_dir", "outputs/data_cache"),
                max_train_samples=req["max_train_samples"],
                max_eval_samples=data_cfg.get("max_eval_samples"),
                preprocessing_config=data_cfg.get("preprocessing"), seed=req["seed"],
            )
            texts = list(extract_texts(ds["train"]))
            eval_rows = len(ds.get("eval", []))
            loader_ref = ref(load_arabic_dataset)
        load_ms = round((time.perf_counter() - t_load) * 1000, 1)
        total_chars = sum(len(t) for t in texts)
        self._log(f"corpus loaded: {len(texts)} texts, {total_chars} chars in {load_ms} ms")
        corpus_info = {
            "dataset_name": req["dataset_name"], "rows": len(texts), "chars": total_chars,
            "eval_rows": eval_rows, "max_train_samples": req["max_train_samples"],
            "preprocessing": data_cfg.get("preprocessing"), "load_ms": load_ms,
            "loader": loader_ref,
        }

        # ---- step 0: config -------------------------------------------
        t0 = time.perf_counter()
        effective = self._effective(tok)
        tr.step("config", "Tokenizer instance & parameters", AraRooPatTokenizer.__init__, {
            "effective": effective, "real_defaults": REAL_DEFAULTS,
            "overridden": [k for k, v in effective.items() if REAL_DEFAULTS.get(k) != v],
            "num_texts": len(texts),
            "camel_python": str(bridge._camel_python or _resolve_camel_python()),
            "corpus": corpus_info, "mode": "train",
            "cache_policy": req["cache_policy"], "write_cache": req["write_cache"],
            "_scale": self._scale(None, None,
                                  f"Corpus mode — {len(texts):,} texts loaded through the pipeline's "
                                  f"load_arabic_dataset with configs/base.yaml preprocessing."),
        }, t0, notes=[
            "vocab_size is ignored by araroopat — the final size is max_roots + max_patterns + fixed slots.",
            "This job runs on its own CAMeL subprocess; the small-text tab keeps the shared one.",
        ])

        # ---- pre-pass part 1: NFKC → split → chunks → dedup (whole corpus)
        self._stage("chunk", 0, len(texts), "NFKC → split → alpha chunks → Counter, every text")
        t_chunk = time.perf_counter()
        word_counts: Counter = Counter()
        total_words = 0
        changed_texts = 0
        sample_idx = sorted(_pick(rng, list(range(len(texts))), SAMPLE_TEXTS))
        sample_set = set(sample_idx)
        norm_rows: List[Dict[str, Any]] = []
        split_rows: List[Dict[str, Any]] = []
        word_rows: List[Dict[str, Any]] = []
        for i, t in enumerate(texts):
            n = unicodedata.normalize("NFKC", t)
            if n != t:
                changed_texts += 1
            ws = n.split()
            total_words += len(ws)
            for w in ws:
                for chunk in _extract_alpha_chunks(w):
                    word_counts[chunk] += 1
            if i in sample_set:
                changes = []
                if n != t:
                    for ch in sorted(set(t) - set(n)):
                        changes.append({"before": ch, "before_cp": _cp(ch), "before_name": _cp_name(ch),
                                        "after": unicodedata.normalize("NFKC", ch)})
                norm_rows.append({"i": i, "before": t, "after": n, "changed": n != t, "changes": changes})
                split_rows.append({"i": i, "words": ws})
                for w in ws[:SAMPLE_WORDS_PER_TEXT]:
                    word_rows.append({
                        "line": i, "word": w,
                        "chars": [{"ch": c, "cls": _classify_char(c), "cp": _cp(c)} for c in w],
                        "runs": [{"text": chunk, "cls": cls, "kept": cls == "alpha",
                                  "ends_with_fem": chunk.endswith("ة")} for cls, chunk in _split_runs(w)],
                        "chunks": _extract_alpha_chunks(w),
                    })
            if i % 2000 == 0:
                self._tick(i, len(texts))
        unique_words = list(word_counts.keys())
        chunk_ms = round((time.perf_counter() - t_chunk) * 1000, 1)
        self._log(f"chunk pass: {total_words:,} words → {len(unique_words):,} unique chunks in {chunk_ms} ms")

        t0 = time.perf_counter()
        tr.step("nfkc", "Unicode NFKC normalization", A.AraRooPatTokenizer._corpus_prepass, {
            "rows": norm_rows,
            "changed_total": changed_texts, "total": len(texts),
            "_scale": self._scale(len(norm_rows), len(texts),
                                  f"{len(norm_rows)} seeded sample texts shown (corpus rows "
                                  f"{', '.join(map(str, sample_idx))}); NFKC changed {changed_texts:,} of "
                                  f"{len(texts):,} texts."),
        }, t0, notes=["Run once per text inside _corpus_prepass, before whitespace splitting."])
        tr.step("split", "Whitespace split", "builtins.str.split", {
            "rows": split_rows, "total_words": total_words,
            "_scale": self._scale(sum(len(r["words"]) for r in split_rows), total_words,
                                  "Sample texts shown; total_words is corpus-wide."),
        }, t0, notes=["str.split() with no argument: any run of Unicode whitespace is a boundary."])
        tr.step("chunks", "Character classes → Arabic alpha chunks", _extract_alpha_chunks, {
            "words": word_rows,
            "class_legend": {
                "alpha": "ARABIC_LETTERS ∪ ARABIC_DIACRITICS (minus ة)",
                "fem": "ة — closes the alpha run it follows; never a [CHAR_*], always [CLITICE_ة]",
                "digit": "0-9 and ٠-٩ (DIGIT_INVENTORY)",
                "punct": "ASCII punctuation + Arabic/typographic (PUNCT_INVENTORY)",
                "space": "whitespace (already removed by split)",
                "other": "anything else: Latin letters, emoji, … → UNK at encode time",
            },
            "pass_ms": chunk_ms,
            "_scale": self._scale(len(word_rows), total_words,
                                  f"First {SAMPLE_WORDS_PER_TEXT} words of each sample text. The whole pass "
                                  f"(NFKC + split + chunking + counting) over {len(texts):,} texts took {chunk_ms/1000:.0f} s."),
        }, t0, notes=[
            "Only 'alpha' runs go to the analyzer. Digit/punct runs are handled by fixed inventories at encode time.",
            "_classify_char is shared with _encode_word, so the pre-pass and encode() agree on the exact "
            "strings sent to CAMeL — this is what makes the analysis cache hit.",
        ])
        top_counts = word_counts.most_common(TOP_COUNTS)
        sample_chunks_from_texts = list(dict.fromkeys(c for r in word_rows for c in r["chunks"]))
        shown = list(dict.fromkeys([w for w, _ in top_counts] + sample_chunks_from_texts))
        tr.step("dedup", "Deduplicate → word_counts", "collections.Counter", {
            "word_counts": [{"word": w, "count": word_counts[w]} for w in shown],
            "unique": len(unique_words), "occurrences": sum(word_counts.values()),
            "_scale": self._scale(len(shown), len(unique_words),
                                  f"Top {TOP_COUNTS} chunks by count, then the sample texts' chunks. "
                                  f"{len(unique_words):,} unique chunks cover {sum(word_counts.values()):,} occurrences."),
        }, t0, notes=[
            "Counter keeps first-seen order; unique_words = list(word_counts.keys()).",
            "The analyzer is called once per *unique* chunk — this is the whole point of the pre-pass.",
        ])

        # ---- pre-pass part 2: the cache decision (the real rule) --------
        cache_dir = TOKENIZERS_DIR / req["cache_dir"]
        cache_file = cache_dir / "corpus_analysis.pkl"
        self._stage("cache", detail=str(cache_file.relative_to(REPO_ROOT)))
        t0 = time.perf_counter()
        entries: Optional[List[CorpusEntry]] = None
        decision = ""
        reasons: List[str] = []
        key_found: Any = None
        cached_n = None
        coverage: Optional[Dict[str, Any]] = None
        cached_by_word: Dict[str, CorpusEntry] = {}
        to_analyze: List[str] = unique_words
        if req["cache_policy"] == "ignore":
            decision = "bypassed"
            reasons.append("cache_policy=ignore — the pre-pass runs again even if the file covers the corpus")
        elif not cache_file.exists():
            decision = "miss"
            reasons.append("no corpus_analysis.pkl at that path")
        else:
            try:
                with cache_file.open("rb") as f:
                    payload = pickle.load(f)
                key_found = payload.get("key") if isinstance(payload, dict) else None
                try:
                    cached, reuse = tok._reusable_cache_entries(payload)
                except ValueError:
                    cached, reuse = None, "mismatch"
                if cached is None:
                    decision = "miss"
                    reasons.append("cache format/key mismatch — analyses were produced by a different "
                                   "_CACHE_FORMAT / preposition inventory / peeler setting")
                else:
                    cached_n = len(payload["entries"])
                    if reuse == "migrate":
                        reasons.append(f"format {key_found[0]} → {tok._CACHE_FORMAT}: migratable — "
                                       f"{cached_n - len(cached):,} entries the spelling-faithful walk can change are "
                                       f"re-analysed, {len(cached):,} reused")
                    cached_words = {e.word for e in cached}
                    missing = [w for w in unique_words if w not in cached_words]
                    coverage = {"cached_unique": len(cached_words), "needed_unique": len(unique_words),
                                "missing": len(missing), "missing_examples": missing[:20],
                                "migrated": cached_n - len(cached) if reuse == "migrate" else 0}
                    if not missing:
                        decision = "hit"
                        entries = [e for e in cached if e.word in word_counts]
                        reasons.append(f"cached_words ⊇ unique_words → reuse, filtered to the {len(entries):,} "
                                       f"entries this corpus needs")
                    else:
                        decision = "partial"
                        cached_by_word = {e.word: e for e in cached}
                        to_analyze = missing
                        reasons.append(f"key matches but {len(missing):,} of {len(unique_words):,} chunks are not in the "
                                       f"cache → only those go to CAMeL; the union is written back")
            except Exception as e:  # noqa: BLE001
                decision = "miss"
                cached_by_word = {}
                to_analyze = unique_words
                reasons.append(f"cache load failed ({type(e).__name__}: {e}) — re-running pre-pass")
        cache_ms = round((time.perf_counter() - t0) * 1000, 1)
        self._log(f"cache: {decision} ({'; '.join(reasons)}) in {cache_ms} ms")
        tr.step("cache", "On-disk cache check (the real decision)", A.AraRooPatTokenizer._corpus_prepass, {
            "cache_corpus_analysis": tok.cache_corpus_analysis,
            "default_cache_file": str(cache_file.relative_to(REPO_ROOT)),
            "default_cache_exists": cache_file.exists(),
            "rule": "key must match (a migratable older format keeps every entry the newer rule cannot change); "
                    "chunks the cache holds are reused, chunks it lacks go to CAMeL (partial), the union is written back",
            "decision": decision, "reasons": reasons, "policy": req["cache_policy"],
            "write_cache": req["write_cache"],
            "key_expected": list(tok._cache_key()), "key_found": list(key_found) if key_found else None,
            "cached_entries": cached_n, "coverage": coverage,
            "file_bytes": cache_file.stat().st_size if cache_file.exists() else None,
            "file_mtime": cache_file.stat().st_mtime if cache_file.exists() else None,
            "load_ms": cache_ms,
            "_scale": self._scale(None, None, f"Decision: {decision}."),
        }, t0, notes=[
            "The pickle stores post-processed CorpusEntry records, so it is only valid for the analysis "
            "logic + preposition inventory + peeler flags that produced it (the key).",
            "The explorer exposes read policy and write flag separately; the real train() couples both "
            "in cache_corpus_analysis.",
        ])

        # ---- pre-pass part 3: CAMeL (fresh) -----------------------------
        tap = _WireTap(bridge)
        ipc_batches: List[Dict[str, Any]] = []
        batch_ms: List[float] = []
        prepass_ms = None
        cand_hist: Counter = Counter()
        n_batches = (len(to_analyze) + _BATCH_SIZE - 1) // _BATCH_SIZE
        if entries is None:
            self._stage("prepass", 0, len(to_analyze),
                        "CAMeL pre-pass, batches of 256 unique chunks" + (" (only the chunks the cache lacks)" if decision == "partial" else ""))
            sampled_batches = sorted({0, n_batches // 2, max(0, n_batches - 1)})
            t_pp = time.perf_counter()
            new_entries: List[CorpusEntry] = []
            with tap:
                for bi, start in enumerate(range(0, len(to_analyze), _BATCH_SIZE)):
                    batch = to_analyze[start:start + _BATCH_SIZE]
                    tb = time.perf_counter()
                    analyses_b = backend.analyze_many(batch, batch_size=_BATCH_SIZE)
                    ms = round((time.perf_counter() - tb) * 1000, 2)
                    lines = tap.take()
                    for w, a in zip(batch, analyses_b):
                        new_entries.append(CorpusEntry.from_analysis(w, a))
                    if len(batch_ms) < BATCH_MS_CAP:
                        batch_ms.append(ms)
                    if bi in sampled_batches:
                        req_line = next((l["line"] for l in lines if l["dir"] == "→ server"), "")
                        resp_line = next((l["line"] for l in lines if l["dir"] == "← server"), "")
                        cpw: List[int] = []
                        try:
                            resp = json.loads(resp_line)
                            cpw = [len(c) for c in (resp.get("result") or [])]
                        except (json.JSONDecodeError, AttributeError, TypeError):
                            pass
                        cand_hist.update(cpw)
                        ipc_batches.append({
                            "index": bi, "size": len(batch), "words": batch, "duration_ms": ms,
                            "request": req_line, "response": resp_line, "candidates_per_word": cpw,
                            "extra_lines": max(0, len(lines) - 2),
                            "source": "live pre-pass batch",
                        })
                    if bi % 4 == 0:
                        self._tick(min(len(to_analyze), start + len(batch)), len(to_analyze))
            prepass_ms = round((time.perf_counter() - t_pp) * 1000, 1)
            merged = {**cached_by_word, **{e.word: e for e in new_entries}}
            entries = [merged[w] for w in unique_words]   # corpus order, exactly what a fresh run yields
            self._log(f"pre-pass: {sum(1 for e in new_entries if e.analyzed):,} / {len(new_entries):,} analyzed "
                      f"in {prepass_ms/1000:.0f} s" + (f" ({len(cached_by_word):,} reused from the cache)" if decision == "partial" else "")
                      + f"; peeler {backend.peel_stats}")
            if req["write_cache"]:
                self._stage("write_cache", detail=str(cache_file.relative_to(REPO_ROOT)))
                cache_dir.mkdir(parents=True, exist_ok=True)
                with cache_file.open("wb") as f:
                    pickle.dump({"key": tok._cache_key(), "entries": list(merged.values())}, f)
                with (cache_dir / "corpus_analysis.json").open("w", encoding="utf-8") as f:
                    json.dump([e.to_dict() for e in entries[:10000]], f, ensure_ascii=False, indent=2)
                self._log(f"cache written: {cache_file}")
        cache_usable = decision == "hit" or (entries is not None and req["write_cache"]
                                             and decision in ("miss", "partial", "bypassed"))

        # ---- sample replay through the bridge (both hit and fresh) -----
        self._stage("replay", detail="re-analyzing a seeded sample of chunks live for the gate cards")
        by_word = {e.word: e for e in entries}
        misses = [e.word for e in entries if (not e.analyzed) or getattr(e, "peeled", False)]
        k = req["sample_size"]
        sample = list(dict.fromkeys(
            _pick(rng, unique_words, k) + _pick(rng, misses, max(5, k // 2))))
        t0 = time.perf_counter()
        with tap:
            tb = time.perf_counter()
            results = bridge.analyze(sample)
            replay_ms = round((time.perf_counter() - tb) * 1000, 2)
            lines = tap.take()
        raw_by_word = dict(zip(sample, results))
        ipc_batches.append({
            "index": len(ipc_batches), "size": len(sample), "words": sample, "duration_ms": replay_ms,
            "request": next((l["line"] for l in lines if l["dir"] == "→ server"), ""),
            "response": next((l["line"] for l in lines if l["dir"] == "← server"), ""),
            "candidates_per_word": [len(raw_by_word[w]) for w in sample],
            "source": "sample replay (live, for the cards below)",
        })
        tr.step("ipc", "Batched NDJSON round-trip to the CAMeL subprocess", bridge.analyze, {
            "batch_size": _BATCH_SIZE, "batches": ipc_batches,
            "server_module": "src/arabic_eval/tools/araroopat_camel_server.py",
            "server_pipeline": [
                "MLEDisambiguator.disambiguate(words) — analyzer + MLE ranking, one call per batch",
                "for each word: keep every scored analysis, top-scored first",
                "_trim(analysis) — keep only the 12 fields the client consumes (incl. asp for the peeler)",
                "json.dumps(ensure_ascii=False) → one line on stdout",
            ],
            "total_batches": n_batches if prepass_ms is not None else 0,
            "total_unique": len(unique_words), "prepass_ms": prepass_ms, "batch_ms": batch_ms,
            "candidates_histogram": dict(sorted(cand_hist.items())),
            "source": ("live pre-pass (only the chunks the cache lacked)" if decision == "partial" else "live pre-pass") if prepass_ms is not None else "cache hit — no pre-pass IPC; the batch below is a sample replay",
            "_scale": self._scale(len(ipc_batches), n_batches,
                                  (f"{n_batches:,} batches went over the pipe in {prepass_ms/1000:.0f} s; "
                                   f"3 are shown with their literal lines, plus the sample replay." if prepass_ms is not None else
                                   f"Cache hit: the real run sent nothing to CAMeL. The one batch shown is "
                                   f"{len(sample)} sampled chunks re-analyzed live so the gate cards are real.")),
        }, t0, notes=[
            "One request per 256 unique words. Each response sublist is the MLE-ranked candidate list "
            "for one word; an empty sublist means CAMeL has no analysis at all.",
        ])

        t0 = time.perf_counter()
        analyses, val_rows = trace_validate_words(sample, raw_by_word, backend)
        for r in val_rows:
            e = by_word.get(r["word"])
            r["prepass_entry"] = _entry_summary(e) if e else None
            # Native validation only; a peeled entry legitimately disagrees here and is settled in the peel card.
            r["agrees_with_prepass"] = (_agrees(e, analyses[r["word"]]) if e and not getattr(e, "peeled", False) else None)
        self._stage("records", detail="one compact record per unique chunk, every CorpusEntry field (the cards page through them)")
        t_rec = time.perf_counter()
        self.words = WordCategoryIndex(entries, word_counts)
        self._log(f"records: {len(self.words.rows):,} in {time.perf_counter() - t_rec:.1f} s; "
                  + ", ".join(f"{k} {v:,}" for k, v in self.words.path_totals().items()))
        n_analyzed = self.words.analyzed_count
        path_totals = self.words.path_totals()
        tr.step("validate", "Client-side validation: _first_valid → _dict_to_analysis", _dict_to_analysis, {
            "words": val_rows, "analyzed": n_analyzed, "total": len(entries),
            "path_totals": path_totals,
            "replay_disagreements": sum(1 for r in val_rows if r["agrees_with_prepass"] is False),
            "_scale": self._scale(len(val_rows), len(entries),
                                  f"Totals are corpus-wide ({n_analyzed:,} analyzed of {len(entries):,} unique chunks); "
                                  f"the {len(val_rows)} word cards are the live sample replay, each compared "
                                  f"with its pre-pass entry ('agrees' badge). Click a counter to page through every record on that path."),
        }, t0, notes=[
            "Candidates are walked in MLE order; the first one that survives every gate wins.",
            "'#' is kept in the root token: a radical whose surface letter lives in the pattern.",
        ])

        t0 = time.perf_counter()
        peel_rows = trace_peel_words(sample, analyses, backend)
        for r in peel_rows:
            e = by_word.get(r["word"])
            r["prepass_entry"] = _entry_summary(e) if e else None
            r["agrees_with_prepass"] = _agrees(e, analyses[r["word"]]) if e else None
        tr.step("peel", "Clitic peeler: closed-list slicing of the words CAMeL rejected", B.peel_candidates, {
            "enabled": backend.enable_peeler, "words": peel_rows,
            "rescued": path_totals["peeled"], "total": path_totals["peeled"] + path_totals["LIT"],
            "_scale": self._scale(len(peel_rows), path_totals["peeled"] + path_totals["LIT"],
                                  f"Corpus-wide: {path_totals['peeled']:,} chunks rescued by the peeler, "
                                  f"{path_totals['LIT']:,} exhausted → character path. Cards = the sample's native misses, replayed live. "
                                  f"Click a counter to page through every rescued / exhausted record."),
        }, t0, notes=[
            "The peeler may only remove what CAMeL cannot represent (أ, a second pronoun, كمو/همو).",
            "Least-peeled candidate first; residual readings are walked in MLE order (top=32).",
        ])

        t0 = time.perf_counter()
        sample_entries = [by_word[w] for w in sample if w in by_word]
        tr.step("entries", "CorpusEntry records (the pre-pass output)", CorpusEntry.from_analysis, {
            "entries": [e.to_dict() for e in sample_entries],
            "totals": {"entries": len(entries), **path_totals},
            "browsable": True,
            "_scale": self._scale(len(entries), len(entries),
                                  f"All {len(entries):,} records (from the {'cache' if decision == 'hit' else 'live pre-pass'}), "
                                  f"paged below by occurrences; filter by pre-pass path or substring."),
        }, t0, notes=[
            "proclitics = (prc3, prc2, prc1, prc0) minus empties, enclitics = (ة, enc0) minus empties in emission order.",
        ])

        # ---- step 2: frequency tables (whole corpus) -------------------
        self._stage("freq", detail="frequency tables over every entry")
        t0 = time.perf_counter()
        root_freq: Counter = Counter()
        pat_freq: Counter = Counter()
        proclitic_freq: Counter = Counter()
        enclitic_freq: Counter = Counter()
        particle_freq: Counter = Counter()
        func_freq: Counter = Counter()
        contrib: Dict[str, Dict[str, List[str]]] = {"root": {}, "pat": {}, "prc": {}, "enc": {}, "prep": {}, "func": {}}

        def add_contrib(kind: str, key: str, word: str) -> None:
            lst = contrib[kind].setdefault(key, [])
            if len(lst) < 6:
                lst.append(word)

        for e in entries:
            if not e.analyzed:
                continue
            if e.particle:
                kind = "func" if getattr(e, "particle_kind", "prep") == "func" else "prep"
                (func_freq if kind == "func" else particle_freq)[e.particle] += 1
                add_contrib(kind, e.particle, e.word)
            elif e.root and e.pattern:
                root_freq[e.root] += 1
                add_contrib("root", e.root, e.word)
                pat_freq[e.pattern] += 1
                add_contrib("pat", e.pattern, e.word)
            else:
                continue
            for c in e.proclitics:
                proclitic_freq[c] += 1
                add_contrib("prc", c, e.word)
            for c in e.enclitics:
                enclitic_freq[c] += 1
                add_contrib("enc", c, e.word)

        def ftable(cnt: Counter, kind: str, cap: int = TOP_FREQ) -> List[Dict[str, Any]]:
            rows = []
            for k_, v in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:cap]:
                row = {"key": k_, "freq": v, "words": contrib[kind].get(k_, [])}
                if kind == "pat":
                    row["wazn"] = wazn_of(k_)
                rows.append(row)
            return rows

        sizes = {"root_freq": len(root_freq), "pat_freq": len(pat_freq), "proclitic_freq": len(proclitic_freq),
                 "enclitic_freq": len(enclitic_freq), "preposition_freq": len(particle_freq), "func_freq": len(func_freq)}
        tr.step("freq", "Frequency tables (one count per unique word, not per occurrence)", A.AraRooPatTokenizer.train, {
            "root_freq": ftable(root_freq, "root"), "pat_freq": ftable(pat_freq, "pat"),
            "proclitic_freq": ftable(proclitic_freq, "prc", 10_000), "enclitic_freq": ftable(enclitic_freq, "enc", 10_000),
            "preposition_freq": ftable(particle_freq, "prep", 10_000), "func_freq": ftable(func_freq, "func", 10_000), "sizes": sizes,
            "_scale": self._scale(TOP_FREQ, sizes,
                                  f"Top {TOP_FREQ} roots (of {len(root_freq):,}) and patterns (of {len(pat_freq):,}) "
                                  f"by unique-word frequency, 6 contributing words each; clitic and preposition tables are complete. "
                                  f"The root and pattern tables below page through every candidate and can be searched."),
        }, t0, notes=[
            "Prepositions ([PREP_*]) and function words ([FUNC_*]) contribute their clitics to the clitic tables but no root or pattern.",
            "Unit: entries are unique words, so a root seen in 3 distinct words has freq 3 even if one occurred 50 times.",
        ])

        # ---- step 3: vocab (real) -------------------------------------
        self._stage("vocab")
        t0 = time.perf_counter()
        tok._build_vocab(root_freq, pat_freq, proclitic_freq, enclitic_freq)
        vocab = tok._vocab
        tr.step("vocab", "Assemble the vocab in deterministic ID order", A.AraRooPatTokenizer._build_vocab,
                self._vocab_step_data(tok, root_freq, pat_freq, proclitic_freq, enclitic_freq), t0, notes=[
            "Fixed slots (specials, LIT markers, PREP, FUNC, CHAR, DIGIT, PUNCT) are always present; only clitics/roots/patterns depend on the corpus.",
            "The budget loop uses `break`, not `continue`: the first item below min_freq ends the loop for everything after it.",
        ])

        self._stage("categories", detail="encode-time category of every unique chunk")
        self.words.attach_vocab(tok, root_freq, pat_freq)
        self._log("categories: " + ", ".join(f"{k} {v:,}" for k, v in self.words.counts.items()))
        self.freq = FreqIndex.from_counters(root_freq, pat_freq, proclitic_freq, enclitic_freq, particle_freq, contrib, vocab, func_freq)

        # ---- step 4: reconstruction (real) -----------------------------
        self._stage("reconstruction", detail="_build_reconstruction over every entry (generator for unseen pairs)")
        t0 = time.perf_counter()
        with tap:
            tok._build_reconstruction(entries, word_counts)
            gen_lines = tap.take()
        real_reco = dict(tok._reconstruction)
        written_real: Dict[Tuple[str, str], Counter] = {}
        diac_real: Dict[Tuple[str, str], Counter] = {}
        all_pairs: set = set()
        skipped: Counter = Counter()
        source_counts: Counter = Counter()
        for e in entries:
            if not (e.analyzed and e.root and e.pattern):
                skipped["not analyzed" if not e.analyzed else "missing root/pattern"] += 1
                continue
            rt, pt = f"{PFX_ROOT}{e.root}{SFX}", f"{PFX_PAT}{e.pattern}{SFX}"
            if rt not in vocab or pt not in vocab:
                skipped["root cut by budget" if rt not in vocab else "pattern cut by budget"] += 1
                continue
            all_pairs.add((e.root, e.pattern))
            real = entry_realization(e, tok.use_diacritized_surface)
            if real is None:
                skipped["no usable surface"] += 1
                continue
            source_counts[real[1]] += 1
            (written_real if real[1] == "written" else diac_real).setdefault(
                (e.root, e.pattern), Counter())[real[0]] += word_counts.get(e.word, 1)
        corpus_real = {k: True for k in set(written_real) | set(diac_real)}
        pass1 = []
        for e in sample_entries:
            row: Dict[str, Any] = {"word": e.word}
            if not (e.analyzed and e.root and e.pattern):
                row["skipped"] = "not analyzed" if not e.analyzed else "missing root/pattern"
            else:
                rt, pt = f"{PFX_ROOT}{e.root}{SFX}", f"{PFX_PAT}{e.pattern}{SFX}"
                if rt not in vocab or pt not in vocab:
                    row["skipped"] = f"{rt if rt not in vocab else pt} not in vocab"
                else:
                    surface = e.surface or ""
                    pro = _trace_proclitic_stack(surface, e.proclitics)
                    enc_ops = _trace_enclitic_stack(pro["output"], e.enclitics)
                    s = enc_ops[-1]["after"] if enc_ops else pro["output"]
                    inflected = _strip_clitic_surfaces(surface, e.proclitics, e.enclitics)
                    bare = strip_diacritics(e.word)
                    written_stem = _strip_clitic_surfaces(bare, e.proclitics, e.enclitics)
                    rejoined = join_word(e.proclitics, written_stem, e.enclitics) if written_stem else ""
                    real = entry_realization(e, tok.use_diacritized_surface)
                    row.update({"root": e.root, "pattern": e.pattern, "surface": surface, "camel_stem": e.stem,
                                "proclitic_strip": pro, "enclitic_strip": enc_ops, "inflected": inflected,
                                "written": {"chunk": bare, "stem": written_stem, "rejoined": rejoined,
                                            "reproduces": rejoined == bare},
                                "form": real[0] if real else None, "source": real[1] if real else None,
                                "weight": word_counts.get(e.word, 1),
                                "matches_real": inflected == s and (
                                    real is None or real[0] == (written_stem if real[1] == "written" else
                                                                (inflected if tok.use_diacritized_surface
                                                                 else strip_diacritics(inflected))))})
            pass1.append(row)
        pass2 = []
        sample_pairs = sorted({(e.root, e.pattern) for e in sample_entries
                               if e.analyzed and e.root and e.pattern and (e.root, e.pattern) in corpus_real})
        for (root, pat) in sample_pairs:
            rid, pid = vocab[f"{PFX_ROOT}{root}{SFX}"], vocab[f"{PFX_PAT}{pat}{SFX}"]
            w_cnt, d_cnt = written_real.get((root, pat)), diac_real.get((root, pat))
            value = pick_realization(w_cnt, d_cnt)
            pass2.append({"root": root, "pattern": pat, "root_id": rid, "pat_id": pid,
                          "realizations": [{"form": f, "count": n, "source": "written"} for f, n in (w_cnt or Counter()).most_common(8)]
                                          + [{"form": f, "count": n, "source": "diac"} for f, n in (d_cnt or Counter()).most_common(8)],
                          "chosen": value, "value": value, "tier": 1, "source": "written" if w_cnt else "diac",
                          "matches_real": real_reco.get((rid, pid)) == value})
        unresolved = sorted(p for p in all_pairs if p not in corpus_real)
        pass3 = []
        for (root, pat) in unresolved[:PASS3_CAP]:
            rid, pid = vocab[f"{PFX_ROOT}{root}{SFX}"], vocab[f"{PFX_PAT}{pat}{SFX}"]
            gen = backend._generate_cache.get((root, pat))
            naive = naive_pattern_fill(root, pat)
            s = gen or naive
            value = (s if tok.use_diacritized_surface else strip_diacritics(s)) if s else None
            pass3.append({"root": root, "pattern": pat, "root_id": rid, "pat_id": pid,
                          "generator_result": gen, "naive_fill": naive,
                          "tier": 2 if gen else (3 if naive else None), "value": value,
                          "matches_real": real_reco.get((rid, pid)) == value})
        tier2 = sum(1 for p in unresolved if backend._generate_cache.get(p))
        reco_items = sorted(real_reco.items())
        reco_sample = _pick(rng, reco_items, RECO_TABLE_SAMPLE)
        tr.step("reconstruction", "Reconstruction lookup: (root_id, pat_id) → inflected stem",
                A.AraRooPatTokenizer._build_reconstruction, {
                    "pass1": pass1, "pass2": pass2, "pass3": pass3,
                    "table": [{"root_id": k_[0], "pat_id": k_[1], "value": v,
                               "root": _inner(tok._reverse_vocab[k_[0]]), "pattern": _inner(tok._reverse_vocab[k_[1]])}
                              for k_, v in sorted(reco_sample)],
                    "size": len(real_reco),
                    "totals": {"pairs": len(all_pairs), "with_corpus_surface": len(corpus_real),
                               "pairs_from_written": len(written_real),
                               "pairs_diac_only": len(set(diac_real) - set(written_real)),
                               "entries_by_source": dict(source_counts),
                               "unresolved": len(unresolved), "tier2_generator": tier2,
                               "tier3_naive": len(unresolved) - tier2, "skipped": dict(skipped),
                               "generator_wire_lines": len(gen_lines)},
                    "_scale": self._scale(len(pass1), len(entries),
                                          f"{len(real_reco):,} table entries from {len(all_pairs):,} distinct pairs; "
                                          f"{len(unresolved):,} pairs had no corpus surface (generator {tier2:,}, naive {len(unresolved)-tier2:,}). "
                                          f"Pass 1/2 rows = the sample; pass 3 = first {min(PASS3_CAP, len(unresolved))}; "
                                          f"table = {len(reco_sample)} random entries (search finds any)."),
                }, t0, wire=gen_lines[:WIRE_CAP], notes=[
            "Pass 1 strips the clitic surfaces from the chunk AS WRITTEN (diacritics removed) and keeps that form when "
            "the decoder's own joins give the chunk back (source 'written'); otherwise from CAMeL's diac, NOT from its "
            "stem field (source 'diac'). With use_diacritized_surface the diac form is the only source.",
            "Pass 2 keeps the most frequent WRITTEN form per pair, weighted by corpus occurrences; diac forms only for a "
            "pair with no written form — so a pair decodes to the spelling the writers used (2026-09-22).",
            "Pass 3 only fires for pairs with no usable surface: CAMeL generator (tier 2), then naive slot substitution (tier 3).",
        ])

        # ---- step 5: metadata (real) -----------------------------------
        self._stage("metadata")
        t0 = time.perf_counter()
        tok._build_metadata(root_freq, pat_freq, proclitic_freq, enclitic_freq, entries, particle_freq, func_freq)
        tr.step("metadata", "Provenance metadata (vocab_metadata.json)", A.AraRooPatTokenizer._build_metadata,
                self._metadata_step_data(tok), t0,
                notes=["Answers 'where did this token come from?' without re-running the pre-pass."])

        # ---- step 6: probe round trip ----------------------------------
        self._stage("probe")
        probe_texts = [ln for ln in req["probe_text"].splitlines() if ln.strip()]
        with tap:
            trace_probe_roundtrip(tr, tok, backend, tap, probe_texts, None, None, real_reco)
        self._annotate_probe(tr, live_ipc=(decision == "hit"),
                             why="the entries came from the pickle, so the job's analyzer cache is empty")

        # ---- verify -----------------------------------------------------
        t0 = time.perf_counter()
        if req["verify"] and cache_usable:
            self._stage("verify", detail="un-instrumented train() from the cache (re-chunks the corpus)")
            fresh = self._make_tokenizer(req["params"], cache_flag=True)
            fresh.train(texts, cache_path=str(cache_dir))
            tr.step("verify", "Cross-check against an un-instrumented train()", A.AraRooPatTokenizer.train, {
                "vocab_equal": fresh._vocab == tok._vocab, "reconstruction_equal": fresh._reconstruction == tok._reconstruction,
                "fresh_vocab_size": fresh.vocab_size, "traced_vocab_size": tok.vocab_size,
                "_scale": self._scale(None, None, "The real train() ran again on the same corpus with the cache it can now hit."),
            }, t0, notes=["Proves the trace above is the real algorithm, not a paraphrase of it."])
        else:
            why = ("verify unticked" if not req["verify"] else
                   "no reusable cache (cache_policy=ignore and write_cache=false) — a re-run would repeat the whole CAMeL pre-pass")
            tr.step("verify", "Cross-check against an un-instrumented train()", A.AraRooPatTokenizer.train, {
                "skipped": True, "why": why, "traced_vocab_size": tok.vocab_size,
                "_scale": self._scale(None, None, f"Skipped: {why}."),
            }, t0, notes=["Proves the trace above is the real algorithm, not a paraphrase of it."])

        self.trace = {
            "mode": "train", "input": req["probe_text"], "texts": probe_texts, "params": effective,
            "corpus": corpus_info, "request": {k: v for k, v in req.items() if k != "probe_text"},
            "total_ms": round((time.perf_counter() - tr.t0) * 1000, 1), "steps": tr.steps,
            "all_ok": self._all_ok(tr.steps),
            "vocab_size": tok.vocab_size, "reconstruction_size": len(tok._reconstruction),
        }
        self.tokenizer = tok

    # ------------------------------------------------------------------
    # mode = saved
    # ------------------------------------------------------------------
    def _trace_saved(self) -> None:
        req = self.request
        rng = random.Random(req["seed"])
        tr = _Trace()
        path = TOKENIZERS_DIR / req["saved_dir"]
        self._stage("load", detail=str(path.relative_to(REPO_ROOT)))
        t0 = time.perf_counter()
        tok = self._make_tokenizer({}, cache_flag=False)
        tok.load(path)
        # load() drops the backend; rebuild it on the job's bridge with the loaded inventory.
        tok._backend = MorphAnalyzer(generator_timeout_ms=tok.generator_timeout_ms, bridge=self.bridge,
                                     particles=frozenset(tok.prepositions), enable_peeler=tok.clitic_peeler,
                                     peel_bare_alef=tok.peel_bare_alef, func_words=frozenset(tok.func_words))
        backend: MorphAnalyzer = tok._backend
        bridge = self.bridge
        assert bridge is not None
        bridge._ensure_started()
        bridge.analyze(["إحماء"])
        load_ms = round((time.perf_counter() - t0) * 1000, 1)
        self._log(f"loaded {path.name}: vocab {tok.vocab_size}, reconstruction {len(tok._reconstruction)} in {load_ms} ms")
        files = {p.name: p.stat().st_size for p in sorted(path.iterdir()) if p.is_file()}
        effective = self._effective(tok)
        meta = tok._metadata or {}
        tr.step("config", "Tokenizer instance & parameters", AraRooPatTokenizer.load, {
            "effective": effective, "real_defaults": REAL_DEFAULTS,
            "overridden": [k for k, v in effective.items() if REAL_DEFAULTS.get(k) != v],
            "num_texts": 0, "camel_python": str(bridge._camel_python or _resolve_camel_python()),
            "mode": "saved", "saved_dir": str(path.relative_to(REPO_ROOT)), "files": files, "load_ms": load_ms,
            "metadata_config": meta.get("config", {}),
            "_scale": self._scale(None, None,
                                  f"Loaded from {path.relative_to(REPO_ROOT)} via load(): config.json → vocab.json → "
                                  f"reconstruction.pkl → vocab_metadata.json. No corpus in memory."),
        }, t0, notes=["Parameters come from config.json of the saved directory; the corpus that produced it is not reloaded."])

        unavailable_note = ("Not reconstructable from a saved tokenizer: this step consumes the corpus / the "
                            "CorpusEntry records, which save() does not persist. Run 'Train on the corpus' — "
                            "with a covering pre-pass cache it is minutes, not hours.")
        for sid, title, fn in (
            ("nfkc", "Unicode NFKC normalization", A.AraRooPatTokenizer._corpus_prepass),
            ("split", "Whitespace split", "builtins.str.split"),
            ("chunks", "Character classes → Arabic alpha chunks", _extract_alpha_chunks),
            ("dedup", "Deduplicate → word_counts", "collections.Counter"),
        ):
            tr.step(sid, title, fn, {"unavailable": True, "reason": unavailable_note}, time.perf_counter())
        cache_file = DEFAULT_CACHE_DIR / "corpus_analysis.pkl"
        tr.step("cache", "On-disk cache check", A.AraRooPatTokenizer._corpus_prepass, {
            "cache_corpus_analysis": False, "default_cache_file": str(cache_file.relative_to(REPO_ROOT)),
            "default_cache_exists": cache_file.exists(), "decision": "not consulted",
            "reasons": ["a saved tokenizer is loaded from its own artifacts; the pre-pass cache is only read by train()"],
            "policy": None, "write_cache": False, "key_expected": list(tok._cache_key()), "key_found": None,
            "cached_entries": None, "coverage": None,
            "file_bytes": cache_file.stat().st_size if cache_file.exists() else None,
            "file_mtime": cache_file.stat().st_mtime if cache_file.exists() else None, "load_ms": 0,
            "rule": "if cached_words ⊇ set(unique_words): reuse (filtered to current words) else re-run",
            "_scale": self._scale(None, None, "Not consulted in saved mode."),
        }, time.perf_counter())
        for sid, title, fn in (
            ("ipc", "Batched NDJSON round-trip to the CAMeL subprocess", bridge.analyze),
            ("validate", "Client-side validation: _first_valid → _dict_to_analysis", _dict_to_analysis),
            ("peel", "Clitic peeler: closed-list slicing of the words CAMeL rejected", B.peel_candidates),
            ("entries", "CorpusEntry records (the pre-pass output)", CorpusEntry.from_analysis),
        ):
            tr.step(sid, title, fn, {"unavailable": True, "reason": unavailable_note}, time.perf_counter())

        # freq from metadata (kept tokens only)
        t0 = time.perf_counter()
        roots_meta = meta.get("roots", {})
        pats_meta = meta.get("patterns", {})

        def mrows(d: Dict[str, Any], ex_key: str, is_pat: bool = False) -> List[Dict[str, Any]]:
            rows = []
            for k_, v in sorted(d.items(), key=lambda kv: (-kv[1].get("freq", 0), kv[0]))[:TOP_FREQ]:
                ex = v.get(ex_key, [])
                words = [e_[1] if isinstance(e_, list) else e_ for e_ in ex][:6]
                row = {"key": k_, "freq": v.get("freq", 0), "words": words}
                if is_pat:
                    row["wazn"] = wazn_of(k_)
                rows.append(row)
            return rows

        def crows(d: Dict[str, int]) -> List[Dict[str, Any]]:
            return [{"key": k_, "freq": v, "words": []} for k_, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))]

        preps_meta = meta.get("prepositions", {})
        func_meta = meta.get("func_words", {})
        tr.step("freq", "Frequency tables (from vocab_metadata.json — kept tokens only)", A.AraRooPatTokenizer.train, {
            "root_freq": mrows(roots_meta, "example_words"), "pat_freq": mrows(pats_meta, "examples", True),
            "proclitic_freq": crows(meta.get("proclitic_freq", {})), "enclitic_freq": crows(meta.get("enclitic_freq", {})),
            "preposition_freq": [{"key": k_, "freq": v.get("freq", 0), "words": []} for k_, v in preps_meta.items()],
            "func_freq": [{"key": k_, "freq": v.get("freq", 0), "words": []} for k_, v in func_meta.items()],
            "sizes": {"root_freq": len(roots_meta), "pat_freq": len(pats_meta)},
            "from_metadata": True,
            "_scale": self._scale(TOP_FREQ, {"roots": len(roots_meta), "patterns": len(pats_meta)},
                                  "Frequencies of the tokens that made it into the vocab, read back from vocab_metadata.json. "
                                  "Candidates cut by the budget are not in the artifacts."),
        }, t0)
        self.freq = FreqIndex.from_metadata(meta, tok._vocab)

        t0 = time.perf_counter()
        root_freq = Counter({r: v.get("freq", 0) for r, v in roots_meta.items()})
        pat_freq = Counter({p: v.get("freq", 0) for p, v in pats_meta.items()})
        proclitic_freq = Counter(meta.get("proclitic_freq", {}))
        enclitic_freq = Counter(meta.get("enclitic_freq", {}))
        vdata = self._vocab_step_data(tok, root_freq, pat_freq, proclitic_freq, enclitic_freq, from_metadata=True)
        tr.step("vocab", "Vocab as loaded from vocab.json (ID order)", A.AraRooPatTokenizer.load, vdata, t0, notes=[
            "Budget tables show the kept tokens with their recorded frequency; the cut candidates were never saved.",
        ])

        t0 = time.perf_counter()
        real_reco = dict(tok._reconstruction)
        reco_sample = _pick(rng, sorted(real_reco.items()), RECO_TABLE_SAMPLE)
        tr.step("reconstruction", "Reconstruction lookup as loaded (reconstruction.pkl)", A.AraRooPatTokenizer.load, {
            "pass1": [], "pass2": [], "pass3": [],
            "table": [{"root_id": k_[0], "pat_id": k_[1], "value": v,
                       "root": _inner(tok._reverse_vocab.get(k_[0], "?")), "pattern": _inner(tok._reverse_vocab.get(k_[1], "?"))}
                      for k_, v in sorted(reco_sample)],
            "size": len(real_reco), "from_artifacts": True,
            "_scale": self._scale(len(reco_sample), len(real_reco),
                                  f"{len(real_reco):,} entries loaded; {len(reco_sample)} random ones shown (search finds any). "
                                  f"The three passes need the corpus entries and are not in the artifacts."),
        }, t0)

        t0 = time.perf_counter()
        tr.step("metadata", "Provenance metadata (vocab_metadata.json)", A.AraRooPatTokenizer.load,
                self._metadata_step_data(tok), t0)

        self._stage("probe")
        probe_texts = [ln for ln in req["probe_text"].splitlines() if ln.strip()]
        tap = _WireTap(bridge)
        with tap:
            trace_probe_roundtrip(tr, tok, backend, tap, probe_texts, None, None, real_reco)
        self._annotate_probe(tr, live_ipc=True, why="a loaded tokenizer starts with an empty analyzer cache")

        t0 = time.perf_counter()
        again = AraRooPatTokenizer()
        again.load(path)
        tr.step("verify", "Cross-check: load() the directory again", A.AraRooPatTokenizer.load, {
            "vocab_equal": again._vocab == tok._vocab, "reconstruction_equal": again._reconstruction == tok._reconstruction,
            "fresh_vocab_size": again.vocab_size, "traced_vocab_size": tok.vocab_size, "from_artifacts": True,
            "_scale": self._scale(None, None, "A second load() of the same directory must agree with the first."),
        }, t0)

        self.trace = {
            "mode": "saved", "input": req["probe_text"], "texts": probe_texts, "params": effective,
            "saved_dir": str(path.relative_to(REPO_ROOT)), "request": {k: v for k, v in req.items() if k != "probe_text"},
            "total_ms": round((time.perf_counter() - tr.t0) * 1000, 1), "steps": tr.steps,
            "all_ok": self._all_ok(tr.steps),
            "vocab_size": tok.vocab_size, "reconstruction_size": len(tok._reconstruction),
        }
        self.tokenizer = tok

    # ------------------------------------------------------------------
    # shared step builders
    # ------------------------------------------------------------------
    @staticmethod
    def _vocab_step_data(tok: AraRooPatTokenizer, root_freq: Counter, pat_freq: Counter,
                         proclitic_freq: Counter, enclitic_freq: Counter,
                         from_metadata: bool = False) -> Dict[str, Any]:
        vocab = tok._vocab
        ranges: List[Dict[str, Any]] = []
        for t_, i in sorted(vocab.items(), key=lambda kv: kv[1]):
            fam = _split_key(t_)
            if ranges and ranges[-1]["family"] == fam:
                ranges[-1]["end"] = i
                ranges[-1]["count"] += 1
            else:
                ranges.append({"family": fam, "start": i, "end": i, "count": 1})

        def budget(items: Counter, max_k: int, min_f: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
            rows = []
            kept = 0
            stopped = False
            counts: Counter = Counter()
            for k_, f in sorted(items.items(), key=lambda kv: (-kv[1], kv[0])):
                if stopped:
                    rows.append({"key": k_, "freq": f, "kept": False, "reason": "after break"})
                    counts["after_break"] += 1
                    continue
                if kept >= max_k:
                    stopped = True
                    rows.append({"key": k_, "freq": f, "kept": False, "reason": f"budget max={max_k} reached"})
                    counts["cut_budget"] += 1
                    continue
                if f < min_f:
                    stopped = True
                    rows.append({"key": k_, "freq": f, "kept": False, "reason": f"freq {f} < min_freq {min_f}"})
                    counts["cut_minfreq"] += 1
                    continue
                rows.append({"key": k_, "freq": f, "kept": True, "reason": ""})
                kept += 1
            counts["kept"] = kept
            counts["candidates"] = len(rows)
            return rows, dict(counts)

        def truncate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if len(rows) <= BUDGET_HEAD + 2 * BUDGET_WINDOW:
                return rows
            cut = next((i for i, r in enumerate(rows) if not r["kept"]), len(rows))
            head = rows[:BUDGET_HEAD]
            window = rows[max(BUDGET_HEAD, cut - BUDGET_WINDOW): cut + BUDGET_WINDOW]
            return head + window

        root_rows, root_counts = budget(root_freq, tok.max_roots, tok.min_root_freq)
        pat_rows, pat_counts = budget(pat_freq, tok.max_patterns, tok.min_pattern_freq)
        pats_meta = (tok._metadata or {}).get("patterns", {})
        vocab_list = []
        for t_, i in sorted(vocab.items(), key=lambda kv: kv[1]):
            fam = _split_key(t_)
            row = {"id": i, "token": t_, "family": fam, "inner": _inner(t_)}
            if fam == "pat":
                d = describe_pattern(row["inner"], (pats_meta.get(row["inner"]) or {}).get("examples"))
                row.update({"wazn": d["wazn"], "gloss": d["gloss"], "gloss_tier": d["gloss_tier"],
                            "filled": (d["filled"] or {}).get("stem")})
            vocab_list.append(row)
        clitic_order = lambda cnt: [k for k, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])) if k]  # noqa: E731
        return {
            "vocab_size": len(vocab), "ranges": ranges,
            "layout": [
                {"family": "special", "source": "SPECIAL_TOKENS_ORDERED", "items": SPECIAL_TOKENS_ORDERED},
                {"family": "lit", "source": "TOK_LIT_BEGIN / TOK_LIT_END", "items": [TOK_LIT_BEGIN, TOK_LIT_END]},
                {"family": "prop", "source": "TOK_PROP_BEGIN / TOK_PROP_END", "items": [TOK_PROP_BEGIN, TOK_PROP_END]},
                {"family": "cliticp", "source": "proclitic_freq sorted by (-freq, surface)", "items": clitic_order(proclitic_freq)},
                {"family": "clitice", "source": "[CLITICE_ة] fixed first, then enclitic_freq sorted by (-freq, surface)",
                 "items": [B.TAA_MARBUTA] + [c for c in clitic_order(enclitic_freq) if c != B.TAA_MARBUTA]},
                {"family": "prep", "source": "tok.prepositions (fixed order, corpus-independent)", "items": list(tok.prepositions)},
                {"family": "func", "source": "tok.func_words (fixed order, corpus-independent)", "items": list(tok.func_words)},
                {"family": "char", "source": "CHAR_INVENTORY (sorted letters + sorted diacritics)", "items": CHAR_INVENTORY},
                {"family": "digit", "source": "DIGIT_INVENTORY", "items": DIGIT_INVENTORY},
                {"family": "punct", "source": "PUNCT_INVENTORY", "items": PUNCT_INVENTORY},
                {"family": "root", "source": f"root_freq top-{tok.max_roots}, freq ≥ {tok.min_root_freq}", "items": None},
                {"family": "pat", "source": f"pat_freq top-{tok.max_patterns}, freq ≥ {tok.min_pattern_freq}", "items": None},
            ],
            "root_budget": truncate(root_rows), "pattern_budget": truncate(pat_rows),
            "budget_counts": {"root": root_counts, "pattern": pat_counts},
            "vocab": vocab_list, "special_token_map": tok._special_token_map,
            "from_metadata": from_metadata,
            "_scale": {
                "shown": {"root_budget": len(truncate(root_rows)), "pattern_budget": len(truncate(pat_rows))},
                "total": {"root_candidates": len(root_rows), "pattern_candidates": len(pat_rows)},
                "note": (f"Full vocab ({len(vocab):,} tokens) is shipped for the palette and search. Budget tables show the "
                         f"top {BUDGET_HEAD} candidates plus ±{BUDGET_WINDOW} rows around the cut: roots {root_counts.get('kept',0):,} kept "
                         f"of {len(root_rows):,} candidates, patterns {pat_counts.get('kept',0):,} of {len(pat_rows):,}."
                         + (" Candidate lists come from vocab_metadata.json (kept tokens only)." if from_metadata else "")),
            },
        }

    @staticmethod
    def _metadata_step_data(tok: AraRooPatTokenizer) -> Dict[str, Any]:
        meta = tok._metadata or {}
        roots = dict(sorted(meta.get("roots", {}).items(), key=lambda kv: (-kv[1].get("freq", 0), kv[0]))[:META_TOP])
        pats = dict(sorted(meta.get("patterns", {}).items(), key=lambda kv: (-kv[1].get("freq", 0), kv[0]))[:META_TOP])
        peeled = meta.get("peeled") or {}
        return {
            "roots": roots, "patterns": pats,
            "proclitic_freq": meta.get("proclitic_freq", {}), "enclitic_freq": meta.get("enclitic_freq", {}),
            "config": meta.get("config", {}), "prepositions": meta.get("prepositions", {}),
            "func_words": meta.get("func_words", {}),
            "peeled": {"count": peeled.get("count"), "examples": (peeled.get("examples") or [])[:20]},
            "totals": {"roots": len(meta.get("roots", {})), "patterns": len(meta.get("patterns", {}))},
            "_scale": {"shown": META_TOP, "total": {"roots": len(meta.get("roots", {})), "patterns": len(meta.get("patterns", {}))},
                       "note": f"Top {META_TOP} roots and patterns by frequency; search below reaches every entry."},
        }

    def _annotate_probe(self, tr: _Trace, live_ipc: bool, why: str) -> None:
        """Scale banners for the two probe steps; fix the inherited 'zero IPC' note when it does not hold."""
        enc, dec = tr.steps[-2], tr.steps[-1]
        base = "Run on the probe text, not the corpus: the same encode → decode proof as tab 01."
        if live_ipc:
            n = enc["data"].get("ipc_calls", 0)
            enc["notes"][0] = (f"IPC calls > 0 are expected here (unlike tab 01): {why}, so every alpha chunk of "
                               f"the probe is analyzed live by CAMeL at encode time — {n} round-trip(s) on the pipe below.")
            enc["data"]["_scale"] = self._scale(None, None, base + f" {n} live CAMeL calls: {why}.")
        else:
            enc["data"]["_scale"] = self._scale(None, None, base + " The probe's chunks were analyzed in the fresh pre-pass, so encode() needs no IPC.")
        dec["data"]["_scale"] = self._scale(None, None, base + " Differences in the aligned diff are decode()'s own: CAMeL's canonical hamza spelling, punctuation and digits emitted as separate units, tanween dropped.")

    @staticmethod
    def _all_ok(steps: List[Dict[str, Any]]) -> bool:
        ok = True
        for st in steps:
            d = st["data"]
            if d.get("vocab_equal") is False or d.get("reconstruction_equal") is False:
                ok = False
            for key in ("words", "pass1", "pass2", "pass3"):
                for r in d.get(key) or []:
                    if isinstance(r, dict) and r.get("matches_real") is False:
                        ok = False
        return ok


# ---------------------------------------------------------------------------
# Word categories: where does a chunk end up at encode time, and why
# ---------------------------------------------------------------------------

# Order = display order. The LIT reasons mirror the all-or-nothing rules of
# ``AraRooPatTokenizer._emit_alpha`` (root, pattern AND every clitic token must
# exist, else the whole chunk goes to the character path).
CATEGORIES: List[Tuple[str, str]] = [
    ("root_pat", "ROOT+PAT"),
    ("root_pat_peeled", "ROOT+PAT (peeled)"),
    ("prep", "PREP"),
    ("func", "FUNC"),
    ("clitic", "CLITIC"),
    ("prop", "PROP"),
    ("lit_no_analysis", "LIT: no analysis"),
    ("lit_root_cut", "LIT: root cut"),
    ("lit_pattern_cut", "LIT: pattern cut"),
    ("lit_clitic_missing", "LIT: clitic / particle token missing"),
]
CATEGORY_LABEL = dict(CATEGORIES)
_CAT_IDX = {k: i for i, (k, _) in enumerate(CATEGORIES)}


def categorize(vocab: Dict[str, int], analyzed: bool, particle: Optional[str], root: Optional[str],
               pattern: Optional[str], proclitics: Tuple[str, ...], enclitics: Tuple[str, ...],
               peeled: bool = False, particle_kind: str = "prep",
               clitic_only: bool = False, proper: bool = False,
               proper_mode: str = "unrooted") -> Tuple[str, str, List[str]]:
    """``(category, why, missing_tokens)`` for one alpha chunk — the exact rule of ``_emit_alpha``.

    ``enclitics`` is the emission-order tuple (fem ة + pronouns) — what both
    ``CorpusEntry.enclitics`` and ``Analysis.enclitics`` hold; for a particle
    it is the pronoun list (particles have no ة). ``particle_kind`` picks the
    closed group (``prep`` → [PREP_*], ``func`` → [FUNC_*]); ``clitic_only``
    marks a pronoun-hosted preposition (له) emitted as clitic tokens only;
    ``proper`` a database proper noun, routed to [PROP_*] under
    ``proper_mode`` (``unrooted``: only when it has no root; ``all``: always)
    or when its root / pattern was cut — the markers are fixed slots, so a
    name never reaches LIT.
    """
    if not analyzed:
        return "lit_no_analysis", "no accepted CAMeL analysis (native miss, peeler exhausted or every candidate rejected by a gate)", []
    cl_p = [f"{A.PFX_CLITICP}{c}{SFX}" for c in proclitics if c]
    cl_e = [f"{A.PFX_CLITICE}{c}{SFX}" for c in enclitics if c]
    prop_markers = TOK_PROP_BEGIN in vocab and TOK_PROP_END in vocab
    if proper and (proper_mode == "all" or not root):
        if prop_markers:
            return "prop", ("proper noun (CAMeL database noun_prop" + ("" if root else ", no root") +
                            "): its characters between [PROP_BEGIN] / [PROP_END], clitics outside"), []
        return "lit_no_analysis", "proper noun, but this vocab has no [PROP_*] markers — plain literal", []
    if particle:
        is_func = particle_kind == "func"
        prep_tok = f"{A.PFX_FUNC if is_func else A.PFX_PREP}{particle}{SFX}"
        missing = [t for t in [prep_tok] + cl_p + cl_e if t not in vocab]
        if not missing:
            if is_func:
                return "func", "closed-class function word: one [FUNC_*] token, clitics outside", []
            return "prep", "closed-class particle: one [PREP_*] token, clitics outside", []
        return "lit_clitic_missing", f"{' '.join(missing)} not in the vocab — the all-or-nothing rule sends the whole chunk to LIT", missing
    if clitic_only:
        missing = [t for t in cl_p + cl_e if t not in vocab]
        if not missing and cl_p and cl_e:
            return "clitic", "pronoun-hosted preposition: proclitic + pronoun tokens, no host word", []
        return "lit_clitic_missing", f"{' '.join(missing) or 'clitic-only word without both sides'} not in the vocab — the whole chunk goes to LIT", missing
    if not (root and pattern):
        return "lit_no_analysis", "analysis without root or pattern", []
    root_tok, pat_tok = f"{PFX_ROOT}{root}{SFX}", f"{PFX_PAT}{pattern}{SFX}"
    missing = [t for t in [root_tok, pat_tok] + cl_p + cl_e if t not in vocab]
    if not missing:
        return ("root_pat_peeled" if peeled else "root_pat"), ("analysis rescued by the clitic peeler; " if peeled else "") + "root, pattern and every clitic token are in the vocab", []
    if proper and prop_markers:
        return "prop", f"{' '.join(missing)} not in the vocab — a proper noun keeps the [PROP_*] path instead of LIT", missing
    if root_tok in missing:
        return "lit_root_cut", f"{root_tok} is not in the vocab" + (f" (nor {pat_tok})" if pat_tok in missing else "") + " — cut by the budget or below min_root_freq", missing
    if pat_tok in missing:
        return "lit_pattern_cut", f"{pat_tok} is not in the vocab — cut by the budget or below min_pattern_freq", missing
    return "lit_clitic_missing", f"{' '.join(missing)} not in the vocab — the all-or-nothing rule sends the whole chunk to LIT", missing


def categorize_analysis(vocab: Dict[str, int], a: Optional[B.Analysis],
                        proper_mode: str = "unrooted") -> Tuple[str, str, List[str]]:
    if a is None:
        return categorize(vocab, False, None, None, None, (), ())
    return categorize(vocab, True, a.particle, a.root, a.pattern, a.proclitics, a.enclitics, bool(a.peeled),
                      a.particle_kind, bool(a.clitic_only), bool(a.proper), proper_mode)


def categorize_entry(vocab: Dict[str, int], e: CorpusEntry,
                     proper_mode: str = "unrooted") -> Tuple[str, str, List[str]]:
    return categorize(vocab, e.analyzed, e.particle, e.root, e.pattern, tuple(e.proclitics or ()),
                      tuple(e.enclitics or ()), bool(getattr(e, "peeled", False)),
                      getattr(e, "particle_kind", "prep"), bool(getattr(e, "clitic_only", False)),
                      bool(getattr(e, "proper", False)), proper_mode)


# Pre-pass paths: what the analyzer produced for a chunk, before any budget.
# Exclusive (a peeled particle counts as "peeled"), so the counts sum to the
# number of unique chunks. Display order. ``prop`` is the *rootless* proper
# noun (NTWS name); a name with a root sits on ``root_pat`` with its
# ``proper`` flag, whatever the tokenizer's ``proper_nouns`` mode does with
# it at encode time.
PATHS: List[Tuple[str, str]] = [
    ("root_pat", "ROOT+PAT"),
    ("prep", "PREP"),
    ("func", "FUNC"),
    ("clitic", "CLITIC"),
    ("prop", "PROP"),
    ("peeled", "peeled"),
    ("lit", "LIT"),
]
PATH_LABEL = dict(PATHS)
_PATH_IDX = {k: i for i, (k, _) in enumerate(PATHS)}
# Aliases the cards use: the validate card's "analyzed" / "rejected" counters.
PATH_ALIASES: Dict[str, Tuple[str, ...]] = {
    "analyzed": ("root_pat", "prep", "func", "clitic", "prop", "peeled"),
    "rejected": ("lit",),
}
_PATH_TOTALS_KEY = {"root_pat": "ROOT+PAT", "prep": "PREP", "func": "FUNC", "clitic": "CLITIC",
                    "prop": "PROP", "peeled": "peeled", "lit": "LIT"}


def prepass_path(analyzed: bool, particle: Optional[str], peeled: bool, particle_kind: str = "prep",
                 clitic_only: bool = False, proper: bool = False, root: Optional[str] = None) -> str:
    """Exclusive pre-pass path of one entry (see ``PATHS``)."""
    if not analyzed:
        return "lit"
    if peeled:
        return "peeled"
    if particle:
        return "func" if particle_kind == "func" else "prep"
    if clitic_only:
        return "clitic"
    if proper and not root:
        return "prop"
    return "root_pat"


class WordCategoryIndex:
    """Every unique chunk of the corpus as one compact record (train mode).

    Built from the CorpusEntry list right after the pre-pass so the
    CorpusEntry objects (2.6 GB on the full corpus) can be dropped: one
    tuple per chunk holding every CorpusEntry field, shared strings
    interned (measured ≈ 260 MB for 1.09 M chunks). Rows are sorted by
    occurrences. Two classifications per row: the **pre-pass path**
    (``prepass_path``: what the analyzer produced) and, once
    ``attach_vocab`` has run, the **encode-time category** (``categorize``:
    what ``encode()`` will do under the built vocab). Also keeps the root /
    pattern frequency + rank tables so a budget cut can be explained exactly.

    ``query`` pages through any path / category / substring / peeled
    filter; the last filtered index list is cached so paging through a
    substring search does not rescan the million rows per page.
    """

    # row layout (tuples; the encode-time category lives in ``self.cats``)
    W, N, ROOT, PAT, PEELED, ANALYZED, PARTICLE, PRAW, STEM, SURF, PRC, ENC, PATH, KIND, CO, PROP = range(16)

    def __init__(self, entries: List[CorpusEntry], word_counts: Counter) -> None:
        intern: Dict[str, str] = {}

        def I(s: Optional[str]) -> Optional[str]:  # noqa: E743
            if not s:
                return None
            return intern.setdefault(s, s)

        rows: List[Tuple[Any, ...]] = []
        for e in entries:
            peeled = bool(getattr(e, "peeled", False))
            kind = getattr(e, "particle_kind", "prep") or "prep"
            co = bool(getattr(e, "clitic_only", False))
            proper = bool(getattr(e, "proper", False))
            rows.append((
                e.word, word_counts.get(e.word, 0), I(e.root), I(e.pattern), peeled, bool(e.analyzed),
                I(e.particle), I(getattr(e, "pattern_raw", None)), I(e.stem), e.surface or None,
                tuple(I(c) for c in (e.proclitics or ()) if c), tuple(I(c) for c in (e.enclitics or ()) if c),
                _PATH_IDX[prepass_path(e.analyzed, e.particle, peeled, kind, co, proper, e.root)], I(kind), co,
                proper,
            ))
        rows.sort(key=lambda r: (-r[1], r[0]))
        self.rows = rows
        self.pos = {r[0]: i for i, r in enumerate(rows)}
        self.by_path: Dict[str, List[int]] = {k: [] for k, _ in PATHS}
        for i, r in enumerate(rows):
            self.by_path[PATHS[r[self.PATH]][0]].append(i)
        self.path_counts = {k: len(v) for k, v in self.by_path.items()}
        self.path_occurrences = {k: sum(rows[i][self.N] for i in v) for k, v in self.by_path.items()}
        self.analyzed_count = sum(self.path_counts[k] for k in PATH_ALIASES["analyzed"])
        self.peeled_count = self.path_counts["peeled"]
        # filled by attach_vocab
        self.cats: Optional[array] = None
        self.by_cat: Dict[str, List[int]] = {}
        self.counts: Dict[str, int] = {}
        self.occurrences: Dict[str, int] = {}
        self.root_freq: Counter = Counter()
        self.pat_freq: Counter = Counter()
        self.root_rank: Dict[str, int] = {}
        self.pat_rank: Dict[str, int] = {}
        self.limits: Dict[str, int] = {}
        self._filter_cache: Optional[Tuple[Tuple[Any, ...], List[int]]] = None

    def attach_vocab(self, tok: AraRooPatTokenizer, root_freq: Counter, pat_freq: Counter) -> None:
        """Classify every row the way ``encode()`` will under ``tok._vocab``."""
        vocab = tok._vocab
        cats = array("b", bytes(len(self.rows)))
        by_cat: Dict[str, List[int]] = {k: [] for k, _ in CATEGORIES}
        for i, r in enumerate(self.rows):
            cat, _, _ = categorize(vocab, r[self.ANALYZED], r[self.PARTICLE], r[self.ROOT], r[self.PAT],
                                   r[self.PRC], r[self.ENC], r[self.PEELED], r[self.KIND] or "prep", r[self.CO],
                                   r[self.PROP], getattr(tok, "proper_nouns", "unrooted"))
            cats[i] = _CAT_IDX[cat]
            by_cat[cat].append(i)
        self.cats = cats
        self.by_cat = by_cat
        self.counts = {k: len(v) for k, v in by_cat.items()}
        self.occurrences = {k: sum(self.rows[i][self.N] for i in v) for k, v in by_cat.items()}
        self.root_freq, self.pat_freq = root_freq, pat_freq
        self.root_rank = {k: i + 1 for i, (k, _) in enumerate(sorted(root_freq.items(), key=lambda kv: (-kv[1], kv[0])))}
        self.pat_rank = {k: i + 1 for i, (k, _) in enumerate(sorted(pat_freq.items(), key=lambda kv: (-kv[1], kv[0])))}
        self.limits = {"max_roots": tok.max_roots, "max_patterns": tok.max_patterns,
                       "min_root_freq": tok.min_root_freq, "min_pattern_freq": tok.min_pattern_freq}
        self._filter_cache = None

    def path_totals(self) -> Dict[str, int]:
        """The validate card's counters: ``{"ROOT+PAT", "PREP", "FUNC", "CLITIC", "peeled", "LIT"}`` (exclusive, sum = total)."""
        return {_PATH_TOTALS_KEY[k]: self.path_counts[k] for k, _ in PATHS}

    def _row_dict(self, i: int, full: bool = False) -> Dict[str, Any]:
        r = self.rows[i]
        ci = self.cats[i] if self.cats is not None else None
        pk = PATHS[r[self.PATH]][0]
        d: Dict[str, Any] = {"word": r[self.W], "count": r[self.N],
                             "category": CATEGORIES[ci][0] if ci is not None else None,
                             "label": CATEGORIES[ci][1] if ci is not None else None,
                             "root": r[self.ROOT], "pattern": r[self.PAT], "peeled": r[self.PEELED],
                             "path": pk, "path_label": PATH_LABEL[pk]}
        if full:
            d.update({"analyzed": r[self.ANALYZED], "particle": r[self.PARTICLE], "particle_kind": r[self.KIND],
                      "clitic_only": r[self.CO], "proper": r[self.PROP], "pattern_raw": r[self.PRAW],
                      "stem": r[self.STEM], "surface": r[self.SURF],
                      "proclitics": list(r[self.PRC]), "enclitics": list(r[self.ENC])})
        return d

    def cached(self, word: str) -> Optional[Dict[str, Any]]:
        i = self.pos.get(word)
        return self._row_dict(i) if i is not None else None

    def record(self, word: str) -> Optional[Dict[str, Any]]:
        """The full pre-pass record of one chunk (every CorpusEntry field), or None."""
        i = self.pos.get(word)
        return self._row_dict(i, full=True) if i is not None else None

    def summary(self) -> Dict[str, Any]:
        return {"categories": [{"key": k, "label": lab, "unique": self.counts.get(k, 0), "occurrences": self.occurrences.get(k, 0)}
                               for k, lab in CATEGORIES],
                "paths": [{"key": k, "label": lab, "unique": self.path_counts[k], "occurrences": self.path_occurrences[k]}
                          for k, lab in PATHS],
                "path_aliases": {k: list(v) for k, v in PATH_ALIASES.items()},
                "unique": len(self.rows), "occurrences": sum(r[self.N] for r in self.rows),
                "analyzed": self.analyzed_count, "peeled": self.peeled_count, "limits": self.limits}

    def query(self, category: Optional[str] = None, q: str = "", peeled: Optional[bool] = None,
              limit: int = 100, offset: int = 0, path: Optional[str] = None,
              full: bool = False) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q = strip_diacritics((q or "").strip())
        if category and category not in self.by_cat:
            raise ValueError(f"unknown category {category!r}")
        path_keys: Optional[Tuple[str, ...]] = None
        if path:
            path_keys = PATH_ALIASES.get(path) or ((path,) if path in self.by_path else None)
            if path_keys is None:
                raise ValueError(f"unknown path {path!r}")
        limit = max(1, min(500, limit))
        offset = max(0, offset)
        # base index list: the narrowest precomputed list, the rest filtered on the fly
        path_set: Optional[set] = None
        if category:
            idx: Any = self.by_cat[category]
            if path_keys:
                path_set = {_PATH_IDX[k] for k in path_keys}
        elif path_keys and len(path_keys) == 1:
            idx = self.by_path[path_keys[0]]
        elif path_keys:
            idx = range(len(self.rows))
            path_set = {_PATH_IDX[k] for k in path_keys}
        else:
            idx = range(len(self.rows))
        if not q and peeled is None and path_set is None:
            total = len(idx)
            page = list(idx[offset:offset + limit])
        else:
            key = (category, path, q, peeled)
            if self._filter_cache and self._filter_cache[0] == key:
                hits = self._filter_cache[1]
            else:
                rows = self.rows
                hits = [i for i in idx
                        if (path_set is None or rows[i][self.PATH] in path_set)
                        and (peeled is None or rows[i][self.PEELED] == peeled)
                        and (not q or q in strip_diacritics(rows[i][self.W]))]
                self._filter_cache = (key, hits)
            total = len(hits)
            page = hits[offset:offset + limit]
        return {"category": category, "path": path, "q": q, "peeled": peeled, "full": full,
                "total": total, "offset": offset, "limit": limit,
                "rows": [self._row_dict(i, full) for i in page], "ms": round((time.perf_counter() - t0) * 1000, 1)}

    def budget_reason(self, kind: str, key: Optional[str], vocab: Dict[str, int]) -> Dict[str, Any]:
        """Why a root / pattern is or is not in the vocab, in the budget loop's own terms."""
        if not key:
            return {"kind": kind, "key": None, "in_vocab": False, "reason": "no such token"}
        freq_tab, rank_tab = (self.root_freq, self.root_rank) if kind == "root" else (self.pat_freq, self.pat_rank)
        max_k = self.limits["max_roots" if kind == "root" else "max_patterns"]
        min_f = self.limits["min_root_freq" if kind == "root" else "min_pattern_freq"]
        tok_ = f"{PFX_ROOT if kind == 'root' else PFX_PAT}{key}{SFX}"
        freq, rank = freq_tab.get(key), rank_tab.get(key)
        out = {"kind": kind, "key": key, "token": tok_, "in_vocab": tok_ in vocab, "freq": freq, "rank": rank,
               "candidates": len(freq_tab), "max": max_k, "min_freq": min_f}
        if tok_ in vocab:
            out["reason"] = f"kept — rank {rank:,} of {len(freq_tab):,} candidates, freq {freq} ≥ min_freq {min_f}, within max {max_k:,}"
        elif freq is None:
            out["reason"] = "never produced by an analyzed corpus word — it has no frequency, so it could not be a candidate"
        elif freq < min_f:
            out["reason"] = f"freq {freq} < min_freq {min_f} (rank {rank:,} of {len(freq_tab):,}) — the budget loop breaks at the first such item"
        elif rank is not None and rank > max_k:
            out["reason"] = f"rank {rank:,} > max {max_k:,} (freq {freq}) — beyond the budget"
        else:
            out["reason"] = f"cut — rank {rank}, freq {freq} (an earlier item ended the loop)"
        return out


def _root_regex(q: str) -> Optional[re.Pattern]:
    """Root-letter query → regex: ``#`` matches any radical, a weak letter matches itself or a masked radical."""
    parts = []
    for ch in q:
        if ch == B.WEAK_RADICAL_MARK:
            parts.append(".")
        elif ch in _WEAK_CLASS:
            parts.append(f"[{re.escape(ch)}{B.WEAK_RADICAL_MARK}]")
        else:
            parts.append(re.escape(ch))
    try:
        return re.compile("".join(parts))
    except re.error:
        return None


class FreqIndex:
    """The complete frequency tables of the freq step, searchable and paged.

    One row per candidate (rank, key, freq, contributing words, kept / cut
    in the vocab; wazn for patterns), per kind: ``root``, ``pat``, ``prc``,
    ``enc``, ``prep``. Train mode builds it from the counters the freq step
    computed (``from_counters``); saved mode from ``vocab_metadata.json``
    (``from_metadata`` — kept tokens only, as the card says). Root queries
    use the token search's wildcard rule (``قول`` finds ``ق#ل``); pattern
    queries match the CAMeL slot string *or* the wazn, tashkeel ignored.
    """

    KINDS = ("root", "pat", "prc", "enc", "prep", "func")
    _PFX = {"root": PFX_ROOT, "pat": PFX_PAT, "prc": A.PFX_CLITICP, "enc": A.PFX_CLITICE, "prep": A.PFX_PREP,
            "func": A.PFX_FUNC}

    def __init__(self, tables: Dict[str, Dict[str, int]], words: Dict[str, Dict[str, List[str]]],
                 vocab: Dict[str, int], from_metadata: bool = False) -> None:
        self.from_metadata = from_metadata
        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        for kind in self.KINDS:
            cnt = tables.get(kind) or {}
            ws = words.get(kind) or {}
            pfx = self._PFX[kind]
            rows = []
            for rank, (k, v) in enumerate(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])), 1):
                row: Dict[str, Any] = {"rank": rank, "key": k, "freq": v, "words": list(ws.get(k, []))[:6],
                                       "in_vocab": f"{pfx}{k}{SFX}" in vocab}
                if kind == "pat":
                    row["wazn"] = wazn_of(k)
                    row["_skel"] = strip_diacritics(k)
                    row["_wskel"] = strip_diacritics(row["wazn"])
                rows.append(row)
            self.tables[kind] = rows
        self.sizes = {k: len(v) for k, v in self.tables.items()}
        self.kept = {k: sum(1 for r in v if r["in_vocab"]) for k, v in self.tables.items()}

    @classmethod
    def from_counters(cls, root_freq: Counter, pat_freq: Counter, proclitic_freq: Counter, enclitic_freq: Counter,
                      particle_freq: Counter, contrib: Dict[str, Dict[str, List[str]]],
                      vocab: Dict[str, int], func_freq: Optional[Counter] = None) -> "FreqIndex":
        return cls({"root": root_freq, "pat": pat_freq, "prc": proclitic_freq, "enc": enclitic_freq,
                    "prep": particle_freq, "func": func_freq or Counter()}, contrib, vocab)

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any], vocab: Dict[str, int]) -> "FreqIndex":
        roots, pats, preps = meta.get("roots", {}), meta.get("patterns", {}), meta.get("prepositions", {})
        funcs = meta.get("func_words", {})

        def ex_words(v: Dict[str, Any], key: str) -> List[str]:
            return [e_[1] if isinstance(e_, list) else e_ for e_ in (v.get(key) or [])][:6]

        tables = {"root": {k: v.get("freq", 0) for k, v in roots.items()},
                  "pat": {k: v.get("freq", 0) for k, v in pats.items()},
                  "prc": dict(meta.get("proclitic_freq", {})), "enc": dict(meta.get("enclitic_freq", {})),
                  "prep": {k: v.get("freq", 0) for k, v in preps.items()},
                  "func": {k: v.get("freq", 0) for k, v in funcs.items()}}
        words = {"root": {k: ex_words(v, "example_words") for k, v in roots.items()},
                 "pat": {k: ex_words(v, "examples") for k, v in pats.items()}}
        return cls(tables, words, vocab, from_metadata=True)

    def query(self, kind: str, q: str = "", kept: Optional[bool] = None, limit: int = 50,
              offset: int = 0) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if kind not in self.tables:
            raise ValueError(f"unknown kind {kind!r} (one of {', '.join(self.KINDS)})")
        rows = self.tables[kind]
        q = (q or "").strip()
        limit = max(1, min(500, limit))
        offset = max(0, offset)
        match = None
        if q:
            if kind == "root":
                rx = _root_regex(strip_diacritics(q))
                match = (lambda r: bool(rx.search(r["key"]))) if rx else (lambda r: False)
            elif kind == "pat":
                qs = strip_diacritics(q)
                match = lambda r: qs in r["_skel"] or qs in r["_wskel"]  # noqa: E731
            else:
                match = lambda r: q in r["key"]  # noqa: E731
        if match is None and kept is None:
            hits = rows
        else:
            hits = [r for r in rows if (match is None or match(r)) and (kept is None or r["in_vocab"] == kept)]
        page = [{k: v for k, v in r.items() if not k.startswith("_")} for r in hits[offset:offset + limit]]
        return {"kind": kind, "q": q, "kept": kept, "total": len(hits), "offset": offset, "limit": limit,
                "candidates": len(rows), "kept_total": self.kept[kind], "max_freq": rows[0]["freq"] if rows else 0,
                "from_metadata": self.from_metadata, "rows": page, "ms": round((time.perf_counter() - t0) * 1000, 1)}


# ---------------------------------------------------------------------------
# Encode plain text / trace one word, on a finished job
# ---------------------------------------------------------------------------

def _stream(tok: AraRooPatTokenizer, out: Any) -> List[Dict[str, Any]]:
    return [{"pos": i, "id": tid, "token": tok._reverse_vocab.get(tid, "?"),
             "family": _split_key(tok._reverse_vocab.get(tid, "?")),
             "inner": _inner(tok._reverse_vocab.get(tid, "?")), "metric_string": m}
            for i, (tid, m) in enumerate(zip(out.input_ids, out.tokens))]


def encode_text(job: "CorpusTraceJob", text: str) -> Dict[str, Any]:
    """The real ``encode()`` of plain text + the category of every alpha chunk + the round-trip."""
    tok, backend = job.tokenizer, job.tokenizer._backend  # type: ignore[union-attr]
    assert tok is not None and backend is not None
    t0 = time.perf_counter()
    with _WireTap(backend._bridge) as tap:
        out = tok.encode(text)
        wire = tap.take()
    stream = _stream(tok, out)
    norm = unicodedata.normalize("NFKC", text)
    counts: Counter = Counter()
    for w in norm.split():
        for chunk in _extract_alpha_chunks(w):
            counts[chunk] += 1
    chunks = []
    cat_occ: Counter = Counter()
    for chunk, n in counts.items():
        a = backend._analyze_cache.get(chunk)   # filled by encode()
        cat, why, missing = categorize_analysis(tok._vocab, a, tok.proper_nouns)
        cat_occ[cat] += n
        row = {"chunk": chunk, "count": n, "category": cat, "label": CATEGORY_LABEL[cat], "why": why, "missing": missing,
               "root": (a.root or None) if a else None, "pattern": (a.pattern or None) if a else None,
               "particle": a.particle if a else None, "particle_kind": a.particle_kind if a else None,
               "clitic_only": bool(a.clitic_only) if a else False, "proper": bool(a.proper) if a else False,
               "peeled": bool(a.peeled) if a else False,
               "proclitics": list(a.proclitics) if a else [], "enclitics": list(a.enclitics) if a else []}
        if job.words is not None:
            c = job.words.cached(chunk)
            row["corpus"] = c and {"category": c["category"], "count": c["count"]}
        chunks.append(row)
    chunks.sort(key=lambda r: (-r["count"], r["chunk"]))
    decoded = tok.decode(out.input_ids)
    src_words = norm.split()
    return {
        "text": text, "num_tokens": len(out.input_ids), "num_words": len(src_words), "num_chunks": len(counts),
        "stream": stream, "chunks": chunks,
        "categories": [{"key": k, "label": lab, "occurrences": cat_occ.get(k, 0),
                        "unique": sum(1 for r in chunks if r["category"] == k)} for k, lab in CATEGORIES],
        "ipc_calls": sum(1 for l in wire if l["dir"] == "→ server"), "wire": wire[:WIRE_CAP],
        "decoded": decoded,
        "exact_match": " ".join(src_words) == decoded,
        "match_ignoring_diacritics": strip_diacritics(" ".join(src_words)) == strip_diacritics(decoded),
        "match_ignoring_spacing": strip_diacritics("".join(src_words)) == strip_diacritics(decoded).replace(" ", ""),
        "ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def trace_word_process(job: "CorpusTraceJob", word: str) -> Dict[str, Any]:
    """Replay one chunk live: CAMeL candidates + gates, peeler, vocab / budget check, emitted tokens."""
    tok, backend = job.tokenizer, job.tokenizer._backend  # type: ignore[union-attr]
    assert tok is not None and backend is not None
    t0 = time.perf_counter()
    norm = unicodedata.normalize("NFKC", (word or "").strip())
    chunks = [c for w in norm.split() for c in _extract_alpha_chunks(w)]
    if not chunks:
        raise ValueError("no Arabic alpha chunk in the word")
    chunk = chunks[0]
    bridge = backend._bridge
    with _WireTap(bridge) as tap:
        raw = bridge.analyze([chunk])
        analyses, val_rows = trace_validate_words([chunk], {chunk: raw[0]}, backend)
        peel_rows = trace_peel_words([chunk], analyses, backend)
        wire = tap.take()
        out = tok.encode(chunk)
        wire += tap.take()
    a = analyses[chunk]
    cat, why, missing = categorize_analysis(tok._vocab, a, tok.proper_nouns)
    vocab = tok._vocab
    check: Dict[str, Any] = {"root": None, "pattern": None, "particle": None, "clitics": [], "clitic_only": False,
                             "proper": False, "prop_path": False, "proper_mode": tok.proper_nouns}
    if a is not None:
        check["clitic_only"] = bool(a.clitic_only)
        check["proper"] = bool(a.proper)
        check["prop_path"] = tok._routes_to_prop(a.proper, a.root)
        if a.particle:
            t_ = f"{A.PFX_FUNC if a.particle_kind == 'func' else A.PFX_PREP}{a.particle}{SFX}"
            check["particle"] = {"token": t_, "in_vocab": t_ in vocab, "kind": a.particle_kind}
        elif a.proper and not a.root:
            pass   # nothing to look up: the characters go between the fixed [PROP_*] markers
        else:
            if job.words is not None:
                check["root"] = job.words.budget_reason("root", a.root, vocab)
                check["pattern"] = job.words.budget_reason("pattern", a.pattern, vocab)
            else:
                rt, pt = f"{PFX_ROOT}{a.root}{SFX}", f"{PFX_PAT}{a.pattern}{SFX}"
                check["root"] = {"kind": "root", "key": a.root, "token": rt, "in_vocab": rt in vocab,
                                 "reason": "in the loaded vocab" if rt in vocab else "not in the loaded vocab (no frequency tables for a saved tokenizer)"}
                check["pattern"] = {"kind": "pattern", "key": a.pattern, "token": pt, "in_vocab": pt in vocab,
                                    "reason": "in the loaded vocab" if pt in vocab else "not in the loaded vocab (no frequency tables for a saved tokenizer)"}
        for c in a.proclitics:
            t_ = f"{A.PFX_CLITICP}{c}{SFX}"
            check["clitics"].append({"token": t_, "kind": "proclitic", "in_vocab": t_ in vocab})
        encl = a.pronoun_enclitics if a.particle else a.enclitics
        for c in encl:
            t_ = f"{A.PFX_CLITICE}{c}{SFX}"
            check["clitics"].append({"token": t_, "kind": "enclitic", "in_vocab": t_ in vocab})
    cached = job.words.cached(chunk) if job.words is not None else None
    prepass = job.words.record(chunk) if job.words is not None else None
    agrees_prepass = None
    if prepass is not None:
        live = _analysis_summary(a)
        agrees_prepass = (prepass["analyzed"] == live["analyzed"]) and (
            not live["analyzed"] or all(prepass[k] == live[k] for k in ("root", "pattern", "proclitics", "enclitics", "particle", "clitic_only")))
    return {
        "word": word, "chunk": chunk, "other_chunks": chunks[1:],
        "prepass": prepass, "agrees_with_prepass": agrees_prepass,
        "validate": val_rows[0], "peel": peel_rows[0] if peel_rows else None,
        "analysis": _analysis_summary(a), "category": cat, "label": CATEGORY_LABEL[cat], "why": why,
        "missing": missing, "vocab_check": check,
        "stream": _stream(tok, out), "decoded": tok.decode(out.input_ids),
        "cached": cached, "agrees_with_corpus": (cached["category"] == cat) if cached else None,
        "wire": wire[:WIRE_CAP], "ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------------------
# Token search
# ---------------------------------------------------------------------------

_ARABIC_RE = re.compile(r"[؀-ۿ]")
_WEAK_CLASS = "اويءأإآؤئى"


class TokenIndex:
    """In-memory search over a trained/loaded tokenizer's vocab + reconstruction table.

    Fields per token: id, token, family, inner, freq, examples, and for
    ``[PAT_*]`` the wazn / skeleton / gloss. Queries: an integer id; a
    bracketed token prefix; Arabic text matched against inners (with ``#``
    as a radical wildcard and weak letters matching a masked radical), wazn,
    skeletons and example words; a whole word also looks up reconstruction
    surfaces and is encoded live ("this word becomes these tokens").
    """

    def __init__(self, tok: AraRooPatTokenizer) -> None:
        self.tok = tok
        meta = tok._metadata or {}
        roots_meta, pats_meta = meta.get("roots", {}), meta.get("patterns", {})
        prc, enc, preps = meta.get("proclitic_freq", {}), meta.get("enclitic_freq", {}), meta.get("prepositions", {})
        funcs = meta.get("func_words", {})
        self.rows: List[Dict[str, Any]] = []
        for t_, i in sorted(tok._vocab.items(), key=lambda kv: kv[1]):
            fam = _split_key(t_)
            inner = _inner(t_)
            row: Dict[str, Any] = {"id": i, "token": t_, "family": fam, "inner": inner, "freq": None, "examples": []}
            if fam == "root":
                m = roots_meta.get(inner) or {}
                row["freq"] = m.get("freq")
                row["examples"] = list(m.get("example_words", []))[:5]
            elif fam == "pat":
                m = pats_meta.get(inner) or {}
                d = describe_pattern(inner, m.get("examples"))
                row.update({"freq": m.get("freq"), "wazn": d["wazn"], "skeleton": d["skeleton"], "gloss": d["gloss"],
                            "gloss_tier": d["gloss_tier"], "weak_slots": d["weak_slots"],
                            "filled": (d["filled"] or {}).get("stem"),
                            "examples": [e_[1] if isinstance(e_, list) else e_ for e_ in (m.get("examples") or [])][:5]})
            elif fam == "cliticp":
                row["freq"] = prc.get(inner)
            elif fam == "clitice":
                row["freq"] = enc.get(inner)
            elif fam == "prep":
                row["freq"] = (preps.get(inner) or {}).get("freq")
            elif fam == "func":
                row["freq"] = (funcs.get(inner) or {}).get("freq")
            self.rows.append(row)
        self.by_id = {r["id"]: r for r in self.rows}
        self.families = Counter(r["family"] for r in self.rows)
        self.surface_to_pairs: Dict[str, List[Tuple[int, int]]] = {}
        for (rid, pid), surf in tok._reconstruction.items():
            self.surface_to_pairs.setdefault(strip_diacritics(surf), []).append((rid, pid))

    @staticmethod
    def _root_regex(q: str) -> Optional[re.Pattern]:
        return _root_regex(q)

    def search(self, q: str, family: Optional[str] = None, limit: int = 100, offset: int = 0,
               encode_lookup: bool = True) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q = (q or "").strip()
        limit = max(1, min(500, limit))
        hits: List[Tuple[float, Dict[str, Any], str]] = []
        encode: Optional[Dict[str, Any]] = None
        if not q:
            rows = [r for r in self.rows if not family or r["family"] == family]
            hits = [(0.0, r, "") for r in rows]
        elif q.lstrip("-").isdigit():
            r = self.by_id.get(int(q))
            if r and (not family or r["family"] == family):
                hits.append((100.0, r, "id"))
        else:
            ql = q.lower()
            is_token = q.startswith("[") or q.startswith("<")
            arabic = bool(_ARABIC_RE.search(q))
            q_nodiac = strip_diacritics(q)
            root_re = self._root_regex(q_nodiac) if arabic and " " not in q else None
            seen: set = set()

            def add(score: float, r: Dict[str, Any], why: str) -> None:
                if family and r["family"] != family:
                    return
                if r["id"] in seen:
                    return
                seen.add(r["id"])
                hits.append((score, r, why))

            for r in self.rows:
                if is_token:
                    if ql in r["token"].lower():
                        add(90.0 if r["token"].lower().startswith(ql) else 60.0, r, "token")
                    continue
                inner = r["inner"]
                inner_nd = strip_diacritics(inner)
                if inner == q or inner_nd == q_nodiac:
                    add(100.0, r, "exact")
                    continue
                if r["family"] == "root" and root_re is not None and root_re.search(inner):
                    add(80.0 if root_re.fullmatch(inner) else 65.0, r, "root (# wildcard)")
                    continue
                if r["family"] == "pat":
                    if q in (r.get("wazn") or "") or q_nodiac in (r.get("skeleton") or ""):
                        add(75.0 if q_nodiac == r.get("skeleton") else 55.0, r, "wazn")
                        continue
                    if not arabic and r.get("gloss") and ql in r["gloss"].lower():
                        add(45.0, r, "gloss")
                        continue
                if arabic and q_nodiac in inner_nd:
                    add(50.0, r, "contains")
                    continue
                if not arabic and ql in r["token"].lower():
                    add(40.0, r, "token")
                    continue
                if arabic and any(q_nodiac in strip_diacritics(w) for w in r["examples"]):
                    add(35.0, r, "example word")
            if arabic and " " not in q and not is_token:
                for (rid, pid) in self.surface_to_pairs.get(q_nodiac, []):
                    rr, pr = self.by_id.get(rid), self.by_id.get(pid)
                    if rr:
                        add(70.0, rr, f"reconstructs '{q_nodiac}'")
                    if pr:
                        add(70.0, pr, f"reconstructs '{q_nodiac}'")
                if encode_lookup and len(q) <= 60:
                    try:
                        out = self.tok.encode(q)
                        encode = {"text": q, "stream": [
                            {"pos": i, "id": tid, "token": self.tok._reverse_vocab.get(tid, "?"),
                             "family": _split_key(self.tok._reverse_vocab.get(tid, "?")),
                             "inner": _inner(self.tok._reverse_vocab.get(tid, "?")), "metric_string": m}
                            for i, (tid, m) in enumerate(zip(out.input_ids, out.tokens))],
                            "decoded": self.tok.decode(out.input_ids)}
                    except Exception as e:  # noqa: BLE001
                        encode = {"text": q, "error": f"{type(e).__name__}: {e}"}
        hits.sort(key=lambda x: (-x[0], -(x[1]["freq"] or 0), x[1]["id"]))
        page = hits[offset:offset + limit]
        return {
            "query": q, "family": family, "total": len(hits), "offset": offset, "limit": limit,
            "hits": [{**r, "score": s, "why": why} for s, r, why in page],
            "families": dict(self.families), "encode": encode,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }


def save_trained(tok: AraRooPatTokenizer, name: str, overwrite: bool = False) -> Dict[str, Any]:
    """``tok.save()`` under ``outputs/tokenizers/<name>``; refuses to overwrite unless asked."""
    if not _NAME_RE.match(name or ""):
        raise ValueError("name must match [A-Za-z0-9_.-]{1,80}")
    path = TOKENIZERS_DIR / name
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path.relative_to(REPO_ROOT)} exists — tick overwrite to replace it")
    t0 = time.perf_counter()
    tok.save(path)
    files = {p.name: p.stat().st_size for p in sorted(path.iterdir()) if p.is_file()}
    return {"path": str(path.relative_to(REPO_ROOT)), "files": files,
            "ms": round((time.perf_counter() - t0) * 1000, 1)}
