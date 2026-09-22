"""AraRooPat decode canonicalization (2026-09-22): the word comes back as written.

CAMeL's analyzer normalises hamza / ى / ة on input and the MLE disambiguator ranks
readings without looking at the spelling the writer used, so the top reading of
الإعرابية was الأَعْرابِيَّة and that of همزة was هَمَزَهُ (verb + 3ms ه). Three fixes,
all pinned here on recorded readings (pure Python, no bridge) plus two live checks:

* ``prefer_faithful`` — a rooted top reading that does not spell the word yields to
  the first ranked candidate that does;
* ``reconcile_final_taa`` — no faithful candidate and the mismatch is the ة/ه slot →
  the slot follows the written letter;
* ``entry_realization`` — the reconstruction surface of a (root, pattern) pair is the
  most frequent *written* stem (clitics stripped from the chunk, accepted when the
  decoder's joins reproduce it), CAMeL's ``diac`` only as a fallback;
* U+200B and the other ``Cf`` characters are dropped at encode instead of ``<unk>``.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import pytest

from arabic_eval.tokenizers.araroopat import (
    PFX_CLITICE,
    PFX_PAT,
    PFX_ROOT,
    SFX,
    TOK_UNK,
    AraRooPatTokenizer,
    _stale_under_spelling_walk,
    entry_realization,
    join_word,
    normalize_text,
    pick_realization,
)
from arabic_eval.tokenizers.araroopat_backend import (
    FUNC_INVENTORY,
    PREPOSITION_INVENTORY,
    TAA_MARBUTA,
    Analysis,
    CorpusEntry,
    MorphAnalyzer,
    _dict_to_analysis,
    _strip_diac,
    is_rooted_analysis,
    needs_spelling_walk,
    prefer_faithful,
    reconcile_final_taa,
    spelling_faithful,
)

PREPS = frozenset(PREPOSITION_INVENTORY)
FUNCS = frozenset(FUNC_INVENTORY)


def cam(lex: str, diac: str, root: str, pattern: str, stem: str, pos: str = "noun", **clitics: str) -> Dict[str, str]:
    d = {"lex": lex, "diac": diac, "root": root, "pattern": pattern, "stem": stem,
         "pos": pos, "prc3": "0", "prc2": "0", "prc1": "0", "prc0": "0", "enc0": "0"}
    d.update(clitics)
    return d


# Ranked readings as the bridge returned them on 2026-09-22 (MLE order, top first).
RECORDED: Dict[str, List[Dict[str, str]]] = {
    # الإعراب: the Bedouins first, the grammatical term (the spelling written) fifth.
    "الإعراب": [
        cam("عَرَب", "الأَعْراب", "ع.ر.ب", "الأَ1ْ2ا3", "أَعْراب", prc0="Al_det"),
        cam("عَرَب", "الأَعْرابَ", "ع.ر.ب", "الأَ1ْ2ا3َ", "أَعْراب", prc0="Al_det"),
        cam("إِعْراب", "الإِعْراب", "ع.ر.ب", "الإِ1ْ2ا3", "إِعْراب", prc0="Al_det"),
    ],
    # الإعرابية: CAMeL knows only the Bedouin adjective — no faithful candidate.
    "الإعرابية": [
        cam("أَعْرابِيّ", "الأَعْرابِيَّة", "ع.ر.ب", "الأَ1ْ2ا3ِيَّة", "أَعْرابِيّ", pos="adj", prc0="Al_det"),
        cam("أَعْرابِيّ", "الأَعْرابِيَّةَ", "ع.ر.ب", "الأَ1ْ2ا3ِيَّةَ", "أَعْرابِيّ", pos="adj", prc0="Al_det"),
    ],
    "الأعرابية": [
        cam("أَعْرابِيّ", "الأَعْرابِيَّة", "ع.ر.ب", "الأَ1ْ2ا3ِيَّة", "أَعْرابِيّ", pos="adj", prc0="Al_det"),
    ],
    # همزة: the verb + 3ms object first (ة→ه normalisation), the feminine noun fourth.
    "همزة": [
        cam("هَمَز", "هَمَزَهُ", "ه.م.ز", "1َ2َ3َهُ", "هَمَز", pos="verb", enc0="3ms_dobj"),
        cam("هَمْز", "هَمْزه", "ه.م.ز", "1َ2ْ3ه", "هَمْز", enc0="3ms_poss"),
        cam("هَمْزَة", "هَمْزَة", "ه.م.ز", "1َ2ْ3َة", "هَمْز"),
        cam("هَمْزَة", "هَمْزَةً", "ه.م.ز", "1َ2ْ3َةً", "هَمْز"),
    ],
    "همزه": [
        cam("هَمَز", "هَمَزَهُ", "ه.م.ز", "1َ2َ3َهُ", "هَمَز", pos="verb", enc0="3ms_dobj"),
    ],
    # تزعمتة: a ة typed for a ه — only ه readings exist → the slot is reconciled.
    "تزعمتة": [
        cam("تَزَعَّم", "تَزَعَّمَتْهُ", "ز.ع.م", "تَ1َ2َّ3َتْهُ", "تَزَعَّم", pos="verb", enc0="3ms_dobj"),
    ],
    # المتضرره: a ه written for ة — only ة readings exist → the slot follows the writer.
    "المتضرره": [
        cam("مُتَضَرِّر", "المُتَضَرِّرَة", "ض.ر.ر", "المُتَ1َ2ِّ3َة", "مُتَضَرِّر", pos="adj", prc0="Al_det"),
    ],
    # ابعد: a dropped hamza — no candidate spells it, the top reading stays.
    "ابعد": [
        cam("أَبْعَد", "أَبْعَدَ", "ب.ع.د", "أَ1ْ2َ3َ", "أَبْعَد", pos="verb"),
    ],
    # مدرسته: faithful (ة realised as ت before the pronoun) — the walk must not fire.
    "مدرسته": [
        cam("مَدْرَسَة", "مَدْرَسَته", "د.ر.س", "مَ1ْ2َ3َته", "مَدْرَس", enc0="3ms_poss"),
    ],
    "كتاب": [cam("كِتاب", "كِتاب", "ك.ت.ب", "1ِ2ا3", "كِتاب")],
    "الجملة": [cam("جُمْلَة", "الجُمْلَةُ", "ج.م.ل", "ال1ُ2ْ3َةُ", "جُمْل", prc0="Al_det")],
    # Closed-class matches modulo alef (the format-10 POS gate). CAMeL's top readings:
    "كان": [cam("كان", "كانَ", "ك.#.ن", "1ا3َ", "كان", pos="verb")],                       # the verb "was", not كأن
    "وكان": [cam("كان", "وَكانَ", "ك.#.ن", "وَ1ا3َ", "كان", pos="verb", prc2="wa_conj")],
    "كأن": [cam("كَأَنَّ", "كَأَنَّ", "", "", "", pos="verb_pseudo")],
    "آن": [cam("آن", "آن", "#.#.ن", "آ3", "آن", pos="noun")],                             # "time", not أن
    "إنما": [cam("إِنَّما", "إِنَّما", "", "", "", pos="verb_pseudo", enc0="ma_rel")],         # إن + ما, not أن
    "الى": [cam("إِلَى", "إِلَى", "", "", "", pos="prep")],                                 # a dropped hamza canonicalises
    "انها": [cam("أَنَّ", "أَنَّها", "", "", "", pos="conj_sub", enc0="3fs_pron")],
}


def analyses(word: str) -> List[Analysis]:
    return [a for a in (_dict_to_analysis(d, PREPS, word) for d in RECORDED[word]) if a is not None]


class _RecordedBridge:
    """Serves the recorded readings; records the ``top`` of every request."""

    def __init__(self) -> None:
        self.requests: List[tuple] = []

    def analyze(self, words: List[str], top: int = 1) -> List[List[Dict[str, str]]]:
        self.requests.append((tuple(words), top))
        return [RECORDED.get(w, [])[:max(1, top)] for w in words]

    def generate(self, root: str, pattern: str) -> Optional[str]:
        return None


# ---------------------------------------------------------------------------
# The helpers
# ---------------------------------------------------------------------------

class TestSpellingFaithful:
    def test_faithful_compares_undiacritized_surfaces(self):
        a = analyses("الإعراب")[0]
        assert not spelling_faithful(a, "الإعراب") and spelling_faithful(a, "الأعراب")
        assert spelling_faithful(analyses("مدرسته")[0], "مدرسته")

    def test_only_rooted_readings_walk(self):
        top = analyses("الإعراب")[0]
        assert is_rooted_analysis(top) and needs_spelling_walk(top, "الإعراب")
        assert not needs_spelling_walk(top, "الأعراب")
        particle = MorphAnalyzer._first_valid([], "في", PREPS)
        assert particle.particle == "في" and not is_rooted_analysis(particle)
        assert not needs_spelling_walk(particle, "فى") and not needs_spelling_walk(None, "x")


class TestPreferFaithful:
    def test_hamza_seat_picks_the_reading_actually_written(self):
        cands = analyses("الإعراب")
        chosen = prefer_faithful(cands[0], cands, "الإعراب")
        assert (chosen.pattern, chosen.lemma) == ("إِ1ْ2ا3", "إِعْراب")
        # the same candidates with the other spelling keep the top reading
        assert prefer_faithful(cands[0], cands, "الأعراب") is cands[0]

    def test_taa_marbuta_picks_the_feminine_noun_over_the_verb_plus_pronoun(self):
        cands = analyses("همزة")
        assert cands[0].enc0 == "ه" and cands[0].pos == "verb"
        chosen = prefer_faithful(cands[0], cands, "همزة")
        assert (chosen.pattern, chosen.fem, chosen.enc0, chosen.pos) == ("1َ2ْ3َ", TAA_MARBUTA, None, "noun")

    def test_written_ha_keeps_the_verb_reading(self):
        cands = analyses("همزه")
        assert prefer_faithful(cands[0], cands, "همزه") is cands[0]

    def test_no_faithful_candidate_keeps_the_top_reading(self):
        cands = analyses("ابعد")
        assert prefer_faithful(cands[0], cands, "ابعد") is cands[0]
        cands = analyses("الإعرابية")
        assert prefer_faithful(cands[0], cands, "الإعرابية") is cands[0]


class TestReconcileFinalTaa:
    def test_written_taa_read_as_pronoun_becomes_the_suffix(self):
        a = analyses("تزعمتة")[0]
        assert (a.enc0, a.fem) == ("ه", None)
        r = reconcile_final_taa(a, "تزعمتة")
        assert (r.enc0, r.fem, r.enclitics) == (None, TAA_MARBUTA, (TAA_MARBUTA,))
        assert r.surface.endswith(TAA_MARBUTA) and _strip_diac(r.surface) == "تزعمتة"
        assert r.pattern == a.pattern and r.root == a.root

    def test_written_ha_read_as_feminine_becomes_the_pronoun(self):
        a = analyses("المتضرره")[0]
        assert (a.fem, a.enc0) == (TAA_MARBUTA, None)
        r = reconcile_final_taa(a, "المتضرره")
        assert (r.fem, r.enc0, r.enclitics) == (None, "ه", ("ه",))
        assert _strip_diac(r.surface) == "المتضرره"

    @pytest.mark.parametrize("word", ["همزه", "مدرسته", "كتاب", "الإعراب"])
    def test_no_op_when_the_slot_already_agrees_or_is_not_the_mismatch(self, word):
        a = analyses(word)[0]
        assert reconcile_final_taa(a, word) is a

    def test_particles_and_none_pass_through(self):
        particle = MorphAnalyzer._first_valid([], "في", PREPS)
        assert reconcile_final_taa(particle, "فيه") is particle
        assert prefer_faithful(None, [], "x") is None


class TestClosedClassAlefGate:
    """A closed-class match that holds only modulo alef needs a closed-class POS."""

    def test_the_verb_kana_is_not_the_particle_kaanna(self):
        a = _dict_to_analysis(RECORDED["كان"][0], PREPS, "كان", FUNCS)
        assert a is not None and a.particle is None and (a.root, a.pattern) == ("ك#ن", "1ا3َ")
        b = _dict_to_analysis(RECORDED["وكان"][0], PREPS, "وكان", FUNCS)
        assert b.particle is None and b.proclitics == ("و",)
        c = _dict_to_analysis(RECORDED["كأن"][0], PREPS, "كأن", FUNCS)
        assert (c.particle, c.particle_kind) == ("كأن", "func")

    def test_open_class_readings_never_fold_onto_a_particle(self):
        a = _dict_to_analysis(RECORDED["آن"][0], PREPS, "آن", FUNCS)
        assert a is not None and a.particle is None        # a noun; the rooted gates decide it

    def test_dropped_hamza_still_canonicalises_for_closed_class_readings(self):
        assert _dict_to_analysis(RECORDED["الى"][0], PREPS, "الى", FUNCS).particle == "إلى"
        a = _dict_to_analysis(RECORDED["انها"][0], PREPS, "انها", FUNCS)
        assert (a.particle, a.enc0) == ("أن", "ها")

    def test_exact_spelling_beats_the_alphabetically_first_fold(self):
        a = _dict_to_analysis(RECORDED["إنما"][0], PREPS, "إنما", FUNCS)
        assert (a.particle, a.enc0) == ("إن", "ما")

    def test_migration_from_format_9_re_analyses_inexact_particles_only(self):
        tok = AraRooPatTokenizer()
        kana = CorpusEntry.from_analysis("كان", MorphAnalyzer._first_valid(
            [cam("كَأَنَّ", "كَأَنَّ", "", "", "", pos="verb_pseudo")], "كان", PREPS, FUNCS))   # the pre-fix reading
        assert kana.particle == "كأن"
        exact = CorpusEntry.from_analysis("كأن", _dict_to_analysis(RECORDED["كأن"][0], PREPS, "كأن", FUNCS))
        rooted_unfaithful = _entry("الإعراب")
        entries = [kana, exact, rooted_unfaithful, _entry("كتاب")]
        kept9, d9 = tok._reusable_cache_entries({"key": (9,) + tok._cache_key()[1:], "entries": entries})
        assert d9 == "migrate" and [e.word for e in kept9] == ["كأن", "الإعراب", "كتاب"]
        kept8, d8 = tok._reusable_cache_entries({"key": (8,) + tok._cache_key()[1:], "entries": entries})
        assert d8 == "migrate" and [e.word for e in kept8] == ["كأن", "كتاب"]


class TestNativePathWalks:
    def test_second_round_trip_only_for_unfaithful_rooted_words(self):
        bridge = _RecordedBridge()
        ma = MorphAnalyzer(bridge=bridge, enable_peeler=False)
        out = ma.analyze_many(["كتاب", "الإعراب", "همزة", "مدرسته", "الجملة", "ابعد"])
        by = dict(zip(["كتاب", "الإعراب", "همزة", "مدرسته", "الجملة", "ابعد"], out))
        assert by["الإعراب"].pattern == "إِ1ْ2ا3" and by["همزة"].fem == TAA_MARBUTA and by["ابعد"].pattern == "أَ1ْ2َ3َ"
        assert by["مدرسته"].enclitics == (TAA_MARBUTA, "ه") and by["كتاب"].pattern == "1ِ2ا3"
        tops = [r for r in bridge.requests if r[1] == 1]
        walks = [r for r in bridge.requests if r[1] > 1]
        assert len(tops) == 1 and len(walks) == 1
        assert set(walks[0][0]) == {"الإعراب", "همزة", "ابعد"}   # one batched top-32 request, faithful words excluded
        # cached: a second call issues nothing
        ma.analyze_many(["همزة", "الإعراب"])
        assert len(bridge.requests) == 2
        assert ma.analyze("همزة").fem == TAA_MARBUTA


# ---------------------------------------------------------------------------
# The reconstruction surface
# ---------------------------------------------------------------------------

def _entry(word: str, reading: Optional[Analysis] = None) -> CorpusEntry:
    return CorpusEntry.from_analysis(word, reading if reading is not None else analyses(word)[0])


class TestJoinWord:
    def test_mirrors_the_decoder(self):
        assert join_word((), "مدرس", (TAA_MARBUTA,)) == "مدرسة"
        assert join_word((), "مدرس", (TAA_MARBUTA, "ه")) == "مدرسته"
        assert join_word(("و", "ل", "ال"), "كتاب", ()) == "وللكتاب"
        assert join_word(("ب",), "كتاب", ("ها",)) == "بكتابها"


class TestEntryRealization:
    def test_written_form_wins_and_is_the_writers_spelling(self):
        # الإعرابية read as الأَعْرابِيَّة: the pair's surface comes from the chunk, not the diac.
        e = _entry("الإعرابية")
        assert e.proclitics == ("ال",) and e.enclitics == (TAA_MARBUTA,)
        assert entry_realization(e, False) == ("إعرابي", "written")
        assert entry_realization(_entry("الأعرابية"), False) == ("أعرابي", "written")

    def test_diac_fallback_when_the_written_strip_does_not_reproduce(self):
        # A ه pronoun reading on a written ة that nothing reconciled: the ه cannot be
        # stripped from the chunk, so the written strip is rejected and diac is used.
        e = _entry("تزعمتة")              # un-reconciled reading: enclitics ("ه",)
        assert e.enclitics == ("ه",)
        assert entry_realization(e, False) == ("تزعمت", "diac")
        # after reconciliation the written form reproduces
        e2 = _entry("تزعمتة", reconcile_final_taa(analyses("تزعمتة")[0], "تزعمتة"))
        assert entry_realization(e2, False) == ("تزعمت", "written")

    def test_diacritized_table_always_uses_diac(self):
        # the strip also takes the case vowel / shadda after the ة with it
        assert entry_realization(_entry("الإعرابية"), True) == ("أَعْرابِي", "diac")

    def test_taa_before_pronoun(self):
        assert entry_realization(_entry("مدرسته"), False) == ("مدرس", "written")

    def test_nothing_usable(self):
        e = CorpusEntry(word="", analyzed=True, root="كتب", pattern="1ِ2ا3", surface="")
        assert entry_realization(e, False) is None


class TestPickRealization:
    def test_written_before_diac_and_weighted_majority(self):
        assert pick_realization(Counter({"إعرابي": 3, "أعرابي": 5}), Counter({"أعرابي": 100})) == "أعرابي"
        assert pick_realization(Counter({"إعرابي": 7, "أعرابي": 5}), None) == "إعرابي"
        assert pick_realization(None, Counter({"x": 1})) == "x"
        assert pick_realization(None, None) is None

    def test_ties_break_on_the_string(self):
        assert pick_realization(Counter({"ب": 2, "ا": 2}), None) == "ا"


class _NoGen:
    def generate(self, root: str, pattern: str) -> Optional[str]:
        return None


def _tok_with_pair() -> AraRooPatTokenizer:
    tok = AraRooPatTokenizer()
    tok._backend = _NoGen()
    tok._build_vocab(Counter({"عرب": 5, "همز": 3}), Counter({"أَ1ْ2ا3ِيَّ": 5, "1َ2ْ3َ": 3}),
                     Counter({"ال": 5}), Counter({TAA_MARBUTA: 5}))
    return tok


class TestBuildReconstruction:
    def test_majority_written_form_weighted_by_occurrences(self):
        tok = _tok_with_pair()
        v = tok._vocab
        entries = [_entry("الإعرابية"), _entry("الأعرابية")]
        # by chunk types it is a tie broken on the string (أ < إ); occurrences decide
        tok._build_reconstruction(entries, Counter({"الإعرابية": 40, "الأعرابية": 3}))
        assert tok._reconstruction[(v[f"{PFX_ROOT}عرب{SFX}"], v[f"{PFX_PAT}أَ1ْ2ا3ِيَّ{SFX}"])] == "إعرابي"
        tok._build_reconstruction(entries, Counter({"الإعرابية": 2, "الأعرابية": 30}))
        assert tok._reconstruction[(v[f"{PFX_ROOT}عرب{SFX}"], v[f"{PFX_PAT}أَ1ْ2ا3ِيَّ{SFX}"])] == "أعرابي"

    def test_decode_returns_the_written_spelling(self):
        tok = _tok_with_pair()
        tok._build_reconstruction([_entry("الإعرابية"), _entry("الأعرابية")], Counter({"الإعرابية": 40, "الأعرابية": 3}))
        v = tok._vocab
        ids = [v["[CLITICP_ال]"], v[f"{PFX_ROOT}عرب{SFX}"], v[f"{PFX_PAT}أَ1ْ2ا3ِيَّ{SFX}"], v[f"{PFX_CLITICE}ة{SFX}"]]
        assert tok.decode(ids) == "الإعرابية"


# ---------------------------------------------------------------------------
# Format characters at encode
# ---------------------------------------------------------------------------

class TestFormatCharacters:
    def test_normalize_text_drops_cf_and_keeps_nfkc(self):
        assert normalize_text("​الجملة‌ صح‏يحة﻿") == "الجملة صحيحة"
        assert normalize_text("ﻻ") == normalize_text("لا")          # NFKC still applies
        assert normalize_text("كتاب") == "كتاب"

    def test_encode_has_no_unk_for_zero_width_space(self):
        tok = _tok_with_pair()
        bridge = _RecordedBridge()
        tok._backend = MorphAnalyzer(bridge=bridge, enable_peeler=False)
        plain = tok.encode("الإعرابية").input_ids
        with_zwsp = tok.encode("​الإعرابية").input_ids
        assert with_zwsp == plain and tok._vocab[TOK_UNK] not in with_zwsp


# ---------------------------------------------------------------------------
# Cache migration 8 → 9
# ---------------------------------------------------------------------------

class TestCacheMigration:
    def test_stale_only_for_rooted_unfaithful_native_entries(self):
        assert _stale_under_spelling_walk(_entry("الإعراب"))
        assert _stale_under_spelling_walk(_entry("همزة"))
        assert not _stale_under_spelling_walk(_entry("كتاب"))
        assert not _stale_under_spelling_walk(_entry("مدرسته"))
        assert not _stale_under_spelling_walk(CorpusEntry(word="x", analyzed=False))
        particle = CorpusEntry.from_analysis("فى", MorphAnalyzer._first_valid([], "في", PREPS))
        assert not _stale_under_spelling_walk(particle)
        peeled = CorpusEntry(word="أتكتب", analyzed=True, root="كتب", pattern="تَ1ْ2ُ3", surface="تَكْتُبُ", peeled=True)
        assert not _stale_under_spelling_walk(peeled)

    def test_format_8_payload_is_migrated_not_discarded(self):
        tok = AraRooPatTokenizer()
        key8 = (8,) + tok._cache_key()[1:]
        entries = [_entry("الإعراب"), _entry("كتاب"), _entry("همزة"), CorpusEntry(word="zzz", analyzed=False)]
        kept, decision = tok._reusable_cache_entries({"key": key8, "entries": entries})
        assert decision == "migrate" and [e.word for e in kept] == ["كتاب", "zzz"]
        kept9, decision9 = tok._reusable_cache_entries({"key": tok._cache_key(), "entries": entries})
        assert decision9 == "match" and len(kept9) == 4

    def test_other_keys_are_rejected(self):
        tok = AraRooPatTokenizer()
        with pytest.raises(ValueError):
            tok._reusable_cache_entries({"key": (7,) + tok._cache_key()[1:], "entries": []})
        other = AraRooPatTokenizer(prepositions=["من", "إلى"])
        with pytest.raises(ValueError):
            tok._reusable_cache_entries({"key": (8,) + other._cache_key()[1:], "entries": []})
        with pytest.raises(ValueError):
            tok._reusable_cache_entries({"key": (99,) + tok._cache_key()[1:], "entries": []})
        with pytest.raises(ValueError):
            tok._reusable_cache_entries([1, 2, 3])


# ---------------------------------------------------------------------------
# Live: the two measured cases through the real CAMeL bridge
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_analyzer():
    from pathlib import Path
    if not (Path(__file__).resolve().parents[1] / ".venv-camel").exists():
        pytest.skip(".venv-camel not set up")
    return MorphAnalyzer()


class TestLive:
    def test_hamza_seat_and_taa_marbuta(self, live_analyzer):
        a = live_analyzer.analyze("الإعراب")
        assert (a.pattern, a.proclitics) == ("إِ1ْ2ا3", ("ال",))
        h = live_analyzer.analyze("همزة")
        assert (h.fem, h.enc0, h.pattern) == (TAA_MARBUTA, None, "1َ2ْ3َ")
        assert live_analyzer.analyze("همزه").enc0 == "ه"
        assert live_analyzer.analyze("المتضرره").enc0 == "ه"
