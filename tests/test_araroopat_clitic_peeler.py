"""AraRooPat clitic peeler: clitic *combinations* the CAMeL database lacks.

calima-msa-r13 is table-driven. Verified 2026-09-13 (see the diagnosis in
docs/HANDOFF_araroopat_open_issues.md): no prefix row carries the
interrogative أ (prc3), there is no ``enc1`` feature at all (no suffix row
holds two pronouns), and the classical lengthened كمو/همو are absent. So
أنلزمكموها (Qurʾān 11:28) — and every double-object verb and every
interrogative — went to the character fallback.

The peeler strips those clitics from a closed list and sends only the
residual to CAMeL. Tiers A–C are pure Python: CAMeL's *valid readings* for
every whole word and residual consulted here (``top=32``, MLE order) were
recorded from the live bridge into tests/data/araroopat_camel_recorded.json
and are served by a fake bridge that honours ``top`` exactly like the
server, so the real ``MorphAnalyzer`` code runs. Tier D hits the live bridge
and is skipped without ``.venv-camel``. Regenerate the fixture with the
recording snippet in docs/HANDOFF_araroopat_open_issues.md §6b whenever
camel-tools or the database changes.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from arabic_eval.tokenizers.araroopat import (
    PFX_CLITICE,
    PFX_CLITICP,
    PFX_ROOT,
    SFX,
    TOK_LIT_BEGIN,
    AraRooPatTokenizer,
    join_proclitics,
)
from arabic_eval.tokenizers.araroopat_backend import (
    LENGTHENED_ENCLITICS,
    PEEL_ENC_SECOND,
    PREPOSITION_INVENTORY,
    Analysis,
    CorpusEntry,
    MorphAnalyzer,
    PeelCandidate,
    _dict_to_analysis,
    merge_peeled,
    peel_candidates,
    peel_compatible,
)

RECORDED: Dict[str, List[Dict[str, str]]] = json.loads(
    (Path(__file__).parent / "data" / "araroopat_camel_recorded.json").read_text("utf-8")
)


class _RecordedBridge:
    """Serves the recorded CAMeL readings; unknown words get no analysis.

    Mirrors the server contract: ``top`` caps the readings per word and
    defaults to 1, so the native path sees exactly one candidate.
    """

    def analyze(self, words: List[str], top: int = 1) -> List[List[Dict[str, str]]]:
        return [RECORDED.get(w, [])[:max(1, top)] for w in words]

    def generate(self, root: str, pattern: str) -> Optional[str]:
        return None


def _analyzer(peeler: bool = True) -> MorphAnalyzer:
    return MorphAnalyzer(bridge=_RecordedBridge(), enable_peeler=peeler)


# ---------------------------------------------------------------------------
# The stress set: (word, proclitics, residual stem, enclitics, note).
# The canonical failing word is first. Enclitics are in emission order and
# follow the tokenizer's conventions: ة is its own enclitic and the 1sg
# object ني keeps its ن in the pattern (CAMeL: 1s_dobj → ي), as it does for
# every natively analysed word (أعطيتني).
# ---------------------------------------------------------------------------

STRESS: List[Tuple[str, Tuple[str, ...], str, Tuple[str, ...], str]] = [
    # A. interrogative أ — r13 has no prc3 prefix row at all
    ("أنلزمكموها", ("أ",), "نلزم", ("كمو", "ها"), "Qurʾān 11:28 — the canonical case"),
    ("أتكتب", ("أ",), "تكتب", (), "أ + verb"),
    ("أفتكتبه", ("أ", "ف"), "تكتب", ("ه",), "أ + ف + verb + object"),
    ("أوتكتبها", ("أ", "و"), "تكتب", ("ها",), "أ + و + verb + object"),
    ("أبالكتاب", ("أ", "ب", "ال"), "كتاب", (), "أ + ب + ال + noun"),
    ("أفبالكتاب", ("أ", "ف", "ب", "ال"), "كتاب", (), "four proclitics"),
    ("أوللكتاب", ("أ", "و", "ل", "ال"), "كتاب", (), "لل contraction under أ"),
    ("أفتعلمونها", ("أ", "ف"), "تعلمون", ("ها",), "أ + ف + 2mp verb + object"),
    ("أوبكتابكم", ("أ", "و", "ب"), "كتاب", ("كم",), "أ + و + ب + noun + possessive"),
    ("أفبمدرستكم", ("أ", "ف", "ب"), "مدرست", ("ة", "كم"), "ة realised as ت before the pronoun"),
    # B. two object pronouns — r13 has no enc1 feature
    ("أنلزمكمها", ("أ",), "نلزم", ("كم", "ها"), "MSA spelling of the canonical case"),
    ("نلزمكموها", (), "نلزم", ("كمو", "ها"), "lengthened, no hamza"),
    ("نلزمكمها", (), "نلزم", ("كم", "ها"), "double object, MSA"),
    ("أعطيتكه", (), "أعطيت", ("ك", "ه"), "2ms + 3ms"),
    ("سلمتكها", (), "سلمت", ("ك", "ها"), "2ms + 3fs"),
    ("أعطيناكموها", (), "أعطينا", ("كمو", "ها"), "1p subject + lengthened"),
    ("أعطيتنيه", (), "أعطيت", ("ني", "ه"), "1s + 3ms"),
    ("يعطيكموه", (), "يعطي", ("كمو", "ه"), "imperfect + lengthened"),
    ("سأعطيكها", ("س",), "أعطي", ("ك", "ها"), "س + double object"),
    ("فسنلزمكموها", ("ف", "س"), "نلزم", ("كمو", "ها"), "ف + س + lengthened"),
    ("لأعطينكموها", ("ل",), "أعطين", ("كمو", "ها"), "emphatic ل + energetic + lengthened"),
    ("لأحدثنهموه", ("ل",), "أحدثن", ("همو", "ه"), "corpus word: lengthened همو"),
    # Need a reading below rank 1 of the residual (server top>1, P7 fix)
    ("ألزمناهموها", (), "ألزمنا", ("همو", "ها"), "residual ألزمنا: PV + SUBJ:1P is reading #6"),
    ("وسأعطيكموه", ("و", "س"), "أعطي", ("كمو", "ه"), "residual وسأعطي: active أُعْطِي is reading #4"),
    # C. controls CAMeL handles natively — must be byte-identical to before
    ("وبالكتاب", ("و", "ب", "ال"), "كتاب", (), "control"),
    ("فبكتابه", ("ف", "ب"), "كتاب", ("ه",), "control"),
    ("وللمدرسة", ("و", "ل", "ال"), "مدرس", ("ة",), "control"),
    ("فكالشمس", ("ف", "ك", "ال"), "شمس", (), "control"),
    ("وبمدرستهم", ("و", "ب"), "مدرست", ("ة", "هم"), "control"),
    ("وكتابكما", ("و",), "كتاب", ("كما",), "control"),
    ("فلكتابهن", ("ف", "ل"), "كتاب", ("هن",), "control"),
    ("سنكتبها", ("س",), "نكتب", ("ها",), "control"),
    ("وسيكتبونها", ("و", "س"), "يكتبون", ("ها",), "control"),
    ("فسيعطيكم", ("ف", "س"), "يعطي", ("كم",), "control"),
    ("أعطيتموها", (), "أعطيتمو", ("ها",), "control: تموها IS a suffix row"),
    ("كتابهما", (), "كتاب", ("هما",), "control"),
    ("يدرسها", (), "يدرس", ("ها",), "control"),
]
PEELED_WORDS = [w for w, *_ in STRESS[:24]]
CONTROL_WORDS = [w for w, *_ in STRESS[24:]]
# CAMeL reads this natively in a way the peeler must not override.
NATIVE_AMBIGUOUS = "أستكتبه"      # Form X استكتب, a legitimate reading
# For an unseen word the MLE model scores every reading 1.0, so rank 1 is
# database order. These two need a lower-ranked reading of the residual:
# the peeler fetches every valid reading (top=32) and walks them in order.
NEEDS_LOWER_READING = {
    "ألزمناهموها": ("ألزمنا", 6, "noun + 1p_poss and PV + 1p_dobj come first"),
    "وسأعطيكموه": ("وسأعطي", 4, "three passive وَسَأُعْطَى readings come first — surface guard rejects them"),
}


def _pieces(a: Analysis) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    return a.proclitics, a.enclitics


# ---------------------------------------------------------------------------
# Tier A — the peeler alone (no analyzer)
# ---------------------------------------------------------------------------

class TestCandidates:
    def test_canonical_case_first(self):
        cands = peel_candidates("أنلزمكموها")
        assert PeelCandidate(("أ",), "نلزم", ("كمو", "ها")) in cands
        # Least peeled first: the canonical full slicing is the last resort.
        assert cands[0] == PeelCandidate(("أ",), "نلزمكموها", ())
        assert cands[-1] == PeelCandidate(("أ",), "نلزم", ("كمو", "ها"))

    def test_least_peeled_first(self):
        cands = peel_candidates("أبالكتاب")
        assert [c.proclitics for c in cands] == [("أ",), ("أ", "ب"), ("أ", "ب", "ال")]

    @pytest.mark.parametrize("word,stack", [
        ("أكتب", ("أ",)), ("أفكتب", ("أ", "ف")), ("أوكتب", ("أ", "و")),
        ("أسكتب", ("أ", "س")), ("أفسكتب", ("أ", "ف", "س")), ("أوبكتب", ("أ", "و", "ب")),
        ("أفبالكتب", ("أ", "ف", "ب", "ال")), ("أوللكتب", ("أ", "و", "ل", "ال")),
        ("للكتب", ("ل", "ال")), ("بالكتب", ("ب", "ال")),
    ])
    def test_proclitic_grammar_accepts(self, word, stack):
        assert stack in {c.proclitics for c in peel_candidates(word)}

    @pytest.mark.parametrize("word,stack", [
        ("فأكتب", ("ف", "أ")),        # أ is outermost
        ("الكتاب", ("ا", "ل")),       # ال is one clitic
        ("وفكتب", ("و", "ف")),        # two conjunctions
        ("سالكتب", ("س", "ال")),      # س then ال is not a stack
        ("بسكتب", ("ب", "س")),        # one prc1 slot
    ])
    def test_proclitic_grammar_rejects(self, word, stack):
        assert stack not in {c.proclitics for c in peel_candidates(word)}

    @pytest.mark.parametrize("word,stack", [
        ("كتبكها", ("ك", "ها")), ("كتبكه", ("ك", "ه")), ("كتبنيه", ("ني", "ه")),
        ("كتبكموها", ("كمو", "ها")), ("كتبكموه", ("كمو", "ه")), ("كتبهموها", ("همو", "ها")),
        ("كتبها", ("ها",)), ("كتبكم", ("كم",)), ("كتبي", ("ي",)),
    ])
    def test_enclitic_grammar_accepts(self, word, stack):
        assert stack in {c.enclitics for c in peel_candidates(word)}

    @pytest.mark.parametrize("word,stack", [
        ("كتبكهاهم", ("ك", "ها", "هم")),   # at most two
        ("كتبهك", ("ه", "ك")),             # second pronoun is 3rd person only
        ("كتبناني", ("نا", "ني")),         # "us me": 1st person cannot follow
        ("كتبيه", ("ي", "ه")),             # 1sg object is ني, not the possessive ي
        ("كتبكمو", ("كمو",)),              # lengthened only before a pronoun
    ])
    def test_enclitic_grammar_rejects(self, word, stack):
        assert stack not in {c.enclitics for c in peel_candidates(word)}
        assert all(len(c.enclitics) <= 2 for c in peel_candidates(word))

    def test_second_slot_is_third_person(self):
        assert set(PEEL_ENC_SECOND) == {"ه", "ها", "هم", "هن", "هما"}
        assert LENGTHENED_ENCLITICS == ("كمو", "همو")

    @pytest.mark.parametrize("word", ["أب", "وك", "بها", "أ", "ه", "كم"])
    def test_residual_never_too_short(self, word):
        assert all(len(c.residual) >= 2 for c in peel_candidates(word))
        if len(word) <= 2:
            assert peel_candidates(word) == []

    @pytest.mark.parametrize("word", [w for w, *_ in STRESS])
    def test_reversible_by_construction(self, word):
        for c in peel_candidates(word):
            assert join_proclitics(list(c.proclitics)) + c.residual + "".join(c.enclitics) == word


# ---------------------------------------------------------------------------
# Tier B — MorphAnalyzer with the recorded bridge
# ---------------------------------------------------------------------------

class TestAnalyzer:
    @pytest.mark.parametrize("word,procs,stem,encs,note", STRESS, ids=[w for w, *_ in STRESS])
    def test_stress_set_exact_segmentation(self, word, procs, stem, encs, note):
        a = _analyzer().analyze(word)
        assert a is not None, f"{word} ({note}) fell to LIT"
        assert _pieces(a) == (procs, encs), note
        assert a.root and a.pattern
        assert a.peeled == (word in PEELED_WORDS)

    def test_batched_path_matches_single(self):
        words = [w for w, *_ in STRESS] + [NATIVE_AMBIGUOUS, "بيرنيني"]
        single = [_analyzer().analyze(w) for w in words]
        batched = _analyzer().analyze_many(words)
        assert [(a and _pieces(a), a and a.peeled) for a in single] \
            == [(a and _pieces(a), a and a.peeled) for a in batched]

    def test_native_reading_is_never_overridden(self):
        a = _analyzer().analyze(NATIVE_AMBIGUOUS)
        assert a is not None and not a.peeled
        assert a.pattern == "أَسْتَ1ْ2ِ3" and a.enclitics == ("ه",)   # Form X استكتب + ه

    def test_controls_identical_with_peeler_off(self):
        on, off = _analyzer(True), _analyzer(False)
        for w in CONTROL_WORDS + [NATIVE_AMBIGUOUS]:
            assert on.analyze(w) == off.analyze(w)

    def test_peeler_off_sends_class_to_lit(self):
        off = _analyzer(False)
        assert all(off.analyze(w) is None for w in PEELED_WORDS)

    @pytest.mark.parametrize("word", sorted(NEEDS_LOWER_READING))
    def test_lower_ranked_residual_readings_are_walked(self, word):
        residual, rank, why = NEEDS_LOWER_READING[word]
        readings = RECORDED[residual]
        assert len(readings) >= rank, why
        # With only the top reading (the pre-P7 behaviour) the word is LIT...
        ma = _analyzer()
        ma.RESIDUAL_TOP = 1
        assert ma.analyze(word) is None, why
        # ...with every reading it resolves, on exactly the expected one.
        ma = _analyzer()
        a = ma.analyze(word)
        assert a is not None and a.peeled
        chosen = next(r for r in ma._candidates_cache[residual]
                      if r.pattern == a.pattern and r.pos == a.pos and r.surface in a.surface)
        assert ma._candidates_cache[residual].index(chosen) == rank - 1

    def test_native_path_still_sees_one_candidate(self):
        # Widening `top` is scoped to residuals: the whole-word path keeps
        # its top-1 semantics so native admission does not change (P7b).
        seen = []
        class _Spy(_RecordedBridge):
            def analyze(self, words, top=1):
                seen.append(top)
                return super().analyze(words, top)
        ma = MorphAnalyzer(bridge=_Spy())
        ma.analyze("كتاب")
        ma.analyze("أنلزمكموها")
        assert seen[0] == 1 and seen[1] == 1 and seen[2] == MorphAnalyzer.RESIDUAL_TOP

    @pytest.mark.parametrize("word,why", [
        ("بيرنيني", "ب + يرني + ني — ب needs a nominal, not a verb"),
        ("أندرسون", "transliteration; no residual analysis"),
        ("المجتهدي", "ال + المجتهد + ي — article and possessive are incompatible, and "
                    "CAMeL models ال + one pronoun natively: a native miss is informative"),
        ("عدناني", "عدّ + نا + ني — 'us me': second pronoun must be 3rd person"),
        ("أوديس", "أ + و + ديس — a peeled conjunction does not license أ on a bare noun"),
    ])
    def test_wrong_peels_are_rejected(self, word, why):
        assert _analyzer().analyze(word) is None, why

    def test_surface_mismatch_is_rejected(self):
        # CAMeL reads the residual قرضة as قرض + ه (its ة/ه normalisation);
        # accepting it would decode القرضة as القرضه.
        assert _analyzer().analyze("القرضة") is None

    def test_bare_alef_interrogative_is_opt_in(self):
        # Under alef normalisation the interrogative surfaces as ا. Off by
        # default: word-initial ا is hamzat-wasl / article far more often,
        # and enabling it doubled the false-peel rate (المسا → ا + لمس).
        assert _analyzer().analyze("اتكتب") is None
        on = MorphAnalyzer(bridge=_RecordedBridge(), peel_bare_alef=True)
        a = on.analyze("اتكتب")
        assert a is not None and a.proclitics == ("ا",) and a.root == "كتب"
        # Even when on, a lone ا followed by ل is the article, never the interrogative.
        assert all("ا" not in c.proclitics for c in peel_candidates("المسا", bare_alef=True))
        assert any(c.proclitics == ("ا",) for c in peel_candidates("اتكتب", bare_alef=True))
        assert all("ا" not in c.proclitics for c in peel_candidates("اتكتب"))
        assert on.analyze("المسا") is None

    def test_known_false_peel_class_is_documented(self):
        # أوهم (root و-ه-م) is missing from the r13 lexicon, so أوهموه
        # ("they deluded him") is read as أ + و + همّوا + ه — itself a
        # well-formed parse. Pinned so a change in behaviour is noticed.
        a = _analyzer().analyze("أوهموه")
        assert a is not None and a.peeled and a.proclitics == ("أ", "و")

    def test_peel_is_logged_and_counted(self, caplog):
        ma = _analyzer()
        with caplog.at_level(logging.DEBUG, logger="arabic_eval.tokenizers.araroopat.backend"):
            ma.analyze("أنلزمكموها")
            ma.analyze("أندرسون")
        msgs = [r.getMessage() for r in caplog.records if "clitic peeler" in r.getMessage()]
        assert any("أنلزمكموها" in m and "كمو+ها" in m for m in msgs)
        assert any("أندرسون" in m and "character path" in m for m in msgs)
        assert ma.peel_stats == {"peeled": 1, "exhausted": 1}

    def test_merge_keeps_camel_inner_clitics_and_rebuilds_surface(self):
        ma = _analyzer()
        residual = ma._native_many(["بالكتاب"])[0]
        cand = PeelCandidate(("أ",), "بالكتاب", ())
        assert peel_compatible(cand, residual)
        merged = merge_peeled("أبالكتاب", cand, residual)
        assert merged.proclitics == ("أ", "ب", "ال") and merged.surface.startswith("أ")
        assert merged.pattern == residual.pattern and merged.peeled

    def test_corpus_entry_round_trips_peeled_flag(self):
        a = _analyzer().analyze("سلمتكها")
        e = CorpusEntry.from_analysis("سلمتكها", a)
        assert e.peeled and e.enclitics == ("ك", "ها")
        assert CorpusEntry.from_dict(e.to_dict()) == e
        assert CorpusEntry.from_dict({"word": "x", "analyzed": False}).peeled is False


# ---------------------------------------------------------------------------
# Tier C — encode / decode through the tokenizer
# ---------------------------------------------------------------------------

def _stub_tokenizer(words: List[str], drop_tokens: Tuple[str, ...] = ()) -> AraRooPatTokenizer:
    """Train-equivalent build from the recorded analyses of ``words``."""
    tok = AraRooPatTokenizer(min_root_freq=1, min_pattern_freq=1)
    tok._backend = _analyzer()
    entries = [CorpusEntry.from_analysis(w, a) for w, a in zip(words, tok._backend.analyze_many(words))]
    root_freq, pat_freq, proc_freq, enc_freq = Counter(), Counter(), Counter(), Counter()
    for e in entries:
        if not e.analyzed:
            continue
        root_freq[e.root] += 1
        pat_freq[e.pattern] += 1
        for c in e.proclitics:
            proc_freq[c] += 1
        for c in e.enclitics:
            enc_freq[c] += 1
    for t in drop_tokens:
        enc_freq.pop(t, None)
    tok._build_vocab(root_freq, pat_freq, proc_freq, enc_freq)
    tok._build_reconstruction(entries)
    return tok


ALL_WORDS = [w for w, *_ in STRESS]


@pytest.fixture(scope="module")
def tok() -> AraRooPatTokenizer:
    return _stub_tokenizer(ALL_WORDS)


def _stream(tok: AraRooPatTokenizer, text: str) -> List[str]:
    return [tok._reverse_vocab[i] for i in tok.encode(text).input_ids][1:-1]


class TestEncodeDecode:
    def test_canonical_token_stream(self, tok):
        assert _stream(tok, "أنلزمكموها") == [
            "[CLITICP_أ]", "[ROOT_لزم]", "[PAT_نَ1ْ2َ3]", "[CLITICE_كمو]", "[CLITICE_ها]",
        ]

    def test_double_object_stream(self, tok):
        pat = tok._backend.analyze("سلمتكها").pattern
        assert _stream(tok, "سلمتكها") == [
            "[ROOT_سلم]", f"[PAT_{pat}]", "[CLITICE_ك]", "[CLITICE_ها]",
        ]

    def test_fem_before_pronoun_under_hamza(self, tok):
        assert _stream(tok, "أفبمدرستكم") == [
            "[CLITICP_أ]", "[CLITICP_ف]", "[CLITICP_ب]", "[ROOT_درس]", "[PAT_مَ1ْ2َ3َ]",
            "[CLITICE_ة]", "[CLITICE_كم]",
        ]

    @pytest.mark.parametrize("text", ALL_WORDS + [
        "أنلزمكموها وأنتم لها كارهون",
        "سلمتكها أمس ، أفتعلمونها ؟",
        NATIVE_AMBIGUOUS, "بيرنيني",
    ])
    def test_roundtrip(self, tok, text):
        assert tok.decode(tok.encode(text).input_ids) == text

    def test_lengthened_forms_are_tokens_not_normalised(self, tok):
        assert f"{PFX_CLITICE}كمو{SFX}" in tok._vocab
        assert f"{PFX_CLITICE}همو{SFX}" in tok._vocab
        # Never rewritten to كم/هم: the surface is what decodes.
        assert tok.decode(tok.encode("يعطيكموه").input_ids) == "يعطيكموه"
        assert "[CLITICE_كمو]" in _stream(tok, "يعطيكموه")

    def test_oov_clitic_sends_whole_word_to_lit_and_still_roundtrips(self):
        small = _stub_tokenizer(ALL_WORDS, drop_tokens=("كمو",))
        assert f"{PFX_CLITICE}كمو{SFX}" not in small._vocab
        stream = _stream(small, "أنلزمكموها")
        assert stream[0] == TOK_LIT_BEGIN and PFX_ROOT not in "".join(stream)
        assert small.decode(small.encode("أنلزمكموها").input_ids) == "أنلزمكموها"
        # ...while the MSA spelling, whose clitics are all in vocab, is analysed.
        assert _stream(small, "أنلزمكمها")[0] == f"{PFX_CLITICP}أ{SFX}"

    def test_metric_strings(self, tok):
        # ROOT carries the root letters and PAT the inflected residual (so
        # ROOT+PAT words never concatenate back — same as every native word);
        # the clitic tokens carry their surface.
        out = tok.encode("أنلزمكموها")
        assert out.tokens[1:-1] == ["أ", "لزم", "نلزم", "كمو", "ها"]

    def test_cache_key_changed_and_flags_persist(self, tmp_path, tok):
        assert AraRooPatTokenizer._CACHE_FORMAT >= 4
        assert tok._cache_key()[2:] == (True, False)
        tok.save(tmp_path)
        loaded = AraRooPatTokenizer()
        loaded.load(tmp_path)
        assert loaded.clitic_peeler is True and loaded.peel_bare_alef is False
        assert AraRooPatTokenizer(clitic_peeler=False)._cache_key() != tok._cache_key()
        assert AraRooPatTokenizer(peel_bare_alef=True)._cache_key() != tok._cache_key()


# ---------------------------------------------------------------------------
# Tier D — live CAMeL bridge (skipped without .venv-camel)
# ---------------------------------------------------------------------------

_CAMEL_VENV = Path(__file__).resolve().parents[1] / ".venv-camel" / "bin" / "python"


@pytest.mark.skipif(not _CAMEL_VENV.exists(), reason="needs .venv-camel with camel-tools")
class TestLiveCamel:
    @pytest.fixture(scope="class")
    def live(self) -> MorphAnalyzer:
        return MorphAnalyzer()

    @pytest.mark.parametrize("word,procs,stem,encs,note", STRESS, ids=[w for w, *_ in STRESS])
    def test_live_stress_set(self, live, word, procs, stem, encs, note):
        a = live.analyze(word)
        assert a is not None, f"{word} ({note}) fell to LIT on the live bridge"
        assert _pieces(a) == (procs, encs), note

    def test_live_recorded_readings_are_current(self, live):
        # Guards the fixture: if a camel-tools/DB upgrade changes any answer
        # the recorded file must be regenerated (see the module docstring).
        words = sorted(RECORDED)
        fresh = live._bridge.analyze(words, top=MorphAnalyzer.RESIDUAL_TOP)
        particles = live.particles
        for w, cands in zip(words, fresh):
            valid = [d for d in cands if _dict_to_analysis(d, particles, w) is not None]
            assert valid == RECORDED[w], w

    def test_live_reading_order_is_deterministic_across_processes(self):
        # The MLE model scores every reading of an unseen word 1.0 and the
        # analyzer's order behind the tie follows string hashing, which
        # differs per process. The server sorts ties on content so the same
        # word always yields the same top-1 and the same residual walk.
        import subprocess, sys as _sys
        code = (
            "import sys, json; sys.path.insert(0, 'src'); import logging; logging.disable(logging.CRITICAL)\n"
            "from arabic_eval.tokenizers.araroopat_bridge import get_shared_bridge\n"
            "r = get_shared_bridge().analyze(['ومال', 'ألزمنا', 'أعطي', 'كتاب'], top=32)\n"
            "print(json.dumps([[d['diac'] + '|' + d['pattern'] + '|' + d['enc0'] for d in x] for x in r], ensure_ascii=False))\n"
        )
        runs = {
            subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parents[1]).stdout.strip().splitlines()[-1]
            for _ in range(2)
        }
        assert len(runs) == 1

    def test_live_server_honours_top(self, live):
        one = live._bridge.analyze(["ألزمنا"])[0]
        many = live._bridge.analyze(["ألزمنا"], top=MorphAnalyzer.RESIDUAL_TOP)[0]
        assert len(one) == 1 and len(many) > 1 and many[0] == one[0]
