"""AraRooPat proper nouns between [PROP_BEGIN] / [PROP_END] (added 2026-09-16).

A word whose CAMeL reading is a *database* ``noun_prop`` is a proper noun:
names without a root (NTWS — باريس, كوريا, أكتوبر, بن) are emitted as their
characters between the two markers with the clitics outside; names with a
root (محمد, مصر, القاهرة) keep ROOT+PAT under the default ``proper_nouns:
unrooted`` mode and take the markers under ``all``. CAMeL's NOAN_PROP
backoff (root ``O``, pattern ``backoff``) stamps ``noun_prop`` on every
out-of-vocabulary word and is *not* a name. Pure-Python, stub analyzer,
real bridge dicts captured on 2026-09-16.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Optional

import pytest

from arabic_eval.tokenizers.araroopat import (
    PFX_CHAR,
    PFX_CLITICE,
    PFX_CLITICP,
    PFX_PAT,
    PFX_ROOT,
    PROPER_NOUN_MODES,
    SFX,
    TOK_LIT_BEGIN,
    TOK_LIT_END,
    TOK_PROP_BEGIN,
    TOK_PROP_END,
    AraRooPatTokenizer,
)
from arabic_eval.tokenizers.araroopat_backend import (
    FUNC_INVENTORY,
    PREPOSITION_INVENTORY,
    Analysis,
    CorpusEntry,
    MorphAnalyzer,
    PeelCandidate,
    _dict_to_analysis,
    is_db_proper,
    peel_compatible,
)
from arabic_eval.tokenizers.araroopat_corpus_trace import categorize_analysis, prepass_path

PREPS = frozenset(PREPOSITION_INVENTORY)
FUNCS = frozenset(FUNC_INVENTORY)


def cam(lex: str, diac: str, root: str, pattern: str, pos: str = "noun_prop", **clitics: str) -> Dict[str, str]:
    d = {"lex": lex, "diac": diac, "root": root, "pattern": pattern, "pos": pos,
         "stem": "", "prc3": "0", "prc2": "0", "prc1": "0", "prc0": "0", "enc0": "0"}
    d.update(clitics)
    return d


CAMEL: Dict[str, list] = {
    # rootless (NTWS) names
    "باريس": [cam("بارِيس", "بارِيس", "NTWS", "NTWS")],
    "بن": [cam("بِن", "بِن", "NTWS", "NTWS")],
    "كوريا": [cam("كُورِيا", "كُورِيا", "NTWS", "NTWS")],
    "وسوريا": [cam("سُورِيا", "وَسُورِيا", "NTWS", "وَNTWS", prc2="wa_conj")],
    "بباريس": [cam("بارِيس", "بِبارِيس", "NTWS", "بِNTWS", prc1="bi_prep")],
    "لكوريا": [cam("كُورِيا", "لِكُورِيا", "NTWS", "لِNTWS", prc1="li_prep")],
    "أبي": [cam("أَبِي", "أَبِي", "NTWS", "NTWS")],
    "إسرائيل": [cam("إِسْرائِيل", "إِسْرائِيلَ", "NTWS", "NTWSَ")],
    # a rootless name whose surface ends in ة
    "فاطمه": [cam("فاطِمَه", "فاطِمَه", "NTWS", "NTWS")],
    "أنقرة": [cam("أَنْقَرَة", "أَنْقَرَة", "NTWS", "NTWS")],
    "بأنقرة": [cam("أَنْقَرَة", "بِأَنْقَرَة", "NTWS", "بِNTWS", prc1="bi_prep")],
    # rooted names
    "محمد": [cam("مُحَمَّد", "مُحَمَّد", "ح.م.د", "مُ1َ2َّ3")],
    "ومحمد": [cam("مُحَمَّد", "وَمُحَمَّد", "ح.م.د", "وَمُ1َ2َّ3", prc2="wa_conj")],
    "مصر": [cam("مِصْر", "مِصْر", "م.ص.ر", "1ِ2ْ3"), cam("مُصِرّ", "مُصِرَّ", "ص.ر.ر", "مُ1ِ2َّ", pos="adj")],
    "بمكة": [cam("مَكَّة", "بِمَكَّة", "م.ك.ك", "بِ1َ2َّة", prc1="bi_prep")],
    "للقاهرة": [cam("قاهِرَة", "لِلقاهِرَةِ", "ق.ه.ر", "لِل1ا2ِ3َةِ", prc1="li_prep", prc0="Al_det")],
    # backoff guesses (out-of-vocabulary words): not names
    "تويتر": [cam("تويتر", "تويتر", "O", "backoff")],
    "معيلات": [cam("معيلات", "معيلات", "O", "backoff")],
    # the deferral: a rootless name yields to a clitic-only reading, not to a root reading
    "بك": [cam("بَك", "بَك", "NTWS", "NTWS"), cam("بِ", "بِكَ", "ب", "1", pos="prep", prc0="na", enc0="2ms_pron")],
    "باريس2": [cam("بارِيس", "بارِيس", "NTWS", "NTWS"), cam("أَرِيس", "بِأَرِيس", "#.ر.س", "بِأَ2ِي3", pos="noun", prc1="bi_prep")],
    # ordinary words
    "كتاب": [cam("كِتاب", "كِتاب", "ك.ت.ب", "1ِ2ا3", pos="noun")],
    "الكتاب": [cam("كِتاب", "الكِتاب", "ك.ت.ب", "ال1ِ2ا3", pos="noun", prc0="Al_det")],
}


def analyze(word: str) -> Optional[Analysis]:
    return MorphAnalyzer._first_valid(CAMEL[word], word, PREPS, FUNCS)


class TestProperGate:
    def test_database_noun_prop_is_recognized(self):
        assert is_db_proper(CAMEL["باريس"][0]) and is_db_proper(CAMEL["محمد"][0])
        assert not is_db_proper(CAMEL["تويتر"][0])          # backoff guess
        assert not is_db_proper(CAMEL["كتاب"][0])           # noun

    @pytest.mark.parametrize("word,procs", [("باريس", ()), ("بن", ()), ("وسوريا", ("و",)), ("بباريس", ("ب",)),
                                            ("لكوريا", ("ل",)), ("أبي", ()), ("إسرائيل", ())])
    def test_rootless_name(self, word, procs):
        a = analyze(word)
        assert a is not None and a.proper and a.root == "" and a.pattern == "" and a.particle is None
        assert a.proclitics == procs and a.pos == "noun_prop"

    def test_rootless_name_factors_its_taa_marbuta(self):
        a = analyze("أنقرة")
        assert a.proper and a.fem == "ة" and a.enclitics == ("ة",)
        a = analyze("بأنقرة")
        assert a.proper and a.proclitics == ("ب",) and a.fem == "ة"
        assert analyze("فاطمه").fem is None       # ه, not ة

    def test_rooted_name_keeps_its_root_and_the_flag(self):
        a = analyze("محمد")
        assert a.proper and (a.root, a.pattern) == ("حمد", "مُ1َ2َّ3")
        a = analyze("للقاهرة")
        assert a.proper and a.root == "قهر" and a.proclitics == ("ل", "ال") and a.fem == "ة"

    def test_backoff_guess_is_not_a_name(self):
        assert analyze("تويتر") is None and analyze("معيلات") is None
        assert _dict_to_analysis(CAMEL["تويتر"][0], PREPS, "تويتر", FUNCS) is None

    def test_ordinary_words_are_untouched(self):
        assert analyze("كتاب").proper is False and analyze("الكتاب").root == "كتب"

    def test_was_lit_before(self):
        """Without the proper flag an NTWS reading is rejected — the pre-2026-09-16 behaviour."""
        d = dict(CAMEL["باريس"][0]); d["pos"] = "noun"
        assert _dict_to_analysis(d, PREPS, "باريس", FUNCS) is None

    def test_rootless_name_yields_only_to_a_closed_class_reading(self):
        a = analyze("بك")
        assert a.clitic_only and not a.proper                      # بِ + ك wins
        a = analyze("باريس2")
        assert a.proper and a.root == ""                            # the بِ + أَرِيس root reading does not


class TestPeelerInteraction:
    def test_no_peel_onto_a_name(self):
        name = analyze("باريس")
        assert peel_compatible(PeelCandidate(proclitics=("أ",), residual="باريس", enclitics=()), name) is False
        rooted = analyze("محمد")
        assert peel_compatible(PeelCandidate(proclitics=("أ",), residual="محمد", enclitics=("ه",)), rooted) is False


class _StubAnalyzer:
    def analyze(self, word: str) -> Optional[Analysis]:
        return MorphAnalyzer._first_valid(CAMEL.get(word, []), word, PREPS, FUNCS)

    def generate(self, root: str, pattern: str) -> Optional[str]:
        return None


def _stub_tokenizer(**kwargs) -> AraRooPatTokenizer:
    tok = AraRooPatTokenizer(**kwargs)
    tok._backend = _StubAnalyzer()
    tok._build_vocab(
        root_freq=Counter({"كتب": 5, "حمد": 3, "قهر": 2, "مكك": 2}),
        pat_freq=Counter({"1ِ2ا3": 5, "مُ1َ2َّ3": 3, "1ا2ِ3َ": 2, "1َ2َّ": 2}),
        proclitic_freq=Counter({"و": 9, "ال": 8, "ل": 7, "ب": 6}),
        enclitic_freq=Counter({"ه": 9, "ها": 8, "ك": 3}),
    )
    v = tok._vocab
    tok._reconstruction = {
        (v[f"{PFX_ROOT}كتب{SFX}"], v[f"{PFX_PAT}1ِ2ا3{SFX}"]): "كتاب",
        (v[f"{PFX_ROOT}حمد{SFX}"], v[f"{PFX_PAT}مُ1َ2َّ3{SFX}"]): "محمد",
        (v[f"{PFX_ROOT}قهر{SFX}"], v[f"{PFX_PAT}1ا2ِ3َ{SFX}"]): "قاهر",
        (v[f"{PFX_ROOT}مكك{SFX}"], v[f"{PFX_PAT}1َ2َّ{SFX}"]): "مك",
    }
    return tok


def _stream(tok, text):
    out = tok.encode(text)
    return out, [tok._reverse_vocab[i] for i in out.input_ids if tok._reverse_vocab[i] not in ("<s>", "</s>")]


class TestVocabLayout:
    def test_markers_follow_the_literal_markers(self):
        tok = _stub_tokenizer()
        assert (tok._vocab[TOK_LIT_BEGIN], tok._vocab[TOK_LIT_END]) == (4, 5)
        assert (tok._vocab[TOK_PROP_BEGIN], tok._vocab[TOK_PROP_END]) == (6, 7)
        assert min(i for t, i in tok._vocab.items() if t.startswith(PFX_CLITICP)) == 8

    def test_mode_is_validated_and_persisted(self, tmp_path):
        assert PROPER_NOUN_MODES == ("unrooted", "all")
        with pytest.raises(ValueError):
            AraRooPatTokenizer(proper_nouns="names")
        tok = _stub_tokenizer(proper_nouns="all")
        tok.save(tmp_path)
        loaded = AraRooPatTokenizer()
        loaded.load(tmp_path)
        assert loaded.proper_nouns == "all"

    def test_cache_format_bumped(self):
        assert AraRooPatTokenizer._CACHE_FORMAT >= 8


class TestEncodeDecode:
    def test_rootless_name_between_markers(self):
        tok = _stub_tokenizer()
        _, stream = _stream(tok, "باريس")
        assert stream == [TOK_PROP_BEGIN] + [f"{PFX_CHAR}{c}{SFX}" for c in "باريس"] + [TOK_PROP_END]

    def test_clitics_outside(self):
        tok = _stub_tokenizer()
        _, stream = _stream(tok, "بباريس")
        assert stream[:2] == [f"{PFX_CLITICP}ب{SFX}", TOK_PROP_BEGIN] and stream[-1] == TOK_PROP_END
        _, stream = _stream(tok, "بأنقرة")
        assert stream[0] == f"{PFX_CLITICP}ب{SFX}" and stream[-2:] == [TOK_PROP_END, f"{PFX_CLITICE}ة{SFX}"]
        assert f"{PFX_CHAR}ة{SFX}" not in stream

    def test_rooted_name_by_mode(self):
        tok = _stub_tokenizer()                       # unrooted (default)
        _, stream = _stream(tok, "ومحمد")
        assert stream == [f"{PFX_CLITICP}و{SFX}", f"{PFX_ROOT}حمد{SFX}", f"{PFX_PAT}مُ1َ2َّ3{SFX}"]
        tok = _stub_tokenizer(proper_nouns="all")
        _, stream = _stream(tok, "ومحمد")
        assert stream == [f"{PFX_CLITICP}و{SFX}", TOK_PROP_BEGIN] + [f"{PFX_CHAR}{c}{SFX}" for c in "محمد"] + [TOK_PROP_END]
        _, stream = _stream(tok, "للقاهرة")
        assert stream[:3] == [f"{PFX_CLITICP}ل{SFX}", f"{PFX_CLITICP}ال{SFX}", TOK_PROP_BEGIN]
        assert stream[-2:] == [TOK_PROP_END, f"{PFX_CLITICE}ة{SFX}"]

    def test_budget_cut_name_falls_to_prop_not_lit(self):
        tok = _stub_tokenizer()
        del tok._vocab[f"{PFX_ROOT}حمد{SFX}"]
        tok._reverse_vocab = {i: t for t, i in tok._vocab.items()}
        _, stream = _stream(tok, "محمد")
        assert stream[0] == TOK_PROP_BEGIN and TOK_LIT_BEGIN not in stream

    def test_unknown_and_backoff_words_stay_lit(self):
        tok = _stub_tokenizer()
        for w in ("تويتر", "معيلات", "غريب"):
            _, stream = _stream(tok, w)
            assert stream[0] == TOK_LIT_BEGIN and stream[-1] == TOK_LIT_END

    def test_split_must_round_trip_or_the_chunk_goes_in_whole(self):
        """A proclitic the surface does not literally carry is not split off."""
        tok = _stub_tokenizer()
        CAMEL["ءباريس"] = [cam("بارِيس", "بِبارِيس", "NTWS", "بِNTWS", prc1="bi_prep")]   # claimed ب, surface has ء
        try:
            _, stream = _stream(tok, "ءباريس")
            assert stream[0] == TOK_PROP_BEGIN and f"{PFX_CLITICP}ب{SFX}" not in stream
            assert tok.decode(tok.encode("ءباريس").input_ids) == "ءباريس"
        finally:
            del CAMEL["ءباريس"]

    def test_oov_clitic_token_keeps_the_chunk_whole(self):
        tok = _stub_tokenizer()
        del tok._vocab[f"{PFX_CLITICP}ب{SFX}"]
        tok._reverse_vocab = {i: t for t, i in tok._vocab.items()}
        _, stream = _stream(tok, "بباريس")
        assert stream[0] == TOK_PROP_BEGIN and tok.decode(tok.encode("بباريس").input_ids) == "بباريس"

    @pytest.mark.parametrize("text", [
        "باريس", "بن", "وسوريا", "بباريس", "لكوريا", "أبي", "أنقرة", "بأنقرة", "فاطمه",
        "محمد", "ومحمد", "مصر", "بمكة", "للقاهرة", "كتاب باريس", "بن محمد بن كوريا", "تويتر باريس الكتاب",
    ])
    @pytest.mark.parametrize("mode", PROPER_NOUN_MODES)
    def test_roundtrip(self, text, mode):
        tok = _stub_tokenizer(proper_nouns=mode)
        assert tok.decode(tok.encode(text).input_ids) == text

    def test_metric_strings_concatenate_to_the_word(self):
        tok = _stub_tokenizer()
        out = tok.encode("بأنقرة")
        assert "".join(t for t in out.tokens if t) == "بأنقرة"

    def test_decode_malformed_streams(self):
        tok = _stub_tokenizer()
        v = tok._vocab
        chars = [v[f"{PFX_CHAR}{c}{SFX}"] for c in "بن"]
        assert tok.decode([v[TOK_PROP_BEGIN]] + chars) == ""                       # unclosed: nothing flushed
        assert tok.decode([v[TOK_PROP_BEGIN]] + chars + [v[TOK_LIT_END]]) == "بن"   # either END closes either BEGIN
        assert tok.decode([v[TOK_LIT_BEGIN]] + chars + [v[TOK_PROP_END]]) == "بن"
        assert tok.decode([v[TOK_PROP_END]]) == ""                                  # END alone: empty literal, dropped
        assert tok.decode([v[f"{PFX_CLITICP}و{SFX}"], v[TOK_PROP_BEGIN]] + chars + [v[TOK_PROP_END], v[f"{PFX_CLITICE}ه{SFX}"]]) == "وبنه"

    def test_old_tokenizer_without_markers_falls_back_to_lit(self):
        tok = _stub_tokenizer()
        for t in (TOK_PROP_BEGIN, TOK_PROP_END):
            del tok._vocab[t]
        tok._reverse_vocab = {i: t for t, i in tok._vocab.items()}
        _, stream = _stream(tok, "بباريس")
        assert stream[0] == f"{PFX_CLITICP}ب{SFX}" or stream[0] == TOK_LIT_BEGIN
        assert TOK_LIT_BEGIN in stream and tok.decode(tok.encode("بباريس").input_ids) == "بباريس"


class TestCorpusEntry:
    def test_round_trip_and_defaults(self):
        e = CorpusEntry.from_analysis("بباريس", analyze("بباريس"))
        assert e.analyzed and e.proper and e.root is None and e.pattern is None and e.proclitics == ("ب",)
        assert CorpusEntry.from_dict(e.to_dict()) == e
        d = e.to_dict(); del d["proper"]
        assert CorpusEntry.from_dict(d).proper is False
        r = CorpusEntry.from_analysis("محمد", analyze("محمد"))
        assert r.proper and r.root == "حمد"

    def test_train_counts_by_mode(self):
        entries = [CorpusEntry.from_analysis(w, analyze(w)) for w in ("محمد", "باريس", "كتاب")]
        counted = {}
        for mode in PROPER_NOUN_MODES:
            tok = AraRooPatTokenizer(proper_nouns=mode)
            counted[mode] = [e.word for e in entries if e.analyzed and not tok._routes_to_prop(e.proper, e.root)]
        assert counted == {"unrooted": ["محمد", "كتاب"], "all": ["كتاب"]}


class TestCorpusTraceHelpers:
    def test_prepass_path(self):
        assert prepass_path(True, None, False, proper=True, root=None) == "prop"
        assert prepass_path(True, None, False, proper=True, root="حمد") == "root_pat"
        assert prepass_path(False, None, False, proper=False) == "lit"

    @pytest.mark.parametrize("mode", PROPER_NOUN_MODES)
    def test_categorize_agrees_with_encode(self, mode):
        tok = _stub_tokenizer(proper_nouns=mode)
        for w in ("باريس", "بباريس", "محمد", "ومحمد", "للقاهرة", "كتاب", "تويتر", "بأنقرة"):
            cat, _, _ = categorize_analysis(tok._vocab, tok._backend.analyze(w), mode)
            _, stream = _stream(tok, w)
            expected = ("prop" if TOK_PROP_BEGIN in stream else
                        "root_pat" if any(t.startswith(PFX_ROOT) for t in stream) else "lit_no_analysis")
            assert cat == expected, (w, mode, stream)

    def test_budget_cut_name_is_prop(self):
        tok = _stub_tokenizer()
        del tok._vocab[f"{PFX_ROOT}حمد{SFX}"]
        cat, why, missing = categorize_analysis(tok._vocab, analyze("محمد"))
        assert cat == "prop" and missing == [f"{PFX_ROOT}حمد{SFX}"]
