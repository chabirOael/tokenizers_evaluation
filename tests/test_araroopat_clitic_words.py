"""AraRooPat pronoun-hosted prepositions as clitic-only words (added 2026-09-16).

له = لِ + ه, بها = بِ + ها, ولهم = و + ل + هم: CAMeL reads them as POS
``prep`` with lemma لِ / بِ / كَ and the pronoun in ``enc0``; the tokenizer
emits ``[CLITICP_ل] [CLITICE_ه]`` and nothing else, and the decoder closes
buffered proclitics into a word when an enclitic follows them directly.
Also pins the 2026-09-16 inventory additions (مع → PREP, فيما / ذات → FUNC
with ذات keeping its possessive). Pure-Python, stub analyzer, real bridge
dicts captured on 2026-09-16.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Optional

import pytest

from arabic_eval.tokenizers.araroopat import (
    PFX_CLITICE,
    PFX_CLITICP,
    PFX_FUNC,
    PFX_PREP,
    PFX_ROOT,
    SFX,
    AraRooPatTokenizer,
)
from arabic_eval.tokenizers.araroopat_backend import (
    FUNC_INVENTORY,
    PREPOSITION_INVENTORY,
    PRONOUN_HOSTING_PROCLITICS,
    Analysis,
    CorpusEntry,
    MorphAnalyzer,
    _clitic_only_analysis,
    _dict_to_analysis,
)

PREPS = frozenset(PREPOSITION_INVENTORY)
FUNCS = frozenset(FUNC_INVENTORY)


def cam(lex: str, diac: str, root: str = "", pattern: str = "", pos: str = "prep",
        **clitics: str) -> Dict[str, str]:
    d = {"lex": lex, "diac": diac, "root": root, "pattern": pattern, "pos": pos,
         "stem": "", "prc3": "0", "prc2": "0", "prc1": "0", "prc0": "na", "enc0": "0"}
    d.update(clitics)
    return d


CAMEL: Dict[str, list] = {
    "له": [cam("لِ", "لَهُ", "ل", "1", enc0="3ms_pron")],
    "به": [cam("بِ", "بِهِ", "ب", "1", enc0="3ms_pron")],
    "بها": [cam("بِ", "بِها", "ب", "1", enc0="3fs_pron")],
    "لها": [cam("لِ", "لَها", "ل", "1", enc0="3fs_pron"), cam("لَها", "لَها", "ل.ه.#", "1َ2ا", pos="verb", prc0="0")],
    "لهم": [cam("لِ", "لَهُم", "ل", "1", enc0="3mp_pron"), cam("لَهِم", "لَهِم", "ل.ه.م", "1َ2ِ3", pos="noun", prc0="0")],
    "لنا": [cam("لِ", "لَنا", "ل", "1", enc0="1p_pron")],
    "لك": [cam("لِ", "لَكَ", "ل", "1", enc0="2ms_pron")],
    "لي": [cam("لِ", "لِي", "ل", "1", enc0="1s_pron")],
    "ولهم": [cam("لِ", "وَلَهُم", "ل", "1", prc2="wa_conj", enc0="3mp_pron")],
    "فبها": [cam("بِ", "فَبِها", "ب", "1", prc2="fa_conj", enc0="3fs_pron")],
    # noun_prop / abbrev first, the prep reading second: the candidate walk must reach it
    "بك": [cam("بَك", "بَك", "NTWS", "NTWS", pos="noun_prop", prc0="0"), cam("بِ", "بِكَ", "ب", "1", enc0="2ms_pron")],
    "بي": [cam("بِي", "بِي", "NTWS", "NTWS", pos="abbrev", prc0="na"), cam("بِ", "بِي", "ب", "1", enc0="1s_pron")],
    # new PREP / FUNC entries
    "مع": [cam("مَع", "مَع", "مع", "12")],
    "معهم": [cam("مَع", "مَعَهُم", "مع", "12", enc0="3mp_pron")],
    "خلال": [cam("خِلالَ", "خِلالَ", "خلل", "1ِ2ا3َ")],
    "فيما": [cam("ما", "فِيما", "ف.#", "1ِيما", pos="conj_sub"), cam("فِيما", "فِيما", "NTWS", "NTWS", pos="noun_prop", prc0="0")],
    "ذات": [cam("ذات", "ذات", "NTWS", "NTWS", pos="noun", prc0="0")],
    "ذاتها": [cam("ذات", "ذاتِها", "NTWS", "NTWS", pos="noun", prc0="0", enc0="3fs_poss")],
    "بذاته": [cam("ذات", "بِذاتِهِ", "NTWS", "NTWS", pos="noun", prc1="bi_prep", prc0="0", enc0="3ms_poss")],
    # controls
    "لكتاب": [cam("كِتاب", "لِكِتاب", "ك.ت.ب", "لِ1ِ2ا3", pos="noun", prc1="li_prep", prc0="0")],
    "كتابه": [cam("كِتاب", "كِتابُهُ", "ك.ت.ب", "1ِ2ا3ُهُ", pos="noun", prc0="0", enc0="3ms_poss")],
    "ربنا": [cam("رَبّ", "رَبّنا", "ر.ب.ب", "1َ2ّنا", pos="noun", prc0="0", enc0="1p_poss")],
    "همه": [cam("هَمّ", "هَمُّهُ", "ه.م.م", "1َ2ُّهُ", pos="noun", prc0="0", enc0="3ms_poss")],
}


def analyze(word: str) -> Optional[Analysis]:
    return MorphAnalyzer._first_valid(CAMEL[word], word, PREPS, FUNCS)


class TestCliticOnlyAnalysis:
    @pytest.mark.parametrize("word,procs,enc", [
        ("له", ("ل",), "ه"), ("به", ("ب",), "ه"), ("بها", ("ب",), "ها"), ("لها", ("ل",), "ها"),
        ("لهم", ("ل",), "هم"), ("لنا", ("ل",), "نا"), ("لك", ("ل",), "ك"), ("لي", ("ل",), "ي"),
        ("ولهم", ("و", "ل"), "هم"), ("فبها", ("ف", "ب"), "ها"),
        ("بك", ("ب",), "ك"), ("بي", ("ب",), "ي"),
    ])
    def test_pronoun_hosted_preposition(self, word, procs, enc):
        a = analyze(word)
        assert a is not None and a.clitic_only and a.particle is None
        assert a.root == "" and a.pattern == ""
        assert (a.proclitics, a.enc0) == (procs, enc)

    def test_lemma_set_is_closed(self):
        assert PRONOUN_HOSTING_PROCLITICS == ("ل", "ب", "ك")
        assert _clitic_only_analysis(cam("مِن", "مِنهُ", "م.ن", "1ِ2", enc0="3ms_pron"), "منه") is None
        assert _clitic_only_analysis(cam("لِ", "لَهُ", "ل", "1", enc0="3ms_pron"), "له") is not None

    def test_needs_a_pronoun_and_a_free_prc1(self):
        assert _clitic_only_analysis(cam("لِ", "لِ", "ل", "1"), "ل") is None
        assert _clitic_only_analysis(cam("لِ", "لِلَهُ", "ل", "1", prc1="li_prep", enc0="3ms_pron"), "لله") is None
        assert _clitic_only_analysis(cam("لِ", "لَهُ", "ل", "1", pos="noun", enc0="3ms_pron"), "له") is None

    def test_surface_must_match(self):
        assert _clitic_only_analysis(cam("لِ", "لَهُ", "ل", "1", enc0="3ms_pron"), "لهما") is None

    def test_hosted_clitics_are_untouched(self):
        a = analyze("لكتاب")
        assert a is not None and not a.clitic_only and a.root == "كتب" and a.prc1 == "ل"
        a = analyze("كتابه")
        assert a is not None and not a.clitic_only and a.enc0 == "ه"

    def test_was_lit_before(self):
        """Without the rule the prep reading has a 1-radical root and is rejected."""
        d = dict(CAMEL["له"][0])
        d["pos"] = "adv"   # not prep → the clitic-only rule does not fire → root gate rejects
        assert _dict_to_analysis(d, PREPS, "له", FUNCS) is None


class TestInventoryAdditions:
    def test_new_prepositions(self):
        for w in ("مع", "تجاه", "خلال", "بلا"):
            assert w in PREPOSITION_INVENTORY
        a = analyze("مع")
        assert (a.particle, a.particle_kind) == ("مع", "prep")
        a = analyze("معهم")
        assert (a.particle, a.enc0) == ("مع", "هم")
        assert analyze("خلال").particle == "خلال"

    def test_fused_fima_is_a_function_word(self):
        a = analyze("فيما")
        assert (a.particle, a.particle_kind) == ("فيما", "func")

    def test_dhat_keeps_its_possessive(self):
        assert analyze("ذات").particle == "ذات"
        a = analyze("ذاتها")
        assert (a.particle, a.enc0, a.particle_kind) == ("ذات", "ها", "func")
        a = analyze("بذاته")
        assert (a.prc1, a.particle, a.enc0) == ("ب", "ذات", "ه")
        # the carve-out is scoped: other nouns with a possessive stay on their path
        assert analyze("ربنا").particle is None
        assert analyze("همه").particle is None and analyze("همه").root == "همم"


# ---------------------------------------------------------------------------
# Tokenizer: encode / decode / persistence
# ---------------------------------------------------------------------------

class _StubAnalyzer:
    def analyze(self, word: str) -> Optional[Analysis]:
        cands = CAMEL.get(word, [])
        return MorphAnalyzer._first_valid(cands, word, PREPS, FUNCS)

    def generate(self, root: str, pattern: str) -> Optional[str]:
        return None


def _stub_tokenizer() -> AraRooPatTokenizer:
    tok = AraRooPatTokenizer()
    tok._backend = _StubAnalyzer()
    tok._build_vocab(
        root_freq=Counter({"كتب": 5, "همم": 2}),
        pat_freq=Counter({"1ِ2ا3": 5, "1َ2ُّ": 2}),
        proclitic_freq=Counter({"و": 9, "ال": 8, "ل": 7, "ب": 6, "ف": 2}),
        enclitic_freq=Counter({"ه": 9, "ها": 8, "هم": 7, "نا": 3, "ك": 3, "ي": 3}),
    )
    tok._reconstruction = {
        (tok._vocab[f"{PFX_ROOT}كتب{SFX}"], tok._vocab["[PAT_1ِ2ا3]"]): "كتاب",
        (tok._vocab[f"{PFX_ROOT}همم{SFX}"], tok._vocab["[PAT_1َ2ُّ]"]): "هم",
    }
    return tok


class TestEncodeDecode:
    def _stream(self, tok, text):
        out = tok.encode(text)
        return out, [tok._reverse_vocab[i] for i in out.input_ids
                     if tok._reverse_vocab[i] not in ("<s>", "</s>")]

    def test_clitic_tokens_only(self):
        tok = _stub_tokenizer()
        _, stream = self._stream(tok, "له")
        assert stream == [f"{PFX_CLITICP}ل{SFX}", f"{PFX_CLITICE}ه{SFX}"]
        _, stream = self._stream(tok, "ولهم")
        assert stream == [f"{PFX_CLITICP}و{SFX}", f"{PFX_CLITICP}ل{SFX}", f"{PFX_CLITICE}هم{SFX}"]
        _, stream = self._stream(tok, "بك")
        assert stream == [f"{PFX_CLITICP}ب{SFX}", f"{PFX_CLITICE}ك{SFX}"]

    def test_same_proclitic_token_as_a_hosted_word(self):
        tok = _stub_tokenizer()
        _, a = self._stream(tok, "له")
        _, b = self._stream(tok, "لكتاب")
        assert a[0] == b[0] == f"{PFX_CLITICP}ل{SFX}"

    @pytest.mark.parametrize("text", [
        "له", "به", "بها", "لها", "لهم", "لنا", "لك", "لي", "ولهم", "فبها", "بك", "بي",
        "كتابه له", "لكتاب له", "له كتابه", "مع معهم خلال", "فيما ذات ذاتها بذاته",
        "ربنا همه", "له له له",
    ])
    def test_roundtrip(self, text):
        tok = _stub_tokenizer()
        assert tok.decode(tok.encode(text).input_ids) == text

    def test_enclitic_never_attaches_to_the_previous_word(self):
        """كتابه له: the ه of له must close ل, not glue onto كتابه."""
        tok = _stub_tokenizer()
        out = tok.encode("كتابه له")
        assert tok.decode(out.input_ids) == "كتابه له"

    def test_metric_strings_concatenate_to_the_word(self):
        tok = _stub_tokenizer()
        out = tok.encode("ولهم")
        assert [t for t in out.tokens if t] == ["و", "ل", "هم"]

    def test_oov_clitic_sends_whole_chunk_to_lit(self):
        tok = _stub_tokenizer()
        del tok._vocab[f"{PFX_CLITICE}نا{SFX}"]
        tok._reverse_vocab = {i: t for t, i in tok._vocab.items()}
        out, stream = self._stream(tok, "لنا")
        assert stream[0] == "[LIT_BEGIN]" and tok.decode(out.input_ids) == "لنا"

    def test_new_prep_and_func_tokens(self):
        tok = _stub_tokenizer()
        _, stream = self._stream(tok, "معهم")
        assert stream == [f"{PFX_PREP}مع{SFX}", f"{PFX_CLITICE}هم{SFX}"]
        _, stream = self._stream(tok, "فيما")
        assert stream == [f"{PFX_FUNC}فيما{SFX}"]
        _, stream = self._stream(tok, "بذاته")
        assert stream == [f"{PFX_CLITICP}ب{SFX}", f"{PFX_FUNC}ذات{SFX}", f"{PFX_CLITICE}ه{SFX}"]

    def test_corpus_entry_round_trip(self):
        e = CorpusEntry.from_analysis("ولهم", analyze("ولهم"))
        assert e.analyzed and e.clitic_only and e.particle is None and e.root is None
        assert (e.proclitics, e.enclitics) == (("و", "ل"), ("هم",))
        assert CorpusEntry.from_dict(e.to_dict()) == e
        d = e.to_dict(); del d["clitic_only"]
        assert CorpusEntry.from_dict(d).clitic_only is False

    def test_cache_format_bumped(self):
        assert AraRooPatTokenizer._CACHE_FORMAT >= 7


class TestPeelerInteraction:
    def test_no_peel_onto_a_clitic_only_residual(self):
        """أبي is 'my father', not أ + بِ + ي: the interrogative never lands on a pronoun-hosted preposition."""
        from arabic_eval.tokenizers.araroopat_backend import PeelCandidate, peel_compatible
        residual = _clitic_only_analysis(cam("بِ", "بِي", "ب", "1", enc0="1s_pron"), "بي")
        assert residual is not None and residual.clitic_only
        assert peel_compatible(PeelCandidate(proclitics=("أ",), residual="بي", enclitics=()), residual) is False
        # a real particle residual still takes the interrogative (أفي)
        particle = MorphAnalyzer._first_valid([], "في", PREPS, FUNCS)
        assert peel_compatible(PeelCandidate(proclitics=("أ",), residual="في", enclitics=()), particle) is True
