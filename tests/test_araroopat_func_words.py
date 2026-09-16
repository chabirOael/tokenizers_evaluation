"""AraRooPat closed-class function words → one ``[FUNC_*]`` token each (added 2026-09-16).

Pure-Python: no CAMeL bridge, no Java. The analysis dicts are what the
bridge returns for these words (captured from camel-tools on 2026-09-16);
encode/decode run against a stub analyzer that serves them. Also pins the
alef-insensitive matching that both closed groups now use.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Optional

import pytest

from arabic_eval.tokenizers.araroopat import (
    PFX_CHAR,
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
    Analysis,
    CorpusEntry,
    MorphAnalyzer,
    _dict_to_analysis,
    canonical_particle,
)

PREPS = frozenset(PREPOSITION_INVENTORY)
FUNCS = frozenset(FUNC_INVENTORY)


def cam(lex: str, diac: str, root: str = "", pattern: str = "", pos: str = "part",
        **clitics: str) -> Dict[str, str]:
    d = {"lex": lex, "diac": diac, "root": root, "pattern": pattern, "pos": pos,
         "stem": "", "prc3": "0", "prc2": "0", "prc1": "0", "prc0": "0", "enc0": "0"}
    d.update(clitics)
    return d


# Real bridge output (top reading) for the words the tests use.
CAMEL: Dict[str, Dict[str, str]] = {
    "هم": cam("هُم", "هُم", pos="pron"),
    "هي": cam("هِيَ", "هِيَ", pos="pron"),
    "ما": cam("ما", "ما", pos="pron_rel"),
    "لا": cam("لا", "لا", pos="part_neg", prc0="na"),
    "الذي": cam("الَّذِي", "الَّذِي", pos="pron_rel"),
    "التي": cam("الَّذِي", "الَّتِي", pos="pron_rel"),        # lemma الذي, surface التي
    "الذين": cam("الَّذِي", "الَّذِينَ", pos="pron_rel"),
    "هذا": cam("هٰذا", "هٰذا", pos="pron_dem"),
    "هذه": cam("هٰذا", "هٰذِهِ", pos="pron_dem"),
    "أن": cam("أَنَّ", "أَنَّ", pos="conj_sub", prc0="na"),
    "ان": cam("أَنَّ", "أَنَّ", pos="conj_sub", prc0="na"),    # alef-folded input, same reading
    "أنهم": cam("أَنَّ", "أَنَّهُم", pos="conj_sub", prc0="na", enc0="3mp_pron"),
    "انهم": cam("أَنَّ", "أَنَّهُم", pos="conj_sub", prc0="na", enc0="3mp_pron"),
    "لكن": cam("لٰكِنَّ", "لٰكِنَّ", "ل.ك.ن", "1ٰ2ِ3ْ", pos="conj", prc0="na"),
    "لكنها": cam("لٰكِنَّ", "لٰكِنَّها", "ل.ك.ن", "1ٰ2ِ3َّها", pos="conj", prc0="na", enc0="3fs_pron"),
    "قد": cam("قَدْ", "قَدْ", pos="part_verb", prc0="na"),
    "لقد": cam("قَدْ", "لَقَدْ", pos="part_verb", prc1="la_emph", prc0="na"),
    "سوف": cam("سَوْفَ", "سَوْفَ", "س.#.ف", "1َوْ3َ", pos="part_fut", prc0="na"),
    "أو": cam("أَو", "أَو", "#.#.ن", "أَوْ", pos="conj", prc0="na"),
    "إذا": cam("إِذا", "إِذا", "#.ذ.#", "إِ2ا", pos="conj", prc0="na"),
    "إياه": cam("إِيّا", "إِيّاهُ", pos="part", prc0="na", enc0="3ms_pron"),
    "وهم": cam("هُم", "وَهُم", pos="pron", prc2="wa_conj"),
    "كم": cam("كَم", "كَم", pos="pron_rel"),
    # prepositions, one of them alef-folded
    "إلى": cam("إِلَى", "إِلَى", "#.ل.#", "إِ2َى", pos="prep"),
    "الى": cam("إِلَى", "إِلَى", "#.ل.#", "إِ2َى", pos="prep"),
    "من": cam("مِن", "مِن", "م.ن", "1ِ2", pos="prep"),
    # content words sharing letters with function words
    "كتاب": cam("كِتاب", "كِتاب", "ك.ت.ب", "1ِ2ا3", pos="noun"),
    "الكتاب": cam("كِتاب", "الكِتاب", "ك.ت.ب", "ال1ِ2ا3", pos="noun", prc0="Al_det"),
    "همة": cam("هِمَّة", "هِمَّة", "ه.م.م", "1ِ2َّ3َة", pos="noun"),
    "وهمي": cam("وَهْم", "وَهْمِي", "و.ه.م", "1َ2ْ3ِي", pos="noun", enc0="1s_poss"),
}


def analyze(word: str) -> Optional[Analysis]:
    return _dict_to_analysis(CAMEL[word], PREPS, word, FUNCS)


# ---------------------------------------------------------------------------
# The intercept
# ---------------------------------------------------------------------------

class TestFuncIntercept:
    @pytest.mark.parametrize("word", ["هم", "هي", "ما", "لا", "الذي", "هذا", "قد", "سوف", "أو", "إذا", "لكن", "كم"])
    def test_bare_function_word(self, word):
        a = analyze(word)
        assert a is not None and (a.particle, a.particle_kind) == (word, "func")
        assert a.root == "" and a.pattern == ""
        assert not any((a.prc3, a.prc2, a.prc1, a.prc0, a.enc0))

    def test_surface_forms_are_their_own_entries(self):
        """التي / الذين / هذه lemmatize to الذي / هذا but are listed as surfaces."""
        assert analyze("التي").particle == "التي"
        assert analyze("الذين").particle == "الذين"
        assert analyze("هذه").particle == "هذه"

    def test_words_that_were_lit_or_bogus_before(self):
        assert _dict_to_analysis(CAMEL["هم"], PREPS, "هم") is None          # 2 radicals, no func inventory → LIT
        bogus = _dict_to_analysis(CAMEL["أو"], PREPS, "أو")
        assert bogus is not None and bogus.root == "##ن"                       # the bogus root+wazn
        assert analyze("أو").particle == "أو"

    def test_enclitic_is_split_off(self):
        a = analyze("أنهم")
        assert (a.particle, a.enc0, a.particle_kind) == ("أن", "هم", "func")
        a = analyze("لكنها")
        assert (a.particle, a.enc0) == ("لكن", "ها")
        a = analyze("إياه")
        assert (a.particle, a.enc0) == ("إيا", "ه")

    def test_proclitic_is_split_off(self):
        a = analyze("وهم")
        assert (a.prc2, a.particle) == ("و", "هم")
        a = analyze("لقد")
        assert (a.prc1, a.particle) == ("ل", "قد")

    def test_prepositions_win(self):
        a = analyze("من")
        assert (a.particle, a.particle_kind) == ("من", "prep")

    def test_content_words_are_left_alone(self):
        assert analyze("كتاب").particle is None
        assert analyze("الكتاب").particle is None
        assert analyze("همة").particle is None      # noun, lemma همة
        assert analyze("وهمي").particle is None     # possessive → noun reading

    def test_surface_fallback_when_camel_has_nothing(self):
        a = MorphAnalyzer._first_valid([], "هؤلاء", PREPS, FUNCS)
        assert a is not None and (a.particle, a.particle_kind) == ("هؤلاء", "func")
        assert MorphAnalyzer._first_valid([], "كتاب", PREPS, FUNCS) is None

    def test_inventory_is_configurable_and_off_by_default_for_the_bare_call(self):
        assert _dict_to_analysis(CAMEL["هذا"], PREPS, "هذا", frozenset({"هم"})) is None
        assert _dict_to_analysis(CAMEL["هذا"], PREPS, "هذا") is None


class TestAlefInsensitiveMatching:
    def test_canonical_particle(self):
        assert canonical_particle("الى", PREPS) == "إلى"
        assert canonical_particle("إلى", PREPS) == "إلى"
        assert canonical_particle("ان", FUNCS) == "أن"       # أ sorts before إ
        assert canonical_particle("إن", FUNCS) == "إن"       # exact spelling always wins
        assert canonical_particle("علي", PREPS) is None      # ى/ي stay strict
        assert canonical_particle("", PREPS) is None and canonical_particle("ان", frozenset()) is None

    def test_folded_preposition_is_intercepted(self):
        """Regression: the alef-folded training corpus never produced [PREP_إلى]."""
        a = analyze("الى")
        assert a is not None and (a.particle, a.particle_kind) == ("إلى", "prep")

    def test_folded_function_word_with_enclitic(self):
        a = analyze("انهم")
        assert (a.particle, a.enc0) == ("أن", "هم")
        assert analyze("ان").particle == "أن"

    def test_yeh_spelled_bare_forms_stay_strict(self):
        assert _dict_to_analysis(cam("عَلَى", "عَلَى", "ع.ل.#", "1َ2َى", pos="prep"), PREPS, "علي", FUNCS).particle is None


# ---------------------------------------------------------------------------
# Tokenizer: layout, encode / decode, persistence
# ---------------------------------------------------------------------------

class _StubAnalyzer:
    def analyze(self, word: str) -> Optional[Analysis]:
        d = CAMEL.get(word)
        return _dict_to_analysis(d, PREPS, word, FUNCS) if d else \
            MorphAnalyzer._first_valid([], word, PREPS, FUNCS)

    def generate(self, root: str, pattern: str) -> Optional[str]:
        return None


def _stub_tokenizer() -> AraRooPatTokenizer:
    tok = AraRooPatTokenizer()
    tok._backend = _StubAnalyzer()
    tok._build_vocab(
        root_freq=Counter({"كتب": 5}),
        pat_freq=Counter({"1ِ2ا3": 5}),
        proclitic_freq=Counter({"و": 5, "ال": 5, "ل": 2}),
        enclitic_freq=Counter({"ه": 5, "هم": 5, "ها": 3}),
    )
    tok._reconstruction = {
        (tok._vocab[f"{PFX_ROOT}كتب{SFX}"], tok._vocab["[PAT_1ِ2ا3]"]): "كتاب",
    }
    return tok


class TestCorpusEntry:
    def test_kind_round_trips_through_dict(self):
        e = CorpusEntry.from_analysis("أنهم", analyze("أنهم"))
        assert e.analyzed and (e.particle, e.particle_kind) == ("أن", "func") and e.root is None
        assert e.enclitics == ("هم",)
        assert CorpusEntry.from_dict(e.to_dict()) == e
        # a pre-func cache entry (no kind key) reads back as a preposition
        d = CorpusEntry.from_analysis("في", _dict_to_analysis(cam("فِي", "فِي", "ف.#", "1ِي", pos="prep"), PREPS, "في")).to_dict()
        del d["particle_kind"]
        assert CorpusEntry.from_dict(d).particle_kind == "prep"


class TestVocabLayout:
    def test_func_range_sits_between_preps_and_chars(self):
        tok = _stub_tokenizer()
        ids = tok._vocab
        prep_ids = [i for t, i in ids.items() if t.startswith(PFX_PREP)]
        func_ids = [i for t, i in ids.items() if t.startswith(PFX_FUNC)]
        first_char = min(i for t, i in ids.items() if t.startswith(PFX_CHAR))
        assert max(prep_ids) < min(func_ids) and max(func_ids) < first_char
        assert max(func_ids) - min(func_ids) + 1 == len(func_ids) == len(FUNC_INVENTORY)
        assert [tok._reverse_vocab[i] for i in sorted(func_ids)] == [f"{PFX_FUNC}{w}{SFX}" for w in FUNC_INVENTORY]

    def test_inventories_must_not_overlap(self):
        with pytest.raises(ValueError, match="already listed"):
            AraRooPatTokenizer(func_words=["هم", "هم"])
        with pytest.raises(ValueError, match="already listed"):
            AraRooPatTokenizer(func_words=["من"])          # a preposition
        with pytest.raises(ValueError, match="modulo alef"):
            AraRooPatTokenizer(func_words=["الى"])         # shadowed by [PREP_إلى]
        AraRooPatTokenizer(func_words=["أن", "إن"])         # within-group fold collision is fine

    def test_inventory_is_configurable(self):
        tok = AraRooPatTokenizer(func_words=["هذا", "هذه"])
        tok._build_vocab(Counter(), Counter(), Counter(), Counter())
        assert [t for t in tok._vocab if t.startswith(PFX_FUNC)] == [f"{PFX_FUNC}هذا{SFX}", f"{PFX_FUNC}هذه{SFX}"]


class TestEncodeDecode:
    def _stream(self, tok, text):
        out = tok.encode(text)
        return out, [tok._reverse_vocab[i] for i in out.input_ids
                     if tok._reverse_vocab[i] not in ("<s>", "</s>")]

    def test_each_bare_function_word_is_one_token(self):
        tok = _stub_tokenizer()
        for w in ("هم", "هي", "ما", "لا", "الذي", "التي", "هذا", "هذه", "قد", "سوف", "أو", "إذا", "لكن"):
            out, stream = self._stream(tok, w)
            assert stream == [f"{PFX_FUNC}{w}{SFX}"], (w, stream)
            assert tok.decode(out.input_ids) == w

    def test_clitics_ride_outside(self):
        tok = _stub_tokenizer()
        _, stream = self._stream(tok, "وهم")
        assert stream == [f"{PFX_CLITICP}و{SFX}", f"{PFX_FUNC}هم{SFX}"]
        _, stream = self._stream(tok, "أنهم")
        assert stream == [f"{PFX_FUNC}أن{SFX}", f"{PFX_CLITICE}هم{SFX}"]
        _, stream = self._stream(tok, "لقد")
        assert stream == [f"{PFX_CLITICP}ل{SFX}", f"{PFX_FUNC}قد{SFX}"]

    @pytest.mark.parametrize("text", [
        "هم", "وهم", "أنهم", "لكنها", "لقد", "إياه", "هذه", "الذين",
        "هذا الكتاب الذي كتب", "لا أو لكن قد سوف", "من الكتاب إلى هم",
    ])
    def test_roundtrip(self, text):
        tok = _stub_tokenizer()
        assert tok.decode(tok.encode(text).input_ids) == text

    def test_folded_input_decodes_to_the_canonical_spelling(self):
        tok = _stub_tokenizer()
        out, stream = self._stream(tok, "انهم")
        assert stream == [f"{PFX_FUNC}أن{SFX}", f"{PFX_CLITICE}هم{SFX}"]
        assert tok.decode(out.input_ids) == "أنهم"
        out, stream = self._stream(tok, "الى")
        assert stream == [f"{PFX_PREP}إلى{SFX}"] and tok.decode(out.input_ids) == "إلى"

    def test_metric_string_is_the_surface_core(self):
        tok = _stub_tokenizer()
        out = tok.encode("أنهم")
        assert [t for t in out.tokens if t] == ["أن", "هم"]
        out = tok.encode("انهم")
        assert "".join(t for t in out.tokens if t) == "انهم"      # the *input* surface, so offsets still align

    def test_oov_clitic_sends_whole_chunk_to_lit(self):
        tok = _stub_tokenizer()
        del tok._vocab[f"{PFX_CLITICE}ها{SFX}"]
        tok._reverse_vocab = {i: t for t, i in tok._vocab.items()}
        out, stream = self._stream(tok, "لكنها")
        assert stream[0] == "[LIT_BEGIN]" and f"{PFX_FUNC}لكن{SFX}" not in stream
        assert tok.decode(out.input_ids) == "لكنها"

    def test_pre_func_tokenizer_falls_back_to_lit(self):
        """A saved tokenizer without the [FUNC_*] range keeps encoding the words as before (LIT)."""
        tok = _stub_tokenizer()
        tok.func_words = ()
        tok._vocab = {t: i for t, i in tok._vocab.items() if not t.startswith(PFX_FUNC)}
        tok._reverse_vocab = {i: t for t, i in tok._vocab.items()}
        out, stream = self._stream(tok, "هذا")
        assert stream[0] == "[LIT_BEGIN]" and tok.decode(out.input_ids) == "هذا"


class TestPersistence:
    def test_config_round_trip(self, tmp_path):
        tok = _stub_tokenizer()
        tok._metadata = {"roots": {}, "patterns": {}, "config": {}}
        tok.save(tmp_path)
        again = AraRooPatTokenizer(func_words=["زائف"])
        again.load(tmp_path)
        assert again.func_words == FUNC_INVENTORY and again.prepositions == PREPOSITION_INVENTORY
        assert f"{PFX_FUNC}هذا{SFX}" in again._vocab
        assert again._cache_key()[-1] == FUNC_INVENTORY and again._CACHE_FORMAT >= 5

    def test_metadata_carries_the_group(self):
        tok = _stub_tokenizer()
        tok._build_metadata(Counter(), Counter(), Counter(), Counter(), [], Counter(), Counter({"هذا": 7}))
        assert tok._metadata["func_words"]["هذا"] == {"id": tok._vocab[f"{PFX_FUNC}هذا{SFX}"], "freq": 7}
        assert tok._metadata["func_words"]["هم"]["freq"] == 0
        assert tok._metadata["config"]["func_words"] == list(FUNC_INVENTORY)
