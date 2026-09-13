"""AraRooPat: the tāʾ marbūṭa (ة) is an enclitic, never a character.

Pure-Python: no CAMeL bridge, no Java. Analysis dicts are what the bridge
returns for these words (verified against camel-tools on 2026-09-13);
encode/decode run against a stub analyzer that serves them.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Optional

import pytest

from arabic_eval.tokenizers.araroopat import (
    CHAR_INVENTORY,
    PFX_CHAR,
    PFX_CLITICE,
    PFX_ROOT,
    SFX,
    AraRooPatTokenizer,
    _classify_char,
    _extract_alpha_chunks,
    _split_runs,
    _strip_clitic_surfaces,
)
from arabic_eval.tokenizers.araroopat_backend import (
    PREPOSITION_INVENTORY,
    TAA_MARBUTA,
    Analysis,
    CorpusEntry,
    MorphAnalyzer,
    _dict_to_analysis,
    strip_enclitics_from_end,
    strip_fem_from_pattern,
)

PARTICLES = frozenset(PREPOSITION_INVENTORY)


def cam(lex: str, diac: str, root: str, pattern: str, stem: str, pos: str = "noun",
        **clitics: str) -> Dict[str, str]:
    d = {"lex": lex, "diac": diac, "root": root, "pattern": pattern, "stem": stem,
         "pos": pos, "prc3": "0", "prc2": "0", "prc1": "0", "prc0": "0", "enc0": "0"}
    d.update(clitics)
    return d


CAMEL: Dict[str, Dict[str, str]] = {
    "مدرسة": cam("مَدْرَسَة", "مَدْرَسَةِ", "د.ر.س", "مَ1ْ2َ3َةِ", "مَدْرَس"),
    "المدرسة": cam("مَدْرَسَة", "المَدْرَسَةِ", "د.ر.س", "المَ1ْ2َ3َةِ", "مَدْرَس", prc0="Al_det"),
    "مدرسته": cam("مَدْرَسَة", "مَدْرَسَته", "د.ر.س", "مَ1ْ2َ3َته", "مَدْرَس", enc0="3ms_poss"),
    "مدرستها": cam("مَدْرَسَة", "مَدْرَسَتها", "د.ر.س", "مَ1ْ2َ3َتها", "مَدْرَس", enc0="3fs_poss"),
    "ومدرستها": cam("مَدْرَسَة", "وَمَدْرَسَتها", "د.ر.س", "وَمَ1ْ2َ3َتها", "مَدْرَس",
                    prc2="wa_conj", enc0="3fs_poss"),
    "معلمة": cam("مُعَلِّم", "مُعَلِّمَةُ", "ع.ل.م", "مُ1َ2ِّ3َةُ", "مُعَلِّم"),
    "معلمتك": cam("مُعَلِّم", "مُعَلِّمَتكَ", "ع.ل.م", "مُ1َ2ِّ3َتكَ", "مُعَلِّم", enc0="2ms_poss"),
    "حياته": cam("حَياة", "حَياتِهِ", "ح.#.#", "1َياَتِهِ", "حَيا", enc0="3ms_poss"),
    "قضاة": cam("قاضِي", "قُضاةِ", "ق.ض.#", "1ُ2اَةِ", "قُضا"),
    "ثلاثة": cam("ثَلاث", "ثَلاثَةِ", "ث.ل.ث", "1َ2ا3َةِ", "ثَلاث", pos="noun_num"),
    # Must NOT be touched.
    "بيته": cam("بَيْت", "بَيْتِهِ", "ب.#.ت", "1َيْ3ِهِ", "بَيْت", enc0="3ms_poss"),
    "كتبته": cam("كَتَب", "كَتَبْتُهُ", "ك.ت.ب", "1َ2َ3ْتُهُ", "كَتَب", pos="verb", enc0="3ms_dobj"),
    "كتاب": cam("كِتاب", "كِتاب", "ك.ت.ب", "1ِ2ا3", "كِتاب"),
}


def analyze(word: str) -> Optional[Analysis]:
    return _dict_to_analysis(CAMEL[word], PARTICLES, word)


# ---------------------------------------------------------------------------
# Character classes and chunking
# ---------------------------------------------------------------------------

class TestCharacterClasses:
    def test_taa_marbuta_is_its_own_class_not_a_char(self):
        assert _classify_char(TAA_MARBUTA) == "fem"
        assert TAA_MARBUTA not in CHAR_INVENTORY
        assert _classify_char("ت") == "alpha" and "ت" in CHAR_INVENTORY

    def test_alpha_run_absorbs_trailing_taa_and_ends_there(self):
        assert _split_runs("مدرسة") == [("alpha", "مدرسة")]
        assert _split_runs("مدرسةكبيرة") == [("alpha", "مدرسة"), ("alpha", "كبيرة")]
        assert _extract_alpha_chunks("مدرسةكبيرة") == ["مدرسة", "كبيرة"]

    def test_lone_taa_is_an_alpha_chunk(self):
        assert _split_runs("ة") == [("alpha", "ة")]

    def test_mixed_runs(self):
        assert _split_runs("مدرسة2024،") == [("alpha", "مدرسة"), ("digit", "2024"), ("punct", "،")]


# ---------------------------------------------------------------------------
# Pattern stripping
# ---------------------------------------------------------------------------

class TestStripFem:
    @pytest.mark.parametrize("word,pattern", [
        ("مدرسة", "مَ1ْ2َ3َ"), ("المدرسة", "مَ1ْ2َ3َ"), ("معلمة", "مُ1َ2ِّ3َ"),
        ("قضاة", "1ُ2اَ"), ("ثلاثة", "1َ2ا3َ"),
    ])
    def test_word_final_taa_is_stripped(self, word, pattern):
        a = analyze(word)
        assert (a.pattern, a.fem) == (pattern, TAA_MARBUTA)

    @pytest.mark.parametrize("word,pattern,enc", [
        ("مدرسته", "مَ1ْ2َ3َ", "ه"), ("مدرستها", "مَ1ْ2َ3َ", "ها"),
        ("ومدرستها", "مَ1ْ2َ3َ", "ها"), ("معلمتك", "مُ1َ2ِّ3َ", "ك"), ("حياته", "1َياَ", "ه"),
    ])
    def test_taa_before_pronoun_is_the_same_suffix(self, word, pattern, enc):
        a = analyze(word)
        assert (a.pattern, a.fem, a.enc0) == (pattern, TAA_MARBUTA, enc)
        assert a.enclitics == (TAA_MARBUTA, enc)

    def test_radical_taa_is_kept(self):
        a = analyze("بيته")
        assert (a.pattern, a.fem) == ("1َيْ3", None)

    def test_verbal_subject_taa_is_kept(self):
        a = analyze("كتبته")
        assert (a.pattern, a.fem) == ("1َ2َ3ْت", None)

    def test_same_pattern_with_and_without_suffix(self):
        """The point of the exercise: one [PAT_*] entry, not two."""
        assert analyze("مدرسة").pattern == analyze("مدرسته").pattern

    def test_helper_is_pure(self):
        assert strip_fem_from_pattern("1ِ2ا3", (), None, "كِتاب", "noun", "كِتاب") == ("1ِ2ا3", None)
        assert strip_fem_from_pattern("", (), None, "", "", "") == ("", None)


class TestStripEnclitics:
    def test_outermost_first_and_taa_as_ta(self):
        assert strip_enclitics_from_end("مَدْرَسَته", ("ة", "ه")) == "مَدْرَس"
        assert strip_enclitics_from_end("مَدْرَسَةِ", ("ة",)) == "مَدْرَس"
        assert strip_enclitics_from_end("بَيْتِهِ", ("ه",)) == "بَيْت"

    def test_inflected_stem_for_reconstruction_has_no_suffix(self):
        for word in ("مدرسة", "مدرسته", "ومدرستها"):
            a = analyze(word)
            proc = tuple(c for c in (a.prc3, a.prc2, a.prc1, a.prc0) if c)
            assert _strip_clitic_surfaces(a.surface, proc, a.enclitics) == "مَدْرَس"  # == CAMeL stem

    def test_corpus_entry_carries_emission_order(self):
        e = CorpusEntry.from_analysis("مدرسته", analyze("مدرسته"))
        assert e.enclitics == ("ة", "ه")
        assert CorpusEntry.from_dict(e.to_dict()) == e


# ---------------------------------------------------------------------------
# Encode / decode with a stub analyzer
# ---------------------------------------------------------------------------

class _StubAnalyzer:
    def analyze(self, word: str) -> Optional[Analysis]:
        d = CAMEL.get(word)
        return _dict_to_analysis(d, PARTICLES, word) if d else \
            MorphAnalyzer._first_valid([], word, PARTICLES)

    def generate(self, root: str, pattern: str) -> Optional[str]:
        return None


def _stub_tokenizer() -> AraRooPatTokenizer:
    tok = AraRooPatTokenizer()
    tok._backend = _StubAnalyzer()
    tok._build_vocab(
        root_freq=Counter({"درس": 5, "علم": 4, "ب#ت": 3, "كتب": 3, "ثلث": 2}),
        pat_freq=Counter({"مَ1ْ2َ3َ": 5, "مُ1َ2ِّ3َ": 4, "1َيْ3": 3, "1َ2َ3ْت": 3, "1َ2ا3َ": 2, "1ِ2ا3": 2}),
        proclitic_freq=Counter({"و": 5, "ال": 5}),
        enclitic_freq=Counter({"ة": 9, "ه": 5, "ها": 3, "ك": 2}),
    )
    v = tok._vocab
    tok._reconstruction = {
        (v["[ROOT_درس]"], v["[PAT_مَ1ْ2َ3َ]"]): "مدرس",
        (v["[ROOT_علم]"], v["[PAT_مُ1َ2ِّ3َ]"]): "معلم",
        (v["[ROOT_ب#ت]"], v["[PAT_1َيْ3]"]): "بيت",
        (v["[ROOT_كتب]"], v["[PAT_1َ2َ3ْت]"]): "كتبت",
        (v["[ROOT_ثلث]"], v["[PAT_1َ2ا3َ]"]): "ثلاث",
        (v["[ROOT_كتب]"], v["[PAT_1ِ2ا3]"]): "كتاب",
    }
    return tok


class TestVocab:
    def test_no_char_taa_and_fixed_clitice_slot(self):
        tok = _stub_tokenizer()
        assert f"{PFX_CHAR}ة{SFX}" not in tok._vocab
        fem_id = tok._vocab[f"{PFX_CLITICE}ة{SFX}"]
        other = [i for t, i in tok._vocab.items() if t.startswith(PFX_CLITICE) and i != fem_id]
        assert fem_id < min(other)

    def test_slot_exists_even_when_corpus_has_no_taa(self):
        tok = AraRooPatTokenizer()
        tok._build_vocab(Counter(), Counter(), Counter(), Counter())
        assert f"{PFX_CLITICE}ة{SFX}" in tok._vocab


class TestEncodeDecode:
    def _stream(self, tok, text):
        out = tok.encode(text)
        return out, [tok._reverse_vocab[i] for i in out.input_ids][1:-1]

    def test_root_pat_path_emits_suffix_token(self):
        tok = _stub_tokenizer()
        _, stream = self._stream(tok, "مدرسة")
        assert stream == ["[ROOT_درس]", "[PAT_مَ1ْ2َ3َ]", "[CLITICE_ة]"]
        _, stream = self._stream(tok, "مدرسته")
        assert stream == ["[ROOT_درس]", "[PAT_مَ1ْ2َ3َ]", "[CLITICE_ة]", "[CLITICE_ه]"]

    def test_lit_path_emits_suffix_token_after_lit_end(self):
        tok = _stub_tokenizer()
        _, stream = self._stream(tok, "شركة")   # no analysis → LIT
        assert stream == ["[LIT_BEGIN]", "[CHAR_ش]", "[CHAR_ر]", "[CHAR_ك]", "[LIT_END]", "[CLITICE_ة]"]

    def test_lone_taa(self):
        tok = _stub_tokenizer()
        out, stream = self._stream(tok, "ة")
        assert stream == ["[LIT_BEGIN]", "[LIT_END]", "[CLITICE_ة]"]
        assert tok.decode(out.input_ids) == "ة"
        # ...and it does not glue onto the previous word.
        out = tok.encode("كتاب ة")
        assert tok.decode(out.input_ids) == "كتاب ة"

    @pytest.mark.parametrize("text", [
        "مدرسة", "المدرسة", "مدرسته", "مدرستها", "ومدرستها", "معلمة", "معلمتك",
        "حياته", "قضاة", "ثلاثة", "بيته", "كتبته", "شركة", "شركته",
        "المعلمة في مدرستها", "كتاب",
    ])
    def test_roundtrip(self, text):
        tok = _stub_tokenizer()
        assert tok.decode(tok.encode(text).input_ids) == text

    def test_mid_word_taa_splits_the_word(self):
        tok = _stub_tokenizer()
        assert tok.decode(tok.encode("مدرسةكتاب").input_ids) == "مدرسة كتاب"

    def test_no_unk_for_taa_anywhere(self):
        tok = _stub_tokenizer()
        unk = tok.special_tokens["unk_token"]
        for text in ("مدرسة", "شركة", "ة", "مدرسةكتاب", "شركته"):
            assert unk not in tok.encode(text).input_ids

    def test_metric_strings_concatenate_to_the_word(self):
        tok = _stub_tokenizer()
        for text in ("مدرسة", "مدرسته", "شركة", "شركته"):
            out = tok.encode(text)
            # ROOT token's metric string (the bare root) is not part of the surface.
            toks = [t for t, i in zip(out.tokens, out.input_ids)
                    if t and not tok._reverse_vocab[i].startswith(PFX_ROOT)]
            assert "".join(toks) == text

    def test_decode_taa_becomes_ta_before_pronoun_for_lit_words_too(self):
        tok = _stub_tokenizer()
        v = tok._vocab
        ids = [v["[LIT_BEGIN]"], v["[CHAR_ش]"], v["[CHAR_ر]"], v["[CHAR_ك]"], v["[LIT_END]"],
               v["[CLITICE_ة]"], v["[CLITICE_ه]"]]
        assert tok.decode(ids) == "شركته"
