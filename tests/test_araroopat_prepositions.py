"""AraRooPat closed-class prepositions → one ``[PREP_*]`` token each.

Pure-Python: no CAMeL bridge, no Java. The analysis dicts are what the
bridge returns for these words (verified against camel-tools on
2026-09-13); encode/decode run against a stub analyzer that serves them.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Optional

import pytest

from arabic_eval.tokenizers.araroopat import (
    PFX_CHAR,
    PFX_CLITICE,
    PFX_CLITICP,
    PFX_PREP,
    PFX_ROOT,
    SFX,
    AraRooPatTokenizer,
)
from arabic_eval.tokenizers.araroopat_backend import (
    PREPOSITION_INVENTORY,
    Analysis,
    CorpusEntry,
    MorphAnalyzer,
    _dict_to_analysis,
    join_particle_enclitic,
)

PARTICLES = frozenset(PREPOSITION_INVENTORY)


def cam(lex: str, diac: str, root: str = "", pattern: str = "", pos: str = "prep",
        **clitics: str) -> Dict[str, str]:
    """A trimmed bridge dict with CAMeL's '0' for absent clitic slots."""
    d = {"lex": lex, "diac": diac, "root": root, "pattern": pattern, "pos": pos,
         "stem": "", "prc3": "0", "prc2": "0", "prc1": "0", "prc0": "0", "enc0": "0"}
    d.update(clitics)
    return d


# Real bridge output for the words the tests use.
CAMEL: Dict[str, Dict[str, str]] = {
    "من": cam("مِن", "مِن", "م.ن", "1ِ2"),
    "إلى": cam("إِلَى", "إِلَى", "#.ل.#", "إِ2َى"),
    "في": cam("فِي", "فِي", "ف.#", "1ِي"),
    "حتى": cam("حَتَّى", "حَتَّى", "ح.ت.ت", "1َ2َّى"),
    "منذ": cam("مُنْذُ", "مُنْذُ", "NTWS", "NTWS", pos="conj"),
    "حاشا": cam("حاش", "حاشا", "ح.#.ش", "1ا3ا", pos="verb"),
    "لعل": cam("لَعَلَّ", "لَعَلَّ", "ع.ل.ل", "لَ1َ2َّ", pos="verb_pseudo", prc1="la_emph"),
    "عليه": cam("عَلَى", "عَلَيهِ", "ع.ل.#", "1َ2َيهِ", enc0="3ms_pron"),
    "منهم": cam("مِن", "مِنهُم", "م.ن", "1ِ2هُم", enc0="3mp_pron"),
    "وإلى": cam("إِلَى", "وَإِلَى", "#.ل.#", "وَإِ2َى", prc2="wa_conj"),
    "لعله": cam("لَعَلَّ", "لَعَلَّهُ", "ع.ل.ل", "لَ1َ2َّهُ", pos="verb_pseudo",
                prc1="la_emph", enc0="3ms_pron"),
    "مما": cam("مِن", "مِمّا", "م.ن", "1ِمّا", enc0="mA_rel"),
    "ممن": cam("مِن", "مِمَّن", "م.ن", "1ِمَّ2", enc0="man_rel"),
    "عمن": cam("عَمَّن", "عَمَّن", "ع.م.ن", "1َ2َّ3", enc0="man_rel"),
    "مني": cam("مِن", "مِنِّي", "م.ن", "1ِ2ِّي", enc0="1s_pron"),
    # ي-spelled bare forms: CAMeL normalizes the lemma to ى.
    "علي": cam("عَلَى", "عَلَى", "ع.ل.#", "1َ2َى"),
    "إلي": cam("إِلَى", "إِلَى", "#.ل.#", "إِ2َى"),
    # Nouns that share letters with particles.
    "ربنا": cam("رَبّ", "رَبّنا", "ر.ب.ب", "1َ2ّنا", pos="noun", enc0="1p_poss"),
    "كتاب": cam("كِتاب", "كِتاب", "ك.ت.ب", "1ِ2ا3", pos="noun"),
    "الكتاب": cam("كِتاب", "الكِتاب", "ك.ت.ب", "ال1ِ2ا3", pos="noun", prc0="Al_det"),
    "لكي": cam("كَي", "لِكَي", "ك.#", "لِ1َيْ", pos="conj"),
}


def analyze(word: str) -> Optional[Analysis]:
    return _dict_to_analysis(CAMEL[word], PARTICLES, word)


# ---------------------------------------------------------------------------
# The intercept in _dict_to_analysis
# ---------------------------------------------------------------------------

class TestParticleIntercept:
    @pytest.mark.parametrize("word", ["من", "إلى", "في", "حتى", "منذ", "حاشا", "لعل"])
    def test_bare_particle(self, word):
        a = analyze(word)
        assert a is not None and a.particle == word
        assert a.root == "" and a.pattern == ""
        assert not any((a.prc3, a.prc2, a.prc1, a.prc0, a.enc0))

    def test_two_radical_words_were_lit_before(self):
        """The intercept runs before the ≥3-radical gate."""
        assert _dict_to_analysis(CAMEL["من"], frozenset()) is None
        assert analyze("من").particle == "من"

    def test_enclitic_is_split_off(self):
        a = analyze("عليه")
        assert (a.particle, a.enc0) == ("على", "ه")
        a = analyze("منهم")
        assert (a.particle, a.enc0) == ("من", "هم")

    def test_proclitic_is_split_off(self):
        a = analyze("وإلى")
        assert (a.prc2, a.particle) == ("و", "إلى")

    def test_spurious_proclitic_on_lacalla_is_dropped(self):
        """CAMeL reads لعل as لَ + عَلَّ; emitting that ل would decode to للعل."""
        assert analyze("لعل").prc1 is None
        a = analyze("لعله")
        assert (a.prc1, a.particle, a.enc0) == (None, "لعل", "ه")

    @pytest.mark.parametrize("word,particle,enc", [
        ("مما", "من", "ما"), ("ممن", "من", "من"), ("عمن", "عن", "من"),
    ])
    def test_assimilated_forms(self, word, particle, enc):
        a = analyze(word)
        assert (a.particle, a.enc0) == (particle, enc)

    def test_yeh_spelled_bare_forms_are_not_particles(self):
        """علي is also the name Ali; إلي must decode back verbatim, not as إلى.

        They fall through to the ordinary root+pattern gate instead."""
        assert analyze("علي").particle is None
        assert analyze("إلي").particle is None

    def test_noun_readings_are_left_alone(self):
        assert analyze("ربنا") is None or analyze("ربنا").particle is None
        assert analyze("الكتاب").particle is None
        assert analyze("كتاب").particle is None

    def test_unreported_proclitic_is_not_invented(self):
        """CAMeL gives no clitic for لكي, so it is not كي with a ل."""
        assert analyze("لكي") is None

    def test_surface_fallback_when_camel_has_nothing(self):
        a = MorphAnalyzer._first_valid([], "حتى", PARTICLES)
        assert a is not None and a.particle == "حتى"
        assert MorphAnalyzer._first_valid([], "كتاب", PARTICLES) is None

    def test_inventory_is_configurable(self):
        assert _dict_to_analysis(CAMEL["إلى"], frozenset({"من"}), "إلى").particle is None


# ---------------------------------------------------------------------------
# Decode-side join (inverse of the enc0 split)
# ---------------------------------------------------------------------------

class TestJoinParticleEnclitic:
    @pytest.mark.parametrize("p,e,expected", [
        ("إلى", "ه", "إليه"), ("على", "هم", "عليهم"), ("حتى", "ه", "حتيه"),
        ("من", "ه", "منه"), ("عن", "ها", "عنها"), ("لعل", "ه", "لعله"),
        ("من", "ما", "مما"), ("من", "من", "ممن"), ("عن", "ما", "عما"), ("عن", "من", "عمن"),
        ("من", "ي", "مني"), ("إلى", "ي", "إلي"), ("في", "ي", "في"), ("على", "ي", "علي"),
    ])
    def test_join(self, p, e, expected):
        assert join_particle_enclitic(p, e) == expected

    def test_encode_acceptance_is_the_inverse(self):
        for word in ("عليه", "منهم", "مما", "ممن", "عمن", "مني", "لعله"):
            a = analyze(word)
            assert join_particle_enclitic(a.particle, a.enc0) == word


# ---------------------------------------------------------------------------
# CorpusEntry / vocab layout / encode+decode with a stub analyzer
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
        root_freq=Counter({"كتب": 5, "ربب": 3}),
        pat_freq=Counter({"1ِ2ا3": 5, "1َ2": 3}),
        proclitic_freq=Counter({"و": 5, "ال": 5}),
        enclitic_freq=Counter({"ه": 5, "هم": 5, "ما": 5, "من": 2, "ي": 2, "نا": 1}),
    )
    tok._reconstruction = {
        (tok._vocab[f"{PFX_ROOT}كتب{SFX}"], tok._vocab["[PAT_1ِ2ا3]"]): "كتاب",
        (tok._vocab[f"{PFX_ROOT}ربب{SFX}"], tok._vocab["[PAT_1َ2]"]): "رب",
    }
    return tok


class TestCorpusEntry:
    def test_particle_round_trips_through_dict(self):
        e = CorpusEntry.from_analysis("عليه", analyze("عليه"))
        assert e.analyzed and e.particle == "على" and e.root is None
        assert e.enclitics == ("ه",)
        assert CorpusEntry.from_dict(e.to_dict()) == e


class TestVocabLayout:
    def test_prep_range_sits_between_enclitics_and_chars(self):
        tok = _stub_tokenizer()
        ids = {t: i for t, i in tok._vocab.items()}
        last_enc = max(i for t, i in ids.items() if t.startswith(PFX_CLITICE))
        prep_ids = [i for t, i in ids.items() if t.startswith(PFX_PREP)]
        first_char = min(i for t, i in ids.items() if t.startswith(PFX_CHAR))
        assert last_enc < min(prep_ids) and max(prep_ids) < first_char
        # Fixed inventory order → deterministic IDs regardless of corpus.
        assert [tok._reverse_vocab[i] for i in sorted(prep_ids)] == \
            [f"{PFX_PREP}{p}{SFX}" for p in PREPOSITION_INVENTORY]

    def test_rb_is_not_in_the_inventory(self):
        assert "رب" not in PREPOSITION_INVENTORY


class TestEncodeDecode:
    def _stream(self, tok, text):
        out = tok.encode(text)
        return out, [tok._reverse_vocab[i] for i in out.input_ids
                     if tok._reverse_vocab[i] not in ("<s>", "</s>")]

    def test_each_bare_particle_is_one_token(self):
        tok = _stub_tokenizer()
        for p in ("من", "إلى", "في", "حتى", "منذ", "حاشا", "لعل"):
            out, stream = self._stream(tok, p)
            assert stream == [f"{PFX_PREP}{p}{SFX}"]
            assert tok.decode(out.input_ids) == p

    def test_clitics_ride_outside_the_particle(self):
        tok = _stub_tokenizer()
        _, stream = self._stream(tok, "وإلى")
        assert stream == [f"{PFX_CLITICP}و{SFX}", f"{PFX_PREP}إلى{SFX}"]
        _, stream = self._stream(tok, "عليه")
        assert stream == [f"{PFX_PREP}على{SFX}", f"{PFX_CLITICE}ه{SFX}"]

    @pytest.mark.parametrize("text", [
        "عليه", "منهم", "وإلى", "لعله", "مما", "ممن", "عمن", "مني",
        "من الكتاب إلى ربنا", "علي", "إلي", "لكي",
    ])
    def test_roundtrip(self, text):
        tok = _stub_tokenizer()
        out = tok.encode(text)
        assert tok.decode(out.input_ids) == text

    def test_metric_string_is_the_surface_core(self):
        """علي (not the lemma على) so aligned_token_offsets can rebuild عليه."""
        tok = _stub_tokenizer()
        out = tok.encode("عليه")
        assert [t for t in out.tokens if t] == ["علي", "ه"]
        out = tok.encode("مما")
        assert "".join(t for t in out.tokens if t) == "مما"

    def test_oov_clitic_sends_whole_chunk_to_lit(self):
        """No half-tokenized word: an absent [CLITICE_*] means LIT for the chunk."""
        tok = _stub_tokenizer()
        del tok._vocab[f"{PFX_CLITICE}هم{SFX}"]
        tok._reverse_vocab = {i: t for t, i in tok._vocab.items()}
        out, stream = self._stream(tok, "منهم")
        assert stream[0] == "[LIT_BEGIN]" and f"{PFX_PREP}من{SFX}" not in stream
        assert tok.decode(out.input_ids) == "منهم"

    def test_noun_enclitic_join_is_untouched(self):
        """The ى→ي rule is preposition-only; ربنا still concatenates plainly."""
        tok = _stub_tokenizer()
        out = tok.encode("ربنا")
        assert tok.decode(out.input_ids) == "ربنا"
