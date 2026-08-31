"""Unit tests for the morphological metrics refactor.

Covers:
  1. Regression on the 5 pre-existing metrics (root/pattern conservation,
     morpheme integrity, root/pattern bearing token pct).
  2. The new `_clitic_boundaries` pure function (13 cases).
  3. `_morpheme_metrics_for_word` end-to-end with a mock Farasa segmenter,
     including the critical fairness cases:
       - SFR data must accumulate when alignment fails.
       - SFR data must accumulate on single-morpheme words.
       - SFR uses the *raw* (pre-clean) token count, not the cleaned one.
  4. Aggregation behavior in `compute_morphological_metrics` —
     empty sample, Farasa-disabled, all-singletons, alignment-all-fail.
  5. Cross-metric invariants (integrity == 1.0 ⇒ CSA == 1.0; integrity
     and CSA can disagree in general).
  6. Data integrity of `_PROCLITIC_TAGS`/`_ENCLITIC_TAGS` against
     `CAMEL_CLITIC_SURFACE`.

Runs without Java (no real Farasa) and without qalsadi (no real root
extractor for §1) — both are mocked. Should complete in <1s.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from arabic_eval.evaluation.intrinsic_metrics import (
    _clitic_boundaries,
    _empty_morph_metrics,
    _morpheme_metrics_for_word,
    compute_morphological_metrics,
)
from arabic_eval.tokenizers.araroopat_backend import (
    CAMEL_CLITIC_SURFACE,
    ENCLITIC_SURFACES,
    PROCLITIC_SURFACES,
    _ENCLITIC_TAGS,
    _PROCLITIC_TAGS,
)
from arabic_eval.tokenizers.base import BaseTokenizer, TokenizerOutput


# ===========================================================================
# Helpers — minimal mocks
# ===========================================================================

class _FakeSegmenter:
    """Mock MorphemeSegmenter driven by a precomputed dict."""

    def __init__(self, table: Dict[str, Optional[List[str]]]):
        self._table = table

    def segment_word(self, word: str) -> Optional[List[str]]:
        return self._table.get(word)


class _FakeTokenizer(BaseTokenizer):
    """Mock BaseTokenizer where `encode(word)` returns precomputed tokens."""

    def __init__(self, table: Dict[str, List[str]]):
        super().__init__()
        self._table = table

    @property
    def vocab_size(self) -> int:
        return 1000

    @property
    def embedding_type(self) -> str:
        return "standard"

    @property
    def special_tokens(self) -> dict:
        return {"pad_token": 0, "bos_token": 1, "eos_token": 2, "unk_token": 3}

    def train(self, texts, vocab_size=None, **kw):
        pass

    def encode(self, text: str, **kw):
        toks = self._table.get(text, [text])
        ids = list(range(len(toks)))
        return TokenizerOutput(
            input_ids=ids, attention_mask=[1] * len(ids), tokens=toks,
        )

    def decode(self, ids, **kw):
        return ""

    def save(self, p):
        pass

    def load(self, p):
        pass

    def get_embedding_config(self):
        return {}


# ===========================================================================
# §6. Data integrity: tag buckets vs CAMEL_CLITIC_SURFACE
# ===========================================================================

class TestCliticTagBuckets:
    def test_buckets_partition_camel_table(self):
        """Every CAMEL_CLITIC_SURFACE key must appear in exactly one bucket."""
        union = _PROCLITIC_TAGS | _ENCLITIC_TAGS
        table_keys = set(CAMEL_CLITIC_SURFACE.keys())
        assert union == table_keys, (
            f"Bucket coverage mismatch: missing={table_keys - union}, "
            f"extra={union - table_keys}"
        )

    def test_buckets_disjoint(self):
        assert _PROCLITIC_TAGS.isdisjoint(_ENCLITIC_TAGS)

    def test_proclitic_surfaces_include_canonical_set(self):
        # The minimum set every Arabic NLP pipeline expects.
        canon = {"و", "ال", "ف", "ب", "ل", "ك", "س"}
        assert canon <= PROCLITIC_SURFACES

    def test_enclitic_surfaces_include_canonical_set(self):
        canon = {"ه", "ها", "هم", "ك", "نا", "ي", "كم"}
        assert canon <= ENCLITIC_SURFACES

    def test_ka_appears_in_both_surface_sets(self):
        # ك is proclitic ka_prep "like" AND enclitic 2ms pronouns; the
        # shared surface is the whole reason CSA must use position-based
        # disambiguation.
        assert "ك" in PROCLITIC_SURFACES
        assert "ك" in ENCLITIC_SURFACES


# ===========================================================================
# §2. _clitic_boundaries pure function
# ===========================================================================

class TestCliticBoundaries:
    @pytest.mark.parametrize("morphemes,expected", [
        ([], set()),
        (["كتاب"], set()),
        (["و", "كتاب"], {1}),
        (["ال", "كتاب"], {2}),
        (["و", "ال", "كتاب"], {1, 3}),
        (["كتاب", "ها"], {4}),
        (["كتاب", "كم"], {4}),
        (["و", "ال", "كتاب", "ها"], {1, 3, 7}),
        (["ل", "كتاب", "ك"], {1, 5}),
        (["ك", "كتاب"], {1}),
        (["كتاب", "ون"], set()),
        (["وَ", "الْ", "كِتَاب"], {1, 3}),
        (["كتاب", "و", "ها"], {5}),
    ])
    def test_clitic_boundary_cases(self, morphemes, expected):
        assert _clitic_boundaries(morphemes) == expected


# ===========================================================================
# §3. _morpheme_metrics_for_word — mock segmenter
# ===========================================================================

class TestMorphemeMetricsForWord:
    def test_farasa_returns_none(self):
        seg = _FakeSegmenter({"كتاب": None})
        assert _morpheme_metrics_for_word("كتاب", ["كتاب"], seg) is None

    def test_single_morpheme_word(self):
        """SFR must accumulate on single-morpheme words; integrity/CSA = None."""
        seg = _FakeSegmenter({"كتاب": ["كتاب"]})
        out = _morpheme_metrics_for_word("كتاب", ["كتاب"], seg)
        assert out == {
            "integrity": None,
            "csa_respected": None,
            "csa_total": None,
            "morpheme_count": 1,
            "raw_token_count": 1,
        }

    def test_perfect_alignment_perfect_tokenization(self):
        seg = _FakeSegmenter({"والكتاب": ["و", "ال", "كتاب"]})
        out = _morpheme_metrics_for_word(
            "والكتاب", ["و", "ال", "كتاب"], seg
        )
        assert out["integrity"] == 1.0
        assert out["csa_respected"] == 2
        assert out["csa_total"] == 2
        assert out["morpheme_count"] == 3
        assert out["raw_token_count"] == 3

    def test_single_token_no_boundaries_respected(self):
        seg = _FakeSegmenter({"والكتاب": ["و", "ال", "كتاب"]})
        out = _morpheme_metrics_for_word("والكتاب", ["والكتاب"], seg)
        assert out["integrity"] == 0.0
        assert out["csa_respected"] == 0
        assert out["csa_total"] == 2
        assert out["morpheme_count"] == 3
        assert out["raw_token_count"] == 1

    def test_alignment_failure_still_returns_sfr_data(self):
        """The critical fairness test: token/morpheme counts flow through
        even when alignment fails (e.g. ByteLevel BPE artifacts).
        """
        seg = _FakeSegmenter({"والكتاب": ["و", "ال", "كتاب"]})
        # "XYZGARBAGE" cleans to empty (no Arabic letters) → alignment fails
        out = _morpheme_metrics_for_word("والكتاب", ["XYZGARBAGE"], seg)
        assert out["integrity"] is None
        assert out["csa_respected"] is None
        assert out["csa_total"] is None
        # SFR data must still be present.
        assert out["morpheme_count"] == 3
        assert out["raw_token_count"] == 1

    def test_no_clitic_boundaries_returns_csa_none(self):
        """If a word has internal morpheme boundaries but none are clitic
        boundaries, CSA should be None (no clitic boundaries to score)
        while integrity is still scored.
        """
        # Hypothetical: stem split into two non-clitic segments.
        seg = _FakeSegmenter({"كتابون": ["كتاب", "ون"]})
        out = _morpheme_metrics_for_word("كتابون", ["كتاب", "ون"], seg)
        assert out["integrity"] == 1.0
        assert out["csa_respected"] is None
        assert out["csa_total"] is None
        assert out["morpheme_count"] == 2
        assert out["raw_token_count"] == 2

    def test_specials_excluded_from_raw_token_count(self):
        seg = _FakeSegmenter({"كتاب": ["كتاب"]})
        out = _morpheme_metrics_for_word(
            "كتاب", ["<s>", "كتاب", "</s>"], seg
        )
        assert out["raw_token_count"] == 1

    def test_byte_level_tokens_count_for_sfr(self):
        """Charformer-style: tokens that all clean to empty Arabic still
        count toward SFR's numerator (raw count, not cleaned count).
        """
        seg = _FakeSegmenter({"والكتاب": ["و", "ال", "كتاب"]})
        # 14 fake byte tokens that all clean to empty.
        byte_tokens = ["X"] * 14
        out = _morpheme_metrics_for_word("والكتاب", byte_tokens, seg)
        assert out["raw_token_count"] == 14
        assert out["morpheme_count"] == 3


# ===========================================================================
# §1 + §4 + §5. compute_morphological_metrics: regression, aggregation,
#                invariants. Mocks both root extraction and Farasa.
# ===========================================================================

@pytest.fixture
def patched_root_extractor():
    """Patch RootExtractor.extract to a fixed table for predictable tests."""
    table = {
        "والكتاب": "كتب",
        "كتابهم":  "كتب",
        "والكاتب": "كتب",
        "كتاب":    "كتب",
        "مدرسة":   "درس",
        "والمدرسة": "درس",
        "يدرس":    "درس",
    }
    with patch(
        "arabic_eval.evaluation.intrinsic_metrics.RootExtractor"
    ) as RE:
        instance = MagicMock()
        instance.extract.side_effect = lambda w: table.get(w)
        # Secondary (tashaphyne) extractor agrees with the primary here, so
        # root_extractor_agreement is 1.0 and the pre-existing assertions
        # in this module stay unaffected.
        instance.extract_secondary.side_effect = lambda w: table.get(w)
        RE.return_value = instance
        yield


@pytest.fixture
def patched_segmenter():
    """Patch MorphemeSegmenter to a fixed segmentation table."""
    table = {
        "والكتاب":  ["و", "ال", "كتاب"],
        "كتابهم":   ["كتاب", "هم"],
        "والكاتب":  ["و", "ال", "كاتب"],
        "كتاب":     ["كتاب"],
        "مدرسة":    ["مدرسة"],
        "والمدرسة": ["و", "ال", "مدرسة"],
        "يدرس":     ["يدرس"],
    }
    with patch(
        "arabic_eval.evaluation.intrinsic_metrics.MorphemeSegmenter"
    ) as MS:
        instance = MagicMock()
        instance.segment_word.side_effect = lambda w: table.get(w)
        MS.return_value = instance
        yield


def _texts_with_words(words):
    """One word per text — _sample_words operates on whitespace splits."""
    return list(words)


# ----- §4: empty / disabled / aggregation edge cases ----------------------

class TestAggregation:
    def test_empty_sample_returns_full_shape(self):
        em = _empty_morph_metrics()
        # All seven metric keys + sample size, with the new ones included.
        expected_keys = {
            "root_conservation_rate",
            "pattern_conservation_rate",
            "morpheme_integrity_rate",
            "clitic_separation_accuracy",
            "semantic_fragmentation_ratio",
            "root_bearing_token_pct",
            "pattern_bearing_token_pct",
            "morph_sample_size",
            "root_measurable_pct",
            "root_conservation_attainable",
            "morph_alignment_coverage",
            "morph_alignment_ceiling",
            "root_extractor_agreement",
        }
        assert set(em.keys()) == expected_keys
        assert all(em[k] is None for k in expected_keys - {"morph_sample_size"})
        assert em["morph_sample_size"] == 0

    def test_use_farasa_false_disables_three_metrics(
        self, patched_root_extractor
    ):
        tok = _FakeTokenizer({"كتاب": ["كتاب"]})
        m = compute_morphological_metrics(
            tok, _texts_with_words(["كتاب"]),
            sample_size=10, use_farasa=False,
        )
        # Farasa-dependent metrics → None.
        assert m["morpheme_integrity_rate"] is None
        assert m["clitic_separation_accuracy"] is None
        assert m["semantic_fragmentation_ratio"] is None
        # Root/pattern still computed.
        assert m["root_conservation_rate"] is not None

    def test_all_singletons_sfr_real_integrity_none(
        self, patched_root_extractor, patched_segmenter
    ):
        tok = _FakeTokenizer({
            "كتاب": ["كتاب"],
            "مدرسة": ["مد", "رسة"],   # 2 raw tokens, 1 morpheme
            "يدرس": ["يدرس"],
        })
        m = compute_morphological_metrics(
            tok, _texts_with_words(["كتاب", "مدرسة", "يدرس"]),
            sample_size=10,
        )
        # All three words are single-morpheme → integrity / CSA = None.
        assert m["morpheme_integrity_rate"] is None
        assert m["clitic_separation_accuracy"] is None
        # SFR is real: (1 + 2 + 1) / (1 + 1 + 1) = 4/3.
        assert m["semantic_fragmentation_ratio"] == round(4 / 3, 4)


# ----- §1 + §5: regression and invariants on a multi-word sample ---------

class TestEndToEndRegression:
    def test_perfect_morpho_aware_tokenizer(
        self, patched_root_extractor, patched_segmenter
    ):
        """A tokenizer that splits exactly at Farasa boundaries should hit
        integrity = 1.0 and CSA = 1.0 (invariant test §5.1).
        """
        tok = _FakeTokenizer({
            "والكتاب": ["و", "ال", "كتاب"],
            "كتابهم":  ["كتاب", "هم"],
            "والكاتب": ["و", "ال", "كاتب"],
        })
        m = compute_morphological_metrics(
            tok,
            _texts_with_words(["والكتاب", "كتابهم", "والكاتب"]),
            sample_size=10,
        )
        assert m["morpheme_integrity_rate"] == 1.0
        # Invariant §5.1: integrity == 1.0 ⇒ CSA == 1.0.
        assert m["clitic_separation_accuracy"] == 1.0
        # SFR: 8 raw tokens / 8 morphemes = 1.0.
        assert m["semantic_fragmentation_ratio"] == 1.0

    def test_word_level_tokenizer_csa_zero_integrity_zero(
        self, patched_root_extractor, patched_segmenter
    ):
        """CharBERT-style: one token per word → no internal boundaries
        → integrity = 0, CSA = 0, SFR < 1.
        """
        tok = _FakeTokenizer({
            "والكتاب": ["والكتاب"],
            "كتابهم":  ["كتابهم"],
            "والكاتب": ["والكاتب"],
        })
        m = compute_morphological_metrics(
            tok,
            _texts_with_words(["والكتاب", "كتابهم", "والكاتب"]),
            sample_size=10,
        )
        assert m["morpheme_integrity_rate"] == 0.0
        assert m["clitic_separation_accuracy"] == 0.0
        # 3 raw tokens / 8 morphemes = 0.375.
        assert m["semantic_fragmentation_ratio"] == round(3 / 8, 4)

    def test_csa_can_disagree_with_integrity(
        self, patched_root_extractor, patched_segmenter
    ):
        """Invariant §5.2: integrity < 1.0 does NOT force CSA < 1.0.

        Construct: tokenization respects clitic boundaries but breaks a
        stem-internal boundary → CSA = 1.0 but integrity < 1.0.
        """
        # We need a word where Farasa returns ≥3 morphemes including a
        # stem-internal split. Use a synthetic case and a tokenization
        # that respects only the clitic boundaries.
        with patch(
            "arabic_eval.evaluation.intrinsic_metrics.RootExtractor"
        ) as RE, patch(
            "arabic_eval.evaluation.intrinsic_metrics.MorphemeSegmenter"
        ) as MS:
            RE.return_value.extract.side_effect = lambda w: "كتب"
            MS.return_value.segment_word.side_effect = (
                lambda w: ["و", "كت", "اب"]
            )  # و is proclitic; كت/اب is a stem-internal split
            # Tokenizer respects only the clitic boundary (و|كتاب), not the
            # stem split (كت|اب).
            tok = _FakeTokenizer({"وكتاب": ["و", "كتاب"]})
            m = compute_morphological_metrics(
                tok, _texts_with_words(["وكتاب"]), sample_size=10,
            )
        assert m["clitic_separation_accuracy"] == 1.0
        assert m["morpheme_integrity_rate"] == 0.5  # 1 of 2 boundaries

    def test_alignment_failure_does_not_break_sfr(
        self, patched_root_extractor, patched_segmenter
    ):
        """Even when every word fails alignment, SFR must still report a
        real number (raw counts are alignment-free).
        """
        tok = _FakeTokenizer({
            # Tokens that can't reconstruct (clean to empty).
            "والكتاب": ["XYZ"],
            "كتابهم":  ["UVW"],
        })
        m = compute_morphological_metrics(
            tok, _texts_with_words(["والكتاب", "كتابهم"]), sample_size=10,
        )
        # Integrity / CSA cannot be computed.
        assert m["morpheme_integrity_rate"] is None
        assert m["clitic_separation_accuracy"] is None
        # SFR: 2 raw tokens / 5 morphemes = 0.4.
        assert m["semantic_fragmentation_ratio"] == round(2 / 5, 4)


# ===========================================================================
# §7. Ceiling / coverage diagnostics
#
# These four fields exist to stop the four headline metrics being misread:
#   - root_measurable_pct           how much of the sample is winnable at all
#   - root_conservation_attainable  RPS over only the winnable part
#   - morph_alignment_coverage      what share of words integrity/CSA saw
#   - root_extractor_agreement      how stable the ground truth itself is
# ===========================================================================

def _patched_extractors(primary, secondary=None):
    """Patch RootExtractor with explicit primary/secondary tables."""
    secondary = primary if secondary is None else secondary
    ctx = patch("arabic_eval.evaluation.intrinsic_metrics.RootExtractor")
    RE = ctx.__enter__()
    instance = MagicMock()
    instance.extract.side_effect = lambda w: primary.get(w)
    instance.extract_secondary.side_effect = lambda w: secondary.get(w)
    RE.return_value = instance
    return ctx


class TestCeilingDiagnostics:
    def test_weak_root_is_not_measurable(self):
        """قال has root قول, but و never surfaces — no tokenizer can win it."""
        ctx = _patched_extractors({"قال": "قول", "كتاب": "كتب"})
        try:
            tok = _FakeTokenizer({"قال": ["قال"], "كتاب": ["كتاب"]})
            m = compute_morphological_metrics(
                tok, _texts_with_words(["قال", "كتاب"]),
                sample_size=10, use_farasa=False,
            )
        finally:
            ctx.__exit__(None, None, None)
        # 1 of 2 words is winnable.
        assert m["root_measurable_pct"] == 50.0
        # Raw rate is dragged down by the unwinnable word ...
        assert m["root_conservation_rate"] == 0.5
        # ... while the attainable rate correctly reads a perfect score.
        assert m["root_conservation_attainable"] == 1.0

    def test_all_sound_roots_measurable(self):
        ctx = _patched_extractors({"كتاب": "كتب", "مدرسة": "درس"})
        try:
            tok = _FakeTokenizer({"كتاب": ["كتاب"], "مدرسة": ["مدرسة"]})
            m = compute_morphological_metrics(
                tok, _texts_with_words(["كتاب", "مدرسة"]),
                sample_size=10, use_farasa=False,
            )
        finally:
            ctx.__exit__(None, None, None)
        assert m["root_measurable_pct"] == 100.0
        assert m["root_conservation_attainable"] == m["root_conservation_rate"] == 1.0

    def test_attainable_never_below_raw(self):
        """Invariant: removing unwinnable words can only help (or tie)."""
        ctx = _patched_extractors({"قال": "قول", "كتاب": "كتب", "مدرسة": "درس"})
        try:
            tok = _FakeTokenizer({
                "قال": ["قال"], "كتاب": ["ك", "تاب"], "مدرسة": ["مدرسة"],
            })
            m = compute_morphological_metrics(
                tok, _texts_with_words(["قال", "كتاب", "مدرسة"]),
                sample_size=10, use_farasa=False,
            )
        finally:
            ctx.__exit__(None, None, None)
        assert m["root_conservation_attainable"] >= m["root_conservation_rate"]

    def test_alignment_coverage_full_for_substring_tokenizer(self):
        ctx = _patched_extractors({"وكتاب": "كتب"})
        try:
            tok = _FakeTokenizer({"وكتاب": ["و", "كتاب"]})
            m = compute_morphological_metrics(
                tok, _texts_with_words(["وكتاب"]), sample_size=10, use_farasa=False,
            )
        finally:
            ctx.__exit__(None, None, None)
        assert m["morph_alignment_coverage"] == 1.0

    def test_alignment_coverage_zero_for_non_substring_tokenizer(self):
        """araroopat-shaped: tokens are a root + a stem, not a cover of the word.

        integrity / CSA are computed only over aligned words, so a coverage
        of 0.0 means those metrics saw none of this tokenizer's real output.
        """
        ctx = _patched_extractors({"يذهب": "ذهب"})
        try:
            tok = _FakeTokenizer({"يذهب": ["ذهب", "يذهب"]})
            m = compute_morphological_metrics(
                tok, _texts_with_words(["يذهب"]), sample_size=10, use_farasa=False,
            )
        finally:
            ctx.__exit__(None, None, None)
        assert m["morph_alignment_coverage"] == 0.0

    def test_extractor_agreement_reports_ground_truth_noise(self):
        ctx = _patched_extractors(
            primary={"والكتاب": "كتب", "مدرسة": "درس"},
            secondary={"والكتاب": "كوب", "مدرسة": "درس"},   # tashaphyne disagrees once
        )
        try:
            tok = _FakeTokenizer({"والكتاب": ["والكتاب"], "مدرسة": ["مدرسة"]})
            m = compute_morphological_metrics(
                tok, _texts_with_words(["والكتاب", "مدرسة"]),
                sample_size=10, use_farasa=False,
            )
        finally:
            ctx.__exit__(None, None, None)
        assert m["root_extractor_agreement"] == 0.5

    def test_alignment_ceiling_excludes_uncleanable_words(self):
        """ى is not in ARABIC_LETTERS, so clean_token_string drops it and the
        word cannot be aligned by any tokenizer — not even a whole-word one."""
        ctx = _patched_extractors({"مستشفى": "شفي", "كتاب": "كتب"})
        try:
            tok = _FakeTokenizer({"مستشفى": ["مستشفى"], "كتاب": ["كتاب"]})
            m = compute_morphological_metrics(
                tok, _texts_with_words(["مستشفى", "كتاب"]),
                sample_size=10, use_farasa=False,
            )
        finally:
            ctx.__exit__(None, None, None)
        assert m["morph_alignment_ceiling"] == 0.5
        assert m["morph_alignment_coverage"] <= m["morph_alignment_ceiling"]
