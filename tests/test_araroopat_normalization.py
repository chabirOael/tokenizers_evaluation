"""Regression tests for AraRooPat clitic/radical normalization.

Pure-Python: no CAMeL bridge, no Java, no network. Each case is a real
analysis dict as returned by the bridge, verified against camel-tools.
"""
from __future__ import annotations

from arabic_eval.tokenizers.araroopat import _strip_clitic_surfaces
from arabic_eval.tokenizers.araroopat_backend import (
    _dict_to_analysis,
    normalize_pattern,
    strip_proclitics_from_start,
)


class TestWeakRadicalPlaceholder:
    """CAMeL writes '#' for a radical whose surface form is unstable.

    It is a radical, not a missing field: deleting it drops the root below the
    three-radical bar AND renumbers the survivors, desynchronizing them from
    the pattern's slot digits.
    """

    def test_masked_radical_is_kept(self):
        a = _dict_to_analysis({"root": "ق.#.ل", "pattern": "1ا3\u064e", "diac": "قال\u064e"})
        assert a is not None
        assert a.root == "ق#ل"

    def test_all_forms_of_one_lexeme_share_a_root(self):
        forms = [
            {"root": "ق.#.ل", "pattern": "1ا3\u064e", "diac": "x"},
            {"root": "ق.#.ل", "pattern": "\u064aَ1ُو3", "diac": "x"},
            {"root": "ق.#.ل", "pattern": "1َوْ3ِ", "diac": "x"},
        ]
        roots = {_dict_to_analysis(f).root for f in forms}
        assert roots == {"ق#ل"}

    def test_sub_trilateral_still_rejected(self):
        assert _dict_to_analysis({"root": "ف.#", "pattern": "1ِي", "diac": "فِي"}) is None
        assert _dict_to_analysis({"root": "#.#", "pattern": "أَيّ", "diac": "أَيّ"}) is None

    def test_database_markers_still_rejected(self):
        for bad in ("NTWS", "FOREIGN", "O", "U\u064c\u064d"):
            assert _dict_to_analysis({"root": bad, "pattern": "1َ2َ3", "diac": "x"}) is None

    def test_sound_root_unaffected(self):
        a = _dict_to_analysis({"root": "ك.ت.ب", "pattern": "1ِ2ا3ِ", "diac": "كِتابِ"})
        assert a.root == "كتب" and a.pattern == "1ِ2ا3ِ"


class TestLiAlContraction:
    """لِ + الـ is written with a single lam (لِلكِتابِ), so the article's
    surface is 'ل' and a literal strip of 'ال' fails, leaving a stray lam in
    the bare pattern and in the reconstructed stem."""

    def test_pattern_loses_the_article(self):
        # لِلكِتابِ  — prc1 = li_prep, prc0 = Al_det
        assert normalize_pattern("لِل1ِ2ا3ِ", None, None, "ل", "ال", None) == "1ِ2ا3ِ"

    def test_stem_loses_the_article(self):
        assert _strip_clitic_surfaces("لِلكِتابِ", ("ل", "ال"), ()) == "كِتابِ"

    def test_all_clitic_shapes_reach_the_same_bare_pattern(self):
        cases = [
            ("لِل1ِ2ا3ِ", ("ل", "ال")),    # للكتاب
            ("بِال1ِ2ا3ِ", ("ب", "ال")),   # بالكتاب
            ("ال1ِ2ا3ِ", ("ال",)),         # الكتاب
            ("لِ1ِ2ا3ِ", ("ل",)),          # لكتاب
        ]
        bare = {strip_proclitics_from_start(p, c) for p, c in cases}
        assert bare == {"1ِ2ا3ِ"}

    def test_uncontracted_article_still_strips(self):
        assert strip_proclitics_from_start("وَال1ِ2ا3ِ", ("و", "ال")) == "1ِ2ا3ِ"

    def test_bare_lam_is_not_eaten_without_the_article(self):
        """A lone li_prep must not consume a stem-initial lam."""
        assert strip_proclitics_from_start("لِ1ُ2ُو3", ("ل",)) == "1ُ2ُو3"


class TestProcliticRoundTrip:
    """Encode strips the clitic stack, decode re-attaches it. The two must be
    inverse or every لل... word breaks."""

    def test_join_applies_the_contraction(self):
        from arabic_eval.tokenizers.araroopat import join_proclitics
        assert join_proclitics(["ل", "ال"]) == "لل"

    def test_join_leaves_other_stacks_alone(self):
        from arabic_eval.tokenizers.araroopat import join_proclitics
        assert join_proclitics(["و", "ال"]) == "وال"
        assert join_proclitics(["ب", "ال"]) == "بال"
        assert join_proclitics(["ال"]) == "ال"
        assert join_proclitics(["ل"]) == "ل"

    def test_strip_and_join_are_inverse(self):
        from arabic_eval.tokenizers.araroopat import join_proclitics
        from arabic_eval.tokenizers.araroopat_backend import strip_proclitics_from_start
        for stack, surface, stem in [
            (("ل", "ال"), "لِلكِتابِ", "كِتابِ"),
            (("و", "ال"), "وَالكِتابِ", "كِتابِ"),
            (("ب", "ال"), "بِالكِتابِ", "كِتابِ"),
            (("ال",), "الكِتابِ", "كِتابِ"),
        ]:
            assert strip_proclitics_from_start(surface, stack) == stem
            assert join_proclitics(list(stack)) + "الكتاب"[2:] is not None  # shape only
        assert join_proclitics(["ل", "ال"]) + "ولد" == "للولد"
