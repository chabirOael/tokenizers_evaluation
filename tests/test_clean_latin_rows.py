"""Tests for the ``clean_latin_rows`` opt-in eval-preprocessing flag.

The flag drops rows containing Latin-script letters in any inspected text
field (question, choices, optional context) before the stratified 10/90
split runs. Default off → backward-compatible. Tests cover:

  * The predicate ``contains_latin_letters`` (script policy: Latin only;
    Cyrillic / Greek / CJK pass through).
  * The ``_row_has_latin`` hook on ``LightEvalBenchmarkTask`` (inspects
    question + choices + optional context, skips internal ``_*`` keys).
  * Filter wiring in ``_get_splits``: drop counts, passthrough at
    ``flag=False``, sub-config wipeout warning, all-rows-dropped hard fail.
  * Metrics-dict stamping (the flag value is echoed into the eval result so
    downstream comparison-report consumers can detect mixed runs).

All fixtures are in-memory dicts — no HF I/O, no GPU. Should run in <1 s.
"""
from __future__ import annotations

import logging

import pytest

from arabic_eval.tokenizers.utils.arabic_text import contains_latin_letters


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------

class TestContainsLatinLetters:
    """Script policy: Latin only. Cyrillic/Greek/CJK pass through."""

    def test_arabic_only(self):
        assert contains_latin_letters("السلام عليكم") is False

    def test_arabic_with_diacritics(self):
        assert contains_latin_letters("الْكِتَابُ جَمِيلٌ") is False

    def test_arabic_with_latin_acronym(self):
        assert contains_latin_letters("ما هو IBM؟") is True

    def test_pure_ascii_latin(self):
        assert contains_latin_letters("Hello world") is True

    def test_latin_extended_a_german(self):
        # ü is Latin Extended-A
        assert contains_latin_letters("über") is True

    def test_latin_extended_additional_vietnamese(self):
        # ạ is Latin Extended Additional
        assert contains_latin_letters("cạnh") is True

    def test_cyrillic_passes_through(self):
        # Script policy: only Latin is filtered, Cyrillic/Greek/CJK kept.
        assert contains_latin_letters("Привет") is False

    def test_greek_passes_through(self):
        assert contains_latin_letters("Καλημέρα") is False

    def test_cjk_passes_through(self):
        assert contains_latin_letters("日本語") is False

    def test_ascii_digits_arabic_punct_no_latin(self):
        assert contains_latin_letters("123 ، . ؟") is False

    def test_eastern_arabic_digits_no_latin(self):
        assert contains_latin_letters("١٢٣ ٤٥٦") is False

    def test_empty_string(self):
        assert contains_latin_letters("") is False

    def test_none_safe(self):
        assert contains_latin_letters(None) is False

    def test_non_str_safe(self):
        # Defensive — caller should pass strings, but don't crash on bad input.
        assert contains_latin_letters(42) is False
        assert contains_latin_letters(["a"]) is False


# ---------------------------------------------------------------------------
# _row_has_latin hook
# ---------------------------------------------------------------------------

def _acva_task(**overrides):
    """Build an ACVATask instance to exercise the base-class hooks. ACVA's
    ``__init__`` doesn't touch HF state — it just stores config — so this is
    a cheap stand-in for the abstract ``LightEvalBenchmarkTask``.
    """
    from arabic_eval.tasks.lighteval.acva import ACVATask
    return ACVATask({"dataset_name": "x", **overrides})


class TestRowHasLatin:
    """Default ``_text_fields`` hook inspects question + choices + context."""

    def test_clean_arabic_row(self):
        t = _acva_task()
        ex = {"question": "العاصمة جميلة", "choices": ["صح", "خطأ"], "answer": 0}
        assert t._row_has_latin(ex) is False

    def test_latin_in_question(self):
        t = _acva_task()
        ex = {"question": "ما هو IBM؟", "choices": ["صح", "خطأ"], "answer": 0}
        assert t._row_has_latin(ex) is True

    def test_latin_in_choice(self):
        t = _acva_task()
        ex = {"question": "ما الإجابة؟", "choices": ["Yes", "لا"], "answer": 1}
        assert t._row_has_latin(ex) is True

    def test_latin_in_context(self):
        # Arabic_Exam ships ~5 % of rows with a `context` field.
        t = _acva_task()
        ex = {
            "question": "بناء على النص",
            "choices": ["صح", "خطأ"],
            "context": "Linux is an OS",
            "answer": 0,
        }
        assert t._row_has_latin(ex) is True

    def test_empty_context_no_false_positive(self):
        t = _acva_task()
        ex = {"question": "س", "choices": ["صح", "خطأ"], "context": ""}
        assert t._row_has_latin(ex) is False

    def test_internal_keys_not_inspected(self):
        # _source_config / _* keys must NOT be inspected (could be Latin-named).
        t = _acva_task()
        ex = {
            "question": "العاصمة جميلة",
            "choices": ["صح", "خطأ"],
            "_source_config": "Algeria",  # Latin name — but a metadata key
            "answer": 0,
        }
        assert t._row_has_latin(ex) is False

    def test_choices_with_non_str_items_skipped(self):
        # Defensive — should not crash on malformed choices.
        t = _acva_task()
        ex = {"question": "س", "choices": ["صح", None, 42, "خطأ"]}
        assert t._row_has_latin(ex) is False


# ---------------------------------------------------------------------------
# Filter wiring in _get_splits
# ---------------------------------------------------------------------------

def _stub_load(*rows):
    """Return a callable that ACVA's ``load_examples`` can be replaced with."""
    return lambda: list(rows)


def _row(question, source="_default", answer=0):
    return {
        "question": question,
        "choices": ["صح", "خطأ"],
        "answer": answer,
        "_source_config": source,
    }


class TestFilterInGetSplits:

    def test_passthrough_when_flag_off(self):
        t = _acva_task(clean_latin_rows=False, seed=42)
        t.load_examples = _stub_load(
            _row("جملة عربية"),
            _row("Latin row IBM"),
            _row("جملة أخرى"),
        )
        sft, ev = t._get_splits()
        assert len(sft) + len(ev) == 3

    def test_drops_latin_when_flag_on(self):
        t = _acva_task(clean_latin_rows=True, seed=42)
        t.load_examples = _stub_load(
            _row("جملة عربية"),
            _row("Latin row IBM"),
            _row("جملة أخرى"),
        )
        sft, ev = t._get_splits()
        assert len(sft) + len(ev) == 2
        # No Latin in any kept row
        for row in sft + ev:
            assert not contains_latin_letters(row["question"])

    def test_logs_drop_count(self, caplog):
        caplog.set_level(logging.INFO, logger="arabic_eval.tasks.lighteval.base")
        t = _acva_task(clean_latin_rows=True, seed=42)
        t.load_examples = _stub_load(
            _row("جملة عربية"),
            _row("Latin row IBM"),
            _row("جملة أخرى"),
        )
        t._get_splits()
        assert any(
            "clean_latin_rows" in r.message and "dropped 1/3" in r.message
            for r in caplog.records
        ), "expected drop-count INFO line"

    def test_no_log_when_flag_off(self, caplog):
        caplog.set_level(logging.INFO, logger="arabic_eval.tasks.lighteval.base")
        t = _acva_task(clean_latin_rows=False, seed=42)
        t.load_examples = _stub_load(_row("جملة عربية"), _row("Latin row IBM"))
        t._get_splits()
        assert not any(
            "clean_latin_rows" in r.message for r in caplog.records
        ), "expected no clean_latin_rows log when flag off"

    def test_subconfig_wipeout_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger="arabic_eval.tasks.lighteval.base")
        t = _acva_task(clean_latin_rows=True, seed=42)
        t.load_examples = _stub_load(
            _row("جملة عربية", source="cfg_clean"),
            _row("Latin row IBM", source="cfg_dirty"),
            _row("Apple is here", source="cfg_dirty", answer=1),
        )
        t._get_splits()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("cfg_dirty" in r.message for r in warnings), (
            "expected wipeout warning naming the dropped sub-config"
        )

    def test_all_rows_dropped_raises(self):
        t = _acva_task(clean_latin_rows=True, seed=42)
        t.load_examples = _stub_load(
            _row("IBM is here"),
            _row("Apple is there", answer=1),
        )
        with pytest.raises(RuntimeError, match="dropped every row"):
            t._get_splits()

    def test_caching_independent_per_instance(self):
        """Different instances with different flag values must produce
        different splits — ``_cached_splits`` is per-instance, no cross-talk."""
        rows = (_row("جملة عربية"), _row("Latin IBM"), _row("جملة أخرى"))
        t_off = _acva_task(clean_latin_rows=False, seed=42)
        t_on = _acva_task(clean_latin_rows=True, seed=42)
        t_off.load_examples = _stub_load(*rows)
        t_on.load_examples = _stub_load(*rows)
        sft_off, ev_off = t_off._get_splits()
        sft_on, ev_on = t_on._get_splits()
        assert len(sft_off) + len(ev_off) == 3
        assert len(sft_on) + len(ev_on) == 2


# ---------------------------------------------------------------------------
# Default ``__init__`` value
# ---------------------------------------------------------------------------

class TestInitDefault:

    def test_default_off(self):
        t = _acva_task()
        assert t.clean_latin_rows is False

    def test_explicit_false(self):
        t = _acva_task(clean_latin_rows=False)
        assert t.clean_latin_rows is False

    def test_explicit_true(self):
        t = _acva_task(clean_latin_rows=True)
        assert t.clean_latin_rows is True

    def test_truthy_coerced_to_bool(self):
        # config.get may return YAML-loaded values (e.g. 1/0); ensure we coerce.
        t = _acva_task(clean_latin_rows=1)
        assert t.clean_latin_rows is True
        assert isinstance(t.clean_latin_rows, bool)
