"""Regression tests for the arabic_exam parser fixes (2026-05-04).

These tests pin down the four bugs corrected in this round:

  * The ``All`` config of ``MBZUAI/ArabicMMLU`` is a strict union of the
    other 40 subject configs and must be excluded — the dataset file passes
    ``ARABIC_EXAM_EXCLUDED_CONFIGS = frozenset({"All"})`` to the loader.
  * The ``Context`` field carries supporting passages for ~5 % of rows
    (Arabic poems, prose excerpts, math word-problem setups). Dropping it
    makes those questions unanswerable; the parser must capture it and the
    eval/SFT prompt formatters must prepend it.
  * ``Option 5`` exists on ~344 rows and 141 rows have ``Answer Key=E``;
    iterating only Options 1–4 truncated the choice list and silently
    dropped E-answer rows via ``answer_idx >= len(choices)``.
  * ``is_few_shot=1`` flags the dataset's own dev-split demonstration rows
    (~120 total). They should not enter the eval pool — we already do our
    own 10/90 SFT split.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arabic_exam_task():
    from arabic_eval.tasks.lighteval.arabic_exam import ArabicExamTask
    return ArabicExamTask({"dataset_name": "x"})


def _row(**overrides):
    """Build a minimal valid raw row, override specific fields per test."""
    base = {
        "Question": "ما عاصمة الجزائر؟",
        "Context": None,
        "Option 1": "الجزائر",
        "Option 2": "وهران",
        "Option 3": "قسنطينة",
        "Option 4": "عنابة",
        "Option 5": None,
        "Answer Key": "A",
        "is_few_shot": 0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Parser — Context field
# ---------------------------------------------------------------------------

def test_context_captured_when_present():
    out = _arabic_exam_task()._parse_example(_row(Context="نص داعم للسؤال"))
    assert out is not None
    assert out["context"] == "نص داعم للسؤال"


def test_context_empty_when_missing():
    """Null/missing Context must result in empty string, not the literal 'None'."""
    out = _arabic_exam_task()._parse_example(_row(Context=None))
    assert out["context"] == ""


def test_context_empty_when_nan():
    """The dataset uses Pandas-derived 'nan' strings on some rows."""
    out = _arabic_exam_task()._parse_example(_row(Context="nan"))
    assert out["context"] == ""


def test_eval_prompt_includes_context_prefix():
    task = _arabic_exam_task()
    ex = task._parse_example(_row(Context="حدد إجابة هذا السؤال بعد قراءة المقطع التالي"))
    rendered = task._format_eval_context(ex)
    assert rendered.startswith("السياق: ")
    assert "حدد إجابة هذا السؤال" in rendered


def test_eval_prompt_byte_identical_when_no_context():
    """No-context fallback must equal the inherited base helper byte-for-byte."""
    from arabic_eval.tasks.lighteval.utils import format_mcq_context
    task = _arabic_exam_task()
    ex = task._parse_example(_row(Context=None))
    assert task._format_eval_context(ex) == format_mcq_context(ex["question"], ex["choices"])


def test_sft_text_includes_context_and_ends_with_gold_letter():
    task = _arabic_exam_task()
    ex = task._parse_example(_row(Context="نص", **{"Answer Key": "C"}))  # C → answer_idx 2
    sft = task._format_sft_text(ex)
    assert "السياق: نص" in sft
    assert sft.endswith(" ج")  # Arabic letter for index 2


# ---------------------------------------------------------------------------
# Parser — Step 3 (Option 5 / Answer Key=E)
# ---------------------------------------------------------------------------

def test_five_option_row_kept():
    out = _arabic_exam_task()._parse_example(
        _row(**{"Option 5": "إجابة خامسة", "Answer Key": "C"})
    )
    assert out is not None
    assert len(out["choices"]) == 5
    assert out["choices"][4] == "إجابة خامسة"
    assert out["answer"] == 2


def test_answer_key_E_recovered():
    """Pre-fix this row was silently dropped; now it must parse."""
    out = _arabic_exam_task()._parse_example(
        _row(**{"Option 5": "إجابة خامسة", "Answer Key": "E"})
    )
    assert out is not None
    assert out["answer"] == 4
    assert out["choices"][4] == "إجابة خامسة"


def test_answer_E_without_option_5_rejected():
    """E-answer with only 4 options must still be rejected (data corruption guard)."""
    out = _arabic_exam_task()._parse_example(_row(**{"Answer Key": "E"}))
    assert out is None


def test_five_option_eval_prompt_renders_all_five_letters():
    task = _arabic_exam_task()
    ex = task._parse_example(_row(**{"Option 5": "ه", "Answer Key": "C"}))
    rendered = task._format_eval_context(ex)
    for letter in ("أ", "ب", "ج", "د", "هـ"):
        assert letter in rendered, f"missing {letter} in 5-option prompt"


def test_five_option_continuations_have_five_entries():
    task = _arabic_exam_task()
    ex = task._parse_example(_row(**{"Option 5": "ه", "Answer Key": "C"}))
    conts = task._build_continuations(ex)
    assert len(conts) == 5
    assert conts == [" أ", " ب", " ج", " د", " هـ"]


# ---------------------------------------------------------------------------
# Parser — Step 4 (is_few_shot filter)
# ---------------------------------------------------------------------------

def test_is_few_shot_row_dropped():
    out = _arabic_exam_task()._parse_example(_row(is_few_shot=1))
    assert out is None


def test_is_few_shot_zero_kept():
    out = _arabic_exam_task()._parse_example(_row(is_few_shot=0))
    assert out is not None


def test_is_few_shot_missing_kept():
    """Robust to schema rows that don't carry the flag at all."""
    raw = _row()
    del raw["is_few_shot"]
    out = _arabic_exam_task()._parse_example(raw)
    assert out is not None


# ---------------------------------------------------------------------------
# Parser — invariants that must still hold
# ---------------------------------------------------------------------------

def test_letter_to_index_mapping():
    """Latin letter A→0, B→1, C→2, D→3, E→4 invariant."""
    task = _arabic_exam_task()
    for letter, idx in (("A", 0), ("B", 1), ("C", 2), ("D", 3)):
        out = task._parse_example(_row(**{"Answer Key": letter}))
        assert out["answer"] == idx, f"{letter} should map to {idx}"


def test_invalid_answer_key_rejected():
    out = _arabic_exam_task()._parse_example(_row(**{"Answer Key": "Z"}))
    assert out is None


def test_empty_question_rejected():
    out = _arabic_exam_task()._parse_example(_row(Question=""))
    assert out is None


def test_no_options_rejected():
    out = _arabic_exam_task()._parse_example(
        _row(**{"Option 1": None, "Option 2": None, "Option 3": None, "Option 4": None})
    )
    assert out is None
