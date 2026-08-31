"""Tests for the LightEval-official prompt formats (post 2026-05-06).

Verifies each task emits prompts byte-identical to the LightEval reference
(``community_tasks/arabic_evals.py`` v0.6.0):

  * ACVA: ``السؤال: {q}\\nالإجابة:`` — bare, no ``###`` markers.
  * AlGhafa: numeric list ``0) {c}\\n1) {c}\\n…``, score choice text.
  * ArabicMMLU/arabic_exam: Arabic-letter list ``أ. {c}\\nب. {c}\\n…``,
    score the letter.

Also verifies the unconditioned PMI query is the bare ``الإجابة:`` (no ``###``).
"""
from __future__ import annotations

import pytest

from arabic_eval.tasks.lighteval.acva import ACVATask
from arabic_eval.tasks.lighteval.alghafa import AlghafaTask
from arabic_eval.tasks.lighteval.arabic_exam import ArabicExamTask
from arabic_eval.tasks.lighteval.utils import (
    ALGHAFA_INSTRUCTION,
    format_acva_context_official,
    format_mcq_context_letter_official,
    format_mcq_context_numeric_official,
)


# ---------------------------------------------------------------------------
# utils.py — direct format-helper tests
# ---------------------------------------------------------------------------

def test_format_acva_context_no_block_markers():
    out = format_acva_context_official("ما هي عاصمة المغرب؟")
    assert out == "السؤال: ما هي عاصمة المغرب؟\nالإجابة:"
    assert "###" not in out


def test_format_mcq_letter_official_arabic_letters():
    out = format_mcq_context_letter_official(
        "ما هي عاصمة مصر؟", ["القاهرة", "الإسكندرية", "أسوان", "بور سعيد"]
    )
    # Instruction prefix
    assert out.startswith(ALGHAFA_INSTRUCTION)
    # Each Arabic letter labels its choice
    for letter, choice in zip("أبجد", ["القاهرة", "الإسكندرية", "أسوان", "بور سعيد"]):
        assert f"{letter}. {choice}" in out
    # No ###, no extra instruction line
    assert "###" not in out
    assert "إختار" not in out
    # Ends with bare answer prefix
    assert out.endswith("الإجابة:")


def test_format_mcq_numeric_official_uses_numbers():
    out = format_mcq_context_numeric_official(
        "أيهما أكبر؟", ["الفيل", "النملة"]
    )
    assert out.startswith(ALGHAFA_INSTRUCTION)
    assert "السؤال: أيهما أكبر؟" in out
    assert "0) الفيل" in out
    assert "1) النملة" in out
    assert "###" not in out
    assert out.endswith("الإجابة:")


# ---------------------------------------------------------------------------
# Task-level prompt tests
# ---------------------------------------------------------------------------

def test_acva_task_prompt_no_block_markers():
    task = ACVATask({})
    ex = {"question": "مدينة الرباط هي عاصمة المغرب.", "choices": ["صح", "خطأ"], "answer": 0}
    ctx = task._format_eval_context(ex)
    assert ctx == "السؤال: مدينة الرباط هي عاصمة المغرب.\nالإجابة:"
    assert "###" not in ctx


def test_acva_task_continuations_score_words():
    task = ACVATask({})
    ex = {"question": "x", "choices": ["صح", "خطأ"], "answer": 0}
    conts = task._build_continuations(ex)
    assert conts == [" صح", " خطأ"]


def test_alghafa_task_uses_numeric_format_for_all_subconfigs():
    """All Alghafa sub-configs should use the same numeric-list format —
    the per-config word/letter dispatch was removed in 2026-05-06."""
    task = AlghafaTask({})
    # Test a 4-way MCQ sub-config (formerly letter-scored)
    ex_mcq = {
        "question": "ما هي عاصمة مصر؟",
        "choices": ["القاهرة", "الرباط", "تونس", "الجزائر"],
        "answer": 0,
        "_source_config": "mcq_exams_test_ar",
    }
    ctx = task._format_eval_context(ex_mcq)
    assert "0) القاهرة" in ctx
    assert "1) الرباط" in ctx
    assert "###" not in ctx
    assert ctx.endswith("الإجابة:")
    conts = task._build_continuations(ex_mcq)
    assert conts == [" القاهرة", " الرباط", " تونس", " الجزائر"]

    # Test a 2-way binary sub-config (formerly word-scored) — same format now
    ex_binary = {
        "question": "هل هذا صحيح؟",
        "choices": ["نعم", "لا"],
        "answer": 0,
        "_source_config": "multiple_choice_facts_truefalse_balanced_task",
    }
    ctx2 = task._format_eval_context(ex_binary)
    assert "0) نعم" in ctx2
    assert "1) لا" in ctx2
    assert ctx2.startswith(ALGHAFA_INSTRUCTION)
    conts2 = task._build_continuations(ex_binary)
    assert conts2 == [" نعم", " لا"]


def test_arabic_exam_task_prompt_official_letter_format():
    task = ArabicExamTask({})
    ex = {
        "question": "ما هي عاصمة الأردن؟",
        "choices": ["عمان", "إربد", "الزرقاء", "العقبة"],
        "answer": 0,
        "context": "",
    }
    ctx = task._format_eval_context(ex)
    assert ctx.startswith(ALGHAFA_INSTRUCTION)
    assert "أ. عمان" in ctx
    assert "ب. إربد" in ctx
    assert "ج. الزرقاء" in ctx
    assert "د. العقبة" in ctx
    assert "###" not in ctx
    assert ctx.endswith("الإجابة:")


def test_arabic_exam_task_with_context_prepended():
    task = ArabicExamTask({})
    ex = {
        "question": "بناء على النص، ما هو الموضوع الرئيسي؟",
        "choices": ["العلم", "التاريخ", "الجغرافيا", "الأدب"],
        "answer": 1,
        "context": "هذا نص قصير عن التاريخ العربي.",
    }
    ctx = task._format_eval_context(ex)
    assert ctx.startswith("السياق: هذا نص قصير عن التاريخ العربي.\n")
    assert "###" not in ctx
    assert "أ. العلم" in ctx


def test_arabic_exam_task_continuations_are_letters():
    task = ArabicExamTask({})
    ex = {
        "question": "x",
        "choices": ["a", "b", "c", "d"],
        "answer": 0,
        "context": "",
    }
    conts = task._build_continuations(ex)
    assert conts == [" أ", " ب", " ج", " د"]


# ---------------------------------------------------------------------------
# PMI unconditioned-query test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_cls", [ACVATask, AlghafaTask, ArabicExamTask])
def test_unconditioned_query_is_bare_answer_prefix(task_cls):
    task = task_cls({})
    ex_dummy = {"question": "x", "choices": ["a", "b"], "answer": 0, "context": ""}
    q = task._unconditioned_query(ex_dummy)
    assert q == "الإجابة:"
    assert "###" not in q
