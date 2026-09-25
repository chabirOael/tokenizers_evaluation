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


# ---------------------------------------------------------------------------
# label_rotation (2026-09-25): letters rotated over the slots on the two
# letter-scored tasks — the diagnostic that separates a letter effect from a
# first-slot effect (docs/report.md §3.11). Rotation 0 is today's prompt.
# ---------------------------------------------------------------------------

import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

from arabic_eval.data.finetune_corpora import QARecord, _format_qa_full  # noqa: E402
from arabic_eval.params_spec import validate_params  # noqa: E402
from arabic_eval.tasks.lighteval.culture_arabic_mmlu import CultureArabicMMLUTask  # noqa: E402
from arabic_eval.tasks.lighteval.utils import (  # noqa: E402
    ARABIC_CHOICE_LETTERS,
    choice_letters,
    read_label_rotation,
)

#: Written by the code *before* label_rotation existed (73bf45b) on real rows:
#: 24 Arabic-Exam rows (2 / 3 / 4 / 5 choices, with and without a passage) and 9
#: Culture-MMLU rows from the v5 dumps (each checked against the dump's prompt
#: and continuations at generation time), and 8 synthetic ``arabic_squad_mcq``
#: training records with their full ``_format_qa_full`` text.
GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "letter_prompt_golden.json").read_text(encoding="utf-8")
)
LETTER_TASKS = {"arabic_exam": ArabicExamTask, "culture_arabic_mmlu": CultureArabicMMLUTask}


@pytest.mark.parametrize("task_name", sorted(LETTER_TASKS))
@pytest.mark.parametrize("config", [{}, {"label_rotation": 0}], ids=["absent", "explicit_0"])
def test_rotation_0_is_byte_identical_to_the_pre_change_prompt(task_name, config):
    task = LETTER_TASKS[task_name](config)
    rows = GOLDEN["tasks"][task_name]
    assert rows
    for r in rows:
        assert task._format_eval_context(r["example"]) == r["context"]
        assert task._build_continuations(r["example"]) == r["continuations"]


def test_training_mcq_template_is_untouched():
    """``_format_mcq_letter_prompt`` never passes a rotation: the synthetic MCQ
    records render byte for byte as before, whatever a task instance is set to."""
    LETTER_TASKS["arabic_exam"]({"label_rotation": 2})   # an instance with a rotation exists
    assert len(GOLDEN["mcq_letter_records"]) == 8
    for r in GOLDEN["mcq_letter_records"]:
        rec = QARecord(**r["record"])
        assert rec.prompt_template == "mcq_letter"
        assert _format_qa_full(rec) == r["full"]


ROT_EXPECTED = {1: "ب ج د أ", 2: "ج د أ ب", 3: "د أ ب ج"}


@pytest.mark.parametrize("task_name", sorted(LETTER_TASKS))
@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotated_letters_on_a_four_choice_row(task_name, k):
    task = LETTER_TASKS[task_name]({"label_rotation": k})
    choices = ["عمان", "إربد", "الزرقاء", "العقبة"]
    ex = {"question": "ما هي عاصمة الأردن؟", "choices": choices, "answer": 0, "context": ""}
    ctx = task._format_eval_context(ex)
    letters = ROT_EXPECTED[k].split()
    option_lines = ctx.split("\n")[-5:-1]
    assert option_lines == [f"{l}. {c}" for l, c in zip(letters, choices)]
    conts = task._build_continuations(ex)
    assert conts == [" " + ARABIC_CHOICE_LETTERS[(i + k) % 4] for i in range(4)]
    assert conts == [" " + l for l in letters]
    # The rest of the prompt is the official one; only the four labels moved.
    official = LETTER_TASKS[task_name]({})._format_eval_context(ex)
    assert ctx.replace("\n".join(option_lines), "") == \
        official.replace("\n".join(f"{l}. {c}" for l, c in zip("أبجد", choices)), "")


def test_five_choice_row_rotates_over_five_letters():
    task = ArabicExamTask({"label_rotation": 2})
    ex = {"question": "س", "choices": ["a", "b", "c", "d", "e"], "answer": 4, "context": ""}
    assert choice_letters(5, 2) == ["ج", "د", "هـ", "أ", "ب"]
    assert task._build_continuations(ex) == [" ج", " د", " هـ", " أ", " ب"]
    assert task._format_eval_context(ex).split("\n")[-6:-1] == ["ج. a", "د. b", "هـ. c", "أ. d", "ب. e"]
    # Two- and three-choice rows rotate over their own letters.
    assert choice_letters(3, 1) == ["ب", "ج", "أ"] and choice_letters(2, 1) == ["ب", "أ"]
    # Beyond five choices the old mapping stands, unrotated.
    assert choice_letters(6, 3) == ["أ", "ب", "ج", "د", "هـ", "5"]


class _FixtureExam(ArabicExamTask):
    """Arabic-Exam with the golden 4-choice rows as its eval list (no HF load)."""

    def load_examples(self):
        rows = [r["example"] for r in GOLDEN["tasks"]["arabic_exam"] if len(r["example"]["choices"]) == 4]
        return [{**ex, "_source_config": "one_config"} for ex in rows]


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_three_shot_demos_carry_the_rotated_letter_of_their_gold_slot(k):
    task = _FixtureExam({"label_rotation": k, "num_fewshot": 3})
    examples = task.get_eval_examples()
    assert len(examples) >= 5
    for ex in examples:
        prompt = task._format_eval_context_with_fewshot(ex)
        demos = task._build_fewshot_examples(ex)
        assert len(demos) == 3
        demo_letters = re.findall(r"الإجابة: (\S+)", prompt)
        assert demo_letters == [choice_letters(4, k)[d["answer"]] for d in demos]
        assert prompt.endswith(task._format_eval_context(ex))
        # Each demo is its own rotated context followed by its gold continuation.
        for d in demos:
            assert task._format_eval_context(d) + task._build_continuations(d)[d["answer"]] in prompt


@pytest.mark.parametrize("bad", [-1, 5, 7, 1.0, "1", True, None])
def test_label_rotation_out_of_range_is_an_error(bad):
    with pytest.raises(ValueError, match="label_rotation"):
        ArabicExamTask({"label_rotation": bad})
    with pytest.raises(ValueError, match="label_rotation"):
        read_label_rotation({"label_rotation": bad})


def test_label_rotation_is_declared_on_the_letter_tasks_only():
    for cls in LETTER_TASKS.values():
        spec = {s.name: s for s in cls.param_spec()}
        s = spec["label_rotation"]
        assert (s.type, s.default, s.min, s.max, s.advanced) == ("int", 0, 0, 4, True)
        assert validate_params(cls.param_spec(), {"label_rotation": 3}, owner=cls.__name__) == []
    for cls in (ACVATask, AlghafaTask):
        assert "label_rotation" not in {s.name for s in cls.param_spec()}
        msgs = validate_params(cls.param_spec(), {"label_rotation": 1}, owner="x")
        assert len(msgs) == 1 and "does not declare" in msgs[0]
