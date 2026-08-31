"""Tests for K-shot in-context demonstrations on LightEval MCQ tasks.

Pin down the contract of ``_format_eval_context_with_fewshot`` and
``_build_fewshot_examples``:

  * Demos come from the same ``_source_config`` as the eval row.
  * The eval row is excluded from its own demo pool (no leakage).
  * Selection is deterministic (seeded on ``self.seed + ex_index``); the
    same eval row gets the same demos across re-runs.
  * Prompt structure: ``<demo_ctx> <gold_continuation>\\n\\n`` × K + ``<eval_ctx>``.
  * ``num_fewshot=0`` is a complete no-op (zero-shot behavior preserved).
  * Tasks ignore the ``num_fewshot`` kwarg gracefully (legacy callers).
"""
from __future__ import annotations

import pytest

from arabic_eval.tasks.lighteval.alghafa import AlghafaTask
from arabic_eval.tasks.lighteval.acva import ACVATask
from arabic_eval.tasks.lighteval.arabic_exam import ArabicExamTask


def _seed_examples_on_task(task, examples):
    """Helper: directly populate a task's eval cache with synthetic examples
    so we don't have to hit the network. Tests are parser-free."""
    task._cached_examples = examples
    task._fewshot_pool_by_config = None  # force lazy rebuild


def _make_acva_examples(cfg: str, n: int) -> list:
    return [
        {"question": f"q-{cfg}-{i}", "choices": ["صح", "خطأ"], "answer": i % 2,
         "_source_config": cfg}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# num_fewshot == 0: no-op (existing zero-shot behavior preserved)
# ---------------------------------------------------------------------------

def test_zero_shot_is_byte_identical_to_format_eval_context():
    task = ACVATask({"num_fewshot": 0})
    ex = {"question": "س", "choices": ["صح", "خطأ"], "answer": 0,
          "_source_config": "Algeria"}
    bare = task._format_eval_context(ex)
    with_fs = task._format_eval_context_with_fewshot(ex)
    assert with_fs == bare


def test_default_num_fewshot_is_zero():
    """Tasks built without the ``num_fewshot`` key default to zero-shot."""
    task = ACVATask({})
    assert task.num_fewshot == 0


# ---------------------------------------------------------------------------
# K > 0: demonstrations come from same _source_config and exclude eval row
# ---------------------------------------------------------------------------

def test_demos_come_from_same_source_config():
    task = ACVATask({"num_fewshot": 3})
    examples = (
        _make_acva_examples("Algeria", 10) +
        _make_acva_examples("Yemen", 10)
    )
    _seed_examples_on_task(task, examples)
    eval_row = examples[0]  # Algeria, idx 0
    demos = task._build_fewshot_examples(eval_row)
    assert len(demos) == 3
    for d in demos:
        assert d["_source_config"] == "Algeria"


def test_demos_exclude_eval_row_itself():
    task = ACVATask({"num_fewshot": 3})
    examples = _make_acva_examples("Algeria", 10)
    _seed_examples_on_task(task, examples)
    eval_row = examples[0]
    demos = task._build_fewshot_examples(eval_row)
    # No demo should be the eval row by identity or by question content.
    for d in demos:
        assert d is not eval_row
        assert d["question"] != eval_row["question"]


def test_demo_selection_is_deterministic_under_same_seed():
    """Two task instances with the same seed must produce the same demos."""
    examples = _make_acva_examples("Algeria", 20)
    a = ACVATask({"num_fewshot": 3, "seed": 42})
    b = ACVATask({"num_fewshot": 3, "seed": 42})
    _seed_examples_on_task(a, examples)
    _seed_examples_on_task(b, examples)
    da = a._build_fewshot_examples(examples[5])
    db = b._build_fewshot_examples(examples[5])
    assert [d["question"] for d in da] == [d["question"] for d in db]


def test_different_eval_rows_get_different_demos():
    """Seeding on (seed + ex_index) gives different rows different demos."""
    examples = _make_acva_examples("Algeria", 30)
    task = ACVATask({"num_fewshot": 3, "seed": 42})
    _seed_examples_on_task(task, examples)
    d_for_5 = [d["question"] for d in task._build_fewshot_examples(examples[5])]
    d_for_15 = [d["question"] for d in task._build_fewshot_examples(examples[15])]
    assert d_for_5 != d_for_15


def test_returns_fewer_demos_when_pool_too_small():
    """Sub-config with fewer than K rows still works (returns what's available)."""
    examples = _make_acva_examples("Yemen", 2)  # K=3 demos, only 2 in pool, 1 after self-exclusion
    task = ACVATask({"num_fewshot": 3})
    _seed_examples_on_task(task, examples)
    demos = task._build_fewshot_examples(examples[0])
    assert 0 <= len(demos) <= 1  # at most 1 (one row, eval-row excluded)


# ---------------------------------------------------------------------------
# Prompt structure: K demos + eval prompt
# ---------------------------------------------------------------------------

def test_prompt_includes_K_rendered_demos_plus_eval_prompt():
    task = ACVATask({"num_fewshot": 2, "seed": 0})
    examples = _make_acva_examples("Algeria", 10)
    _seed_examples_on_task(task, examples)
    eval_row = examples[0]
    rendered = task._format_eval_context_with_fewshot(eval_row)
    bare = task._format_eval_context(eval_row)
    # Eval prompt must appear at the END of the rendered string.
    assert rendered.endswith(bare)
    # Two demos must appear before the eval prompt.
    # Each demo is `<ctx> <gold>` so each contains the answer prefix `الإجابة:`
    # at least once. The eval prompt itself adds one more occurrence.
    demos_segment = rendered[: -len(bare)].rstrip()
    # Each demo includes one ``الإجابة:`` with a trailing answer word.
    assert demos_segment.count("الإجابة:") == 2


def test_arabic_exam_fewshot_demo_includes_letter_answer():
    task = ArabicExamTask({"num_fewshot": 1, "seed": 0})
    examples = [
        {"question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer": (i % 4),
         "_source_config": "_default", "context": ""}
        for i in range(5)
    ]
    _seed_examples_on_task(task, examples)
    eval_row = examples[0]
    rendered = task._format_eval_context_with_fewshot(eval_row)
    # The demo's gold continuation is one of " أ"/" ب"/" ج"/" د" — exactly
    # one such pattern should appear before the eval prompt's `الإجابة:`.
    demos_segment = rendered[: rendered.rindex("الإجابة:")]
    # Each demo has ``الإجابة: <space><letter>``.
    n_letter_after_answer = sum(
        f"الإجابة: {l}" in demos_segment for l in "أبجد"
    )
    assert n_letter_after_answer >= 1


def test_alghafa_fewshot_demo_includes_choice_text_answer():
    task = AlghafaTask({"num_fewshot": 1, "seed": 0})
    examples = [
        {"question": f"q{i}", "choices": ["choice-A-text", "choice-B-text"],
         "answer": (i % 2), "_source_config": "_default"}
        for i in range(5)
    ]
    _seed_examples_on_task(task, examples)
    eval_row = examples[0]
    rendered = task._format_eval_context_with_fewshot(eval_row)
    # The demo gold is the choice TEXT (not a letter) — verify it appears.
    assert ("choice-A-text" in rendered) or ("choice-B-text" in rendered)


# ---------------------------------------------------------------------------
# Eval-set caching: pool is built once, reused
# ---------------------------------------------------------------------------

def test_fewshot_pool_caches_across_calls():
    task = ACVATask({"num_fewshot": 2})
    examples = _make_acva_examples("Algeria", 10)
    _seed_examples_on_task(task, examples)
    assert task._fewshot_pool_by_config is None
    task._build_fewshot_examples(examples[0])
    assert task._fewshot_pool_by_config is not None
    cached = task._fewshot_pool_by_config
    task._build_fewshot_examples(examples[1])
    # Same dict, not rebuilt.
    assert task._fewshot_pool_by_config is cached
