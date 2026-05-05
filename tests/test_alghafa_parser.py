"""Regression tests for the Alghafa parser fix and per-topic scoring dispatch.

These tests pin down the contract corrected in the 2026-05-03 fix:

  * ``label`` is **0-indexed** in OALL/AlGhafa-Native (matches LightEval's
    reference ``alghafa_adapter``). Rows with ``label="0"`` must be kept;
    earlier code dropped 36 % of all rows by treating the field as 1-indexed.
  * ``sol5`` is supported (two grounded-statement sub-configs ship five
    options).
  * Per-topic scoring dispatch keys on ``ex["_source_config"]``: T/F + 2/3-way
    sentiment use word-based scoring, 4-way MCQ + 5-way grounded statement
    keep the inherited letter-based scoring.
  * Char-normalization on the base ``_aggregate_scores`` is the LightEval
    ``LogProbCharNorm`` equivalent — no-op for 1-char continuations,
    discriminative for variable-length answer text.
  * ``evaluate_mcq`` emits ``per_subconfig_accuracy`` so heterogeneous
    benchmarks can be diagnosed without re-running.
"""
from __future__ import annotations

import types

import pytest


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _alghafa_task():
    from arabic_eval.tasks.lighteval.alghafa import AlghafaTask
    return AlghafaTask({"dataset_name": "x"})


def test_label_zero_kept():
    """label='0' is the first option correct, not a malformed row."""
    out = _alghafa_task()._parse_example(
        {"query": "q", "sol1": "a", "sol2": "b", "sol3": "c", "sol4": "d", "label": "0"}
    )
    assert out is not None
    assert out["answer"] == 0
    assert out["choices"] == ["a", "b", "c", "d"]


def test_label_3_maps_to_sol4():
    """0-indexed: label='3' → fourth option."""
    out = _alghafa_task()._parse_example(
        {"query": "q", "sol1": "a", "sol2": "b", "sol3": "c", "sol4": "d", "label": "3"}
    )
    assert out["answer"] == 3


def test_sol5_kept_and_label_4_valid():
    """5-way grounded-statement config: sol5 must be picked up."""
    out = _alghafa_task()._parse_example(
        {"query": "q", "sol1": "a", "sol2": "b", "sol3": "c", "sol4": "d",
         "sol5": "e", "label": "4"}
    )
    assert out is not None
    assert out["choices"] == ["a", "b", "c", "d", "e"]
    assert out["answer"] == 4


def test_label_out_of_range_rejected():
    """label exceeding the choice count → row skipped."""
    out = _alghafa_task()._parse_example(
        {"query": "q", "sol1": "a", "sol2": "b", "label": "2"}
    )
    assert out is None


def test_two_way_config_kept():
    """T/F config has only sol1 / sol2 — the per-key existence check trims sol3+."""
    out = _alghafa_task()._parse_example(
        {"query": "q", "sol1": "a", "sol2": "b", "label": "1"}
    )
    assert out["choices"] == ["a", "b"]
    assert out["answer"] == 1


def test_three_way_config_kept():
    """3-way sentiment config: sol1..sol3, label up to '2'."""
    out = _alghafa_task()._parse_example(
        {"query": "q", "sol1": "a", "sol2": "b", "sol3": "c", "label": "2"}
    )
    assert out["choices"] == ["a", "b", "c"]
    assert out["answer"] == 2


def test_missing_label_rejected():
    out = _alghafa_task()._parse_example(
        {"query": "q", "sol1": "a", "sol2": "b"}
    )
    assert out is None


def test_non_integer_label_rejected():
    out = _alghafa_task()._parse_example(
        {"query": "q", "sol1": "a", "sol2": "b", "label": "abc"}
    )
    assert out is None


def test_empty_query_rejected():
    out = _alghafa_task()._parse_example(
        {"query": "", "sol1": "a", "sol2": "b", "label": "0"}
    )
    assert out is None


# ---------------------------------------------------------------------------
# Per-topic prompt / continuation / SFT dispatch
# ---------------------------------------------------------------------------

WORD_CFG = "multiple_choice_rating_sentiment_no_neutral_task"
LETTER_CFG = "meta_ar_dialects"


def _word_ex():
    return {
        "question": "Q?",
        "choices": ["ايجابي", "سلبي"],
        "answer": 0,
        "_source_config": WORD_CFG,
    }


def _letter_ex():
    return {
        "question": "Q?",
        "choices": ["a", "b", "c", "d"],
        "answer": 1,
        "_source_config": LETTER_CFG,
    }


def test_word_scored_eval_context_drops_choice_listing():
    t = _alghafa_task()
    ctx = t._format_eval_context(_word_ex())
    assert "أ." not in ctx
    assert "### السؤال:" in ctx
    assert "### الإجابة:" in ctx
    # Word-scored prompts must NOT carry the letter-MCQ choices block.
    assert "### الخيارات:" not in ctx


def test_word_scored_continuations_use_choice_text():
    t = _alghafa_task()
    assert t._build_continuations(_word_ex()) == [" ايجابي", " سلبي"]


def test_letter_scored_eval_context_lists_letters():
    t = _alghafa_task()
    ctx = t._format_eval_context(_letter_ex())
    for letter, choice in zip(["أ", "ب", "ج", "د"], ["a", "b", "c", "d"]):
        assert f"{letter}. {choice}" in ctx


def test_letter_scored_continuations_use_letters():
    t = _alghafa_task()
    assert t._build_continuations(_letter_ex()) == [" أ", " ب", " ج", " د"]


def test_missing_source_config_falls_back_to_letter():
    """Defensive: no ``_source_config`` (legacy callers) → letter dispatch."""
    ex = {"question": "Q?", "choices": ["a", "b"], "answer": 0}
    t = _alghafa_task()
    assert t._build_continuations(ex) == [" أ", " ب"]


def test_five_way_letter_continuations():
    ex = {
        "question": "Q?", "choices": ["a", "b", "c", "d", "e"], "answer": 4,
        "_source_config": "multiple_choice_grounded_statement_soqal_task",
    }
    cnt = _alghafa_task()._build_continuations(ex)
    assert len(cnt) == 5 and cnt[:4] == [" أ", " ب", " ج", " د"]


# ---------------------------------------------------------------------------
# Score aggregation (char-norm)
# ---------------------------------------------------------------------------

def test_aggregate_identity_for_one_char_continuations():
    """1-char continuations (letter MCQ): char-norm == identity."""
    t = _alghafa_task()
    out = t._aggregate_scores({}, [" أ", " ب", " ج", " د"], [-3.0, -5.0, -1.0, -7.0])
    assert out == [-3.0, -5.0, -1.0, -7.0]


def test_aggregate_char_norm_for_word_continuations():
    """Variable-length continuations: divide by character count of stripped text."""
    t = _alghafa_task()
    out = t._aggregate_scores({}, [" ايجابي", " سلبي"], [-12.0, -8.0])
    # 'ايجابي' = 6 chars, 'سلبي' = 4 chars
    assert out[0] == pytest.approx(-12.0 / 6)
    assert out[1] == pytest.approx(-8.0 / 4)


def test_aggregate_rescues_length_bias_in_argmax():
    """Equal raw lls + unequal lengths: char-norm picks the longer text."""
    t = _alghafa_task()
    out = t._aggregate_scores({}, [" ايجابي", " سلبي"], [-10.0, -10.0])
    assert out.index(max(out)) == 0  # longer wins under char-norm


# ---------------------------------------------------------------------------
# evaluate_mcq — per-sub-config bucketing + score_margin in failure CSV
# ---------------------------------------------------------------------------

class _FakeAdapter:
    device = "cpu"
    model = None


class _FakeTok:
    embedding_type = "standard"


def _wrapper_with_canned_lls():
    """Return a wrapper whose ``loglikelihood`` always picks index 0
    (first request gets the highest log-likelihood). Lets us deterministically
    drive ``evaluate_mcq`` without a real model."""
    from arabic_eval.tasks.lighteval.base import LightEvalModelWrapper
    w = LightEvalModelWrapper(_FakeAdapter(), _FakeTok(), max_length=128)
    w.loglikelihood = types.MethodType(
        lambda self, reqs: [-1.0 - i for i in range(len(reqs))], w
    )
    return w


def test_evaluate_mcq_emits_per_subconfig_accuracy():
    w = _wrapper_with_canned_lls()
    t = _alghafa_task()
    examples = [
        # 4-way letter-scored, model picks idx 0 → 1 correct + 1 wrong
        {"question": "Q1", "choices": ["a", "b", "c", "d"], "answer": 0,
         "_source_config": LETTER_CFG},
        {"question": "Q2", "choices": ["a", "b", "c", "d"], "answer": 1,
         "_source_config": LETTER_CFG},
        # 2-way word-scored, model picks idx 0 → 1 correct + 1 wrong
        {"question": "Q3", "choices": ["ايجابي", "سلبي"], "answer": 0,
         "_source_config": WORD_CFG},
        {"question": "Q4", "choices": ["ايجابي", "سلبي"], "answer": 1,
         "_source_config": WORD_CFG},
    ]
    metrics, _ = w.evaluate_mcq(examples, task=t)
    assert metrics["accuracy"] == 0.5
    psa = metrics["per_subconfig_accuracy"]
    assert set(psa.keys()) == {LETTER_CFG, WORD_CFG}
    assert psa[LETTER_CFG] == {"accuracy": 0.5, "num_samples": 2}
    assert psa[WORD_CFG] == {"accuracy": 0.5, "num_samples": 2}


def test_evaluate_mcq_failure_record_has_score_margin():
    """Failure CSV must carry both ll_* and score_* columns when aggregation
    is non-identity (word-scored examples)."""
    w = _wrapper_with_canned_lls()
    t = _alghafa_task()
    # Force a wrong prediction — gold is idx 1, model picks idx 0.
    examples = [
        {"question": "Q1", "choices": ["ايجابي", "سلبي"], "answer": 1,
         "_source_config": WORD_CFG},
    ]
    _, failures = w.evaluate_mcq(examples, task=t, collect_failures=True)
    assert len(failures) == 1
    rec = failures[0]
    # raw lls
    assert "ll_0" in rec and "ll_1" in rec and "ll_margin" in rec
    # aggregated scores (after char-norm)
    assert "score_0" in rec and "score_1" in rec and "score_margin" in rec
    # On a word-scored row the two should differ in absolute value because
    # char-norm divides by different character counts.
    assert rec["ll_0"] != rec["score_0"] or rec["ll_1"] != rec["score_1"]


def test_evaluate_mcq_no_task_falls_back_to_default_bucket():
    """Legacy callers (task=None) get a single ``_default`` bucket."""
    w = _wrapper_with_canned_lls()
    examples = [
        {"question": "Q1", "choices": ["a", "b"], "answer": 0},
        {"question": "Q2", "choices": ["a", "b"], "answer": 1},
    ]
    metrics, _ = w.evaluate_mcq(examples, task=None)
    assert metrics["per_subconfig_accuracy"] == {
        "_default": {"accuracy": 0.5, "num_samples": 2}
    }
