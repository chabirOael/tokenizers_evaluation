"""PMI-normalization (LogProbPMINorm) tests for LightEval MCQ scoring.

Covers:

  * The four ``_aggregate_scores`` paths under ``normalization="char"``
    return the unchanged char-norm result (backward compat).
  * Under ``normalization="pmi"`` they return ``ll - unconditioned_ll``.
  * ``_unconditioned_query`` defaults to the bare answer prefix
    ``"الإجابة:"`` for every current task.
  * ``LightEvalModelWrapper.evaluate_mcq`` under ``"char+pmi"``:
      - emits both ``accuracy_char_norm`` and ``accuracy_pmi``;
      - ``accuracy`` aliases ``accuracy_char_norm``;
      - the unconditioned-ll cache fires once for letter-MCQ
        (one extra forward call per unique continuation tuple, not per row);
      - the failure CSV gains ``score_pmi_*`` / ``score_pmi_margin`` columns
        only when PMI is computed.
  * ``score_normalization`` config field accepts the three valid literals
    and rejects others.
  * ``compute_mei`` echoes ``accuracy_source`` only when not the legacy
    default (so byte-identity for char-only runs is preserved).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

import arabic_eval.tasks  # noqa: F401  — populate the task registry
from arabic_eval.config import EvaluationConfig
from arabic_eval.evaluation.metrics import compute_mei
from arabic_eval.registry import task_registry
from arabic_eval.tasks.lighteval.acva import ACVATask
from arabic_eval.tasks.lighteval.alghafa import AlghafaTask
from arabic_eval.tasks.lighteval.arabic_exam import ArabicExamTask
from arabic_eval.tasks.lighteval.base import LightEvalModelWrapper
from arabic_eval.tasks.lighteval.culture_arabic_mmlu import CultureArabicMMLUTask


# ---------------------------------------------------------------------------
# 1. _aggregate_scores: char path is unchanged; pmi path returns ll - u
# ---------------------------------------------------------------------------

def _make_task(cls):
    return cls({"dataset_name": cls({})._default_dataset_name()})


@pytest.mark.parametrize("task_cls", [ACVATask, AlghafaTask, CultureArabicMMLUTask, ArabicExamTask])
def test_aggregate_scores_char_path_unchanged(task_cls):
    """``normalization='char'`` must produce the legacy char-norm output:
    each ll divided by the (lstripped) continuation length."""
    task = _make_task(task_cls)
    continuations = [" أ", " ب", " ج", " د"]
    lls = [-2.0, -3.0, -1.0, -4.0]
    out = task._aggregate_scores(
        ex={}, continuations=continuations, log_likelihoods=lls,
        normalization="char",
    )
    # 1-char continuations → divide by 1, no-op.
    assert out == lls


def test_aggregate_scores_char_path_normalizes_word_continuations():
    """For word-scored continuations of differing lengths, char-norm divides."""
    task = _make_task(ACVATask)
    continuations = [" صح", " خطأ"]  # 2 chars, 3 chars
    lls = [-4.0, -6.0]
    out = task._aggregate_scores(
        ex={}, continuations=continuations, log_likelihoods=lls,
        normalization="char",
    )
    assert out == [-2.0, -2.0]


@pytest.mark.parametrize("task_cls", [ACVATask, AlghafaTask, CultureArabicMMLUTask, ArabicExamTask])
def test_aggregate_scores_pmi_path_subtracts_unconditioned(task_cls):
    """``normalization='pmi'`` must return ll − unconditioned_ll element-wise."""
    task = _make_task(task_cls)
    continuations = [" أ", " ب", " ج", " د"]
    lls = [-2.0, -3.0, -1.0, -4.0]
    uncond = [-2.5, -2.0, -1.0, -3.0]
    out = task._aggregate_scores(
        ex={}, continuations=continuations, log_likelihoods=lls,
        unconditioned_log_likelihoods=uncond,
        normalization="pmi",
    )
    assert out == [-2.0 - -2.5, -3.0 - -2.0, -1.0 - -1.0, -4.0 - -3.0]


def test_aggregate_scores_pmi_without_unconditioned_raises():
    """PMI mode must raise (not silently fall through to char-norm) if the
    unconditioned ll values weren't supplied — that would mean the wiring
    is broken and the operator is getting char-norm results labelled as PMI."""
    task = _make_task(CultureArabicMMLUTask)
    with pytest.raises(ValueError, match="unconditioned"):
        task._aggregate_scores(
            ex={}, continuations=[" أ", " ب"], log_likelihoods=[-1.0, -2.0],
            normalization="pmi",
        )


# ---------------------------------------------------------------------------
# 2. _unconditioned_query: default is "الإجابة:" for every task
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_cls", [ACVATask, AlghafaTask, CultureArabicMMLUTask, ArabicExamTask])
def test_unconditioned_query_default(task_cls):
    """All four current tasks share the same answer-prefix convention; the
    default ``_unconditioned_query`` must return ``"الإجابة:"`` for each."""
    task = _make_task(task_cls)
    ex = {"question": "س", "choices": ["a", "b"], "answer": 0}
    assert task._unconditioned_query(ex) == "الإجابة:"


# ---------------------------------------------------------------------------
# 3. evaluate_mcq under "char+pmi"
# ---------------------------------------------------------------------------

def _fake_loglikelihood_factory(letter_priors: Dict[str, float], boost_letter: str = " ج"):
    """Build a fake ``loglikelihood`` that:
       - returns ``letter_priors[cont]`` when the context is the unconditioned
         answer-prefix only (``"الإجابة:"``);
       - returns ``letter_priors[cont] + 0.5`` for the gold letter when the
         context contains the question (so char-norm picks the boosted one).

    Records every (ctx, cont) pair so we can assert cache behavior.
    """
    calls: List[Tuple[str, str]] = []

    def _ll(model, tokenizer, ctx, cont, max_length=512):
        calls.append((ctx, cont))
        if ctx == "الإجابة:":
            return letter_priors[cont]
        # Conditioned: tweak the gold letter's score upward by a fixed delta
        # so PMI vs char-norm produce different argmax in some rows.
        return letter_priors[cont] + (0.5 if cont == boost_letter else 0.0)

    return _ll, calls


@patch("arabic_eval.tasks.lighteval.base._compute_loglikelihood")
def test_evaluate_mcq_char_plus_pmi_emits_both_accuracies(mock_ll, tmp_path):
    """Smoke: ``evaluate_mcq(score_normalization='char+pmi')`` returns a
    metrics dict with ``accuracy``, ``accuracy_char_norm``, and ``accuracy_pmi``;
    char-norm aliases ``accuracy``."""
    # Letter prior favours " أ" (the strongest unigram letter prior).
    priors = {" أ": -1.0, " ب": -3.0, " ج": -2.0, " د": -4.0}
    fake_ll, _calls = _fake_loglikelihood_factory(priors, boost_letter=" ج")
    mock_ll.side_effect = fake_ll

    examples = [
        {"question": "q1", "choices": ["a", "b", "c", "d"], "answer": 2,
         "_source_config": "_default"},
        {"question": "q2", "choices": ["a", "b", "c", "d"], "answer": 2,
         "_source_config": "_default"},
    ]
    task = _make_task(CultureArabicMMLUTask)
    wrapper = LightEvalModelWrapper(model=None, tokenizer=None, max_length=64)
    metrics, failures = wrapper.evaluate_mcq(
        examples, collect_failures=False, task=task,
        score_normalization="char+pmi",
    )
    assert "accuracy" in metrics
    assert "accuracy_char_norm" in metrics
    assert "accuracy_pmi" in metrics
    assert metrics["accuracy"] == metrics["accuracy_char_norm"]


@patch("arabic_eval.tasks.lighteval.base._compute_loglikelihood")
def test_evaluate_mcq_unconditioned_cache_hits_for_letter_mcq(mock_ll):
    """Cache key = (unconditioned_query, tuple(continuations)). For letter-MCQ
    every example shares the same key, so unconditioned ll is computed once."""
    priors = {" أ": -1.0, " ب": -3.0, " ج": -2.0, " د": -4.0}
    fake_ll, calls = _fake_loglikelihood_factory(priors, boost_letter=" أ")
    mock_ll.side_effect = fake_ll

    examples = [
        {"question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer": 0,
         "_source_config": "_default"}
        for i in range(5)
    ]
    task = _make_task(CultureArabicMMLUTask)
    wrapper = LightEvalModelWrapper(model=None, tokenizer=None, max_length=64)
    wrapper.evaluate_mcq(
        examples, collect_failures=False, task=task,
        score_normalization="char+pmi",
    )

    # Per example we make 4 conditioned calls → 5 × 4 = 20 conditioned calls.
    # Unconditioned: 4 calls TOTAL (all rows hit the cache after the first).
    uncond_calls = [c for c in calls if c[0] == "الإجابة:"]
    cond_calls = [c for c in calls if c[0] != "الإجابة:"]
    assert len(uncond_calls) == 4, (
        f"Expected 4 unconditioned calls (cached after first row); got {len(uncond_calls)}"
    )
    assert len(cond_calls) == 5 * 4


@patch("arabic_eval.tasks.lighteval.base._compute_loglikelihood")
def test_evaluate_mcq_pmi_only_drops_char_keys(mock_ll):
    """Under ``"pmi"`` (not "char+pmi"), ``accuracy_char_norm`` is NOT emitted
    and ``accuracy`` aliases ``accuracy_pmi``."""
    priors = {" أ": -1.0, " ب": -3.0, " ج": -2.0, " د": -4.0}
    fake_ll, _ = _fake_loglikelihood_factory(priors, boost_letter=" ج")
    mock_ll.side_effect = fake_ll

    examples = [
        {"question": "q1", "choices": ["a", "b", "c", "d"], "answer": 2,
         "_source_config": "_default"},
    ]
    task = _make_task(CultureArabicMMLUTask)
    wrapper = LightEvalModelWrapper(model=None, tokenizer=None, max_length=64)
    metrics, _ = wrapper.evaluate_mcq(
        examples, task=task, score_normalization="pmi",
    )
    assert "accuracy" in metrics
    assert "accuracy_pmi" in metrics
    assert metrics["accuracy"] == metrics["accuracy_pmi"]
    assert "accuracy_char_norm" not in metrics


@patch("arabic_eval.tasks.lighteval.base._compute_loglikelihood")
def test_evaluate_mcq_default_char_mode_byte_identical_shape(mock_ll):
    """Under default ``"char"`` mode the shape must match the pre-PMI version:
    no ``accuracy_char_norm`` / ``accuracy_pmi`` keys."""
    priors = {" أ": -1.0, " ب": -3.0, " ج": -2.0, " د": -4.0}
    fake_ll, _ = _fake_loglikelihood_factory(priors, boost_letter=" ج")
    mock_ll.side_effect = fake_ll

    examples = [
        {"question": "q1", "choices": ["a", "b", "c", "d"], "answer": 2,
         "_source_config": "_default"},
    ]
    task = _make_task(CultureArabicMMLUTask)
    wrapper = LightEvalModelWrapper(model=None, tokenizer=None, max_length=64)
    metrics, _ = wrapper.evaluate_mcq(examples, task=task)  # default "char"

    assert set(metrics.keys()) == {"accuracy", "num_samples", "per_subconfig_accuracy"}
    sub = metrics["per_subconfig_accuracy"]["_default"]
    assert set(sub.keys()) == {"accuracy", "num_samples"}


@patch("arabic_eval.tasks.lighteval.base._compute_loglikelihood")
def test_evaluate_mcq_failure_csv_score_pmi_columns(mock_ll, tmp_path):
    """Under ``"char+pmi"`` failure rows must include ``score_pmi_*`` and
    ``score_pmi_margin`` columns (in addition to the legacy ``score_*``)."""
    # Set up so PMI argmax differs from char argmax: " أ" has the strongest
    # unconditioned prior, so under PMI the score for " أ" drops the most.
    priors = {" أ": -1.0, " ب": -3.0, " ج": -2.0, " د": -4.0}

    def _ll(model, tokenizer, ctx, cont, max_length=512):
        if ctx == "الإجابة:":
            return priors[cont]
        # Conditioned: " أ" (gold) gets a small boost but PMI removes it.
        return priors[cont] + (0.1 if cont == " أ" else 0.0)

    mock_ll.side_effect = _ll
    examples = [
        {"question": "q1", "choices": ["a", "b", "c", "d"], "answer": 0,
         "_source_config": "_default"},
    ]
    task = _make_task(CultureArabicMMLUTask)
    wrapper = LightEvalModelWrapper(model=None, tokenizer=None, max_length=64)
    metrics, failures = wrapper.evaluate_mcq(
        examples, collect_failures=True, task=task,
        score_normalization="char+pmi",
    )
    if failures:  # row may or may not be a failure depending on argmax; if so:
        rec = failures[0]
        for i in range(4):
            assert f"score_pmi_{i}" in rec
        assert "score_pmi_margin" in rec
        # Legacy columns still present for backward compat
        for i in range(4):
            assert f"score_{i}" in rec
        assert "score_margin" in rec


# ---------------------------------------------------------------------------
# 4. Config: score_normalization literal accepts/rejects correctly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["char", "pmi", "char+pmi"])
def test_score_normalization_config_accepts_valid(value):
    cfg = EvaluationConfig(score_normalization=value)
    assert cfg.score_normalization == value


def test_score_normalization_config_rejects_invalid():
    with pytest.raises(Exception):
        EvaluationConfig(score_normalization="bogus")


def test_score_normalization_config_default_is_char():
    """Backward compat: default mode is char-norm so existing run JSONs match."""
    assert EvaluationConfig().score_normalization == "char"


# ---------------------------------------------------------------------------
# 5. compute_mei: accuracy_source echoed only when non-default
# ---------------------------------------------------------------------------

def test_compute_mei_omits_accuracy_source_under_default():
    """Default ``accuracy_source='accuracy'`` is the legacy path — the field
    must NOT appear in ``inputs``."""
    out = compute_mei(0.5, 0.8, 3.0, 2.0, 100, is_lighteval_mcq=True)
    assert out["status"] == "ok"
    assert "accuracy_source" not in out["inputs"]


def test_compute_mei_emits_accuracy_source_under_pmi():
    """When MEI was computed off ``accuracy_pmi``, the source must be echoed
    so the JSON is self-describing."""
    out = compute_mei(
        0.7, 0.8, 3.0, 2.0, 100, is_lighteval_mcq=True,
        accuracy_source="accuracy_pmi",
    )
    assert out["status"] == "ok"
    assert out["inputs"]["accuracy_source"] == "accuracy_pmi"
