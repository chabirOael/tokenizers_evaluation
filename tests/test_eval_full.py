"""Test that LightEvalBenchmarkTask returns the full benchmark for evaluation.

Under the 3-phase pipeline, training is task-agnostic — every benchmark row
is in the eval set, no SFT split is taken from the benchmark itself.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.tasks.lighteval.base import LightEvalBenchmarkTask


class _StubTask(LightEvalBenchmarkTask):
    """Minimal LightEval task for split-policy tests; no HF I/O."""

    def __init__(self, config: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
        super().__init__(config)
        self._rows = rows

    def _default_dataset_name(self) -> str:
        return "stub/dataset"

    def _parse_example(self, raw):
        return raw

    def load_examples(self):
        return [
            {**ex, "_source_config": ex.get("_source_config", "_default")}
            for ex in self._rows
        ]

    def _format_eval_context(self, ex):
        return ex.get("question", "")

    def _build_continuations(self, ex):
        return [" A", " B", " C", " D"]

    def _aggregate_scores(self, ex, continuations, log_likelihoods, unconditioned_log_likelihoods=None, normalization="char"):
        return list(log_likelihoods)

    @property
    def name(self) -> str:
        return "stub"


def _rows(n: int) -> List[Dict[str, Any]]:
    return [
        {"question": f"q{i}", "choices": ["A", "B", "C", "D"], "answer": 0,
         "_source_config": "_default"}
        for i in range(n)
    ]


def test_eval_returns_all_rows():
    task = _StubTask({}, rows=_rows(50))
    examples = task.get_eval_examples()
    assert len(examples) == 50


def test_eval_examples_cached():
    task = _StubTask({}, rows=_rows(10))
    a = task.get_eval_examples()
    b = task.get_eval_examples()
    assert a is b


def test_clean_latin_filters_rows_under_eval_only_pipeline():
    """clean_latin_rows still works (independent of split policy)."""
    arabic_rows = [
        {"question": f"س{i}", "choices": ["خ", "ج"], "answer": 0,
         "_source_config": "_default"}
        for i in range(5)
    ]
    latin_rows = [
        {"question": f"q{i}", "choices": ["a", "b"], "answer": 0,
         "_source_config": "_default"}
        for i in range(3)
    ]
    task = _StubTask({"clean_latin_rows": True}, rows=arabic_rows + latin_rows)
    examples = task.get_eval_examples()
    assert len(examples) == 5  # only Arabic rows survive


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
