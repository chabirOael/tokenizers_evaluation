"""Unit tests for the Morphological Efficiency Index (MEI) helper."""
from __future__ import annotations

import math

import pytest

from arabic_eval.evaluation.metrics import compute_mei


class TestComputeMEI:
    def test_happy_path(self):
        out = compute_mei(
            accuracy=0.5,
            rps=0.8,
            compression=3.0,
            inference_time_sec=2.0,
            num_eval_rows=1,
            is_lighteval_mcq=True,
        )
        assert out["status"] == "ok"
        # Per-row form: (acc * rps * comp * N) / time
        assert math.isclose(out["mei"], (0.5 * 0.8 * 3.0 * 1) / 2.0, rel_tol=1e-9)
        assert out["inputs"]["accuracy"] == 0.5
        assert out["inputs"]["rps"] == 0.8
        assert out["inputs"]["compression"] == 3.0
        assert out["inputs"]["inference_time_sec"] == 2.0
        assert out["inputs"]["num_eval_rows"] == 1

    def test_per_row_formula_pins_row_count_factor(self):
        # Doubling rows at constant time should double MEI (per-row time halves).
        base = compute_mei(
            accuracy=0.5, rps=0.8, compression=3.0,
            inference_time_sec=10.0, num_eval_rows=100,
            is_lighteval_mcq=True,
        )
        doubled = compute_mei(
            accuracy=0.5, rps=0.8, compression=3.0,
            inference_time_sec=10.0, num_eval_rows=200,
            is_lighteval_mcq=True,
        )
        assert base["status"] == "ok" and doubled["status"] == "ok"
        assert math.isclose(doubled["mei"], 2.0 * base["mei"], rel_tol=1e-9)
        # Direct formula check on the larger N.
        assert math.isclose(
            doubled["mei"], (0.5 * 0.8 * 3.0 * 200) / 10.0, rel_tol=1e-9
        )

    def test_within_task_ranking_invariant_under_row_count(self):
        # Two tokenizers on the same task share num_eval_rows; the ranking
        # under per-row MEI is identical to per-pass MEI (the row factor
        # is a constant scale that cancels).
        n = 500
        a = compute_mei(0.6, 0.7, 3.5, 12.0, n, is_lighteval_mcq=True)
        b = compute_mei(0.5, 0.9, 4.0, 10.0, n, is_lighteval_mcq=True)
        # Compare to per-pass values (drop the * n factor):
        a_per_pass = (0.6 * 0.7 * 3.5) / 12.0
        b_per_pass = (0.5 * 0.9 * 4.0) / 10.0
        assert (a["mei"] > b["mei"]) == (a_per_pass > b_per_pass)

    def test_task_not_mcq_returns_none_with_status(self):
        out = compute_mei(0.5, 0.8, 3.0, 2.0, 100, is_lighteval_mcq=False)
        assert out["mei"] is None
        assert out["status"] == "task_not_mcq"
        # Inputs are still echoed back so debugging is possible.
        assert out["inputs"]["accuracy"] == 0.5
        assert out["inputs"]["num_eval_rows"] == 100

    @pytest.mark.parametrize(
        "missing_field,expected_status",
        [
            ("accuracy", "missing_accuracy"),
            ("rps", "missing_rps"),
            ("compression", "missing_compression"),
            ("inference_time_sec", "missing_time"),
            ("num_eval_rows", "missing_num_eval_rows"),
        ],
    )
    def test_missing_input_returns_none_with_typed_status(
        self, missing_field, expected_status
    ):
        kwargs = dict(
            accuracy=0.5,
            rps=0.8,
            compression=3.0,
            inference_time_sec=2.0,
            num_eval_rows=100,
            is_lighteval_mcq=True,
        )
        kwargs[missing_field] = None
        out = compute_mei(**kwargs)
        assert out["mei"] is None
        assert out["status"] == expected_status

    def test_zero_time_does_not_divide_by_zero(self):
        out = compute_mei(
            accuracy=0.5, rps=0.8, compression=3.0,
            inference_time_sec=0.0, num_eval_rows=100,
            is_lighteval_mcq=True,
        )
        assert out["mei"] is None
        assert out["status"] == "zero_time"

    def test_negative_time_treated_as_zero_time(self):
        out = compute_mei(
            accuracy=0.5, rps=0.8, compression=3.0,
            inference_time_sec=-1.0, num_eval_rows=100,
            is_lighteval_mcq=True,
        )
        assert out["mei"] is None
        assert out["status"] == "zero_time"

    def test_zero_rows_does_not_divide_by_zero(self):
        out = compute_mei(
            accuracy=0.5, rps=0.8, compression=3.0,
            inference_time_sec=2.0, num_eval_rows=0,
            is_lighteval_mcq=True,
        )
        assert out["mei"] is None
        assert out["status"] == "zero_rows"

    def test_negative_rows_treated_as_zero_rows(self):
        out = compute_mei(
            accuracy=0.5, rps=0.8, compression=3.0,
            inference_time_sec=2.0, num_eval_rows=-1,
            is_lighteval_mcq=True,
        )
        assert out["mei"] is None
        assert out["status"] == "zero_rows"

    def test_accuracy_zero_is_a_valid_zero_mei(self):
        # accuracy=0 is a real result, not missing — MEI should be 0.0, not None.
        out = compute_mei(
            accuracy=0.0, rps=0.8, compression=3.0,
            inference_time_sec=2.0, num_eval_rows=100,
            is_lighteval_mcq=True,
        )
        assert out["status"] == "ok"
        assert out["mei"] == 0.0

    def test_rps_zero_mechanical_floor(self):
        # char-JABER / Charformer hit RPS≈0 mechanically — MEI should be 0.0,
        # not None. Distinguishes "couldn't measure" from "measured zero".
        out = compute_mei(
            accuracy=0.4, rps=0.0, compression=2.0,
            inference_time_sec=10.0, num_eval_rows=100,
            is_lighteval_mcq=True,
        )
        assert out["status"] == "ok"
        assert out["mei"] == 0.0

    def test_missing_priority_order(self):
        # When multiple inputs are missing, the first one we check wins.
        # Ordering: accuracy → rps → compression → time → num_eval_rows
        # → zero_time → zero_rows.
        out = compute_mei(
            accuracy=None, rps=None, compression=None,
            inference_time_sec=None, num_eval_rows=None,
            is_lighteval_mcq=True,
        )
        assert out["status"] == "missing_accuracy"

    def test_zero_time_priority_over_zero_rows(self):
        # zero_time is checked before zero_rows so its status wins when both
        # are degenerate. (Either firing is anomalous — this just pins
        # the documented order.)
        out = compute_mei(
            accuracy=0.5, rps=0.8, compression=3.0,
            inference_time_sec=0.0, num_eval_rows=0,
            is_lighteval_mcq=True,
        )
        assert out["status"] == "zero_time"

    def test_task_not_mcq_short_circuits_missing_inputs(self):
        # If task is not MCQ, status is "task_not_mcq" regardless of input
        # presence — MEI is undefined for non-MCQ tasks period.
        out = compute_mei(
            accuracy=None, rps=None, compression=None,
            inference_time_sec=None, num_eval_rows=None,
            is_lighteval_mcq=False,
        )
        assert out["status"] == "task_not_mcq"

    def test_num_eval_rows_echoed_in_inputs_on_failure(self):
        # All status branches echo num_eval_rows so migration / debugging
        # can read it back without re-running.
        out = compute_mei(
            accuracy=None, rps=0.8, compression=3.0,
            inference_time_sec=2.0, num_eval_rows=500,
            is_lighteval_mcq=True,
        )
        assert out["status"] == "missing_accuracy"
        assert out["inputs"]["num_eval_rows"] == 500
