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
            is_lighteval_mcq=True,
        )
        assert out["status"] == "ok"
        assert math.isclose(out["mei"], (0.5 * 0.8 * 3.0) / 2.0, rel_tol=1e-9)
        assert out["inputs"]["accuracy"] == 0.5
        assert out["inputs"]["rps"] == 0.8
        assert out["inputs"]["compression"] == 3.0
        assert out["inputs"]["inference_time_sec"] == 2.0

    def test_task_not_mcq_returns_none_with_status(self):
        out = compute_mei(0.5, 0.8, 3.0, 2.0, is_lighteval_mcq=False)
        assert out["mei"] is None
        assert out["status"] == "task_not_mcq"
        # Inputs are still echoed back so debugging is possible.
        assert out["inputs"]["accuracy"] == 0.5

    @pytest.mark.parametrize(
        "missing_field,expected_status",
        [
            ("accuracy", "missing_accuracy"),
            ("rps", "missing_rps"),
            ("compression", "missing_compression"),
            ("inference_time_sec", "missing_time"),
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
            is_lighteval_mcq=True,
        )
        kwargs[missing_field] = None
        out = compute_mei(**kwargs)
        assert out["mei"] is None
        assert out["status"] == expected_status

    def test_zero_time_does_not_divide_by_zero(self):
        out = compute_mei(
            accuracy=0.5, rps=0.8, compression=3.0,
            inference_time_sec=0.0, is_lighteval_mcq=True,
        )
        assert out["mei"] is None
        assert out["status"] == "zero_time"

    def test_negative_time_treated_as_zero_time(self):
        out = compute_mei(
            accuracy=0.5, rps=0.8, compression=3.0,
            inference_time_sec=-1.0, is_lighteval_mcq=True,
        )
        assert out["mei"] is None
        assert out["status"] == "zero_time"

    def test_accuracy_zero_is_a_valid_zero_mei(self):
        # accuracy=0 is a real result, not missing — MEI should be 0.0, not None.
        out = compute_mei(
            accuracy=0.0, rps=0.8, compression=3.0,
            inference_time_sec=2.0, is_lighteval_mcq=True,
        )
        assert out["status"] == "ok"
        assert out["mei"] == 0.0

    def test_rps_zero_mechanical_floor(self):
        # char-JABER / Charformer hit RPS≈0 mechanically — MEI should be 0.0,
        # not None. Distinguishes "couldn't measure" from "measured zero".
        out = compute_mei(
            accuracy=0.4, rps=0.0, compression=2.0,
            inference_time_sec=10.0, is_lighteval_mcq=True,
        )
        assert out["status"] == "ok"
        assert out["mei"] == 0.0

    def test_missing_priority_order(self):
        # When multiple inputs are missing, the first one we check wins.
        # Ordering: accuracy → rps → compression → time → zero_time.
        out = compute_mei(
            accuracy=None, rps=None, compression=None, inference_time_sec=None,
            is_lighteval_mcq=True,
        )
        assert out["status"] == "missing_accuracy"

    def test_task_not_mcq_short_circuits_missing_inputs(self):
        # If task is not MCQ, status is "task_not_mcq" regardless of input
        # presence — MEI is undefined for non-MCQ tasks period.
        out = compute_mei(
            accuracy=None, rps=None, compression=None, inference_time_sec=None,
            is_lighteval_mcq=False,
        )
        assert out["status"] == "task_not_mcq"
