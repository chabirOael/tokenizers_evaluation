"""Tests for ``arabic_eval.evaluation.reporter._build_mei_section``.

The MEI section consumes ``results["mei"]`` which is written by the pipeline
as ``{task_name: {mei, status, inputs}}`` (per-task records). Archived runs
predating 2026-05-04 wrote a flat single-record shape with ``status`` /
``mei`` / ``inputs`` at the top level. The reporter must handle both.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.evaluation.reporter import _build_mei_section


def _ok_subrec(mei=1.234, acc=0.5, rps=0.3, comp=2.0, t=100.0, n=1000):
    return {
        "mei": mei,
        "status": "ok",
        "inputs": {
            "accuracy": acc,
            "rps": rps,
            "compression": comp,
            "inference_time_sec": t,
            "num_eval_rows": n,
        },
    }


def _exp(tok_type, mei_block):
    return {"config": {"tokenizer": tok_type}, "mei": mei_block}


# --------------------------------------------------------------------------
# Per-task shape (current pipeline output)
# --------------------------------------------------------------------------

class TestPerTaskShape:
    def test_emits_row_per_task(self):
        experiments = {
            "bpe_32k": _exp("bpe", {
                "acva": _ok_subrec(mei=3.06),
                "alghafa": _ok_subrec(mei=1.69),
                "arabic_exam": _ok_subrec(mei=1.52),
            }),
        }
        out = "\n".join(_build_mei_section(experiments))
        assert "acva" in out and "alghafa" in out and "arabic_exam" in out
        assert "Task" in out  # column header
        # No "not computed" footer when all status == "ok"
        assert "not computed" not in out
        # All three rows for bpe_32k must appear
        assert out.count("bpe_32k") == 3

    def test_multiple_experiments_appear(self):
        experiments = {
            "bpe_32k": _exp("bpe", {"acva": _ok_subrec(mei=3.06)}),
            "wordpiece_32k": _exp("wordpiece", {"acva": _ok_subrec(mei=10.12)}),
        }
        out = "\n".join(_build_mei_section(experiments))
        assert "bpe_32k" in out and "wordpiece_32k" in out
        # Each appears once (single task each).
        assert out.count("bpe_32k") == 1
        assert out.count("wordpiece_32k") == 1

    def test_non_ok_status_goes_to_skipped_list(self):
        experiments = {
            "x": _exp("bpe", {
                "acva": _ok_subrec(mei=3.06),
                "perplexity": {"mei": None, "status": "task_not_mcq", "inputs": {}},
            }),
        }
        out = "\n".join(_build_mei_section(experiments))
        assert "acva" in out
        assert "not computed" in out
        assert "perplexity" in out
        assert "task_not_mcq" in out

    def test_no_mei_key_silently_skipped(self):
        # Experiments without a `mei` key are absent from both table and
        # skipped footer — they're silently dropped (charformer in our sweep).
        experiments = {
            "good": _exp("bpe", {"acva": _ok_subrec(mei=3.06)}),
            "incomplete": {"config": {"tokenizer": "charformer"}},  # no `mei` key
        }
        out = "\n".join(_build_mei_section(experiments))
        assert "good" in out
        assert "incomplete" not in out

    def test_mechanical_flag_asterisk_and_footnote(self):
        experiments = {
            "character_bert": _exp("character_bert", {"acva": _ok_subrec(mei=42.41)}),
        }
        out = "\n".join(_build_mei_section(experiments))
        assert "character_bert*" in out  # asterisk on tokenizer column
        assert "Footnotes" in out

    def test_empty_input_returns_empty_section(self):
        assert _build_mei_section({}) == []

    def test_no_mei_anywhere_returns_empty_section(self):
        experiments = {"x": {"config": {"tokenizer": "bpe"}}}
        assert _build_mei_section(experiments) == []


# --------------------------------------------------------------------------
# Legacy flat shape (pre-2026-05-04 archived runs)
# --------------------------------------------------------------------------

class TestLegacyFlatShape:
    def test_flat_ok_record_renders_one_row(self):
        # Legacy: `mei` is a flat dict with `status` at the top level (no task keys).
        experiments = {
            "old_run": {
                "config": {"tokenizer": "bpe"},
                "mei": _ok_subrec(mei=5.0),
            },
        }
        out = "\n".join(_build_mei_section(experiments))
        assert "old_run" in out
        # Sentinel task name for legacy flat records — keeps the table aligned.
        assert "—" in out
        assert "not computed" not in out

    def test_flat_non_ok_record_goes_to_skipped(self):
        experiments = {
            "old_run": {
                "config": {"tokenizer": "bpe"},
                "mei": {"mei": None, "status": "missing_accuracy", "inputs": {}},
            },
        }
        out = "\n".join(_build_mei_section(experiments))
        assert "not computed" in out
        assert "missing_accuracy" in out


# --------------------------------------------------------------------------
# Regression: the exact bug from outputs/experiments/all_tokenizers_sweep/
# --------------------------------------------------------------------------

class TestRegressionPerTaskNotMistakenAsLegacy:
    """Pre-fix, the per-task dict was read as a flat record. Since none of
    the task keys is named "status", ``record.get("status")`` returned None
    and every experiment was labelled "status=None / not computed" — even
    when every task had ``status: "ok"``."""

    def test_per_task_dict_does_not_get_status_none(self):
        experiments = {
            "bpe_32k": _exp("bpe", {
                "acva": _ok_subrec(),
                "alghafa": _ok_subrec(),
                "arabic_exam": _ok_subrec(),
            }),
        }
        out = "\n".join(_build_mei_section(experiments))
        assert "status=None" not in out
        assert "not computed" not in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
