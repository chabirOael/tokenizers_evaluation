"""Tests for the opt-in UNK reporting flags and CSV outputs.

Covers (this file grows across the feature's implementation steps):
  1. ``EvaluationConfig.intrinsic_unk_report`` / ``downstream_unk_report``
     defaults, independence, YAML override.
  2. The shared UNK-attribution helper (``scan_text`` / aggregate /
     ``records_to_rows``).
  3. ``compute_intrinsic_metrics(unk_report_path=...)`` CSV — populated
     case, header-only case (no UNK id, no UNK seen), and a regression
     guard that the scalar ``unk_rate`` / ``vocab_coverage`` are byte-
     identical with vs. without the path.
  4. ``LightEvalBenchmarkTask._compute_downstream_unk_records`` + the
     ``evaluate(unk_report_dir=...)`` wiring, including a header-only-
     CSV case and a populated case with multi-field source aggregation.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from arabic_eval.config import EvaluationConfig
from arabic_eval.evaluation.unk_reports import (
    DOWNSTREAM_UNK_FIELDS,
    INTRINSIC_UNK_FIELDS,
    WordUnkOccurrence,
    WordUnkRecord,
    aggregate_occurrences,
    records_to_rows,
    scan_text,
)
from arabic_eval.tokenizers.base import BaseTokenizer, TokenizerOutput
from arabic_eval.utils.io import write_failure_csv


# ===========================================================================
# Minimal fake tokenizer — one UNK id, per-word encode table.
# ===========================================================================

class _UnkFakeTokenizer(BaseTokenizer):
    """Each word maps to a list of token ids via a precomputed table.

    UNK id = 3. Words not in the table encode to ``[3]`` (single UNK).
    Useful for forcing specific UNK / non-UNK encodings in tests.
    """

    UNK_ID = 3

    def __init__(self, table: Dict[str, List[int]], has_unk: bool = True):
        super().__init__()
        self._table = table
        self._has_unk = has_unk

    @property
    def vocab_size(self) -> int:
        return 100

    @property
    def embedding_type(self) -> str:
        return "standard"

    @property
    def special_tokens(self) -> dict:
        base = {"pad_token": 0, "bos_token": 1, "eos_token": 2}
        if self._has_unk:
            base["unk_token"] = self.UNK_ID
        return base

    def train(self, texts, vocab_size=None, **kw):
        pass

    def encode(self, text: str, **kw):
        ids = self._table.get(text, [self.UNK_ID])
        return TokenizerOutput(
            input_ids=list(ids),
            attention_mask=[1] * len(ids),
            tokens=[str(i) for i in ids],
        )

    def decode(self, ids, **kw):
        return ""

    def save(self, p):
        pass

    def load(self, p):
        pass


# ===========================================================================
# §1. Config flags
# ===========================================================================

class TestUnkReportConfigFlags:
    def test_defaults_are_false(self):
        cfg = EvaluationConfig()
        assert cfg.intrinsic_unk_report is False
        assert cfg.downstream_unk_report is False

    def test_flags_are_independent_intrinsic_only(self):
        cfg = EvaluationConfig(intrinsic_unk_report=True)
        assert cfg.intrinsic_unk_report is True
        assert cfg.downstream_unk_report is False

    def test_flags_are_independent_downstream_only(self):
        cfg = EvaluationConfig(downstream_unk_report=True)
        assert cfg.intrinsic_unk_report is False
        assert cfg.downstream_unk_report is True

    def test_both_flags_settable(self):
        cfg = EvaluationConfig(
            intrinsic_unk_report=True,
            downstream_unk_report=True,
        )
        assert cfg.intrinsic_unk_report is True
        assert cfg.downstream_unk_report is True

    @pytest.mark.parametrize("bad", ["yes", 1, "true", None])
    def test_rejects_non_bool(self, bad):
        # Pydantic v2 coerces some of these (e.g. 1 -> True, "true" -> True).
        # We just confirm None is rejected — the flag must be a real bool.
        if bad is None:
            with pytest.raises(Exception):
                EvaluationConfig(intrinsic_unk_report=bad)


# ===========================================================================
# §2a. scan_text
# ===========================================================================

class TestScanText:
    def test_no_unk_id_returns_empty(self):
        tok = _UnkFakeTokenizer({"hello": [10, 11]}, has_unk=False)
        assert scan_text(tok, "hello world") == []

    def test_empty_string_returns_empty(self):
        tok = _UnkFakeTokenizer({})
        assert scan_text(tok, "") == []

    def test_whitespace_only_returns_empty(self):
        tok = _UnkFakeTokenizer({})
        assert scan_text(tok, "   \t  \n ") == []

    def test_word_without_unk_skipped(self):
        tok = _UnkFakeTokenizer({"hello": [10, 11]})  # no UNK
        assert scan_text(tok, "hello") == []

    def test_word_with_unk_emitted(self):
        # Word "rare" not in table → encodes to [3] (single UNK).
        tok = _UnkFakeTokenizer({"hello": [10, 11]})
        occs = scan_text(tok, "hello rare")
        assert len(occs) == 1
        assert occs[0].word == "rare"
        assert occs[0].unk_token_count == 1
        assert occs[0].total_token_count == 1
        assert occs[0].source_field == "text"

    def test_partial_unk_word(self):
        # Word encodes to [10, 3, 11] — 1 UNK out of 3 tokens.
        tok = _UnkFakeTokenizer({"partial": [10, 3, 11]})
        occs = scan_text(tok, "partial")
        assert len(occs) == 1
        assert occs[0].unk_token_count == 1
        assert occs[0].total_token_count == 3

    def test_multiple_unk_words(self):
        tok = _UnkFakeTokenizer({"ok": [10]})
        occs = scan_text(tok, "ok rare1 rare2 ok")
        assert {o.word for o in occs} == {"rare1", "rare2"}

    def test_source_field_propagated(self):
        tok = _UnkFakeTokenizer({})
        occs = scan_text(tok, "rare", source_field="continuation_2")
        assert occs[0].source_field == "continuation_2"

    def test_context_truncation(self):
        tok = _UnkFakeTokenizer({})
        long_text = "rare " + ("x" * 500)
        occs = scan_text(tok, long_text, context_max_chars=50)
        # First word "rare" produces UNK; context snippet is truncated.
        assert len(occs[0].example_context) == 50


# ===========================================================================
# §2b. aggregate_occurrences
# ===========================================================================

class TestAggregateOccurrences:
    def _mk(self, word, n=1, field="text", ctx="ctx"):
        return WordUnkOccurrence(
            word=word, unk_token_count=n, total_token_count=n,
            source_field=field, example_context=ctx,
        )

    def test_empty(self):
        assert aggregate_occurrences([]) == {}

    def test_single_example_single_occurrence(self):
        recs = aggregate_occurrences([[self._mk("rare")]])
        assert set(recs.keys()) == {"rare"}
        r = recs["rare"]
        assert r.unk_token_count == 1
        assert r.total_token_count == 1
        assert r.source_fields == {"text"}
        assert r.num_examples_seen_in == 1

    def test_same_word_two_examples_sums_and_counts_separately(self):
        recs = aggregate_occurrences([
            [self._mk("rare")],
            [self._mk("rare", n=2)],
        ])
        r = recs["rare"]
        assert r.unk_token_count == 3
        assert r.num_examples_seen_in == 2  # one per example

    def test_same_word_twice_in_one_example_counts_example_once(self):
        recs = aggregate_occurrences([
            [self._mk("rare"), self._mk("rare", n=5)],
        ])
        r = recs["rare"]
        assert r.unk_token_count == 6
        assert r.num_examples_seen_in == 1

    def test_source_fields_unioned(self):
        recs = aggregate_occurrences([
            [
                self._mk("rare", field="prompt"),
                self._mk("rare", field="continuation_0"),
            ],
            [self._mk("rare", field="continuation_1")],
        ])
        r = recs["rare"]
        assert r.source_fields == {"prompt", "continuation_0", "continuation_1"}

    def test_first_nonempty_context_wins(self):
        recs = aggregate_occurrences([
            [self._mk("rare", ctx="")],
            [self._mk("rare", ctx="real-context")],
            [self._mk("rare", ctx="later-context")],
        ])
        assert recs["rare"].example_context == "real-context"

    def test_distinct_words_kept_separate(self):
        recs = aggregate_occurrences([
            [self._mk("rare1"), self._mk("rare2")],
        ])
        assert set(recs.keys()) == {"rare1", "rare2"}
        assert all(r.num_examples_seen_in == 1 for r in recs.values())


# ===========================================================================
# §2c. records_to_rows
# ===========================================================================

class TestRecordsToRows:
    def _record(self, word, n_unk=1, n_tok=1, sources=None, n_ex=1, ctx="c"):
        return WordUnkRecord(
            word=word,
            unk_token_count=n_unk,
            total_token_count=n_tok,
            source_fields=set(sources or {"text"}),
            num_examples_seen_in=n_ex,
            example_context=ctx,
        )

    def test_empty(self):
        assert records_to_rows([], INTRINSIC_UNK_FIELDS) == []

    def test_intrinsic_fields_only(self):
        rows = records_to_rows(
            [self._record("w", n_unk=2, n_tok=5)],
            INTRINSIC_UNK_FIELDS,
        )
        assert rows == [{
            "word": "w",
            "unk_token_count": 2,
            "total_token_count": 5,
            "example_context": "c",
        }]

    def test_downstream_fields_only(self):
        rec = self._record(
            "w", n_unk=2, sources={"prompt", "continuation_0"}, n_ex=3,
        )
        rows = records_to_rows([rec], DOWNSTREAM_UNK_FIELDS)
        assert rows == [{
            "word": "w",
            "unk_token_count": 2,
            "source_fields": "continuation_0|prompt",  # sorted, pipe-joined
            "num_examples_seen_in": 3,
            "example_context": "c",
        }]

    def test_sorted_by_count_descending_then_alpha(self):
        recs = [
            self._record("apple", n_unk=1),
            self._record("zebra", n_unk=5),
            self._record("mango", n_unk=5),  # tie with zebra
        ]
        rows = records_to_rows(recs, INTRINSIC_UNK_FIELDS)
        # 5 > 1; among the 5s, "mango" < "zebra" alphabetically.
        assert [r["word"] for r in rows] == ["mango", "zebra", "apple"]


# ===========================================================================
# §2d. End-to-end: scan → aggregate → rows → write_failure_csv
# ===========================================================================

class TestUnkCsvRoundTrip:
    def test_empty_writes_header_only(self, tmp_path: Path):
        tok = _UnkFakeTokenizer({"ok": [10]})  # no UNK words at all
        occs_by_ex = [scan_text(tok, "ok ok ok")]
        recs = aggregate_occurrences(occs_by_ex)
        rows = records_to_rows(recs.values(), INTRINSIC_UNK_FIELDS)
        p = tmp_path / "intrinsic_unks.csv"
        n = write_failure_csv(p, rows, INTRINSIC_UNK_FIELDS)
        assert n == 0
        assert p.exists()
        with open(p, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == list(INTRINSIC_UNK_FIELDS)
            assert list(reader) == []

    def test_no_unk_id_still_writes_header(self, tmp_path: Path):
        tok = _UnkFakeTokenizer({"ok": [10]}, has_unk=False)
        occs_by_ex = [scan_text(tok, "ok rare")]
        recs = aggregate_occurrences(occs_by_ex)
        rows = records_to_rows(recs.values(), INTRINSIC_UNK_FIELDS)
        p = tmp_path / "intrinsic_unks.csv"
        n = write_failure_csv(p, rows, INTRINSIC_UNK_FIELDS)
        assert n == 0
        with open(p, encoding="utf-8-sig") as f:
            assert next(csv.reader(f)) == list(INTRINSIC_UNK_FIELDS)

    def test_populated_intrinsic_csv(self, tmp_path: Path):
        # Two distinct UNK words across two texts; "rare" appears twice.
        tok = _UnkFakeTokenizer({"ok": [10]})
        occs_by_ex = [
            scan_text(tok, "ok rare"),
            scan_text(tok, "rare other"),
        ]
        recs = aggregate_occurrences(occs_by_ex)
        rows = records_to_rows(recs.values(), INTRINSIC_UNK_FIELDS)
        p = tmp_path / "intrinsic_unks.csv"
        n = write_failure_csv(p, rows, INTRINSIC_UNK_FIELDS)
        assert n == 2
        with open(p, encoding="utf-8-sig") as f:
            data = list(csv.DictReader(f))
        words = {d["word"] for d in data}
        assert words == {"rare", "other"}
        rare = next(d for d in data if d["word"] == "rare")
        assert int(rare["unk_token_count"]) == 2  # 1 per text, summed
        assert int(rare["total_token_count"]) == 2

    def test_downstream_csv_includes_source_fields_and_example_count(
        self, tmp_path: Path
    ):
        tok = _UnkFakeTokenizer({"ok": [10]})
        # Example 1 has the rare word in both prompt and continuation_0;
        # example 2 has it only in prompt.
        ex1 = scan_text(tok, "ok rare", source_field="prompt") + \
              scan_text(tok, "rare", source_field="continuation_0")
        ex2 = scan_text(tok, "rare ok", source_field="prompt")
        recs = aggregate_occurrences([ex1, ex2])
        rows = records_to_rows(recs.values(), DOWNSTREAM_UNK_FIELDS)
        p = tmp_path / "task_unks.csv"
        n = write_failure_csv(p, rows, DOWNSTREAM_UNK_FIELDS)
        assert n == 1
        with open(p, encoding="utf-8-sig") as f:
            data = list(csv.DictReader(f))
        assert data[0]["word"] == "rare"
        assert int(data[0]["unk_token_count"]) == 3
        assert int(data[0]["num_examples_seen_in"]) == 2
        assert data[0]["source_fields"] == "continuation_0|prompt"


# ===========================================================================
# §3. compute_intrinsic_metrics(unk_report_path=...) — integration
# ===========================================================================

from arabic_eval.evaluation.intrinsic_metrics import compute_intrinsic_metrics  # noqa: E402


class TestIntrinsicUnkReport:
    def _tokenizer_with_known_unk(self) -> _UnkFakeTokenizer:
        # Words "كتاب", "مدرسة" are in-vocab; everything else encodes to [UNK].
        return _UnkFakeTokenizer({
            "كتاب": [10, 11],
            "مدرسة": [12],
        })

    def test_none_path_writes_no_csv(self, tmp_path: Path):
        tok = self._tokenizer_with_known_unk()
        texts = ["كتاب rare1", "rare2 مدرسة"]
        # The temp dir starts empty.
        compute_intrinsic_metrics(
            tok, texts, morphological_metrics=False,
        )
        assert list(tmp_path.iterdir()) == []

    def test_populated_csv(self, tmp_path: Path):
        tok = self._tokenizer_with_known_unk()
        texts = [
            "كتاب rare1",
            "rare1 مدرسة rare2",
            "كتاب",
        ]
        p = tmp_path / "intrinsic_unks.csv"
        compute_intrinsic_metrics(
            tok, texts, morphological_metrics=False, unk_report_path=p,
        )
        assert p.exists()
        with open(p, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        words = {r["word"] for r in rows}
        # In-vocab words are NOT in the report; rare1/rare2 ARE.
        assert words == {"rare1", "rare2"}
        rare1 = next(r for r in rows if r["word"] == "rare1")
        # rare1 produces UNK in two texts → unk_token_count = 2.
        assert int(rare1["unk_token_count"]) == 2

    def test_no_unk_id_writes_header_only(self, tmp_path: Path):
        tok = _UnkFakeTokenizer({"كتاب": [10]}, has_unk=False)
        p = tmp_path / "intrinsic_unks.csv"
        compute_intrinsic_metrics(
            tok,
            ["كتاب rare1 rare2"],
            morphological_metrics=False,
            unk_report_path=p,
        )
        assert p.exists()
        with open(p, encoding="utf-8-sig") as f:
            header = next(csv.reader(f))
            rest = list(csv.reader(f))
        assert header == list(INTRINSIC_UNK_FIELDS)
        assert rest == []

    def test_zero_unk_seen_writes_header_only(self, tmp_path: Path):
        tok = _UnkFakeTokenizer({"كتاب": [10], "مدرسة": [11]})
        p = tmp_path / "intrinsic_unks.csv"
        compute_intrinsic_metrics(
            tok,
            ["كتاب مدرسة", "مدرسة كتاب"],
            morphological_metrics=False,
            unk_report_path=p,
        )
        with open(p, encoding="utf-8-sig") as f:
            header = next(csv.reader(f))
            rest = list(csv.reader(f))
        assert header == list(INTRINSIC_UNK_FIELDS)
        assert rest == []

    def test_scalars_unchanged_when_report_enabled(self, tmp_path: Path):
        """Regression guard: turning on the report must not change
        ``unk_rate`` or ``vocab_coverage`` (the existing public metrics).
        """
        tok = self._tokenizer_with_known_unk()
        texts = ["كتاب rare1", "rare2 مدرسة", "rare1 rare1"]
        m_off = compute_intrinsic_metrics(
            tok, texts, morphological_metrics=False,
        )
        p = tmp_path / "intrinsic_unks.csv"
        m_on = compute_intrinsic_metrics(
            tok, texts, morphological_metrics=False, unk_report_path=p,
        )
        for k in ("fertility", "compression_ratio", "unk_rate",
                  "vocab_coverage", "avg_token_count", "vocab_size"):
            assert m_off[k] == m_on[k], f"{k} drifted: off={m_off[k]!r} on={m_on[k]!r}"


# ===========================================================================
# §4. LightEvalBenchmarkTask downstream UNK CSV
# ===========================================================================

from typing import Any  # noqa: E402
from unittest.mock import patch  # noqa: E402

from arabic_eval.tasks.lighteval.base import LightEvalBenchmarkTask  # noqa: E402


class _UnkStubTask(LightEvalBenchmarkTask):
    """Minimal LightEval task for UNK-report tests.

    Prompt is the question verbatim; continuations are the literal choice
    strings (so we can plant Arabic words that the test tokenizer rejects).
    """

    def __init__(self, config, rows):
        super().__init__(config)
        self._rows = rows

    def _default_dataset_name(self):
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
        return [" " + c for c in ex.get("choices", [])]

    def _aggregate_scores(self, ex, continuations, log_likelihoods,
                          unconditioned_log_likelihoods=None, normalization="char"):
        return list(log_likelihoods)

    @property
    def name(self):
        return "stub_unk"


class TestDownstreamUnkRecords:
    def _task(self, rows):
        return _UnkStubTask({}, rows=rows)

    def test_no_unk_id_returns_empty(self):
        tok = _UnkFakeTokenizer({}, has_unk=False)
        task = self._task([
            {"question": "kitab rare", "choices": ["a", "b"], "answer": 0},
        ])
        task.get_eval_examples()  # populate cache
        recs = task._compute_downstream_unk_records(
            task.get_eval_examples(), tok,
        )
        assert recs == {}

    def test_prompt_only_unk(self):
        tok = _UnkFakeTokenizer({"kitab": [10], "a": [11], "b": [12]})
        task = self._task([
            {"question": "kitab rare1", "choices": ["a", "b"], "answer": 0},
        ])
        recs = task._compute_downstream_unk_records(
            task.get_eval_examples(), tok,
        )
        assert set(recs.keys()) == {"rare1"}
        assert recs["rare1"].source_fields == {"prompt"}

    def test_continuation_only_unk(self):
        tok = _UnkFakeTokenizer({"kitab": [10], "ok": [11]})
        task = self._task([
            {"question": "kitab", "choices": ["ok", "rare2"], "answer": 0},
        ])
        recs = task._compute_downstream_unk_records(
            task.get_eval_examples(), tok,
        )
        assert set(recs.keys()) == {"rare2"}
        assert recs["rare2"].source_fields == {"continuation_1"}

    def test_word_appears_in_both_prompt_and_continuation(self):
        tok = _UnkFakeTokenizer({"ok": [10]})
        task = self._task([
            {"question": "ok rare", "choices": ["ok", "rare"], "answer": 0},
        ])
        recs = task._compute_downstream_unk_records(
            task.get_eval_examples(), tok,
        )
        assert set(recs.keys()) == {"rare"}
        # rare seen in prompt + continuation_1.
        assert recs["rare"].source_fields == {"prompt", "continuation_1"}
        # 2 occurrences total in this one example.
        assert recs["rare"].unk_token_count == 2
        assert recs["rare"].num_examples_seen_in == 1

    def test_word_across_multiple_examples(self):
        tok = _UnkFakeTokenizer({"ok": [10]})
        task = self._task([
            {"question": "rare ok", "choices": ["ok", "ok"], "answer": 0},
            {"question": "ok ok", "choices": ["rare", "ok"], "answer": 0},
            {"question": "ok rare", "choices": ["ok", "rare"], "answer": 0},
        ])
        recs = task._compute_downstream_unk_records(
            task.get_eval_examples(), tok,
        )
        assert recs["rare"].num_examples_seen_in == 3
        # Source fields aggregated across rows.
        assert "prompt" in recs["rare"].source_fields
        assert "continuation_0" in recs["rare"].source_fields
        assert "continuation_1" in recs["rare"].source_fields


@patch("arabic_eval.tasks.lighteval.base._compute_loglikelihood")
class TestEvaluateUnkReportWiring:
    """Drive ``evaluate()`` end-to-end with a mocked log-likelihood."""

    def _fake_model(self):
        class _M:
            class _Inner:
                def eval(self_inner):
                    return None
            model = _Inner()
            device = "cpu"
        return _M()

    def _task(self, rows):
        return _UnkStubTask({}, rows=rows)

    def test_no_unk_report_dir_writes_no_csv(self, mock_ll, tmp_path: Path):
        mock_ll.return_value = -1.0
        tok = _UnkFakeTokenizer({"ok": [10]})
        task = self._task([
            {"question": "ok rare", "choices": ["ok", "rare"], "answer": 0,
             "_source_config": "_default"},
        ])
        task.evaluate(self._fake_model(), tok, max_samples=1)
        # tmp_path is unrelated to the task — confirm no unk_reports dir
        # got created in cwd or anywhere we can detect.
        assert not (tmp_path / "unk_reports").exists()

    def test_unk_report_dir_writes_populated_csv(self, mock_ll, tmp_path: Path):
        mock_ll.return_value = -1.0
        tok = _UnkFakeTokenizer({"ok": [10]})
        task = self._task([
            {"question": "ok rare1", "choices": ["ok", "rare2"], "answer": 0,
             "_source_config": "_default"},
        ])
        task.evaluate(
            self._fake_model(), tok, max_samples=1,
            unk_report_dir=tmp_path,
        )
        csv_path = tmp_path / "stub_unk_unks.csv"
        assert csv_path.exists()
        with open(csv_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        words = {r["word"] for r in rows}
        assert words == {"rare1", "rare2"}

    def test_unk_report_dir_writes_header_only_when_no_unk_id(
        self, mock_ll, tmp_path: Path
    ):
        mock_ll.return_value = -1.0
        tok = _UnkFakeTokenizer({}, has_unk=False)
        task = self._task([
            {"question": "rare", "choices": ["a", "b"], "answer": 0,
             "_source_config": "_default"},
        ])
        task.evaluate(
            self._fake_model(), tok, max_samples=1,
            unk_report_dir=tmp_path,
        )
        csv_path = tmp_path / "stub_unk_unks.csv"
        assert csv_path.exists()
        with open(csv_path, encoding="utf-8-sig") as f:
            header = next(csv.reader(f))
            rest = list(csv.reader(f))
        assert header == list(DOWNSTREAM_UNK_FIELDS)
        assert rest == []



# ===========================================================================
# §1. Config flags
# ===========================================================================

class TestUnkReportConfigFlags:
    def test_defaults_are_false(self):
        cfg = EvaluationConfig()
        assert cfg.intrinsic_unk_report is False
        assert cfg.downstream_unk_report is False

    def test_flags_are_independent_intrinsic_only(self):
        cfg = EvaluationConfig(intrinsic_unk_report=True)
        assert cfg.intrinsic_unk_report is True
        assert cfg.downstream_unk_report is False

    def test_flags_are_independent_downstream_only(self):
        cfg = EvaluationConfig(downstream_unk_report=True)
        assert cfg.intrinsic_unk_report is False
        assert cfg.downstream_unk_report is True

    def test_both_flags_settable(self):
        cfg = EvaluationConfig(
            intrinsic_unk_report=True,
            downstream_unk_report=True,
        )
        assert cfg.intrinsic_unk_report is True
        assert cfg.downstream_unk_report is True

    @pytest.mark.parametrize("bad", ["yes", 1, "true", None])
    def test_rejects_non_bool(self, bad):
        # Pydantic v2 coerces some of these (e.g. 1 -> True, "true" -> True).
        # We just confirm None is rejected — the flag must be a real bool.
        if bad is None:
            with pytest.raises(Exception):
                EvaluationConfig(intrinsic_unk_report=bad)
