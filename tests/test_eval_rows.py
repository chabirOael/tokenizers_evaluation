"""Tests for the per-eval-row Parquet dump and the console's browser over it.

Three layers, each pinned where it can actually go wrong:

  §1 record shape — the two margins mean different things, the truncation
     sentinel is detected, and ``near_tie`` describes a coin-flip decision
     rather than "the model was nearly right" (which every correct row is).
  §2 write → read — accuracy recomputed from a dump equals the accuracy the
     eval loop reported, the prompt stored is byte-identical to the one the
     scorer was handed, filters and pages partition the file exactly, and an
     empty eval still leaves a readable file carrying the schema.
  §3 browser — discovery over a sweep-shaped tree, query paging, CSV export,
     and the path guard.

Everything runs offline: the model is a mocked log-likelihood and the
tokenizer a stub, so no GPU, no network, no benchmark download.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.evaluation.eval_rows import (  # noqa: E402
    NEAR_TIE_MARGIN,
    SENTINEL_LL,
    EvalRowFile,
    EvalRowWriter,
    build_row_record,
    read_progress,
)
from arabic_eval.tasks.lighteval.base import (  # noqa: E402
    LightEvalBenchmarkTask,
    LightEvalModelWrapper,
)
from arabic_eval.tokenizers.base import EmbeddingType, TokenizerOutput  # noqa: E402
from arabic_eval.tools import eval_rows_browser as browser  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubTask(LightEvalBenchmarkTask):
    """Letter-MCQ task over injected rows; prompt ends with the answer prefix."""

    def __init__(self, config: Dict[str, Any], rows: List[Dict[str, Any]]):
        super().__init__(config)
        self._rows = rows

    def _default_dataset_name(self) -> str:
        return "stub/rows"

    def _parse_example(self, raw):
        return raw

    def load_examples(self):
        return [
            {**ex, "_source_config": ex.get("_source_config", "_default")}
            for ex in self._rows
        ]

    def _format_eval_context(self, ex):
        lines = [f"السؤال: {ex['question']}"]
        for i, c in enumerate(ex["choices"]):
            lines.append(f"{i}) {c}")
        lines.append("الإجابة:")
        return "\n".join(lines)

    def _build_continuations(self, ex):
        return [f" {c}" for c in ex["choices"]]

    def _aggregate_scores(self, ex, continuations, log_likelihoods,
                          unconditioned_log_likelihoods=None, normalization="char"):
        if normalization == "pmi":
            assert unconditioned_log_likelihoods is not None
            return [ll - u for ll, u in zip(log_likelihoods, unconditioned_log_likelihoods)]
        return [ll / max(len(c.lstrip()), 1) for c, ll in zip(continuations, log_likelihoods)]

    @property
    def name(self) -> str:
        return "stub_bench"


class _StubTokenizer:
    """Whitespace tokenizer: prompt_units is the word count."""
    embedding_type = EmbeddingType.STANDARD
    vocab_size = 100
    special_tokens = {"pad_token": 0, "unk_token": 3}

    def encode(self, text, max_length=None, truncation=False, padding=False):
        ids = [7] * len(text.split())
        if truncation and max_length:
            ids = ids[:max_length]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids),
                               tokens=text.split())


class _FakeModel:
    class _Inner:
        def eval(self):
            return None
    model = _Inner()
    device = "cpu"


def _rows(n: int = 24, n_choices: int = 4) -> List[Dict[str, Any]]:
    return [
        {"question": f"سؤال {i}", "choices": [f"خيار{j}" for j in range(n_choices)],
         "answer": i % n_choices, "_source_config": f"sub_{i % 3}"}
        for i in range(n)
    ]


def _record(**over) -> Dict[str, Any]:
    """A record with four choices, gold 0, overridable."""
    base = dict(
        row_index=0,
        example={"question": "س", "choices": ["a", "b", "c", "d"],
                 "_source_config": "_default"},
        prompt="السؤال: س\nالإجابة:",
        continuations=[" أ", " ب", " ج", " د"],
        log_likelihoods=[-1.0, -2.0, -3.0, -4.0],
        scores_char=[-1.0, -2.0, -3.0, -4.0],
        scores_pmi=None,
        unconditioned_log_likelihoods=None,
        gold_idx=0, pred_idx=0, pred_idx_char=0, pred_idx_pmi=None,
        prompt_units=5, max_length=512,
    )
    base.update(over)
    return build_row_record(**base)


# ===========================================================================
# §1. Record shape
# ===========================================================================

class TestRecordShape:
    def test_margin_is_pred_minus_gold_and_zero_when_correct(self):
        rec = _record()
        assert rec["correct"] is True
        assert rec["margin"] == 0.0

    def test_margin_is_positive_when_wrong(self):
        # Model picks index 2 (-3.0) while gold is 1 (-2.0): it lost by 1 nat.
        rec = _record(gold_idx=1, pred_idx=2, pred_idx_char=2)
        assert rec["correct"] is False
        assert rec["margin"] == pytest.approx(-1.0)

    def test_decision_margin_is_top1_minus_top2_regardless_of_correctness(self):
        rec = _record()
        assert rec["decision_margin"] == pytest.approx(1.0)  # -1.0 vs -2.0

    def test_near_tie_describes_the_decision_not_the_gold_gap(self):
        """The distinction that makes the flag useful.

        A correct row has ``margin == 0`` but is not a coin flip; a row whose
        top two scores are a hair apart is one, whether or not it was right.
        """
        confident = _record(log_likelihoods=[-1.0, -9.0, -9.0, -9.0],
                            scores_char=[-1.0, -9.0, -9.0, -9.0])
        assert confident["correct"] is True and confident["margin"] == 0.0
        assert confident["near_tie"] is False

        eps = NEAR_TIE_MARGIN / 10
        coinflip = _record(log_likelihoods=[-1.0, -1.0 - eps, -9.0, -9.0],
                           scores_char=[-1.0, -1.0 - eps, -9.0, -9.0])
        assert coinflip["near_tie"] is True

    def test_sentinel_flags_a_fully_truncated_row(self):
        rec = _record(log_likelihoods=[SENTINEL_LL] * 4,
                      scores_char=[SENTINEL_LL] * 4)
        assert rec["sentinel"] is True
        assert rec["all_sentinel"] is True

    def test_partial_sentinel_is_not_all_sentinel(self):
        rec = _record(log_likelihoods=[SENTINEL_LL, -2.0, -3.0, -4.0],
                      scores_char=[SENTINEL_LL, -2.0, -3.0, -4.0])
        assert rec["sentinel"] is True
        assert rec["all_sentinel"] is False

    def test_hit_cap_compares_prompt_units_against_max_length(self):
        assert _record(prompt_units=511, max_length=512)["hit_cap"] is False
        assert _record(prompt_units=512, max_length=512)["hit_cap"] is True
        assert _record(prompt_units=None, max_length=512)["hit_cap"] is False

    def test_disagree_needs_both_scorings(self):
        both = _record(scores_pmi=[-4.0, -1.0, -3.0, -2.0],
                       unconditioned_log_likelihoods=[0.0] * 4,
                       pred_idx=1, pred_idx_char=0, pred_idx_pmi=1)
        assert both["disagree"] is True
        assert _record()["disagree"] is False

    def test_prompt_is_stored_verbatim(self):
        prompt = "السؤال: س\n0) أ\n1) ب\nالإجابة:"
        assert _record(prompt=prompt)["prompt"] == prompt


# ===========================================================================
# §2. Write → read
# ===========================================================================

def _make_dump(tmp_path: Path, n: int = 24, n_choices: int = 4,
               score_normalization: str = "char+pmi", rows=None) -> Path:
    """Run ``evaluate`` over stub rows with a mocked model; return the dump."""
    rows = rows if rows is not None else _rows(n, n_choices)
    out = tmp_path / "eval_rows"

    def _ll(model, tokenizer, ctx, cont, max_length=512):
        # Deterministic and content-dependent: the continuation's own text
        # plus a small bonus keyed to the question, so predictions vary.
        base = -1.0 - 0.5 * len(cont)
        if "الإجابة:" == ctx:          # unconditioned pass
            return base
        return base - 0.3 * (len(ctx) % 5)

    with patch("arabic_eval.tasks.lighteval.base._compute_loglikelihood", side_effect=_ll):
        task = _StubTask({}, rows=rows)
        metrics = task.evaluate(
            _FakeModel(), _StubTokenizer(),
            row_dump_dir=out, score_normalization=score_normalization,
        )
    dump = out / "stub_bench.parquet"
    assert dump.exists()
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return dump


class TestWriteRead:
    def test_every_row_is_written_not_just_failures(self, tmp_path: Path):
        dump = _make_dump(tmp_path, n=24)
        assert EvalRowFile(dump).n_rows == 24

    def test_accuracy_recomputed_from_dump_matches_the_reported_metric(self, tmp_path: Path):
        """The invariant that makes the dump trustworthy as an audit trail."""
        dump = _make_dump(tmp_path, n=24)
        metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        f = EvalRowFile(dump)
        summary = f.summary()
        assert summary["n_rows"] == metrics["num_samples"]
        assert summary["accuracy"] == pytest.approx(metrics["accuracy"], abs=1e-4)
        assert summary["accuracy_pmi"] == pytest.approx(metrics["accuracy_pmi"], abs=1e-4)
        assert summary["accuracy_char"] == pytest.approx(metrics["accuracy_char_norm"], abs=1e-4)

    def test_stored_prompt_is_the_string_the_scorer_received(self, tmp_path: Path):
        """The dump's whole claim: this is what the model was given.

        Rebuild the prompt through the task's own formatter and compare byte
        for byte, so a future prompt change cannot silently desynchronise the
        record from the scoring path.
        """
        rows = _rows(6)
        dump = _make_dump(tmp_path, rows=rows)
        task = _StubTask({}, rows=rows)
        examples = task.get_eval_examples()
        stored = EvalRowFile(dump).rows(range(len(examples)))
        for ex, rec in zip(examples, stored):
            assert rec["prompt"] == task._format_eval_context_with_fewshot(ex)
            assert rec["continuations"] == task._build_continuations(ex)
            assert rec["gold_idx"] == ex["answer"]
            assert rec["question"] == ex["question"]
            assert rec["choices"] == ex["choices"]

    def test_fewshot_prompt_is_captured_with_its_demos(self, tmp_path: Path):
        # One sub-config so the demo pool can always supply K=2 (demos are
        # drawn from the eval row's own sub-config, minus the row itself).
        rows = [{**r, "_source_config": "only"} for r in _rows(8)]
        out = tmp_path / "eval_rows"
        with patch("arabic_eval.tasks.lighteval.base._compute_loglikelihood",
                   side_effect=lambda *a, **k: -1.0):
            task = _StubTask({"num_fewshot": 2}, rows=rows)
            task.evaluate(_FakeModel(), _StubTokenizer(), row_dump_dir=out)
        f = EvalRowFile(out / "stub_bench.parquet")
        task2 = _StubTask({"num_fewshot": 2}, rows=rows)
        examples = task2.get_eval_examples()
        for ex, rec in zip(examples, f.rows(range(len(examples)))):
            expected = task2._format_eval_context_with_fewshot(ex)
            assert rec["prompt"] == expected
            # The demos really are in there: three answer prefixes, not one.
            assert rec["prompt"].count("الإجابة:") == 3
        assert f.metadata["num_fewshot"] == 2

    def test_metadata_records_the_run_context(self, tmp_path: Path):
        f = EvalRowFile(_make_dump(tmp_path))
        meta = f.metadata
        assert meta["task"] == "stub_bench"
        assert meta["score_normalization"] == "char+pmi"
        assert meta["primary"] == "pmi"
        assert meta["unit"] == "tokens"          # standard embedding
        assert meta["embedding_type"] == EmbeddingType.STANDARD
        assert meta["max_length"] == 512
        assert meta["schema_version"] >= 1

    def test_prompt_units_are_measured(self, tmp_path: Path):
        rows = _rows(4)
        f = EvalRowFile(_make_dump(tmp_path, rows=rows))
        task = _StubTask({}, rows=rows)
        for ex, rec in zip(task.get_eval_examples(), f.rows(range(4))):
            # The stub tokenizer emits one id per whitespace word.
            assert rec["prompt_units"] == len(task._format_eval_context(ex).split())

    def test_char_only_mode_leaves_pmi_columns_unset(self, tmp_path: Path):
        f = EvalRowFile(_make_dump(tmp_path, score_normalization="char"))
        assert f.index()["has_correct_char"] is True
        assert f.index()["has_correct_pmi"] is False
        with pytest.raises(ValueError, match="no pmi-normalised scores"):
            f.select(scoring="pmi")

    def test_empty_eval_still_writes_a_readable_file(self, tmp_path: Path):
        dump = _make_dump(tmp_path, rows=[])
        f = EvalRowFile(dump)
        assert f.n_rows == 0
        assert f.summary()["n_rows"] == 0
        assert f.rows([]) == []

    def test_progress_file_appears_during_and_vanishes_after(self, tmp_path: Path):
        path = tmp_path / "t.parquet"
        w = EvalRowWriter(path, metadata={"task": "t"}, row_group_size=2)
        w.write(_record()); w.write(_record()); w.write(_record())
        progress = read_progress(path)
        assert progress is not None and progress["done"] is False
        assert progress["rows"] == 2          # flushed at the row-group boundary
        w.close()
        assert read_progress(path) is None

    def test_close_is_idempotent(self, tmp_path: Path):
        path = tmp_path / "t.parquet"
        w = EvalRowWriter(path, metadata={"task": "t"})
        w.write(_record())
        w.close()
        w.close()
        assert EvalRowFile(path).n_rows == 1


class TestSelection:
    def test_outcome_partitions_the_file(self, tmp_path: Path):
        f = EvalRowFile(_make_dump(tmp_path, n=24))
        right = f.select(outcome="correct")
        wrong = f.select(outcome="wrong")
        assert len(right) + len(wrong) == f.n_rows
        assert not set(right.tolist()) & set(wrong.tolist())
        assert f.summary(right)["accuracy"] == 1.0
        assert f.summary(wrong)["accuracy"] == 0.0

    def test_subconfig_filter(self, tmp_path: Path):
        f = EvalRowFile(_make_dump(tmp_path, n=24))
        counts = {c["name"]: c["n_rows"] for c in f.source_configs()}
        assert sum(counts.values()) == 24
        picked = f.select(source_configs=["sub_1"])
        assert len(picked) == counts["sub_1"]

    def test_text_search_matches_question_and_optionally_prompt(self, tmp_path: Path):
        f = EvalRowFile(_make_dump(tmp_path, n=24))
        assert len(f.select(query="سؤال 7")) == 1
        # The answer prefix is in every prompt but in no question.
        assert len(f.select(query="الإجابة")) == 0
        assert len(f.select(query="الإجابة", search_prompt=True)) == 24

    def test_unknown_flag_and_sort_are_rejected(self, tmp_path: Path):
        f = EvalRowFile(_make_dump(tmp_path, n=8))
        with pytest.raises(ValueError, match="unknown flag"):
            f.select(flags={"not_a_flag": True})
        with pytest.raises(ValueError, match="unknown sort"):
            f.select(sort="sideways")

    def test_sort_uncertain_is_ascending_by_decision_margin(self, tmp_path: Path):
        f = EvalRowFile(_make_dump(tmp_path, n=24))
        idx = f.index()["decision_margin"]
        ordered = [float(idx[i]) for i in f.select(sort="uncertain")]
        assert ordered == sorted(ordered)
        reverse = [float(idx[i]) for i in f.select(sort="confident")]
        assert reverse == sorted(reverse, reverse=True)

    def test_rows_return_in_the_order_asked(self, tmp_path: Path):
        f = EvalRowFile(_make_dump(tmp_path, n=24))
        want = [5, 0, 23, 11]
        assert [r["position"] for r in f.rows(want)] == want
        assert [r["row_index"] for r in f.rows(want)] == want


# ===========================================================================
# §3. Browser (console back-end)
# ===========================================================================

def _sweep_tree(tmp_path: Path) -> Path:
    """A repo-shaped tree: one sweep with two cells, one single-cell run."""
    repo = tmp_path / "repo"
    base = repo / "outputs" / "experiments"
    for cell, tokenizer in (("bpe_32k", "bpe"), ("charformer", "charformer")):
        cell_dir = base / "sweep_demo" / cell
        _make_dump(cell_dir, n=24)
        (cell_dir / "config.json").write_text(json.dumps({
            "tokenizer": {"type": tokenizer, "vocab_size": 32000},
            "model": {"name_or_path": "meta-llama/Llama-3.2-1B"},
        }), encoding="utf-8")
        (cell_dir / "all_metrics.json").write_text(json.dumps({
            "downstream": {"stub_bench": {"accuracy": 0.25, "num_samples": 24}},
        }), encoding="utf-8")
    _make_dump(base / "solo_run", n=8)
    return repo


class TestDiscovery:
    def test_groups_sweep_cells_and_single_runs(self, tmp_path: Path):
        repo = _sweep_tree(tmp_path)
        found = browser.discover(repo)["experiments"]
        names = [e["experiment"] for e in found]
        assert names == ["solo_run", "sweep_demo"]
        sweep = next(e for e in found if e["experiment"] == "sweep_demo")
        assert sweep["single"] is False
        assert [c["cell"] for c in sweep["cells"]] == ["bpe_32k", "charformer"]
        solo = next(e for e in found if e["experiment"] == "solo_run")
        assert solo["single"] is True

    def test_carries_tokenizer_and_reported_accuracy_from_the_cell_config(self, tmp_path: Path):
        repo = _sweep_tree(tmp_path)
        sweep = next(e for e in browser.discover(repo)["experiments"]
                     if e["experiment"] == "sweep_demo")
        cell = sweep["cells"][0]
        assert cell["tokenizer"] == "bpe"
        assert cell["vocab_size"] == 32000
        assert cell["model"] == "meta-llama/Llama-3.2-1B"
        assert cell["reported"]["stub_bench"]["num_samples"] == 24

    def test_task_entry_has_row_count_and_metadata(self, tmp_path: Path):
        repo = _sweep_tree(tmp_path)
        task = browser.discover(repo)["experiments"][0]["cells"][0]["tasks"][0]
        assert task["task"] == "stub_bench"
        assert task["n_rows"] == 8
        assert task["in_progress"] is False
        assert task["metadata"]["task"] == "stub_bench"

    def test_in_flight_task_shows_from_its_progress_file(self, tmp_path: Path):
        """A dump being written has no footer yet, so it is invisible to a
        Parquet reader. The progress file is what keeps it on the list."""
        repo = tmp_path / "repo"
        rows_dir = repo / "outputs" / "experiments" / "running" / "eval_rows"
        w = EvalRowWriter(rows_dir / "acva.parquet", metadata={"task": "acva"},
                          row_group_size=2)
        w.write(_record()); w.write(_record())
        try:
            tasks = browser.discover(repo)["experiments"][0]["cells"][0]["tasks"]
            assert len(tasks) == 1
            assert tasks[0]["in_progress"] is True
            assert tasks[0]["n_rows"] == 2
        finally:
            w.close()

    def test_empty_tree(self, tmp_path: Path):
        assert browser.discover(tmp_path / "nothing")["experiments"] == []


class TestQuery:
    def _repo_and_path(self, tmp_path: Path):
        repo = _sweep_tree(tmp_path)
        return repo, "outputs/experiments/sweep_demo/bpe_32k/eval_rows/stub_bench.parquet"

    def test_pages_partition_the_selection_without_overlap(self, tmp_path: Path):
        repo, rel = self._repo_and_path(tmp_path)
        seen: List[int] = []
        first = browser.query(repo, rel, {"page_size": 10, "page": 1})
        assert first["total"] == 24 and first["pages"] == 3
        for page in range(1, first["pages"] + 1):
            got = browser.query(repo, rel, {"page_size": 10, "page": page})
            seen.extend(r["row_index"] for r in got["rows"])
        assert seen == sorted(seen)
        assert len(seen) == len(set(seen)) == 24

    def test_summary_describes_the_whole_selection_not_the_page(self, tmp_path: Path):
        repo, rel = self._repo_and_path(tmp_path)
        got = browser.query(repo, rel, {"page_size": 5, "page": 1, "outcome": "wrong"})
        assert len(got["rows"]) == 5
        assert got["summary"]["n_rows"] == got["total"] > 5
        assert got["summary"]["accuracy"] == 0.0

    def test_page_clamped_into_range(self, tmp_path: Path):
        repo, rel = self._repo_and_path(tmp_path)
        assert browser.query(repo, rel, {"page": 999, "page_size": 10})["page"] == 3
        assert browser.query(repo, rel, {"page": 0, "page_size": 10})["page"] == 1

    def test_page_size_capped(self, tmp_path: Path):
        repo, rel = self._repo_and_path(tmp_path)
        got = browser.query(repo, rel, {"page_size": 10_000})
        assert got["page_size"] == browser.MAX_PAGE_SIZE

    def test_subconfig_facets_follow_the_filter(self, tmp_path: Path):
        repo, rel = self._repo_and_path(tmp_path)
        got = browser.query(repo, rel, {"subconfig": "sub_0"})
        assert [f["name"] for f in got["subconfigs"]] == ["sub_0"]
        assert got["total"] == sum(f["n_rows"] for f in got["subconfigs"])

    def test_flag_filter_accepts_the_compact_string_form(self, tmp_path: Path):
        repo, rel = self._repo_and_path(tmp_path)
        none_hit = browser.query(repo, rel, {"flags": "all_sentinel"})
        assert none_hit["total"] == 0          # nothing truncated in the stub
        negated = browser.query(repo, rel, {"flags": "-all_sentinel"})
        assert negated["total"] == 24

    def test_bad_inputs_raise_eval_rows_error(self, tmp_path: Path):
        repo, rel = self._repo_and_path(tmp_path)
        for params in ({"flags": "nope"}, {"sort": "sideways"},
                       {"scoring": "vibes"}, {"outcome": "maybe"}):
            with pytest.raises(browser.EvalRowsError):
                browser.query(repo, rel, params)

    def test_single_row_lookup(self, tmp_path: Path):
        repo, rel = self._repo_and_path(tmp_path)
        got = browser.row(repo, rel, 7)
        assert got["row"]["row_index"] == 7
        with pytest.raises(browser.EvalRowsError, match="out of range"):
            browser.row(repo, rel, 999)

    def test_cache_notices_a_rewritten_dump(self, tmp_path: Path):
        """A re-run overwrites the file in place; the browser must not serve
        the previous generation from its handle cache."""
        repo, rel = self._repo_and_path(tmp_path)
        assert browser.query(repo, rel, {})["n_rows_file"] == 24
        cell_dir = repo / "outputs/experiments/sweep_demo/bpe_32k"
        _make_dump(cell_dir, n=12)
        assert browser.query(repo, rel, {})["n_rows_file"] == 12


class TestExportAndSafety:
    def test_export_csv_renders_the_current_selection(self, tmp_path: Path):
        repo = _sweep_tree(tmp_path)
        rel = "outputs/experiments/sweep_demo/bpe_32k/eval_rows/stub_bench.parquet"
        name, text = browser.export_csv(repo, rel, {"outcome": "wrong"})
        assert name.endswith("_rows.csv") and "bpe_32k" in name
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == browser.query(repo, rel, {"outcome": "wrong"})["total"]
        # List columns are flattened for the spreadsheet, prompt kept verbatim.
        assert "|" in rows[0]["choices"]
        assert "الإجابة:" in rows[0]["prompt"]
        assert rows[0]["correct"] == "False"

    def test_path_guard_rejects_escapes_and_non_dumps(self, tmp_path: Path):
        repo = _sweep_tree(tmp_path)
        for bad in ("../../etc/passwd", "outputs/../../secrets.parquet",
                    "outputs/experiments/sweep_demo/bpe_32k/config.json", ""):
            with pytest.raises(browser.EvalRowsError):
                browser.resolve_dump(repo, bad)

    def test_missing_dump_reports_clearly(self, tmp_path: Path):
        with pytest.raises(browser.EvalRowsError, match="no such dump"):
            browser.resolve_dump(_sweep_tree(tmp_path), "outputs/experiments/nope.parquet")


class TestFailureReportInteraction:
    def test_row_dump_supersedes_the_failure_report(self, tmp_path: Path):
        """Both flags on: the dump is written and the failure report is not,
        because the dump already contains every failing row plus its prompt."""
        rows = _rows(12)
        with patch("arabic_eval.tasks.lighteval.base._compute_loglikelihood",
                   side_effect=lambda *a, **k: -1.0):
            task = _StubTask({}, rows=rows)
            task.evaluate(
                _FakeModel(), _StubTokenizer(),
                row_dump_dir=tmp_path / "eval_rows",
                failure_report_dir=tmp_path / "failure_reports",
            )
        assert (tmp_path / "eval_rows" / "stub_bench.parquet").exists()
        assert not (tmp_path / "failure_reports").exists()

    def test_failure_report_alone_still_writes_parquet(self, tmp_path: Path):
        import pyarrow.parquet as pq

        rows = _rows(12)
        with patch("arabic_eval.tasks.lighteval.base._compute_loglikelihood",
                   side_effect=lambda *a, **k: -1.0):
            task = _StubTask({}, rows=rows)
            task.evaluate(
                _FakeModel(), _StubTokenizer(),
                failure_report_dir=tmp_path / "failure_reports",
            )
        out = tmp_path / "failure_reports" / "stub_bench_accuracy_failures.parquet"
        assert out.exists()
        table = pq.read_table(out)
        assert "ll_margin" in table.schema.names
        assert json.loads(table.schema.metadata[b"arabic_eval"])["kind"] == "accuracy_failures"


class TestSummaryScoping:
    def test_headline_accuracy_follows_the_selected_scoring(self, tmp_path: Path):
        """Filtering to "wrong under char-norm" must report 0 % accuracy.

        The two scorings disagree on real data, so a headline pinned to the
        primary column would show a high accuracy next to a selection labelled
        "wrong" — the exact reading a reviewer would take at face value.
        """
        f = EvalRowFile(_make_dump(tmp_path, n=24))
        for scoring in ("primary", "char", "pmi"):
            wrong = f.select(outcome="wrong", scoring=scoring)
            assert f.summary(wrong, scoring=scoring)["accuracy"] == 0.0
            right = f.select(outcome="correct", scoring=scoring)
            assert f.summary(right, scoring=scoring)["accuracy"] == 1.0

    def test_breakdowns_still_report_both_scorings(self, tmp_path: Path):
        f = EvalRowFile(_make_dump(tmp_path, n=24))
        got = f.summary(f.select(outcome="wrong", scoring="char"), scoring="char")
        assert got["accuracy"] == got["accuracy_char"] == 0.0
        assert "accuracy_pmi" in got        # the same rows, scored the other way

    def test_query_summary_agrees_with_its_outcome_filter(self, tmp_path: Path):
        repo = _sweep_tree(tmp_path)
        rel = "outputs/experiments/sweep_demo/bpe_32k/eval_rows/stub_bench.parquet"
        for scoring in ("primary", "char", "pmi"):
            got = browser.query(repo, rel, {"outcome": "wrong", "scoring": scoring})
            assert got["summary"]["accuracy"] == 0.0


class TestReaderGuards:
    def test_availability_booleans_are_not_usable_as_flags(self, tmp_path: Path):
        """``has_correct_pmi`` lives in the index but is not a row mask."""
        f = EvalRowFile(_make_dump(tmp_path, n=8))
        with pytest.raises(ValueError, match="unknown flag"):
            f.select(flags={"has_correct_pmi": True})

    def test_out_of_range_position_is_reported_clearly(self, tmp_path: Path):
        f = EvalRowFile(_make_dump(tmp_path, n=8))
        with pytest.raises(ValueError, match="outside the file"):
            f.rows([0, 99])
