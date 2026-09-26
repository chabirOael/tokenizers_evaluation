"""The ``ae`` analysis helpers (arabic_eval.analysis) on a fixture experiments folder,
plus a reproduction of the report's numbers on the real cells when they are present."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import arabic_eval.analysis as ae
from arabic_eval.analysis import display, docs
from arabic_eval.judge.freeform_judge import paired_bootstrap

N_ROWS = 40


def _write_parquet(path: Path, table: pa.Table, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = table.replace_schema_metadata({b"arabic_eval": json.dumps(meta).encode()})
    pq.write_table(table, path)


def _mcq_dump(path: Path, correct_char, correct_pmi, capped, *, prompts=None, max_length=512) -> None:
    n = len(correct_char)
    conts = [[" أ", " ب", " ج", " د"] for _ in range(n)]
    gold = [i % 4 for i in range(n)]
    pred_c = [g if c else (g + 1) % 4 for g, c in zip(gold, correct_char)]
    pred_p = [g if c else (g + 2) % 4 for g, c in zip(gold, correct_pmi)]
    t = pa.table({
        "row_index": pa.array(list(range(n))[::-1], pa.int32()),       # written out of order on purpose
        "source_config": [("cfg_a" if i % 2 else "cfg_b") for i in range(n)][::-1],
        "prompt": (prompts or [f"سؤال {i}" for i in range(n)])[::-1],
        "continuations": conts[::-1],
        "gold_idx": pa.array(gold[::-1], pa.int8()),
        "pred_idx_char": pa.array(pred_c[::-1], pa.int8()),
        "pred_idx_pmi": pa.array(pred_p[::-1], pa.int8()),
        "correct": list(correct_pmi)[::-1],
        "correct_char": list(correct_char)[::-1],
        "correct_pmi": list(correct_pmi)[::-1],
        "hit_cap": list(capped)[::-1],
        "all_sentinel": [False] * n,
        "sentinel": [False] * n,
        "near_tie": [False] * n,
        "cont_tokens": [[1, 1, 1, 1]] * n,
    })
    _write_parquet(path, t, {"task": "arabic_exam", "max_length": max_length, "num_fewshot": 3, "schema_version": 2})


def _freeform(cell: Path, scores, judge="gemma4_31b") -> None:
    ids = [f"cidar-{i}" for i in range(len(scores))]
    strata = [("short", "medium", "long")[i % 3] for i in range(len(scores))]
    gen = pa.table({"id": ids, "stratum": strata, "instruction": ["اكتب"] * len(ids), "reference": ["مرجع"] * len(ids),
                    "generation": ["جواب"] * len(ids), "stop_reason": ["eos"] * len(ids), "chrf": [10.0] * len(ids)})
    _write_parquet(cell / "eval_rows" / "freeform_cidar.parquet", gen,
                   {"task": "freeform_cidar", "schema_version": 3, "decoding": {"max_output_chars": 2400, "sampling": "greedy",
                                                                                "repetition_penalty": 1.0}})
    jt = pa.table({"id": ids, "stratum": strata, "score": pa.array(scores, pa.int64()),
                   "correctness": pa.array(scores, pa.int64()), "fluency": pa.array(scores, pa.int64()),
                   "instruction_following": pa.array(scores, pa.int64()), "flags": [""] * len(ids),
                   "rationale": ["ok"] * len(ids), "parse_ok": [True] * len(ids), "raw": ["{}"] * len(ids)})
    _write_parquet(cell / "freeform_judge" / f"{judge}.parquet", jt, {"kind": "freeform_judgments"})


def _metrics(cell: Path, *, tokenizer: str, acc_char=None, acc_pmi=None, judge_mean=None, parent=None) -> None:
    am = {"config": {"tokenizer": tokenizer}, "intrinsic": {"fertility": 1.5, "vocab_size": 16000},
          "training": {"embedding_alignment": {"status": "skipped"}, "warmup": {"status": "skipped"},
                       "sft": {"status": "ok", "steps_completed": 100,
                               "train_losses_tail": [[99, 0.5], [100, 0.4]],
                               "eval_history": [{"step": 50, "loss": 0.9, "tokens": 10,
                                                 "per_category": {"free_form": {"loss": 1.0, "tokens": 8}}}]}},
          "downstream": {}}
    if acc_char is not None:
        am["downstream"]["arabic_exam"] = {"num_samples": N_ROWS, "accuracy": acc_pmi,
                                           "accuracy_char_norm": acc_char, "accuracy_pmi": acc_pmi}
    if judge_mean is not None:
        am["downstream"]["freeform_cidar"] = {"status": "ok", "chrf": 10.0, "loop_stop_rate": 0.1,
                                              "judge": {"gemma4_31b": {"score_mean": judge_mean, "score_se": 0.1, "n": 6}}}
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "all_metrics.json").write_text(json.dumps(am))
    cfg = {"tokenizer": {"type": tokenizer, "vocab_size": 16000},
           "model": {"type": "qwen3", "name_or_path": parent or "Qwen/Qwen3-4B-Base"},
           "training": {"phases": {"sft": {"mixture": {"total_examples": 100, "shares": {"extractive": 0.4, "mcq": 0.3, "free_form": 0.3}}}}},
           "evaluation": {"score_normalization": "char+pmi"}}
    (cell / "config.json").write_text(json.dumps(cfg))


@pytest.fixture()
def root(tmp_path):
    rng = np.random.RandomState(0)
    exp = tmp_path / "experiments" / "exp"
    a_char, b_char = rng.rand(N_ROWS) < 0.6, rng.rand(N_ROWS) < 0.4
    a_pmi, b_pmi = rng.rand(N_ROWS) < 0.55, rng.rand(N_ROWS) < 0.5
    cap_a = np.zeros(N_ROWS, bool); cap_a[[3, 7]] = True
    cap_b = np.zeros(N_ROWS, bool); cap_b[[7, 11]] = True
    _mcq_dump(exp / "cellA" / "eval_rows" / "arabic_exam.parquet", a_char, a_pmi, cap_a)
    _mcq_dump(exp / "cellB" / "eval_rows" / "arabic_exam.parquet", b_char, b_pmi, cap_b, max_length=1024)
    _metrics(exp / "cellA", tokenizer="araroopat", acc_char=0.61, acc_pmi=0.55,
             parent="outputs/experiments/exp/base_run/training/warmup")
    _metrics(exp / "cellB", tokenizer="bpe", acc_char=0.42, acc_pmi=0.50)
    _freeform(exp / "ffA", [3, 2, 4, 1, 5, 3])
    _freeform(exp / "ffB", [2, 2, 3, 1, 4, 3])
    _metrics(exp / "ffA", tokenizer="araroopat", judge_mean=3.0)
    _metrics(exp / "ffB", tokenizer="bpe", judge_mean=2.5)
    _metrics(exp / "_superseded" / "old", tokenizer="bpe", acc_char=0.1, acc_pmi=0.1)
    _metrics(tmp_path / "experiments" / "exp2" / "cellA", tokenizer="wordpiece")   # a second 'cellA'
    ae.set_experiments_root(tmp_path / "experiments")
    yield tmp_path / "experiments"
    ae.set_experiments_root(None)


def _files_state(root: Path):
    return {str(p): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


# --------------------------------------------------------------------------
# catalog / resolution
# --------------------------------------------------------------------------

def test_catalog_lists_cells_skips_superseded(root):
    cat = ae.catalog()
    assert set(cat.index) == {"exp/cellA", "exp/cellB", "exp/ffA", "exp/ffB", "exp2/cellA"}
    assert "exp/_superseded/old" in set(ae.catalog(include_superseded=True).index)
    assert cat.loc["exp/cellA", "mcq"] == "arabic_exam"
    assert cat.loc["exp/cellA", "parent"] == "exp/base_run:warmup"
    assert cat.loc["exp/cellB", "rules"] == "arabic_exam: len1024 3shot s2"
    assert cat.loc["exp/ffA", "rules"] == "freeform_cidar: 2400ch greedy"
    assert cat.loc["exp/ffA", "judges"] == "gemma4_31b"
    assert set(ae.catalog("exp2").index) == {"exp2/cellA"}
    full = ae.catalog(full=True)
    assert full.loc["exp/cellA", "sft_mixture"] == "100 @ 40/30/30 (extractive/mcq/free_form)"


def test_resolve_by_name_id_path_and_errors(root):
    assert ae.resolve("cellB").id == "exp/cellB"
    assert ae.resolve("exp/cellA").id == "exp/cellA"
    assert ae.resolve(root / "exp" / "ffA").id == "exp/ffA"
    with pytest.raises(ae.CellNotFound, match="ambiguous"):
        ae.resolve("cellA")
    with pytest.raises(ae.CellNotFound, match="did you mean"):
        ae.resolve("cell")                       # a substring of several ids → suggestions
    with pytest.raises(ae.CellNotFound, match="lists every cell"):
        ae.resolve("zzz")


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------

def test_metrics_uses_char_and_pmi_never_the_bare_accuracy(root):
    m = ae.metrics(["exp/cellA", "cellB", "ffA"])
    assert m.loc["exp/cellA", "arabic_exam.acc_char"] == 0.61        # accuracy (= pmi here) is not taken
    assert m.loc["exp/cellA", "arabic_exam.acc_pmi"] == 0.55
    assert m.loc["exp/ffA", "judge.gemma4_31b.mean"] == 3.0
    assert "intrinsic.fertility" in m.columns
    long = ae.metrics_long(["exp/ffA"], contains="judge")
    assert set(long["key"]) >= {"downstream.freeform_cidar.judge.gemma4_31b.score_mean"}
    assert ae.metric("exp/cellB", "downstream.arabic_exam.accuracy_char_norm") == 0.42
    assert ae.metric("exp/cellB", "downstream.nope", "x") == "x"


def test_rows_histories_and_judges(root):
    rows = ae.mcq_rows("cellB", "arabic_exam", columns=["row_index", "correct_char"])
    assert list(rows.columns) == ["row_index", "correct_char"] and len(rows) == N_ROWS
    ff = ae.freeform_rows("ffA")
    assert ff.index.name == "id" and "gemma4_31b.score" in ff.columns and "gemma4_31b.rationale" not in ff.columns
    assert "gemma4_31b.rationale" in ae.freeform_rows("ffA", rationale=True).columns
    with pytest.raises(FileNotFoundError, match="no judge file"):
        ae.freeform_rows("ffA", judges="gpt56_terra")
    with pytest.raises(FileNotFoundError, match="has no eval_rows/acva.parquet"):
        ae.mcq_rows("cellB", "acva")
    assert ae.training_history("ffA").to_dict("list") == {"step": [99, 100], "loss": [0.5, 0.4]}
    eh = ae.eval_history("ffA")
    assert eh.loc[0, "free_form.loss"] == 1.0
    assert ae.phase_summary("ffA").loc["sft", "steps_completed"] == 100


# --------------------------------------------------------------------------
# paired comparisons
# --------------------------------------------------------------------------

def test_mcq_paired_intersection_and_compare_agree(root):
    full = ae.mcq_paired(["exp/cellA", "cellB"], "arabic_exam", clean_only=False)
    assert list(full.index) == list(range(N_ROWS))                     # ordered by row_index
    assert set(full.index[~full["clean"]]) == {3, 7, 11}
    clean = ae.mcq_paired(["exp/cellA", "cellB"], "arabic_exam")
    assert len(clean) == N_ROWS - 3
    res = ae.mcq_compare(["exp/cellA", "cellB"], tasks=["arabic_exam"], pairs=[("exp/cellA", "cellB")], n_boot=200)
    t = res["tasks"]["arabic_exam"]
    assert t["intersection_rows"] == N_ROWS - 3
    pair = t["pairs"][0]["char"]
    a, b = clean["cellA.char"].to_numpy(), clean["cellB.char"].to_numpy()
    assert pair["delta"] == pytest.approx(a.mean() - b.mean(), abs=1e-4)
    mc = ae.mcnemar(a, b)
    assert (pair["only_a"], pair["only_b"]) == (mc["only_a"], mc["only_b"])
    tbl = ae.mcq_table(res)
    assert set(tbl["cell"]) == {"exp/cellA", "exp/cellB"}
    pt = ae.mcq_pairs_table(res)
    assert set(pt["norm"]) == {"char", "pmi"} and set(pt["group"]) == {"all"}


def test_mcq_paired_refuses_different_prompts(root):
    n = N_ROWS
    _mcq_dump(root / "exp" / "cellX" / "eval_rows" / "arabic_exam.parquet", [True] * n, [True] * n, [False] * n,
              prompts=[f"آخر {i}" for i in range(n)])
    with pytest.raises(ValueError, match="prompts"):
        ae.mcq_paired(["exp/cellA", "exp/cellX"], "arabic_exam")


def test_judge_compare_sign_and_freeform_paired(root):
    fp = ae.freeform_paired(["ffB", "ffA"])
    assert list(fp.columns) == ["stratum", "ffB", "ffA"] and len(fp) == 6
    j = ae.judge_compare("ffB", "ffA", n_boot=500)
    diff = (fp["ffA"] - fp["ffB"]).mean()
    assert j["score"]["delta_mean"] == pytest.approx(diff, abs=1e-4) and diff > 0   # cell − baseline
    d = ae.paired_delta(fp["ffA"], fp["ffB"], n_boot=500)
    assert d["delta"] == j["score"]["delta_mean"] and d["ci_low"] == j["score"]["ci_low"]
    assert d["mean_a"] == pytest.approx(fp["ffA"].mean())


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def test_paired_delta_matches_the_scripts_bootstrap():
    a = {"x": 3.0, "y": 1.0, "z": 5.0, "w": 2.0}
    b = {"x": 2.0, "y": 1.0, "z": 4.0, "q": 9.0}          # q unpaired
    d = ae.paired_delta(a, b, n_boot=300, seed=1)
    ref = paired_bootstrap(b, a, n_boot=300, seed=1)
    assert d["n"] == 3 and d["delta"] == ref["delta_mean"] and (d["ci_low"], d["ci_high"]) == (ref["ci_low"], ref["ci_high"])
    assert d["win_rate"] == ref["win_rate"]
    s1, s2 = pd.Series({"p": 1.0, "q": np.nan, "r": 3.0}), pd.Series({"p": 0.0, "q": 1.0, "r": 1.0})
    assert ae.paired_delta(s1, s2, n_boot=50)["n"] == 2
    arr = ae.paired_delta(np.array([True, True, False]), np.array([False, True, False]), n_boot=50)
    assert arr["delta"] == pytest.approx(1 / 3, abs=1e-4)
    with pytest.raises(ValueError):
        ae.paired_delta([1, 2], [1, 2, 3])


def test_mcnemar_exact_and_bootstrap_ci():
    r = ae.mcnemar([False] * 5 + [True] * 3, [True] * 5 + [True] * 3)
    assert (r["only_a"], r["only_b"]) == (0, 5) and r["p"] == pytest.approx(0.0625)
    assert ae.mcnemar([True, False], [True, False])["p"] is None
    ci = ae.bootstrap_ci([1, 2, 3, 4, np.nan], n_boot=200)
    assert ci["n"] == 4 and ci["value"] == 2.5 and ci["ci_low"] <= 2.5 <= ci["ci_high"]


# --------------------------------------------------------------------------
# report search
# --------------------------------------------------------------------------

def test_report_search_and_section(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.md").write_text(
        "# Report\n\n## 3.11 Letter or slot\n\nThe deficit follows the letter أ, not the slot.\n\n"
        "### Detail\n\nE_letter and E_slot are both reported.\n\n## 3.12 Other\n\nUnrelated text.\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("# Notes\n\nThe letter prior is a caveat.\n", encoding="utf-8")
    monkeypatch.setattr(docs, "repo_root", lambda: tmp_path)
    hits = ae.report_search("letter slot deficit", k=2)
    assert hits and hits[0].heading.startswith("Report › 3.11 Letter or slot") and "أ" in hits[0].text
    assert ae.report_search("letter", source="claude")[0].source == "claude"
    assert ae.report_search("zzz") == []
    sec = ae.report_section("3.11")
    assert "E_letter" in sec and "Unrelated" not in sec
    with pytest.raises(KeyError):
        ae.report_section("9.99")


# --------------------------------------------------------------------------
# show()
# --------------------------------------------------------------------------

def test_show_writes_files_and_reports_to_the_sink(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.express as px
    got = []
    display.set_sink(got.append, tmp_path, prefix="step1")
    try:
        ae.show(px.bar(x=["a", "b"], y=[1, 2]), title="Bars")
        ae.show(pd.DataFrame({"x": range(300), "y": [f"ع{i}" for i in range(300)]}), max_rows=10)
        fig = plt.figure(); plt.plot([1, 2])
        ae.show(fig, title="mpl")
        ae.show("**note**")
        with pytest.raises(TypeError):
            ae.show(42)
    finally:
        display.set_sink(None)
    kinds = [g["kind"] for g in got]
    assert kinds == ["plotly", "table", "image", "markdown"]
    assert (tmp_path / got[0]["path"]).exists() and "Bars" in got[0]["text"] and "bar[2]" in got[0]["text"]
    tbl = got[1]
    assert tbl["n_rows"] == 300 and len(tbl["rows"]) == 10 and tbl["columns"] == ["x", "y"]
    assert len(pd.read_csv(tmp_path / tbl["path"])) == 300
    assert (tmp_path / got[2]["path"]).stat().st_size > 0


def test_plotly_template_uses_the_reference_palette():
    import plotly.io as pio
    display.install_style()
    assert pio.templates.default == "arabic_eval"
    assert list(pio.templates["arabic_eval"].layout.colorway) == display.PALETTE


# --------------------------------------------------------------------------
# API page + read-only guarantee
# --------------------------------------------------------------------------

def test_api_reference_documents_every_helper():
    text = ae.api_reference()
    for fns in ae.API_GROUPS.values():
        for fn in fns:
            assert f"ae.{fn.__name__}(" in text
            assert (fn.__doc__ or "").strip(), fn.__name__
    assert "delta = a − b" in ae.api_reference(full=True) or "a − b" in ae.api_reference(full=True)


def test_helpers_never_write_under_the_experiments_root(root):
    before = _files_state(root)
    ae.catalog(full=True); ae.metrics(); ae.metrics_long()
    ae.mcq_compare(["exp/cellA", "cellB"], tasks=["arabic_exam"], pairs=[("exp/cellA", "cellB")], n_boot=50)
    ae.mcq_paired(["exp/cellA", "cellB"], "arabic_exam"); ae.freeform_rows("ffA"); ae.judge_compare("ffB", "ffA", n_boot=50)
    assert _files_state(root) == before


# --------------------------------------------------------------------------
# the report's numbers, on the real cells (skipped without them)
# --------------------------------------------------------------------------

REAL = Path(__file__).resolve().parents[1] / "outputs" / "experiments" / "qwen_native_vs_araroopat"


@pytest.mark.skipif(not (REAL / "araroopat_3phase_v5_distill_mcq4096" / "eval_rows" / "arabic_exam.parquet").exists(),
                    reason="the v5 MCQ cells are not on this machine")
def test_reproduces_the_report_on_real_cells():
    ae.set_experiments_root(None)
    A, N = "araroopat_3phase_v5_distill_mcq4096", "native_qwen3_sft_v5_distill_mcq4096"
    res = ae.mcq_compare([A, N], tasks=["arabic_exam"], pairs=[(A, N)], n_boot=100)
    t = res["tasks"]["arabic_exam"]
    assert t["intersection_rows"] == 14114
    cells = {c.rsplit("/", 1)[-1]: v for c, v in t["cells"].items()}
    assert round(cells[A]["acc_char_int"], 3) == 0.551 and round(cells[N]["acc_char_int"], 3) == 0.621
    assert round(cells[A]["acc_pmi_int"], 3) == 0.557 and round(cells[N]["acc_pmi_int"], 3) == 0.596
    p = t["pairs"][0]
    assert (p["char"]["only_a"], p["char"]["only_b"]) == (1396, 2386)
    assert (p["pmi"]["only_a"], p["pmi"]["only_b"]) == (1761, 2302)
    j = ae.judge_compare("bpe_16k_3phase_v5_distill", "araroopat_3phase_v5_distill")
    assert (j["score"]["delta_mean"], j["score"]["ci_low"], j["score"]["ci_high"]) == (0.3, 0.152, 0.452)
    m = ae.metrics(["araroopat_3phase_v5_distill", "bpe_16k_3phase_v5_distill", "native_qwen3_sft_v5_distill"])
    assert list(m["judge.gemma4_31b.mean"].round(2)) == [2.43, 2.13, 2.52]
