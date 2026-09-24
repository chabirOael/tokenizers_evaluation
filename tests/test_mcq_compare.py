"""scripts/mcq_compare.py on fixture dumps: the intersection rule, a pair with a
known difference, the McNemar counts, the Alghafa sub-config split, the join
checks, and schema-1 dumps (no ``cont_tokens``)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from arabic_eval.evaluation.eval_rows import SENTINEL_LL, EvalRowWriter, build_row_record  # noqa: E402

_spec = importlib.util.spec_from_file_location("mcq_compare", REPO / "scripts" / "mcq_compare.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

FIXED = "multiple_choice_rating_sentiment_task"      # a fixed-label Alghafa sub-config
CHOICE = "meta_ar_msa"                               # a choice-text one
N = 20
MAXLEN = 64


def _record(i: int, correct_char: bool, correct_pmi: bool, *, capped: bool = False,
            sentinel: bool = False, cfg: str = "_default", prompt: Optional[str] = None):
    gold = 0
    pc = 0 if correct_char else 1
    pp = 0 if correct_pmi else 1
    lls = [SENTINEL_LL, SENTINEL_LL] if sentinel else [-1.0, -2.0]
    sc = [-1.0, -2.0] if pc == 0 else [-2.0, -1.0]
    sp = [-0.5, -1.5] if pp == 0 else [-1.5, -0.5]
    return build_row_record(
        row_index=i, example={"question": f"q{i}", "choices": ["a", "b"], "_source_config": cfg},
        prompt=prompt if prompt is not None else f"prompt {i}", continuations=[" a", " b"],
        log_likelihoods=lls, scores_char=sc, scores_pmi=sp, unconditioned_log_likelihoods=[-0.5, -0.5],
        gold_idx=gold, pred_idx=pp, pred_idx_char=pc, pred_idx_pmi=pp,
        prompt_units=MAXLEN if capped else 10, max_length=MAXLEN,
        cont_tokens=[0, 0] if sentinel else [1, 3], cont_truncated=[False, False],
    )


def _write(exp: Path, cell: str, task: str, records: List[dict], max_length: int = MAXLEN,
           schema1: bool = False) -> None:
    path = exp / cell / "eval_rows" / f"{task}.parquet"
    if schema1:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from arabic_eval.evaluation.eval_rows import eval_row_schema
        sch = eval_row_schema({"task": task, "max_length": max_length, "schema_version": 1})
        sch = pa.schema([f for f in sch if f.name != "cont_tokens"], metadata=sch.metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{k: v for k, v in r.items() if k != "cont_tokens"} for r in records]
        pq.write_table(pa.Table.from_pylist(rows, schema=sch), path)
        return
    with EvalRowWriter(path, metadata={"task": task, "max_length": max_length, "num_fewshot": 3}) as w:
        for r in records:
            w.write(r)


@pytest.fixture()
def exp(tmp_path: Path) -> Path:
    """Three cells over 20 rows of 'acva' and 'alghafa'.

    A: right on every row.  B: right on even rows only.  C: right on every row
    but capped on rows 0-1 (row 1 as an all-sentinel row); B capped on row 2.
    So the intersection is rows 3..19 (17 rows) — 9 odd rows where only A is
    right, 0 where only B is right.
    """
    for task in ("acva", "alghafa"):
        cfgs = [FIXED if i % 2 == 0 else CHOICE for i in range(N)] if task == "alghafa" else ["_default"] * N
        _write(tmp_path, "A", task, [_record(i, True, True, cfg=cfgs[i]) for i in range(N)])
        _write(tmp_path, "B", task, [_record(i, i % 2 == 0, i % 2 == 0, capped=(i == 2), cfg=cfgs[i])
                                     for i in range(N)])
        _write(tmp_path, "C", task, [_record(i, True, True, capped=i in (0, 1), sentinel=(i == 1), cfg=cfgs[i])
                                     for i in range(N)])
    (tmp_path / "A" / "all_metrics.json").write_text(json.dumps(
        {"mei": {"acva": {"mei": 1.25, "status": "ok", "inputs": {"accuracy_source": "accuracy_pmi"}}}}))
    return tmp_path


def test_intersection_drops_rows_capped_in_any_cell(exp):
    res = mc.compare(exp, ["A", "B", "C"], ["acva"], [("A", "B")], n_boot=200)
    t = res["tasks"]["acva"]
    assert t["rows"] == N and t["intersection_rows"] == N - 3
    assert t["removed_by_intersection"]["C"] == {"capped": 2, "capped_only_here": 2}
    assert t["removed_by_intersection"]["B"] == {"capped": 1, "capped_only_here": 1}
    b = t["cells"]["B"]
    assert b["acc_char_all"] == pytest.approx(0.5)
    # rows 3..19: 8 even rows right of 17
    assert b["acc_char_int"] == pytest.approx(8 / 17, abs=1e-6)
    c = t["cells"]["C"]
    assert c["hit_cap_share"] == pytest.approx(2 / N) and c["all_sentinel_share"] == pytest.approx(1 / N)
    assert t["cells"]["A"]["mean_cont_tokens"] == pytest.approx(2.0)
    assert t["cells"]["A"]["mei"]["mei"] == 1.25 and t["cells"]["B"]["mei"] is None


def test_pair_difference_and_mcnemar(exp):
    res = mc.compare(exp, ["A", "B", "C"], ["acva"], [("A", "B")], n_boot=500)
    p = res["tasks"]["acva"]["pairs"][0]
    for norm in ("char", "pmi"):
        s = p[norm]
        assert s["n"] == 17
        assert s["only_a"] == 9 and s["only_b"] == 0
        assert s["delta"] == pytest.approx(9 / 17, abs=1e-4)
        assert s["ci_low"] <= s["delta"] <= s["ci_high"]
        assert s["mcnemar_p"] == pytest.approx(2 * 0.5 ** 9, abs=1e-6)


def test_alghafa_subconfig_split(exp):
    res = mc.compare(exp, ["A", "B", "C"], ["alghafa"], [("A", "B")], n_boot=200)
    t = res["tasks"]["alghafa"]
    b = t["cells"]["B"]
    # B is right on even rows = every FIXED row and no CHOICE row.
    assert b["groups"]["fixed_label"]["acc_char_int"] == 1.0
    assert b["groups"]["choice_text"]["acc_char_int"] == 0.0
    assert b["per_subconfig"][FIXED]["rows"] == 10 and b["per_subconfig"][CHOICE]["rows"] == 10
    # Intersection rows 3..19: FIXED = 4..18 even (8), CHOICE = 3..19 odd (9).
    assert b["per_subconfig"][FIXED]["rows_int"] == 8 and b["per_subconfig"][CHOICE]["rows_int"] == 9
    p = t["pairs"][0]
    assert p["fixed_label_pmi"]["only_a"] == 0 and p["choice_text_pmi"]["only_a"] == 9
    assert "fixed-label" in mc.to_markdown(res)


def test_mcnemar_exact_values():
    assert mc.mcnemar_exact_p(0, 0) is None
    assert mc.mcnemar_exact_p(5, 0) == pytest.approx(0.0625)
    assert mc.mcnemar_exact_p(1, 1) == 1.0
    assert mc.mcnemar_exact_p(3, 7) == mc.mcnemar_exact_p(7, 3)


def test_prompt_mismatch_refuses_the_join(tmp_path):
    _write(tmp_path, "A", "acva", [_record(i, True, True) for i in range(4)])
    _write(tmp_path, "B", "acva", [_record(i, True, True, prompt="other" if i == 2 else None) for i in range(4)])
    with pytest.raises(SystemExit, match="prompts"):
        mc.compare(tmp_path, ["A", "B"], ["acva"])


def test_row_set_mismatch_refuses_the_join(tmp_path):
    _write(tmp_path, "A", "acva", [_record(i, True, True) for i in range(4)])
    _write(tmp_path, "B", "acva", [_record(i, True, True) for i in range(3)])
    with pytest.raises(SystemExit, match="row_index"):
        mc.compare(tmp_path, ["A", "B"], ["acva"])


def test_schema1_dump_and_max_length_flag(tmp_path):
    _write(tmp_path, "new", "acva", [_record(i, True, True) for i in range(6)], max_length=4096)
    _write(tmp_path, "new2", "acva", [_record(i, i < 3, True) for i in range(6)], max_length=4096)
    _write(tmp_path, "old", "acva", [_record(i, True, False, capped=(i == 5)) for i in range(6)],
           max_length=1024, schema1=True)
    res = mc.compare(tmp_path, ["new", "new2", "old"], ["acva"], [("new", "old")], n_boot=100)
    t = res["tasks"]["acva"]
    assert t["majority_max_length"] == 4096
    assert t["cells"]["old"]["flag_max_length"] and not t["cells"]["new"]["flag_max_length"]
    assert t["cells"]["old"]["mean_cont_tokens"] is None
    assert t["intersection_rows"] == 5
    assert "old †" in mc.to_markdown(res)


def test_cli_writes_md_and_json(exp):
    rc = mc.main(["--experiment", str(exp), "--cells", "A", "B", "C", "--tasks", "acva",
                  "--pairs", "A:B", "--n-boot", "100", "--name", "t"])
    assert rc == 0
    assert (exp / "_mcq_compare" / "t.md").exists()
    data = json.loads((exp / "_mcq_compare" / "t.json").read_text())
    assert data["tasks"]["acva"]["intersection_rows"] == 17


def test_missing_dump_is_reported_not_fatal(exp):
    res = mc.compare(exp, ["A", "B"], ["acva", "arabic_exam"])
    assert res["tasks"]["arabic_exam"] == {"missing": ["A", "B"]}


def _label_record(i: int, conts, gold: int, pred_char: int, pred_pmi: int, cfg: str):
    n = len(conts)
    sc = [-2.0] * n; sc[pred_char] = -1.0
    sp = [-2.0] * n; sp[pred_pmi] = -1.0
    return build_row_record(
        row_index=i, example={"question": f"q{i}", "choices": [c.strip() for c in conts], "_source_config": cfg},
        prompt=f"prompt {i}", continuations=conts, log_likelihoods=[-1.0] * n, scores_char=sc, scores_pmi=sp,
        unconditioned_log_likelihoods=[-0.5] * n, gold_idx=gold, pred_idx=pred_pmi, pred_idx_char=pred_char,
        pred_idx_pmi=pred_pmi, prompt_units=10, max_length=MAXLEN, cont_tokens=[1] * n,
        cont_truncated=[False] * n)


def test_label_collapse_is_read_by_label_text_not_position(tmp_path):
    """Alghafa shuffles the choice order per row: a cell that always picks the
    same *label* spreads evenly over positions. The diagnostic must see it."""
    rows = []
    for i in range(20):
        conts = [" ايجابي", " سلبي"] if i % 2 == 0 else [" سلبي", " ايجابي"]
        gold = i % 4 // 2                   # half the golds each label
        pos_pos = conts.index(" ايجابي")
        rows.append(_label_record(i, conts, gold, pred_char=gold, pred_pmi=pos_pos, cfg=FIXED))
    for i in range(20, 24):                 # a choice-text sub-config: not a fixed-label group
        conts = [f" جواب{i}{k}" for k in range(4)] + [f" بديل{i}"]
        rows.append(_label_record(i, conts, 0, 0, 0, cfg=CHOICE))
    _write(tmp_path, "X", "alghafa", rows)
    res = mc.compare(tmp_path, ["X"], ["alghafa"])
    ls = res["tasks"]["alghafa"]["cells"]["X"]["label_shares"]
    assert set(ls) == {FIXED}
    assert ls[FIXED]["top_pmi"] == {"label": "ايجابي", "share": 1.0,
                                    "gold_share": pytest.approx(0.5), "collapse": True}
    assert ls[FIXED]["top_char"]["collapse"] is False
    assert "⚠" in mc.to_markdown(res)


def test_whole_task_fixed_labels_is_one_group(exp):
    res = mc.compare(exp, ["A", "B"], ["acva"])
    assert set(res["tasks"]["acva"]["cells"]["B"]["label_shares"]) == {"_all"}
