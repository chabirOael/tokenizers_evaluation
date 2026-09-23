#!/usr/bin/env python
"""Report tables of the distillation experiment (docs/report.md §3.8, P3–P4 of the 2026-09-23 brief).

For every cell named on the command line (folders of ``--experiment``):

* the consolidated free-form row — judge mean ± SE, histogram, the judge's
  off-topic / repetition / truncated flag rates, chrF, BERTScore-F1, loop / EOS / cap
  rates, mean characters, the raw-text loss and the answer NLL per character of the
  cell's held-out diagnostic (``diag_heldout_rawtext_v1*.json``; the eval-only
  reference cells read their source cell's), and the best eval-mixture loss per
  category (``training.sft.eval_history``) when the cell trained;
* loops by period bucket (1 / 2–3 / 4–10 / > 10 words) and by rule, the share of
  judge-score-1 rows that are loops, loop rate per reference-length tercile;
* answer length per tercile (mean / median characters), the share ≤ 40 characters;
* the stop curve (step | total | extractive | mcq | free_form) and the chosen step.

    .venv/bin/python scripts/distill/report_tables.py --cells araroopat_3phase_v5_distill bpe_16k_3phase_v5_distill ...
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP = REPO_ROOT / "outputs/experiments/qwen_native_vs_araroopat"
# eval-only reference cells read the held-out diagnostic of the cell whose checkpoint they re-evaluate
DIAG_SOURCE = {"native_qwen3_base_m2_c2400": ("native_qwen3_base", "diag_heldout_rawtext_v1_base.json"),
               "native_qwen3_sft_m2_c2400": ("native_qwen3_sft", "diag_heldout_rawtext_v1.json")}
STRATA = ("short", "medium", "long")


def _pq(path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _diag(exp: Path, cell: str) -> Optional[Dict[str, Any]]:
    src, name = DIAG_SOURCE.get(cell, (cell, "diag_heldout_rawtext_v1.json"))
    p = exp / src / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def bucket(period: Optional[int]) -> str:
    if period is None:
        return "?"
    return "1" if period == 1 else "2-3" if period <= 3 else "4-10" if period <= 10 else ">10"


def cell_tables(exp: Path, cell: str, judge: str) -> Dict[str, Any]:
    d = exp / cell
    am = json.loads((d / "all_metrics.json").read_text(encoding="utf-8"))
    ff = am["downstream"]["freeform_cidar"]
    j = (ff.get("judge") or {}).get(judge) or {}
    rows = _pq(d / "eval_rows" / "freeform_cidar.parquet")
    jp = d / "freeform_judge" / f"{judge}.parquet"
    scores = {r["id"]: r["score"] for r in _pq(jp)} if jp.exists() else {}
    diag = _diag(exp, cell)
    out: Dict[str, Any] = {
        "cell": cell, "kind": am.get("kind") or ff.get("kind"),
        "judge_mean": j.get("score_mean"), "judge_se": j.get("score_se"), "hist": j.get("score_hist"),
        "flags": {k: (j.get("flag_rates") or {}).get(k) for k in ("off_topic", "repetition", "truncated")},
        "chrf": ff.get("chrf"), "bertscore_f1": ff.get("bertscore_f1"),
        "loop": ff.get("loop_stop_rate"), "eos": ff.get("eos_rate"), "cap": ff.get("hit_cap_rate"),
        "mean_chars": ff.get("mean_gen_chars"),
        "rawtext_nats_per_char": (diag or {}).get("mix", {}).get("loss_per_char"),
        "answer_nll_per_char": (diag or {}).get("heldout_answers", {}).get("nll_per_answer_char"),
    }
    sft = (am.get("training") or {}).get("sft") or {}
    hist = sft.get("eval_history") or []
    if hist:
        # the early stopper's restored step (it only counts improvements above min_delta), not the raw minimum
        best = next((h for h in hist if h["step"] == sft.get("best_eval_step")), min(hist, key=lambda h: h["loss"]))
        out["stop_curve"] = [{"step": h["step"], "total": h["loss"],
                              **{c: (h.get("per_category") or {}).get(c, {}).get("loss") for c in ("extractive", "mcq", "free_form")}}
                             for h in hist]
        out["best"] = {"step": best["step"], "total": best["loss"],
                       **{c: (best.get("per_category") or {}).get(c, {}).get("loss") for c in ("extractive", "mcq", "free_form")}}
        out["steps_completed"] = sft.get("steps_completed")
        out["early_stopped"] = sft.get("early_stopped")
        out["wall_time_sec"] = sft.get("wall_time_sec")
    loops = [r for r in rows if r["stop_reason"] == "loop"]
    out["loop_buckets"] = {b: sum(1 for r in loops if bucket(r.get("loop_period")) == b) for b in ("1", "2-3", "4-10", ">10")}
    out["loop_rules"] = {k: sum(1 for r in loops if r.get("loop_rule") == k) for k in ("period", "char")}
    ones = [r for r in rows if scores.get(r["id"]) == 1]
    out["score1_loops"] = [sum(1 for r in ones if r["stop_reason"] == "loop"), len(ones)]
    out["loop_by_tercile"] = {s: round(sum(1 for r in rows if r["stratum"] == s and r["stop_reason"] == "loop")
                                       / max(1, sum(1 for r in rows if r["stratum"] == s)), 4) for s in STRATA}
    out["len_by_tercile"] = {s: {"mean": round(statistics.fmean([r["gen_chars"] for r in rows if r["stratum"] == s]), 1),
                                 "median": statistics.median([r["gen_chars"] for r in rows if r["stratum"] == s])}
                             for s in STRATA if any(r["stratum"] == s for r in rows)}
    out["ref_len_by_tercile"] = {s: {"mean": round(statistics.fmean([r["ref_chars"] for r in rows if r["stratum"] == s]), 1),
                                     "median": statistics.median([r["ref_chars"] for r in rows if r["stratum"] == s])}
                                 for s in STRATA if any(r["stratum"] == s for r in rows)}
    out["le40_share"] = round(sum(1 for r in rows if r["gen_chars"] <= 40) / len(rows), 4)
    out["judge_by_tercile"] = j.get("score_by_stratum")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default=str(EXP))
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--judge", default="gemma4_31b")
    ap.add_argument("--out", default=None, help="JSON output")
    args = ap.parse_args(argv)
    exp = Path(args.experiment)
    res = [cell_tables(exp, c, args.judge) for c in args.cells]
    f = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    pct = lambda v: "—" if v is None else f"{100 * v:.1f} %"  # noqa: E731
    print("| cell | judge ± SE | hist 1/2/3/4/5 | off-topic / repetition / truncated | chrF / BERTScore | loop / EOS / cap | mean chars | raw-text nats/char | answer NLL nats/char | best eval-mix (total / ext / mcq / ff) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in res:
        h = r["hist"] or {}
        b = r.get("best")
        print(f"| {r['cell']} | {f(r['judge_mean'])} ± {f(r['judge_se'])} | {'/'.join(str(h.get(str(k), '—')) for k in range(1, 6))} | "
              f"{pct(r['flags']['off_topic'])} / {pct(r['flags']['repetition'])} / {pct(r['flags']['truncated'])} | "
              f"{f(r['chrf'])} / {f(r['bertscore_f1'])} | {pct(r['loop'])} / {pct(r['eos'])} / {pct(r['cap'])} | {f(r['mean_chars'], 0)} | "
              f"{f(r['rawtext_nats_per_char'], 3)} | {f(r['answer_nll_per_char'], 3)} | "
              + (f"{b['total']:.4f} / {b['extractive']:.4f} / {b['mcq']:.4f} / {b['free_form']:.4f} @ {b['step']}" if b else "—") + " |")
    print()
    for r in res:
        print(f"{r['cell']}: loops by period {r['loop_buckets']} rules {r['loop_rules']}; score-1 loops {r['score1_loops'][0]}/{r['score1_loops'][1]}; "
              f"loop by tercile {r['loop_by_tercile']}; ≤40 chars {pct(r['le40_share'])}; judge by tercile {r['judge_by_tercile']}")
        print(f"   length by tercile {r['len_by_tercile']} (references {r['ref_len_by_tercile']})")
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
