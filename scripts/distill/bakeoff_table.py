#!/usr/bin/env python
"""Teacher bake-off table and the choice (P1.3 / P1.7a of the 2026-09-23 distillation brief).

Per candidate pseudo-cell of ``outputs/experiments/teacher_bakeoff/``: judge mean ± SE,
paired Δ vs ``native_qwen3_base_dev`` [95 % CI] (10 000 resamples, seed 0 —
``paired_compare.compare``), the kept rate under the P1.6 filters
(``postprocess.drop_reason`` on the 250 dev answers), loop / cap / EOS rates, mean
characters, tokens per second, the determinism exact-match share (the first 200
prompts generated a second time in a separate process) and the projected wall of the
full generation (``--n-full`` prompts × mean answer tokens ÷ tokens per second, plus
the load).

The decision, applied mechanically:
  * **selection** — among candidates with kept ≥ 80 %, the highest judge mean; ties
    within one SE (of the leader) break by higher kept rate, then mean answer length
    closer to the references' mean, then the smaller model;
  * **throughput** — if the winner projects above 6 h and the runner-up is within one
    SE, the runner-up;
  * **stop point** — when no candidate's judge mean minus one SE exceeds
    ``native_qwen3_base_dev``'s mean, the decision is ``stop``.

    .venv/bin/python scripts/distill/bakeoff_table.py [--judge gemma4_31b]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "judge"))

from paired_compare import compare  # noqa: E402

from arabic_eval.distill.postprocess import drop_reason  # noqa: E402
from arabic_eval.distill.teacher import load_candidates, read_jsonl  # noqa: E402

EXP = REPO_ROOT / "outputs/experiments/teacher_bakeoff"
RAW = REPO_ROOT / "outputs/data_cache/distill/bakeoff"
REFERENCE = "native_qwen3_base_dev"
KEPT_MIN = 0.80
MAX_HOURS = 6.0
N_FULL = 42100


def _judge(cell: Path, judge: str) -> Dict[str, Any]:
    am = json.loads((cell / "all_metrics.json").read_text(encoding="utf-8"))
    d = am["downstream"]["freeform_cidar"]
    j = (d.get("judge") or {}).get(judge) or {}
    return {"metrics": d, "judge": j}


def candidate_row(c, judge: str, prompts: Dict[str, Dict[str, Any]], n_full: int) -> Optional[Dict[str, Any]]:
    cell = EXP / c.slug
    meta_p = RAW / c.slug / "generation_meta.json"
    if not (cell / "all_metrics.json").exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        return {"slug": c.slug, "repo": c.repo, "loaded": bool(meta.get("loaded")), "load_errors": meta.get("load_errors")}
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    raw = read_jsonl(RAW / c.slug / "teacher_raw.jsonl")
    rep_p = RAW / c.slug / "teacher_raw_repeat.jsonl"
    det = None
    if rep_p.exists():
        first = {r["id"]: r["text"] for r in raw}
        rep = read_jsonl(rep_p)
        det = sum(1 for r in rep if first.get(r["id"]) == r["text"]) / len(rep) if rep else None
    kept = sum(1 for r in raw if drop_reason(r["text"], r["finish_reason"], prompts[r["id"]]["instruction"]) is None) / len(raw)
    info = _judge(cell, judge)
    m, j = info["metrics"], info["judge"]
    vs = compare(EXP, REFERENCE, c.slug, judge) if j else None
    mean_tokens = sum(r["n_tokens"] for r in raw) / len(raw)
    tps = meta.get("tok_per_sec")
    proj_h = (n_full * mean_tokens / tps + (meta.get("load_wall_sec") or 0)) / 3600 if tps else None
    return {
        "slug": c.slug, "repo": c.repo, "revision": meta.get("revision"), "loaded": True,
        "loaded_from": meta.get("loaded_from"), "dtype": meta.get("dtype"), "quantization": meta.get("quantization"),
        "load_attempt": meta.get("load_attempt"), "system_role": meta.get("system_role"),
        "cap": (meta.get("sampling") or {}).get("max_tokens"), "load_wall_sec": meta.get("load_wall_sec"),
        "gen_wall_sec": meta.get("gen_wall_sec"), "tok_per_sec": tps, "params_b": c.params_b,
        "determinism_exact": round(det, 4) if det is not None else None,
        "judge_mean": j.get("score_mean"), "judge_se": j.get("score_se"), "judge_hist": j.get("score_hist"),
        "delta_vs_ref": vs["score"] if vs else None,
        "kept": round(kept, 4), "loop": m.get("loop_stop_rate"), "cap_rate": m.get("hit_cap_rate"),
        "eos": m.get("eos_rate"), "mean_chars": m.get("mean_gen_chars"), "mean_ref_chars": m.get("mean_ref_chars"),
        "latin": m.get("latin_rate"), "char_truncated": m.get("char_truncated_rate"), "chrf": m.get("chrf"),
        "bertscore_f1": m.get("bertscore_f1"), "mean_answer_tokens": round(mean_tokens, 1),
        "projected_full_hours": round(proj_h, 2) if proj_h is not None else None,
    }


def decide(rows: List[Dict[str, Any]], ref_mean: float) -> Dict[str, Any]:
    scored = [r for r in rows if r.get("judge_mean") is not None]
    beats = [r["slug"] for r in scored if r["judge_mean"] - (r["judge_se"] or 0) > ref_mean]
    eligible = [r for r in scored if r["kept"] >= KEPT_MIN]
    out: Dict[str, Any] = {"reference_mean": ref_mean, "beat_reference_by_one_se": beats,
                           "eligible_kept_ge_80": [r["slug"] for r in eligible]}
    if not eligible:
        out.update(decision="no_eligible_candidate", winner=None)
        return out
    leader = max(eligible, key=lambda r: r["judge_mean"])
    ties = [r for r in eligible if leader["judge_mean"] - r["judge_mean"] <= (leader["judge_se"] or 0)]
    ties.sort(key=lambda r: (-r["kept"], abs((r["mean_chars"] or 0) - (r["mean_ref_chars"] or 0)), r["params_b"] or 1e9))
    winner = ties[0]
    order = [winner] + sorted([r for r in eligible if r is not winner], key=lambda r: -r["judge_mean"])
    runner = order[1] if len(order) > 1 else None
    out.update(leader=leader["slug"], tie_group=[r["slug"] for r in ties], winner=winner["slug"],
               runner_up=runner["slug"] if runner else None)
    if winner["projected_full_hours"] and winner["projected_full_hours"] > MAX_HOURS and runner is not None \
            and winner["judge_mean"] - runner["judge_mean"] <= (winner["judge_se"] or 0):
        out.update(throughput_rule="applied", winner=runner["slug"])
    else:
        out["throughput_rule"] = "not applied"
    win = next(r for r in rows if r["slug"] == out["winner"])
    out["decision"] = "proceed" if win["judge_mean"] - (win["judge_se"] or 0) > ref_mean else "stop"
    if out["decision"] == "stop" and beats:
        out["note"] = "the chosen teacher does not beat the reference by one SE although another candidate does"
    if not beats:
        out["decision"] = "stop"
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", default="gemma4_31b")
    ap.add_argument("--candidates", default="configs/distill/teacher_candidates.yaml")
    ap.add_argument("--n-full", type=int, default=N_FULL)
    args = ap.parse_args(argv)

    prompts = {p["id"]: p for p in read_jsonl(RAW / "prompts_dev.jsonl")}
    _d, cands = load_candidates(REPO_ROOT / args.candidates)
    rows = [candidate_row(c, args.judge, prompts, args.n_full) for c in cands]
    ref = _judge(EXP / REFERENCE, args.judge)
    ref_row = {"slug": REFERENCE, "judge_mean": ref["judge"].get("score_mean"), "judge_se": ref["judge"].get("score_se"),
               "judge_hist": ref["judge"].get("score_hist"), "loop": ref["metrics"].get("loop_stop_rate"),
               "cap_rate": ref["metrics"].get("hit_cap_rate"), "eos": ref["metrics"].get("eos_rate"),
               "mean_chars": ref["metrics"].get("mean_gen_chars"), "chrf": ref["metrics"].get("chrf"),
               "latin": ref["metrics"].get("latin_rate"), "bertscore_f1": ref["metrics"].get("bertscore_f1"),
               "token_cap": ref["metrics"].get("token_cap")}
    dec = decide(rows, ref_row["judge_mean"])
    f = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    print("| candidate | judge ± SE | Δ vs native_qwen3_base_dev [CI] | kept | loop | cap | EOS | latin | mean chars | tok/s | determinism | projected h |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("judge_mean") is None:
            print(f"| {r['slug']} | not scored (loaded: {r.get('loaded')}) |")
            continue
        d = r["delta_vs_ref"] or {}
        print(f"| {r['slug']} | {f(r['judge_mean'])} ± {f(r['judge_se'])} | {d.get('delta_mean', 0):+.2f} "
              f"[{d.get('ci_low', 0):+.2f}, {d.get('ci_high', 0):+.2f}] | {100 * r['kept']:.1f} % | {100 * r['loop']:.1f} % | "
              f"{100 * r['cap_rate']:.1f} % | {100 * r['eos']:.1f} % | {100 * r['latin']:.1f} % | {r['mean_chars']:.0f} | "
              f"{r['tok_per_sec']:.0f} | {100 * r['determinism_exact']:.1f} % | {f(r['projected_full_hours'])} |")
    print(f"| {REFERENCE} (reference) | {f(ref_row['judge_mean'])} ± {f(ref_row['judge_se'])} | — | — | "
          f"{100 * ref_row['loop']:.1f} % | {100 * ref_row['cap_rate']:.1f} % | {100 * ref_row['eos']:.1f} % | "
          f"{100 * ref_row['latin']:.1f} % | {ref_row['mean_chars']:.0f} | — | — | — |")
    print(json.dumps(dec, indent=2))
    out = EXP / "bakeoff_table.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"judge": args.judge, "candidates": rows, "reference": ref_row, "decision": dec,
                   "rules": {"kept_min": KEPT_MIN, "max_hours": MAX_HOURS, "n_full": args.n_full}}, fh, ensure_ascii=False, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
