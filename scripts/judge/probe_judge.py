#!/usr/bin/env python
"""Probe a judge candidate before it judges anything that counts (docs/report.md §3.14).

    # 1. two identical probe experiments: the first N generations of a real cell (copied, the cell is
    #    untouched) + a 4-row fixture cell whose verdicts are known in advance
    .venv/bin/python scripts/judge/probe_judge.py prepare --judge configs/judges/falcon_h1_34b_local.yaml
    # 2. judge each folder in its own process (cold start measured twice; HF_HUB_OFFLINE=1, nothing downloads)
    HF_HUB_OFFLINE=1 scripts/judge/run_judge.sh --experiment outputs/judge_probes/<date>/<judge>/run1 \
        --judge configs/judges/falcon_h1_34b_local.yaml --no-report
    HF_HUB_OFFLINE=1 scripts/judge/run_judge.sh --experiment outputs/judge_probes/<date>/<judge>/run2 ...
    # 3. parse rate, the four fixture verdicts, run1 == run2 row for row → probe.json next to the runs
    .venv/bin/python scripts/judge/probe_judge.py check --judge configs/judges/falcon_h1_34b_local.yaml

The four fixtures reconstruct the "4/4 probe verdicts" of the 2026-09-18 judge set-up (not committed then):
the reference itself as the answer (score ≥ 4), an empty answer (score 1, flag ``empty``), a repetition loop
(score ≤ 2, flag ``repetition``) and an English answer (score ≤ 2, flag ``wrong_language``), all on the
held-out prompt of the source cell's first row. A candidate passes when ≥ 98 % of its verdicts parse, the
four fixtures land on the expected score, and the second run's score equals the first's on ≥ 95 % of rows
(the S3 repeat rule of docs/report.md §3.14); full-row identity (sub-scores, flags, raw text) is reported
beside it. The probe never touches a real cell.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from arabic_eval.judge.freeform_judge import (  # noqa: E402
    GENERATIONS_FILE, JudgeConfig, _read_parquet, judge_file,
)
from arabic_eval.utils.io import save_json, write_report_table  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CELL = REPO / "outputs/experiments/qwen_native_vs_araroopat/araroopat_3phase_v5_distill"
FIXTURE_CELL = "fixture4"
PASS_PARSE = 0.98
PASS_REPEAT = 0.95

LOOP_UNIT = "الماء ضروري للحياة و"
FIXTURES = {
    # id suffix: (generation builder, expectation)
    "reference": (lambda ref: ref, {"min_score": 4}),
    "empty": (lambda ref: "", {"max_score": 1, "flag": "empty"}),
    "loop": (lambda ref: (LOOP_UNIT + " ") * 40, {"max_score": 2, "flag": "repetition"}),
    "english": (lambda ref: "This answer is written in English and does not address the request in Arabic at all. "
                            "It talks about the weather and about football instead.", {"max_score": 2, "flag": "wrong_language"}),
}


def probe_root(base: Path, judge: str, date: str) -> Path:
    return base / date / judge


def prepare(args: argparse.Namespace) -> int:
    cfg = JudgeConfig.from_yaml(args.judge)
    root = probe_root(Path(args.out), cfg.name, args.date)
    src = Path(args.cell)
    rows, meta = _read_parquet(src / GENERATIONS_FILE)
    fields = list(rows[0].keys())
    first = rows[0]
    fx_rows = []
    for key, (build, _exp) in FIXTURES.items():
        r = dict(first)
        r["id"] = f"fx_{key}"
        r["generation"] = build(first["reference"])
        r["generation_raw"] = r["generation"]
        fx_rows.append(r)
    for run in ("run1", "run2"):
        d = root / run
        if d.exists():
            if not args.force:
                print(f"{d} exists — pass --force to rebuild it")
                return 1
            shutil.rmtree(d)
        cell = d / src.name
        (cell / GENERATIONS_FILE).parent.mkdir(parents=True)
        write_report_table(cell / GENERATIONS_FILE, rows[: args.n], fields, metadata={**meta, "probe_source": str(src), "num_rows": args.n})
        save_json({"probe": True, "source_cell": str(src)}, cell / "all_metrics.json")
        fx = d / FIXTURE_CELL
        (fx / GENERATIONS_FILE).parent.mkdir(parents=True)
        write_report_table(fx / GENERATIONS_FILE, fx_rows, fields, metadata={**meta, "probe_fixture": True, "num_rows": len(fx_rows)})
        save_json({"probe": True, "fixture": True}, fx / "all_metrics.json")
    print(f"prepared {root}/run1 and run2: {src.name} first {args.n} rows + {FIXTURE_CELL} ({len(fx_rows)} rows)")
    return 0


def _rows(cell: Path, judge: str):
    f = judge_file(cell, judge)
    if not f.exists():
        return None, None
    return _read_parquet(f)


def check(args: argparse.Namespace) -> int:
    cfg = JudgeConfig.from_yaml(args.judge)
    root = probe_root(Path(args.out), cfg.name, args.date)
    src_name = Path(args.cell).name
    out = {"judge": cfg.to_json(), "root": str(root), "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": {}}
    by_run = {}
    for run in ("run1", "run2"):
        rec = {}
        allrows = []
        for cell in (src_name, FIXTURE_CELL):
            rows, meta = _rows(root / run / cell, cfg.name)
            if rows is None:
                rec[cell] = None
                continue
            rec[cell] = {"n": len(rows), "seconds": meta.get("seconds"), "parse_ok": sum(1 for r in rows if r["parse_ok"])}
            allrows += [(cell, r) for r in rows]
        n = len(allrows)
        rec["parse_rate"] = round(sum(1 for _, r in allrows if r["parse_ok"]) / n, 4) if n else None
        by_run[run] = {(c, r["id"]): r for c, r in allrows}
        out["runs"][run] = rec
    # fixtures (run1)
    fx = {}
    for key, (_b, exp) in FIXTURES.items():
        r = by_run["run1"].get((FIXTURE_CELL, f"fx_{key}"))
        if r is None:
            fx[key] = {"ok": False, "why": "missing"}
            continue
        flags = [f for f in (r.get("flags") or "").split(",") if f]
        ok = r["parse_ok"] and r["score"] is not None
        if ok and "min_score" in exp:
            ok = r["score"] >= exp["min_score"]
        if ok and "max_score" in exp:
            ok = r["score"] <= exp["max_score"]
        flag_ok = ("flag" not in exp) or (exp["flag"] in flags)
        fx[key] = {"ok": bool(ok), "flag_ok": flag_ok, "score": r["score"], "flags": flags,
                   "expected": exp, "rationale": r.get("rationale")}
    out["fixtures"] = fx
    out["fixtures_score_ok"] = sum(1 for v in fx.values() if v["ok"])
    out["fixtures_flag_ok"] = sum(1 for v in fx.values() if v.get("flag_ok"))
    # exact repeat
    keys = sorted(set(by_run["run1"]) & set(by_run["run2"]))
    fields = ("score", "correctness", "fluency", "instruction_following", "flags", "raw")
    same_score = sum(1 for k in keys if by_run["run1"][k]["score"] == by_run["run2"][k]["score"])
    same_all = sum(1 for k in keys if all(by_run["run1"][k][f] == by_run["run2"][k][f] for f in fields))
    out["repeat"] = {"n": len(keys), "score_exact": round(same_score / len(keys), 4) if keys else None,
                     "row_exact": round(same_all / len(keys), 4) if keys else None,
                     "differing_ids": [f"{c}/{i}" for c, i in keys
                                       if any(by_run["run1"][(c, i)][f] != by_run["run2"][(c, i)][f] for f in fields)]}
    p1 = out["runs"]["run1"].get("parse_rate")
    # S3 (docs/report.md §3.14): the greedy repeat must agree exactly on ≥ 95 % of items — on the score.
    # row_exact (score, sub-scores, flags and raw text all identical) is reported beside it.
    out["pass"] = {"parse": p1 is not None and p1 >= PASS_PARSE,
                   "fixtures": out["fixtures_score_ok"] == len(FIXTURES),
                   "repeat_score_ge_0.95": (out["repeat"]["score_exact"] or 0) >= PASS_REPEAT,
                   "repeat_row_exact_1.0": out["repeat"]["row_exact"] == 1.0}
    if args.timing:
        out["timing"] = json.loads(Path(args.timing).read_text())
    save_json(out, root / "probe.json")
    print(json.dumps({k: out[k] for k in ("runs", "fixtures_score_ok", "fixtures_flag_ok", "repeat", "pass")},
                     ensure_ascii=False, indent=1))
    for k, v in fx.items():
        print(f"  fixture {k}: score {v.get('score')} flags {v.get('flags')} → {'ok' if v['ok'] else 'NOT ok'}"
              f"{'' if v.get('flag_ok') else ' (expected flag missing)'}")
    print(f"→ {root / 'probe.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prepare", "check"])
    ap.add_argument("--judge", required=True, help="judge YAML")
    ap.add_argument("--cell", default=str(DEFAULT_CELL), help="the real cell whose first N generations are probed")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=str(REPO / "outputs/judge_probes"))
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timing", default=None, help="check: a JSON of wall / cold-start seconds to fold into probe.json")
    args = ap.parse_args()
    return prepare(args) if args.action == "prepare" else check(args)


if __name__ == "__main__":
    sys.exit(main())
