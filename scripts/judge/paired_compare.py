#!/usr/bin/env python
"""Paired judge comparison between two cells (of one experiment or of two).

Every variant answers the *same* 250 held-out prompts, so the comparison that
means something is paired: the mean of (B − A) over the shared prompt ids with
a percentile bootstrap CI, and how often B wins / ties / loses. An unpaired
difference of two means throws away the pairing and widens the interval for
nothing.

    .venv/bin/python scripts/judge/paired_compare.py \\
        --experiment outputs/experiments/qwen_native_vs_araroopat \\
        --judge gemma4_31b --cells araroopat_3phase_v3_ffstop araroopat_3phase_v3

``--cells A B`` reads ``<experiment>/<cell>/freeform_judge/<judge>.parquet``
and reports B − A (the first cell is the baseline). ``--sub-scores`` adds the
same bootstrap for correctness / fluency / instruction_following.

Two experiments (2026-09-23, for pairing a decoding-ablation cell with its
greedy twin): give ``--experiment`` twice — the baseline is read from the
first, the cell from the second. Without ``--experiment`` the two cells are
directory paths.

    .venv/bin/python scripts/judge/paired_compare.py \
        --experiment outputs/experiments/qwen_native_vs_araroopat \
        --experiment outputs/experiments/qwen_decoding_ablation \
        --cells araroopat_3phase_v3_ffstop araroopat_3phase_v3_ffstop_rp12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arabic_eval.judge.freeform_judge import paired_bootstrap  # noqa: E402

SUB_SCORES = ("correctness", "fluency", "instruction_following")


def load_scores(path: Path, field: str = "score") -> Dict[str, float]:
    """``{prompt id: score}`` from a judge parquet, skipping unparsed verdicts."""
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["id", field, "parse_ok"])
    out: Dict[str, float] = {}
    for row in table.to_pylist():
        if row.get(field) is None or row.get("parse_ok") is False:
            continue
        out[str(row["id"])] = float(row[field])
    return out


def cell_dir(experiment: Optional[Path], cell: str) -> Path:
    """``<experiment>/<cell>``, or ``cell`` itself as a directory path when no experiment is given."""
    return Path(cell) if experiment is None else Path(experiment) / cell


def judge_path(experiment: Optional[Path], cell: str, judge: str) -> Path:
    p = cell_dir(experiment, cell) / "freeform_judge" / f"{judge}.parquet"
    if not p.exists():
        raise SystemExit(f"no judge file at {p} — run the judge on that cell first")
    return p


def compare(experiment: Optional[Path], cell_a: str, cell_b: str, judge: str, *,
            n_boot: int = 10000, seed: int = 0, sub_scores: bool = False,
            experiment_b: Optional[Path] = None) -> Dict[str, Any]:
    """B − A; ``experiment_b`` (default: ``experiment``) is where cell B lives."""
    exp_b = experiment if experiment_b is None else experiment_b
    a = load_scores(judge_path(experiment, cell_a, judge))
    b = load_scores(judge_path(exp_b, cell_b, judge))
    res: Dict[str, Any] = {
        "experiment": None if experiment is None else str(experiment),
        "experiment_cell": None if exp_b is None else str(exp_b), "judge": judge,
        "baseline": cell_a, "cell": cell_b,
        "n_baseline": len(a), "n_cell": len(b),
        "mean_baseline": round(sum(a.values()) / len(a), 4) if a else None,
        "mean_cell": round(sum(b.values()) / len(b), 4) if b else None,
        "score": paired_bootstrap(a, b, n_boot=n_boot, seed=seed),
    }
    if sub_scores:
        res["sub_scores"] = {}
        for field in SUB_SCORES:
            sa = load_scores(judge_path(experiment, cell_a, judge), field)
            sb = load_scores(judge_path(exp_b, cell_b, judge), field)
            res["sub_scores"][field] = paired_bootstrap(sa, sb, n_boot=n_boot, seed=seed)
    return res


def format_row(res: Dict[str, Any]) -> str:
    s = res["score"]
    return (f"{res['cell']} − {res['baseline']}: "
            f"Δ {s['delta_mean']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]  "
            f"(means {res['mean_cell']:.3f} vs {res['mean_baseline']:.3f}; "
            f"win/tie/loss {s['win_rate']:.0%}/{s['tie_rate']:.0%}/{s['loss_rate']:.0%}; n={s['n']})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", action="append", default=[],
                    help="experiment directory holding the cells; twice = the baseline's, then the cell's; "
                         "omitted = the two cells are directory paths")
    ap.add_argument("--judge", default="gemma4_31b", help="judge name = the parquet stem")
    ap.add_argument("--cells", nargs=2, required=True, metavar=("BASELINE", "CELL"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sub-scores", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full record as JSON")
    args = ap.parse_args(argv)

    if len(args.experiment) > 2:
        ap.error("--experiment takes at most two values (the baseline's folder, then the cell's)")
    exps = [Path(e) for e in args.experiment]
    exp_a = exps[0] if exps else None
    exp_b = exps[-1] if exps else None
    res = compare(exp_a, args.cells[0], args.cells[1], args.judge,
                  n_boot=args.n_boot, seed=args.seed, sub_scores=args.sub_scores, experiment_b=exp_b)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(format_row(res))
        for field, s in (res.get("sub_scores") or {}).items():
            print(f"  {field:<22} Δ {s['delta_mean']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
