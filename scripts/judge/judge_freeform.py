#!/usr/bin/env python
"""Score the free-form generations of a finished experiment (or every cell of a sweep)
with one or more LLM judges, merge the summaries into all_metrics.json and regenerate
the comparison report.

    # local Gemma judge (from .venv-judge; the wrapper sets the environment)
    scripts/judge/run_judge.sh --experiment outputs/experiments/<sweep> --judge configs/judges/gemma4_31b_local.yaml
    # API judge (main venv), both judges, explicit baseline cell
    .venv/bin/python scripts/judge/judge_freeform.py --experiment outputs/experiments/<sweep> \
        --judge configs/judges/gpt56_terra_api.yaml --baseline native_llama
    # re-judge one cell only (the others keep their verdicts and their all_metrics.json)
    scripts/judge/run_judge.sh --experiment outputs/experiments/<sweep> --judge <yaml> \
        --cells araroopat_3phase_v5_distill --overwrite
    # look at the prompt for two rows without calling any judge
    .venv/bin/python scripts/judge/judge_freeform.py --experiment <dir> --judge <yaml> --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from arabic_eval.judge.freeform_judge import (  # noqa: E402
    JudgeConfig, build_messages, discover_cells, format_table, load_generations, make_backend, run,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, help="a cell dir (has all_metrics.json) or a sweep dir of cells")
    ap.add_argument("--judge", action="append", required=True, help="judge YAML (repeatable)")
    ap.add_argument("--cells", nargs="+", default=None, metavar="CELL",
                    help="judge only these cells (default: every cell with generations). The report is still built "
                         "from all cells, and the baseline's verdicts are read from its file when it is not selected.")
    ap.add_argument("--baseline", default=None, help="cell name for the paired comparison (default: the native_* cell)")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N generations per cell")
    ap.add_argument("--overwrite", action="store_true", help="re-judge cells that already have this judge's file")
    ap.add_argument("--no-report", action="store_true", help="do not regenerate comparison_report.txt")
    ap.add_argument("--dry-run", action="store_true", help="print the judge prompt of two rows and exit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfgs = [JudgeConfig.from_yaml(p) for p in args.judge]
    if args.dry_run:
        cells = [c for c in discover_cells(Path(args.experiment)) if load_generations(c)]
        if not cells:
            print("no cell with eval_rows/freeform_cidar.parquet under", args.experiment)
            return 1
        gens, meta = load_generations(cells[0])
        print(f"cell {cells[0].name}: {len(gens)} generations (token cap {meta.get('token_cap')})\n")
        for g in gens[:2]:
            print(build_messages(g["instruction"], g.get("context") or "", g["reference"], g["generation"], cfgs[0].rubric)[0]["content"])
            print("\n" + "=" * 80 + "\n")
        return 0
    backends = {c.name: make_backend(c) for c in cfgs}
    report = run(args.experiment, cfgs, backends, baseline=args.baseline, limit=args.limit,
                 overwrite=args.overwrite, regenerate_report=not args.no_report, cells=args.cells)
    print(format_table(report))
    out = Path(args.experiment) / "freeform_judge_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\nreport → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
