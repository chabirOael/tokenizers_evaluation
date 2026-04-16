#!/usr/bin/env python3
"""CLI: Compare results across multiple experiments.

Usage:
    python scripts/compare_results.py outputs/experiments/*/
    python scripts/compare_results.py --dirs exp1/ exp2/ exp3/ --output report.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.evaluation.reporter import load_experiment_results, generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("dirs", nargs="+", help="Experiment output directories")
    parser.add_argument("--output", type=str, default="outputs/results/comparison_report.txt",
                        help="Output report path")
    args = parser.parse_args()

    experiments = {}
    for d in args.dirs:
        path = Path(d)
        if not path.exists():
            print(f"Warning: {d} does not exist, skipping")
            continue
        name = path.name
        results = load_experiment_results(path)
        if results:
            experiments[name] = results
        else:
            print(f"Warning: No results found in {d}")

    if not experiments:
        print("No experiment results found.")
        return

    report = generate_report(experiments, args.output)
    print(report)


if __name__ == "__main__":
    main()
