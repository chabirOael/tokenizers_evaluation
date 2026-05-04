#!/usr/bin/env python3
"""Recompute MEI in archived ``all_metrics.json`` files under the new
per-row formula.

Old formula: MEI = (accuracy × RPS × compression) / inference_time_sec
New formula: MEI = (accuracy × RPS × compression × num_eval_rows) / inference_time_sec

The four MEI inputs (accuracy, rps, compression, inference_time_sec) are
already preserved in ``mei.inputs`` for every existing run; the row count
is read from ``downstream[<task_type>].num_samples`` (populated by
``LightEvalBenchmarkTask.evaluate``). The recompute is purely arithmetic
— no model or tokenizer needs to be re-run.

Idempotent: re-running on already-migrated JSONs reads the same
``mei.inputs`` and ``downstream.<task>.num_samples`` and writes back an
identical record. After per-experiment JSONs are recomputed, the script
also regenerates ``comparison_report.{txt,json}`` for each sweep
directory so the rendered tables reflect the new numbers.

Usage:
    .venv/bin/python scripts/recompute_mei.py outputs/experiments/
    .venv/bin/python scripts/recompute_mei.py --dry-run outputs/experiments/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.evaluation.metrics import compute_mei
from arabic_eval.evaluation.reporter import (
    generate_report,
    load_experiment_results,
)
from arabic_eval.utils.io import load_json, save_json


# Registry keys for the LightEval MCQ task family. Mirrors what the pipeline
# detects via ``isinstance(task, LightEvalBenchmarkTask)``; we can't import
# the task class here without pulling torch transitively, so use a static
# allowlist. If a new LightEval benchmark is added, append its registry
# key here.
LIGHTEVAL_MCQ_TASKS = frozenset(
    {"acva", "alghafa", "culture_arabic_mmlu", "arabic_exam"}
)


def _recompute_one(metrics_path: Path) -> Tuple[str, Dict[str, Any] | None]:
    """Recompute MEI for one ``all_metrics.json``.

    Returns a (status, new_record) tuple. ``status`` is one of:
      - ``updated`` — recompute succeeded and the file was written
      - ``no_mei_block`` — no ``mei`` key in the JSON (legacy run before MEI)
      - ``no_inputs`` — ``mei`` exists but ``inputs`` is missing/empty
      - ``no_num_samples`` — couldn't find ``num_samples`` in the downstream
        block (predates the field)
      - ``not_lighteval_mcq`` — task isn't in the LightEval MCQ family
      - ``status_changed_to_<X>`` — recompute hit a new typed-status branch
        (e.g. ``zero_rows``); recorded for audit
    """
    data = load_json(metrics_path)
    mei_block = data.get("mei")
    if mei_block is None:
        return "no_mei_block", None

    inputs = mei_block.get("inputs") or {}
    if not inputs:
        return "no_inputs", None

    config = data.get("config") or {}
    task_type = config.get("task")
    if task_type not in LIGHTEVAL_MCQ_TASKS:
        return "not_lighteval_mcq", None

    downstream = (data.get("downstream") or {}).get(task_type) or {}
    num_eval_rows = downstream.get("num_samples")
    if num_eval_rows is None:
        return "no_num_samples", None

    accuracy_source = inputs.get("accuracy_source", "accuracy")
    new_record = compute_mei(
        accuracy=inputs.get("accuracy"),
        rps=inputs.get("rps"),
        compression=inputs.get("compression"),
        inference_time_sec=inputs.get("inference_time_sec"),
        num_eval_rows=num_eval_rows,
        is_lighteval_mcq=True,
        accuracy_source=accuracy_source,
    )
    data["mei"] = new_record
    save_json(data, metrics_path)

    new_status = new_record.get("status")
    if new_status == "ok":
        return "updated", new_record
    return f"status_changed_to_{new_status}", new_record


def _regen_sweep_reports(sweep_root: Path) -> List[Path]:
    """Regenerate comparison_report.{txt,json} for any sweep directory under
    ``sweep_root``. A directory is treated as a sweep iff it contains
    multiple per-experiment subdirectories with ``all_metrics.json``.
    """
    regenerated: List[Path] = []
    for candidate in sorted(sweep_root.rglob("comparison_report.txt")):
        sweep_dir = candidate.parent
        # Collect per-experiment subdirs that contain all_metrics.json
        experiments: Dict[str, Dict[str, Any]] = {}
        for child in sorted(sweep_dir.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "all_metrics.json").exists():
                continue
            results = load_experiment_results(child)
            if results:
                experiments[child.name] = results
        if not experiments:
            continue
        generate_report(experiments, sweep_dir / "comparison_report.txt")
        regenerated.append(sweep_dir / "comparison_report.txt")
    return regenerated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute MEI under the per-row formula."
    )
    parser.add_argument(
        "root",
        type=str,
        default="outputs/experiments",
        nargs="?",
        help="Root directory to walk for all_metrics.json files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk and report counts without writing changes.",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip regenerating comparison_report.{txt,json} after migrating.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root {root} does not exist", file=sys.stderr)
        sys.exit(1)

    counts: Dict[str, int] = {}
    updated_paths: List[Path] = []
    for metrics_path in sorted(root.rglob("all_metrics.json")):
        if args.dry_run:
            data = load_json(metrics_path)
            mei_block = data.get("mei")
            if mei_block is None:
                status = "no_mei_block"
            elif (data.get("config") or {}).get("task") not in LIGHTEVAL_MCQ_TASKS:
                status = "not_lighteval_mcq"
            else:
                ds = (data.get("downstream") or {}).get(
                    (data.get("config") or {}).get("task"), {}
                ) or {}
                if ds.get("num_samples") is None:
                    status = "no_num_samples"
                else:
                    status = "would_update"
        else:
            status, _ = _recompute_one(metrics_path)
            if status == "updated":
                updated_paths.append(metrics_path)

        counts[status] = counts.get(status, 0) + 1

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}MEI recompute summary")
    print(f"  root: {root}")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")

    if not args.dry_run and not args.no_reports:
        regenerated = _regen_sweep_reports(root)
        if regenerated:
            print(f"\nRegenerated {len(regenerated)} comparison report(s):")
            for p in regenerated:
                print(f"  {p}")


if __name__ == "__main__":
    main()
