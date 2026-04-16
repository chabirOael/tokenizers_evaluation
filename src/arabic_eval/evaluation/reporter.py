"""Results aggregation, comparison tables, and reporting."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tabulate import tabulate

from arabic_eval.utils.io import load_json, save_json

logger = logging.getLogger("arabic_eval.evaluation.reporter")


def load_experiment_results(results_dir: str | Path) -> Dict[str, Any]:
    """Load all_metrics.json from an experiment output directory."""
    path = Path(results_dir) / "all_metrics.json"
    if path.exists():
        return load_json(path)
    return {}


def build_comparison_table(
    experiments: Dict[str, Dict[str, Any]],
    metric_keys: Optional[List[str]] = None,
) -> str:
    """Build a comparison table across experiments.

    Args:
        experiments: {experiment_name: results_dict}
        metric_keys: specific metrics to include (default: all)

    Returns:
        Formatted ASCII table string.
    """
    if not experiments:
        return "No experiments to compare."

    # Collect all metric keys
    all_keys = set()
    for results in experiments.values():
        # Flatten nested dicts
        flat = _flatten_metrics(results)
        all_keys.update(flat.keys())

    if metric_keys:
        all_keys = all_keys & set(metric_keys)

    all_keys = sorted(all_keys)
    headers = ["Experiment"] + all_keys

    rows = []
    for name, results in sorted(experiments.items()):
        flat = _flatten_metrics(results)
        row = [name] + [flat.get(k, "—") for k in all_keys]
        rows.append(row)

    return tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".4f")


def _flatten_metrics(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested metrics dict into dot-separated keys."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_metrics(v, key))
        else:
            flat[key] = v
    return flat


def generate_report(
    experiments: Dict[str, Dict[str, Any]],
    output_path: str | Path,
) -> str:
    """Generate a full comparison report and save to file.

    Args:
        experiments: {experiment_name: results_dict}
        output_path: where to save the report

    Returns:
        Report as a string.
    """
    lines = ["=" * 70]
    lines.append("ARABIC TOKENIZER EVALUATION — COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Intrinsic metrics table
    intrinsic_data = {}
    for name, results in experiments.items():
        if "intrinsic" in results:
            intrinsic_data[name] = results["intrinsic"]

    if intrinsic_data:
        lines.append("## Intrinsic Tokenizer Metrics")
        lines.append("")
        lines.append(build_comparison_table(
            intrinsic_data,
            metric_keys=["fertility", "compression_ratio", "unk_rate",
                         "vocab_coverage", "vocab_size"],
        ))
        lines.append("")

    # Downstream metrics tables
    downstream_tasks = set()
    for results in experiments.values():
        if "downstream" in results:
            downstream_tasks.update(results["downstream"].keys())

    for task_name in sorted(downstream_tasks):
        task_data = {}
        for name, results in experiments.items():
            if "downstream" in results and task_name in results["downstream"]:
                task_data[name] = results["downstream"][task_name]

        if task_data:
            lines.append(f"## Downstream Task: {task_name}")
            lines.append("")
            lines.append(build_comparison_table(task_data))
            lines.append("")

    report = "\n".join(lines)

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Also save raw data as JSON
    save_json(experiments, output_path.with_suffix(".json"))

    logger.info("Report saved to %s", output_path)
    return report
