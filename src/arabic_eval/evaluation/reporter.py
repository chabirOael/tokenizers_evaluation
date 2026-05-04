"""Results aggregation, comparison tables, and reporting."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tabulate import tabulate

from arabic_eval.utils.io import load_json, save_json

logger = logging.getLogger("arabic_eval.evaluation.reporter")

# Tokenizers whose root_conservation_rate hits a mechanical extreme by
# construction. MEI uses RPS as a multiplier, so these rows should be flagged
# in comparison tables — same policy as the morphological-metrics docs:
# report and footnote, don't suppress.
RPS_MECHANICAL_FLAGS: Dict[str, str] = {
    "character_bert": "RPS ceiling — never splits a word",
    "araroopat":      "RPS ceiling — ROOT token IS the root letters",
    "char_jaber":     "RPS floor — single chars cannot hold a 3-letter root",
    "charformer":     "RPS floor — single bytes cannot hold an Arabic letter",
}

# Tasks whose accuracy carries known label-quality noise that the metric
# itself cannot remove. ACVA (synthetic-generated T/F questions) ships with
# wrong gold labels for some examples (e.g. "Kabsa is the Saudi national
# dish" → خطأ; "Kabsa is traditional in Syrian cuisine" → صح) and ~0.6 % of
# its eval examples appear with both labels in the same partition. Flag
# these tasks in the per-task comparison table so a reader doesn't read a
# 41 % accuracy as "this tokenizer is bad at Arabic culture knowledge" when
# the gold itself is broken. Same policy as RPS_MECHANICAL_FLAGS: report and
# footnote, don't suppress.
LABEL_NOISY_TASKS: Dict[str, str] = {
    "acva": "synthetic-generation label noise + ~30 % duplicates "
            "+ 51 within-eval label conflicts (see CLAUDE.md → ACVA limitations)",
}


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


def _build_per_subconfig_section(
    task_name: str,
    task_data: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Render a per-sub-config accuracy breakdown for one downstream task.

    Returns ``[]`` when the task carries no ``per_subconfig_accuracy`` block,
    or when it does but only has the ``_default`` sentinel (single-config
    benchmark — would just duplicate the main accuracy column).

    Layout: rows = experiments, columns = sub-configs (sorted alphabetically).
    Cell = ``accuracy (n=N)``. Missing sub-configs render as ``—``. The total
    ``num_samples`` per sub-config doesn't vary across tokenizer experiments
    in a normal sweep (same eval split), but we keep the n on each cell so a
    reader can spot drift if it ever happens.
    """
    # Collect the union of sub-config keys across experiments.
    all_keys: set = set()
    for results in task_data.values():
        psa = results.get("per_subconfig_accuracy") or {}
        all_keys.update(psa.keys())

    # Skip the section when there's nothing meaningful to show.
    if not all_keys or all_keys == {"_default"}:
        return []

    sorted_keys = sorted(all_keys)

    headers = ["Experiment"] + sorted_keys
    rows: List[List[Any]] = []
    for name, results in sorted(task_data.items()):
        psa = results.get("per_subconfig_accuracy") or {}
        row: List[Any] = [name]
        for k in sorted_keys:
            entry = psa.get(k)
            if entry is None:
                row.append("—")
            else:
                acc = entry.get("accuracy")
                n = entry.get("num_samples")
                row.append(f"{acc:.4f} (n={n})" if isinstance(acc, (int, float)) else "—")
        rows.append(row)

    lines: List[str] = []
    lines.append(f"### Per-sub-config breakdown — {task_name}")
    lines.append("")
    lines.append(tabulate(rows, headers=headers, tablefmt="grid"))
    lines.append("")
    return lines


def _build_mei_section(experiments: Dict[str, Dict[str, Any]]) -> List[str]:
    """Render the MEI (Morphological Efficiency Index) comparison block.

    Returns a list of lines (empty if no experiment carries an ``mei`` record).
    Rows where MEI was successfully computed are tabulated; rows where it was
    not are summarized after the table with their ``status``. Tokenizers with
    a mechanical RPS extreme (CharBERT, AraRooPat, char-JABER, Charformer) are
    flagged with an asterisk and footnoted.
    """
    has_any_mei = any("mei" in r for r in experiments.values())
    if not has_any_mei:
        return []

    lines: List[str] = []
    lines.append("## Composite Metric: MEI (Morphological Efficiency Index)")
    lines.append("")
    lines.append(
        "MEI = (accuracy × RPS × compression × num_eval_rows) / inference_time_sec"
    )
    lines.append(
        "    = (accuracy × RPS × compression) / (inference_time_sec / num_eval_rows)"
    )
    lines.append(
        "Per-row time normalization makes MEI comparable across LightEval MCQ "
        "tasks with different eval-set sizes. Defined only for the LightEval "
        "MCQ task family."
    )
    lines.append("")

    headers = [
        "Experiment", "Tokenizer", "MEI", "Accuracy", "RPS",
        "Compression", "Time (s)", "Rows",
    ]
    rows: List[List[Any]] = []
    skipped: List[tuple] = []  # (name, status)
    flagged: Dict[str, str] = {}  # tokenizer_type -> footnote text

    def _fmt(x: Any) -> str:
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    for name, results in sorted(experiments.items()):
        record = results.get("mei")
        if record is None:
            continue
        tok_type = (results.get("config") or {}).get("tokenizer", "?")
        status = record.get("status")
        if status != "ok":
            skipped.append((name, tok_type, status))
            continue
        inputs = record.get("inputs") or {}
        flag = RPS_MECHANICAL_FLAGS.get(tok_type)
        display_tok = f"{tok_type}*" if flag else tok_type
        if flag:
            flagged[tok_type] = flag
        rows.append([
            name,
            display_tok,
            _fmt(record.get("mei")),
            _fmt(inputs.get("accuracy")),
            _fmt(inputs.get("rps")),
            _fmt(inputs.get("compression")),
            _fmt(inputs.get("inference_time_sec")),
            _fmt(inputs.get("num_eval_rows")),
        ])

    if rows:
        lines.append(tabulate(rows, headers=headers, tablefmt="grid"))
        lines.append("")

    if flagged:
        lines.append("Footnotes (mechanical RPS extremes — flag, don't over-interpret):")
        for tok_type, note in sorted(flagged.items()):
            lines.append(f"  * {tok_type}: {note}")
        lines.append("")

    if skipped:
        lines.append("MEI not computed for the following experiments:")
        for name, tok_type, status in skipped:
            lines.append(f"  - {name} ({tok_type}): status={status}")
        lines.append("")

    return lines


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
            heading_suffix = " †" if task_name in LABEL_NOISY_TASKS else ""
            lines.append(f"## Downstream Task: {task_name}{heading_suffix}")
            lines.append("")
            # Filter the heavy ``per_subconfig_accuracy`` block out of the
            # main aggregate table — it would flatten into one column per
            # sub-config and bloat the table to ~20 columns on Alghafa.
            # Rendered separately below.
            main_table_data = {
                name: {k: v for k, v in r.items() if k != "per_subconfig_accuracy"}
                for name, r in task_data.items()
            }
            lines.append(build_comparison_table(main_table_data))
            lines.append("")
            if task_name in LABEL_NOISY_TASKS:
                lines.append(f"† {LABEL_NOISY_TASKS[task_name]}")
                lines.append("")
            # Per-sub-config breakdown — emitted only when the task is
            # heterogeneous (more than one ``_source_config``).
            lines.extend(_build_per_subconfig_section(task_name, task_data))

    # Composite (MEI) section — only experiments where MEI was actually
    # computed (status == "ok") are shown in the main table; everything else
    # is summarized below.
    mei_lines = _build_mei_section(experiments)
    if mei_lines:
        lines.extend(mei_lines)

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
