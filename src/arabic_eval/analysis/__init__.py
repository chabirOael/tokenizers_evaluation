"""``ae`` — analysis helpers over the experiments' results.

    import arabic_eval.analysis as ae
    ae.catalog()                                   # what exists
    ae.metrics(experiment="qwen_native_vs_araroopat")
    res = ae.mcq_compare(["araroopat_3phase_v5_distill_mcq4096", "native_qwen3_sft_v5_distill_mcq4096"],
                         tasks=["arabic_exam"], pairs=[("araroopat_3phase_v5_distill_mcq4096",
                                                        "native_qwen3_sft_v5_distill_mcq4096")])
    ae.judge_compare("bpe_16k_3phase_v5_distill", "araroopat_3phase_v5_distill")

Preloaded as ``ae`` in the experiment console's Analysis tab, whose agent
writes Python against it; usable from any script or notebook of the main venv.
Read-only by design: nothing here writes under ``outputs/experiments``.
Pandas-returning helpers import pandas lazily, so importing the package is cheap.
"""
from __future__ import annotations

import inspect
from typing import Callable, List

from arabic_eval.analysis._root import experiments_root, repo_root, set_experiments_root
from arabic_eval.analysis.inventory import (
    FREEFORM_TASK,
    MCQ_TASKS,
    Cell,
    CellNotFound,
    catalog,
    cell_facts,
    cells,
    experiments,
    files,
    resolve,
)
from arabic_eval.analysis.display import PALETTE, install_style, show
from arabic_eval.analysis.docs import report_search, report_section
from arabic_eval.analysis.loaders import (
    config_json,
    diag,
    dump_metadata,
    eval_history,
    freeform_rows,
    judge_rows,
    mcq_rows,
    metric,
    metrics,
    metrics_json,
    metrics_long,
    phase_summary,
    training_history,
)
from arabic_eval.analysis.paired import (
    freeform_paired,
    judge_compare,
    mcq_compare,
    mcq_pairs_table,
    mcq_paired,
    mcq_table,
)
from arabic_eval.analysis.stats import bootstrap_ci, mcnemar, paired_delta

#: The functions the Analysis agent is told about, grouped as its prompt shows them.
API_GROUPS = {
    "What exists": [catalog, cell_facts, experiments, files, report_search, report_section],
    "Numbers per cell": [metrics, metrics_long, metric, metrics_json, config_json, phase_summary,
                         eval_history, training_history, diag, dump_metadata],
    "Rows": [mcq_rows, freeform_rows, judge_rows],
    "Paired comparisons (the report's rules)": [mcq_compare, mcq_table, mcq_pairs_table, mcq_paired,
                                                freeform_paired, judge_compare],
    "Statistics": [paired_delta, mcnemar, bootstrap_ci],
    "Display": [show],
}


def _first_paragraph(fn: Callable) -> str:
    doc = inspect.getdoc(fn) or ""
    return " ".join(doc.split("\n\n")[0].split())


def _signature(fn: Callable) -> str:
    """``(cells, task, *, clean_only=True)`` — names and defaults, no annotations."""
    parts: List[str] = []
    star_done = False
    for p in inspect.signature(fn).parameters.values():
        if p.kind is p.KEYWORD_ONLY and not star_done:
            parts.append("*")
            star_done = True
        if p.default is p.empty:
            parts.append(p.name)
        else:
            d = p.default
            shown = getattr(d, "__name__", None) if callable(d) else repr(d)
            parts.append(f"{p.name}={shown}")
    return "(" + ", ".join(parts) + ")"


def api_reference(full: bool = False) -> str:
    """The helpers' signatures and summaries (``full=True``: whole docstrings) — the agent's API page."""
    lines: List[str] = []
    for group, fns in API_GROUPS.items():
        lines.append(f"### {group}")
        for fn in fns:
            sig = _signature(fn)
            doc = (inspect.getdoc(fn) or "").strip() if full else _first_paragraph(fn)
            if full:
                doc = "\n    ".join(doc.splitlines())
            lines.append(f"- ae.{fn.__name__}{sig} — {doc}")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = [n for n in dir() if not n.startswith("_")]
