"""Discovery and query layer over the per-eval-row Parquet dumps.

Back-end for the experiment console's *Eval rows* tab. It answers three
questions and nothing else:

  * which dumps exist (experiment → cell → benchmark), and what is in them;
  * which rows match a filter, with the aggregates for that selection;
  * what one page of those rows actually contains.

Everything is derived from the files themselves. Nothing here re-runs an
evaluation, and no number is copied from a summary that could have gone
stale: accuracy for a selection is recomputed from the correctness columns
of the rows in that selection.

Filesystem convention (written by ``LightEvalBenchmarkTask.evaluate``)::

    outputs/experiments/<sweep>/<cell>/eval_rows/<task>.parquet
    outputs/experiments/<experiment>/eval_rows/<task>.parquet      (single cell)

The directory holding ``eval_rows`` is the cell; its parent (when there is
one below ``outputs/experiments``) is the experiment.
"""
from __future__ import annotations

import csv
import io
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from arabic_eval.evaluation.eval_rows import EvalRowFile, is_superseded, read_progress

logger = logging.getLogger("arabic_eval.tools.eval_rows_browser")

#: Open ``EvalRowFile`` handles, keyed by (path, mtime, size). Each holds its
#: filter columns (about a megabyte for a 21 k-row task) and, once a text
#: search has run, the search haystack — so the cache is small but not free.
_CACHE: "OrderedDict[Tuple[str, int, int], EvalRowFile]" = OrderedDict()
_CACHE_MAX = 8

#: Per-cell ``config.json`` / ``all_metrics.json``, keyed by (path, mtime).
_JSON_CACHE: Dict[Tuple[str, int], Any] = {}

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 200
MAX_EXPORT_ROWS = 50_000


class EvalRowsError(ValueError):
    """Bad request against a dump (missing file, unknown filter, escape attempt)."""


# ---------------------------------------------------------------------------
# Paths and caching
# ---------------------------------------------------------------------------

def _outputs_root(repo_root: Path) -> Path:
    return (Path(repo_root) / "outputs").resolve()


def resolve_dump(repo_root: Path, rel: str) -> Path:
    """Resolve a client-supplied dump path, confined to ``outputs/``."""
    if not rel:
        raise EvalRowsError("path required")
    path = (Path(repo_root) / rel).resolve()
    root = _outputs_root(repo_root)
    if root != path and root not in path.parents:
        raise EvalRowsError("only files under outputs/ can be read")
    if path.suffix != ".parquet":
        raise EvalRowsError(f"not a parquet dump: {rel}")
    if not path.is_file():
        raise EvalRowsError(f"no such dump: {rel}")
    return path


def open_dump(repo_root: Path, rel: str) -> EvalRowFile:
    """Open (or reuse) a dump. The cache key includes mtime and size, so a
    re-run that overwrites a dump is picked up without a server restart."""
    path = resolve_dump(repo_root, rel)
    st = path.stat()
    key = (str(path), int(st.st_mtime_ns), int(st.st_size))
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit
    # Drop any stale generations of the same file before inserting.
    for k in [k for k in _CACHE if k[0] == str(path)]:
        _CACHE.pop(k, None)
    handle = EvalRowFile(path)
    _CACHE[key] = handle
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return handle


def _read_json(path: Path) -> Optional[dict]:
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), int(st.st_mtime_ns))
    if key in _JSON_CACHE:
        return _JSON_CACHE[key]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    _JSON_CACHE[key] = data
    return data


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _cell_context(cell_dir: Path) -> Dict[str, Any]:
    """Tokenizer / model facts for a cell, from the config it ran with.

    The dump's own metadata records the tokenizer *class*; the registry key
    and vocab size live in the cell's ``config.json``, which is also the only
    place that knows which model checkpoint produced these rows.
    """
    cfg = _read_json(cell_dir / "config.json") or {}
    tok = cfg.get("tokenizer") or {}
    model = cfg.get("model") or {}
    metrics = _read_json(cell_dir / "all_metrics.json") or {}
    downstream = metrics.get("downstream") or {}
    return {
        "tokenizer": tok.get("type"),
        "vocab_size": tok.get("vocab_size"),
        "model": model.get("name_or_path"),
        "reported": {
            task: {
                "accuracy": m.get("accuracy"),
                "accuracy_pmi": m.get("accuracy_pmi"),
                "accuracy_char_norm": m.get("accuracy_char_norm"),
                "num_samples": m.get("num_samples"),
            }
            for task, m in downstream.items() if isinstance(m, dict)
        },
    }


def discover(repo_root: Path) -> Dict[str, Any]:
    """Every dump under ``outputs/experiments``, grouped experiment → cell → task.

    Cheap by construction: a Parquet row count is a footer read, so this does
    not decode any row. A task still being evaluated has no readable file yet
    and shows up from its progress file instead, with ``in_progress`` set.
    """
    repo_root = Path(repo_root)
    base = repo_root / "outputs" / "experiments"
    experiments: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    if not base.is_dir():
        return {"experiments": []}

    for rows_dir in sorted(base.glob("**/eval_rows")):
        if not rows_dir.is_dir() or is_superseded(rows_dir, base):
            continue
        cell_dir = rows_dir.parent
        rel_parts = cell_dir.relative_to(base).parts
        if len(rel_parts) == 1:
            experiment, cell = rel_parts[0], rel_parts[0]
            single = True
        else:
            experiment, cell = str(Path(*rel_parts[:-1])), rel_parts[-1]
            single = False

        tasks: List[Dict[str, Any]] = []
        seen: set = set()
        for dump in sorted(rows_dir.glob("*.parquet")):
            task = dump.stem
            seen.add(task)
            entry: Dict[str, Any] = {
                "task": task,
                "path": str(dump.relative_to(repo_root)),
                "size": dump.stat().st_size,
                "mtime": dump.stat().st_mtime,
                "in_progress": False,
            }
            try:
                handle = open_dump(repo_root, entry["path"])
                entry["n_rows"] = handle.n_rows
                entry["metadata"] = handle.metadata
            except Exception as e:  # noqa: BLE001 - a half-written file must not hide the rest
                # The file exists from the moment the writer opens it, but has
                # no footer until it closes, so it is unreadable while the task
                # is being evaluated. A progress file next to it says which of
                # the two this is: still running, or a run that died.
                progress = read_progress(dump)
                if progress is not None:
                    entry["in_progress"] = True
                    entry["n_rows"] = progress.get("rows")
                    entry["progress"] = progress
                else:
                    entry["error"] = f"{type(e).__name__}: {e}"
                    entry["n_rows"] = None
            tasks.append(entry)
        # Tasks whose footer is not written yet (eval still running).
        for prog_file in sorted(rows_dir.glob("*.progress.json")):
            task = prog_file.name[: -len(".progress.json")]
            if task in seen:
                continue
            progress = read_progress(prog_file) or {}
            tasks.append({
                "task": task,
                "path": str((rows_dir / f"{task}.parquet").relative_to(repo_root)),
                "in_progress": True,
                "n_rows": progress.get("rows"),
                "progress": progress,
            })
        if not tasks:
            continue

        exp = experiments.setdefault(experiment, {
            "experiment": experiment, "single": single, "cells": [],
        })
        exp["cells"].append({
            "cell": cell,
            "dir": str(cell_dir.relative_to(repo_root)),
            **_cell_context(cell_dir),
            "tasks": tasks,
        })

    out = list(experiments.values())
    for exp in out:
        exp["cells"].sort(key=lambda c: c["cell"])
        exp["n_rows"] = sum(
            t.get("n_rows") or 0 for c in exp["cells"] for t in c["tasks"]
        )
    out.sort(key=lambda e: e["experiment"])
    return {"experiments": out}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

#: Boolean row flags a caller may filter on. Each maps to a column of the dump.
FLAG_NAMES: Sequence[str] = (
    "sentinel", "all_sentinel", "hit_cap", "near_tie", "disagree",
)

_SORTS: Sequence[str] = ("row", "margin_asc", "margin_desc", "uncertain", "confident")


def _parse_flags(raw: Any) -> Dict[str, bool]:
    """Parse the ``flags`` query parameter.

    Accepts a dict, or the comma-separated form the page sends:
    ``"all_sentinel,-near_tie"`` means all_sentinel true and near_tie false.
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        out = {str(k): bool(v) for k, v in raw.items()}
    else:
        out = {}
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            want = not part.startswith("-")
            out[part.lstrip("-+")] = want
    for name in out:
        if name not in FLAG_NAMES:
            raise EvalRowsError(
                f"unknown flag {name!r} (known: {', '.join(FLAG_NAMES)})"
            )
    return out


def _selection(handle: EvalRowFile, params: Dict[str, Any]):
    scoring = str(params.get("scoring") or "primary")
    if scoring not in ("primary", "char", "pmi"):
        raise EvalRowsError(f"unknown scoring {scoring!r}")
    sort = str(params.get("sort") or "row")
    if sort not in _SORTS:
        raise EvalRowsError(f"unknown sort {sort!r} (known: {', '.join(_SORTS)})")
    subconfigs = params.get("subconfig") or params.get("subconfigs")
    if isinstance(subconfigs, str):
        subconfigs = [s for s in subconfigs.split("|") if s]
    gold = params.get("gold_idx")
    pred = params.get("pred_idx")
    try:
        return handle.select(
            outcome=str(params.get("outcome") or "all"),
            scoring=scoring,
            source_configs=subconfigs,
            flags=_parse_flags(params.get("flags")),
            gold_idx=None if gold in (None, "") else int(gold),
            pred_idx=None if pred in (None, "") else int(pred),
            query=str(params.get("q") or ""),
            search_prompt=str(params.get("search_prompt") or "") in ("1", "true", "True", True),
            sort=sort,
        ), scoring
    except ValueError as e:
        raise EvalRowsError(str(e)) from e


def _subconfig_facets(handle: EvalRowFile, positions) -> List[Dict[str, Any]]:
    """Row counts per sub-config *within the current selection*."""
    import numpy as np

    arr = handle.index()["source_config"][positions]
    if not len(arr):
        return []
    names, counts = np.unique(arr, return_counts=True)
    pairs = sorted(zip(names.tolist(), counts.tolist()), key=lambda kv: (-kv[1], kv[0]))
    return [{"name": k, "n_rows": int(v)} for k, v in pairs]


def query(repo_root: Path, rel: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """One page of rows plus the aggregates for the whole filtered selection.

    The aggregates describe every matching row, not just the page — the point
    of the summary strip is to characterise the selection you are browsing.
    """
    handle = open_dump(repo_root, rel)
    positions, scoring = _selection(handle, params)
    total = int(len(positions))

    try:
        page_size = int(params.get("page_size") or DEFAULT_PAGE_SIZE)
    except (TypeError, ValueError):
        raise EvalRowsError("page_size must be an integer") from None
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    pages = max(1, (total + page_size - 1) // page_size)
    try:
        page = int(params.get("page") or 1)
    except (TypeError, ValueError):
        raise EvalRowsError("page must be an integer") from None
    page = max(1, min(page, pages))
    start = (page - 1) * page_size
    window = positions[start:start + page_size]

    return {
        "path": rel,
        "total": total,
        "page": page,
        "pages": pages,
        "page_size": page_size,
        "scoring": scoring,
        "n_rows_file": handle.n_rows,
        "metadata": handle.metadata,
        "summary": handle.summary(positions, scoring=scoring),
        "subconfigs": _subconfig_facets(handle, positions),
        "rows": handle.rows(window),
    }


def describe(repo_root: Path, rel: str) -> Dict[str, Any]:
    """Header for one dump: metadata, row count, sub-configs, whole-file summary."""
    return open_dump(repo_root, rel).describe()


def row(repo_root: Path, rel: str, position: int) -> Dict[str, Any]:
    """One full record by its position in the file."""
    handle = open_dump(repo_root, rel)
    pos = int(position)
    if not (0 <= pos < handle.n_rows):
        raise EvalRowsError(f"position {pos} out of range (0..{handle.n_rows - 1})")
    got = handle.rows([pos])
    if not got:
        raise EvalRowsError(f"could not read row {pos}")
    return {"path": rel, "metadata": handle.metadata, "row": got[0]}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

#: Column order of the CSV export. Flat and spreadsheet-shaped: the list
#: columns are rendered as ``|``-joined text, because the destination is a
#: spreadsheet and a nested array has no cell to live in.
EXPORT_FIELDS: Sequence[str] = (
    "row_index", "source_config", "prompt", "question", "context",
    "choices", "continuations", "gold_idx", "gold_text",
    "pred_idx", "pred_text", "correct",
    "ll", "score_char", "score_pmi", "margin", "decision_margin",
    "prompt_units", "n_choices",
    "sentinel", "all_sentinel", "hit_cap", "near_tie", "disagree",
)


def _flatten_for_csv(rec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in EXPORT_FIELDS:
        v = rec.get(k)
        if isinstance(v, list):
            out[k] = "|".join("" if x is None else str(x) for x in v)
        else:
            out[k] = v
    return out


def export_csv(repo_root: Path, rel: str, params: Dict[str, Any]) -> Tuple[str, str]:
    """Render the current selection as CSV text. Returns (filename, text).

    This is the spreadsheet escape hatch: the dump itself is Parquet, but what
    a reader is *looking at* is usually a few hundred rows they want to hand
    to someone or annotate. Capped so a stray click cannot serialise a whole
    sweep into memory.
    """
    handle = open_dump(repo_root, rel)
    positions, _ = _selection(handle, params)
    if len(positions) > MAX_EXPORT_ROWS:
        positions = positions[:MAX_EXPORT_ROWS]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(EXPORT_FIELDS), extrasaction="ignore")
    writer.writeheader()
    for rec in handle.rows(positions, columns=None):
        writer.writerow(_flatten_for_csv(rec))
    stem = Path(rel).stem
    cell = Path(rel).parent.parent.name
    return f"{cell}_{stem}_rows.csv", buf.getvalue()
