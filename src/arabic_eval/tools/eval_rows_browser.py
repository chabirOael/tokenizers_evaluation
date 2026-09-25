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
        other: List[Dict[str, Any]] = []
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
                # The same directory also holds the free-form generation dump,
                # which has neither a row_index nor a correctness column: every
                # filter and aggregate here is about scored choices. Listing it
                # as a benchmark made one click raise (measured 2026-09-25 on
                # freeform_cidar.parquet). It is reported as what it is, next to
                # the benchmarks, and read by the Free-form tab.
                if not {"row_index", "correct"} <= set(handle.columns):
                    other.append({
                        "task": task, "path": entry["path"], "n_rows": handle.n_rows,
                        "reason": "not an MCQ row dump — the Free-form tab reads this one",
                    })
                    continue
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
        if not tasks and not other:
            continue

        exp = experiments.setdefault(experiment, {
            "experiment": experiment, "single": single, "cells": [],
        })
        exp["cells"].append({
            "cell": cell,
            "dir": str(cell_dir.relative_to(repo_root)),
            **_cell_context(cell_dir),
            "tasks": tasks,
            "other_dumps": other,
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
    "ll", "score_char", "score_pmi", "cont_tokens", "margin", "decision_margin",
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


# ---------------------------------------------------------------------------
# Compare: one benchmark, several cells, side by side
# ---------------------------------------------------------------------------
#
# The single-cell view answers "what did this tokenizer do on this row". The
# comparison answers "what did these tokenizers do on the *same* row", which is
# the question a tokenizer study is actually read by. It is the row-level
# counterpart of ``scripts/mcq_compare.py``: that script reports the numbers
# (accuracy on the truncation-free intersection, a paired bootstrap CI, the
# McNemar counts); this reports the rows those numbers are made of. Nothing
# here computes an interval — the page is descriptive on purpose, and says so.

#: At most this many cells in one comparison: beyond it the table stops being
#: readable and the payload stops being small. The page caps at the same number.
MAX_COMPARE_CELLS = 6

#: How the per-cell row filters combine into one decision about a row. Same
#: vocabulary as the free-form compare, deliberately.
MATCH_MODES: Sequence[str] = ("any", "all", "anchor")

#: Cross-cell outcome filters. With two cells selected, ``only_anchor`` and
#: ``anchor_wrong`` are exactly McNemar's two discordant counts.
AGREEMENTS: Sequence[str] = (
    "all", "agree_correct", "agree_wrong", "disagree", "only_anchor", "anchor_wrong",
)

#: Cross-cell row orders. ``disagree`` puts the rows the cells answered
#: differently first; ``margin_spread_desc`` the rows they were furthest apart
#: in confidence on; ``units_ratio_desc`` the rows where the other cells' prompt
#: is longest relative to the anchor's (the tokenizer-length axis).
COMPARE_SORTS: Sequence[str] = (
    "row", "disagree", "margin_spread_desc",
    "anchor_uncertain", "anchor_confident", "units_ratio_desc",
)

#: Metadata that must agree for the comparison to mean what it looks like. A
#: difference does not block anything — it is reported, the way
#: ``mcq_compare.py`` flags a cell whose dumps used another ``max_length``.
_META_COMPARED: Sequence[str] = (
    "dataset_name", "dataset_config", "max_length", "num_fewshot",
    "clean_latin_rows", "score_normalization", "primary",
)

#: Columns one compare page needs. ``continuations`` is here because the
#: predicted text under char / PMI scoring is that list's entry, not the
#: record's ``pred_text`` (which is the primary scoring's pick).
COMPARE_PAGE_COLUMNS: Sequence[str] = (
    "row_index", "source_config", "question", "continuations",
    "gold_idx", "gold_text", "pred_idx", "pred_text", "pred_idx_char", "pred_idx_pmi",
    "correct", "correct_char", "correct_pmi", "cont_tokens",
    "margin", "decision_margin", "prompt_units", "n_choices",
    "sentinel", "all_sentinel", "hit_cap", "near_tie", "disagree",
)


def _task_name(task: Any) -> str:
    name = str(task or "").strip()
    if not name or not all(c.isalnum() or c in "._-" for c in name) or name.startswith("."):
        raise EvalRowsError(f"not a benchmark name: {task!r}")
    return name


def _dump_of(cell_rel: Any, task: str) -> str:
    """The dump path of one benchmark inside one cell directory."""
    cell = str(cell_rel or "").strip().strip("/")
    if not cell:
        raise EvalRowsError("cell path required")
    return f"{cell}/eval_rows/{task}.parquet"


def compare_tree(repo_root: Path) -> Dict[str, Any]:
    """Every experiment's benchmarks, each with the cells that scored it.

    The single-cell picker goes experiment → cell → benchmark; a comparison has
    to go experiment → benchmark → cells, because the benchmark is what the
    cells must share. Derived from :func:`discover`, so a dump that is still
    being written is listed with the reason it cannot take part rather than
    quietly left out.
    """
    out: List[Dict[str, Any]] = []
    for exp in discover(repo_root)["experiments"]:
        tasks: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        for cell in exp["cells"]:
            for t in cell["tasks"]:
                entry = tasks.setdefault(t["task"], {"task": t["task"], "cells": []})
                reason = None
                if t.get("in_progress"):
                    rows = t.get("n_rows") or 0
                    reason = f"still being evaluated ({rows:,} rows written so far)"
                elif t.get("error"):
                    reason = f"unreadable dump ({t['error']})"
                entry["cells"].append({
                    "cell": cell["cell"],
                    "dir": cell["dir"],
                    "path": t["path"],
                    "tokenizer": cell.get("tokenizer"),
                    "vocab_size": cell.get("vocab_size"),
                    "n_rows": t.get("n_rows"),
                    "available": reason is None,
                    "reason": reason,
                })
        listed = []
        for entry in tasks.values():
            entry["n_cells"] = sum(1 for c in entry["cells"] if c["available"])
            listed.append(entry)
        listed.sort(key=lambda e: e["task"])
        out.append({
            "experiment": exp["experiment"],
            "single": exp["single"],
            "tasks": listed,
            "n_comparable": sum(1 for e in listed if e["n_cells"] >= 2),
        })
    return {"experiments": out}


def _open_cells(repo_root: Path, cells: Sequence[str], task: str):
    """Open one benchmark's dump in each requested cell.

    A cell that cannot take part (no such dump, evaluation still running, file
    unreadable) is set aside with its reason instead of failing the whole
    comparison — that only fails when fewer than two cells are left.
    """
    names = list(dict.fromkeys(str(c).strip().strip("/") for c in cells if str(c).strip()))
    if len(names) < 2:
        raise EvalRowsError("compare needs at least two cells")
    if len(names) > MAX_COMPARE_CELLS:
        raise EvalRowsError(
            f"at most {MAX_COMPARE_CELLS} cells can be compared at once (got {len(names)})"
        )
    opened: "OrderedDict[str, Tuple[str, EvalRowFile]]" = OrderedDict()
    unavailable: List[Dict[str, Any]] = []
    for cell in names:
        rel = _dump_of(cell, task)
        try:
            opened[cell] = (rel, open_dump(repo_root, rel))
        except Exception as e:  # noqa: BLE001 - one bad cell must not hide the others
            path = (Path(repo_root) / rel)
            progress = read_progress(path) if path.exists() else None
            if progress is not None:
                reason = f"still being evaluated ({(progress.get('rows') or 0):,} rows written so far)"
            elif not path.exists():
                reason = f"no {task} dump in this cell"
            else:
                reason = f"{type(e).__name__}: {e}"
            unavailable.append({"cell": cell, "reason": reason})
    if len(opened) < 2:
        detail = "; ".join(f"{u['cell']}: {u['reason']}" for u in unavailable)
        raise EvalRowsError(f"fewer than two cells can be compared on {task} — {detail}")
    return opened, unavailable


def _correct_of(handle: EvalRowFile, scoring: str):
    """Per-row correctness under one normalization, from the filter index."""
    idx = handle.index()
    if scoring in ("char", "pmi"):
        arr = idx.get(f"correct_{scoring}")
        if arr is None or not idx.get(f"has_correct_{scoring}"):
            raise EvalRowsError(
                f"no {scoring}-normalised scores in this dump "
                f"(score_normalization={handle.metadata.get('score_normalization')!r})"
            )
        return arr
    arr = idx.get("correct")
    if arr is None:
        raise EvalRowsError("this dump has no correctness column — it is not an MCQ row dump")
    return arr


def _align_cells(opened) -> Tuple[Any, Dict[str, Any], Dict[str, int], List[str]]:
    """``row_index`` → one file position per cell, over the rows every cell holds.

    ``row_index`` is the join key because it is the only identity a dump
    carries: it is the position of the example in the benchmark's own eval
    list, which is tokenizer-independent. What a cell holds beyond the
    intersection is counted, never dropped in silence.
    """
    import numpy as np

    common = None
    row_ids: Dict[str, Any] = {}
    duplicated: List[str] = []
    for cell, (_, handle) in opened.items():
        arr = handle.index().get("row_index")
        if arr is None:
            raise EvalRowsError(f"{cell}: this dump has no row_index column")
        row_ids[cell] = arr
        uniq = np.unique(arr)
        if len(uniq) != len(arr):
            duplicated.append(cell)
        common = uniq if common is None else np.intersect1d(common, uniq, assume_unique=True)
    positions: Dict[str, Any] = {}
    for cell, arr in row_ids.items():
        order = np.argsort(arr, kind="stable")
        positions[cell] = order[np.searchsorted(arr[order], common)]
    only_in = {
        cell: int(len(np.setdiff1d(np.unique(arr), common, assume_unique=True)))
        for cell, arr in row_ids.items()
    }
    return common, positions, only_in, duplicated


def _compare_warnings(opened, duplicated: Sequence[str], subconfig_mismatch: int) -> List[str]:
    """Everything that makes two columns less comparable than they look."""
    warnings: List[str] = []
    metas = {cell: handle.metadata for cell, (_, handle) in opened.items()}
    for key in _META_COMPARED:
        seen = {cell: meta.get(key) for cell, meta in metas.items()}
        if len({json.dumps(v, sort_keys=True, default=str) for v in seen.values()}) > 1:
            shown = ", ".join(f"{cell}: {v!r}" for cell, v in seen.items())
            warnings.append(f"{key} differs between cells — {shown}")
    units = {cell: meta.get("unit") for cell, meta in metas.items()}
    if len(set(units.values())) > 1:
        shown = ", ".join(f"{cell}: {v}" for cell, v in units.items())
        warnings.append(
            "prompt length is measured in different units — " + shown
            + "; the length columns and the length sort are not comparable across these cells"
        )
    schemas = {cell: meta.get("schema_version") for cell, meta in metas.items()}
    if any(v in (None, 1) for v in schemas.values()):
        old = [cell for cell, v in schemas.items() if v in (None, 1)]
        warnings.append(
            "written before schema 2, so the scored-token counts are absent: " + ", ".join(old)
        )
    if duplicated:
        warnings.append(
            "row_index repeats inside " + ", ".join(duplicated)
            + " — the first occurrence of each is the one compared"
        )
    if subconfig_mismatch:
        warnings.append(
            f"{subconfig_mismatch:,} of the shared rows carry a different sub-config in different "
            "cells — the dumps may not be of the same benchmark rows"
        )
    return warnings


def _has_row_filters(params: Dict[str, Any]) -> bool:
    """Whether any of the single-cell filters is actually set."""
    if str(params.get("outcome") or "all") not in ("all", ""):
        return True
    return bool(
        params.get("subconfig") or params.get("subconfigs") or params.get("flags")
        or str(params.get("q") or "").strip()
        or params.get("gold_idx") not in (None, "")
        or params.get("pred_idx") not in (None, "")
    )


def _compare_plan(repo_root: Path, cells: Sequence[str], task: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Cells opened, rows aligned, filtered and ordered — what a page or an
    export then slices. Kept separate so both read exactly the same selection."""
    import numpy as np

    task = _task_name(task)
    opened, unavailable = _open_cells(repo_root, cells, task)
    names = list(opened)
    anchor = str(params.get("anchor") or "").strip().strip("/") or names[0]
    if anchor not in names:
        raise EvalRowsError(
            f"anchor {anchor!r} is not one of the compared cells ({', '.join(names)})"
        )
    scoring = str(params.get("scoring") or "primary")
    if scoring not in ("primary", "char", "pmi"):
        raise EvalRowsError(f"unknown scoring {scoring!r}")
    match = str(params.get("match") or "any")
    if match not in MATCH_MODES:
        raise EvalRowsError(f"match must be one of {', '.join(MATCH_MODES)}")
    agreement = str(params.get("agreement") or "all")
    if agreement not in AGREEMENTS:
        raise EvalRowsError(f"agreement must be one of {', '.join(AGREEMENTS)}")
    sort = str(params.get("sort") or "row")
    if sort not in COMPARE_SORTS:
        raise EvalRowsError(f"sort must be one of {', '.join(COMPARE_SORTS)}")
    clean_only = str(params.get("clean_only") or "") in ("1", "true", "True", True)

    common, positions, only_in, duplicated = _align_cells(opened)
    n_common = int(len(common))

    correct, dmargin, units, dirty, subconfig = {}, {}, {}, {}, {}
    for cell, (_, handle) in opened.items():
        idx, pos = handle.index(), positions[cell]
        try:
            correct[cell] = _correct_of(handle, scoring)[pos]
        except EvalRowsError as e:
            raise EvalRowsError(f"{cell}: {e}") from None
        dm = idx.get("decision_margin")
        dmargin[cell] = dm[pos] if dm is not None else np.full(n_common, np.nan)
        u = idx.get("prompt_units")
        units[cell] = (
            np.where(u[pos] >= 0, u[pos].astype("float64"), np.nan) if u is not None
            else np.full(n_common, np.nan)
        )
        flags = np.zeros(n_common, dtype=bool)
        for name in ("hit_cap", "sentinel"):
            arr = idx.get(name)
            if arr is not None:
                flags |= arr[pos]
        dirty[cell] = flags
        sub = idx.get("source_config")
        subconfig[cell] = None if sub is None else sub[pos]
    # Same row_index, same sub-config: a cheap check on the join that needs no
    # text read (source_config is a filter column).
    subconfig_mismatch = 0
    if subconfig.get(anchor) is not None:
        for cell, taken in subconfig.items():
            if cell != anchor and taken is not None:
                subconfig_mismatch = max(subconfig_mismatch, int((taken != subconfig[anchor]).sum()))

    # -- filters -----------------------------------------------------------
    keep = np.ones(n_common, dtype=bool)
    if _has_row_filters(params):
        masks = {}
        for cell, (_, handle) in opened.items():
            matching = _selection(handle, {**params, "sort": "row"})[0]
            masks[cell] = np.isin(positions[cell], matching)
        if match == "anchor":
            keep &= masks[anchor]
        elif match == "all":
            keep &= np.logical_and.reduce([masks[c] for c in names])
        else:
            keep &= np.logical_or.reduce([masks[c] for c in names])

    corr = np.stack([correct[c] for c in names])       # (cells, rows)
    n_correct = corr.sum(axis=0)
    ai = names.index(anchor)
    patterns = {
        "agree_correct": n_correct == len(names),
        "agree_wrong": n_correct == 0,
        "disagree": (n_correct > 0) & (n_correct < len(names)),
        "only_anchor": corr[ai] & (n_correct == 1),
        "anchor_wrong": (~corr[ai]) & (n_correct >= 1),
    }
    if clean_only:
        keep &= ~np.logical_or.reduce([dirty[c] for c in names])
    # The agreement filter is applied *last* and kept separate: the per-cell
    # accuracies and the agreement breakdown are reported over the rows the
    # other filters select, so browsing "only the anchor got this right" still
    # shows what the two cells score overall — and every cell of the 2x2 keeps
    # a number to click on.
    selected = np.nonzero(keep)[0]
    kept = np.nonzero(keep & patterns[agreement])[0] if agreement != "all" else selected

    # -- cross-cell axes and ordering -------------------------------------
    dm_stack = np.stack([dmargin[c] for c in names])
    with np.errstate(invalid="ignore"):
        spread = np.nanmax(dm_stack, axis=0) - np.nanmin(dm_stack, axis=0)
    others = [units[c] for c in names if c != anchor]
    anchor_units = units[anchor]
    # A row without a recorded length in some cell has no ratio, rather than a
    # ratio computed from a stand-in value; NaN sorts last in either direction.
    mean_other = np.mean(np.stack(others), axis=0) if others else np.full(n_common, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(anchor_units > 0, mean_other / anchor_units, np.nan)

    if sort != "row" and len(kept):
        rows_key = common[kept].astype("float64")

        def _ordered(values, desc: bool):
            # lexsort's last key is the primary one: rows without the value sort
            # last in either direction, then the value, then row order.
            vals = values[kept]
            bad = np.isnan(vals)
            signed = np.where(bad, 0.0, -vals if desc else vals)
            return kept[np.lexsort((rows_key, signed, bad))]

        if sort == "margin_spread_desc":
            kept = _ordered(spread, True)
        elif sort == "anchor_uncertain":
            kept = _ordered(dmargin[anchor], False)
        elif sort == "anchor_confident":
            kept = _ordered(dmargin[anchor], True)
        elif sort == "units_ratio_desc":
            kept = _ordered(ratio, True)
        elif sort == "disagree":
            mixed = patterns["disagree"][kept].astype("float64")
            sp = spread[kept]
            kept = kept[np.lexsort((common[kept].astype("float64"),
                                    -np.where(np.isnan(sp), 0.0, sp), -mixed))]

    return {
        "task": task, "opened": opened, "names": names, "anchor": anchor, "scoring": scoring,
        "match": match, "agreement": agreement, "sort": sort, "clean_only": clean_only,
        "common": common, "positions": positions, "only_in": only_in, "unavailable": unavailable,
        "kept": kept, "selected": selected, "n_common": n_common, "correct": corr, "n_correct": n_correct,
        "patterns": patterns, "spread": spread, "ratio": ratio,
        "warnings": _compare_warnings(opened, duplicated, subconfig_mismatch),
    }


def _pred_view(rec: Dict[str, Any], scoring: str):
    """The prediction under the selected normalization: (index, text, correct).

    The record's ``pred_text`` is the *primary* scoring's pick, so under char or
    PMI the text has to come from the continuation list.
    """
    idx = rec.get("pred_idx")
    correct = rec.get("correct")
    if scoring in ("char", "pmi"):
        idx = rec.get(f"pred_idx_{scoring}")
        correct = rec.get(f"correct_{scoring}")
    conts = rec.get("continuations") or []
    text = rec.get("pred_text") or ""
    if idx is not None and 0 <= int(idx) < len(conts):
        text = (conts[int(idx)] or "").lstrip()
    return (None if idx is None else int(idx)), text, correct


def _compare_rows(plan: Dict[str, Any], window) -> List[Dict[str, Any]]:
    """Materialise the given rows (indices into the aligned intersection)."""
    names, scoring = plan["names"], plan["scoring"]
    window = [int(i) for i in window]
    if not window:
        return []
    per_cell: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for cell, (_, handle) in plan["opened"].items():
        pos = [int(plan["positions"][cell][i]) for i in window]
        recs = handle.rows(pos, columns=COMPARE_PAGE_COLUMNS)
        per_cell[cell] = {i: rec for i, rec in zip(window, recs)}

    out: List[Dict[str, Any]] = []
    for i in window:
        anchor_rec = per_cell[plan["anchor"]].get(i) or {}
        row: Dict[str, Any] = {
            "row_index": int(plan["common"][i]),
            "source_config": anchor_rec.get("source_config"),
            "question": anchor_rec.get("question"),
            "gold_idx": anchor_rec.get("gold_idx"),
            "gold_text": anchor_rec.get("gold_text"),
            "n_correct": int(plan["n_correct"][i]),
            "n_cells": len(names),
            "agreement": (
                "agree_correct" if plan["patterns"]["agree_correct"][i]
                else "agree_wrong" if plan["patterns"]["agree_wrong"][i] else "disagree"
            ),
            "margin_spread": _finite(plan["spread"][i]),
            "units_ratio": _finite(plan["ratio"][i]),
            "cells": {},
        }
        for cell in names:
            rec = per_cell[cell].get(i)
            if rec is None:
                row["cells"][cell] = {"missing": True}
                continue
            pred_idx, pred_text, correct = _pred_view(rec, scoring)
            gold = rec.get("gold_idx")
            cont = rec.get("cont_tokens") or []
            row["cells"][cell] = {
                "position": rec.get("position"),
                "pred_idx": pred_idx,
                "pred_text": pred_text,
                "correct": correct,
                "margin": rec.get("margin"),
                "decision_margin": rec.get("decision_margin"),
                "prompt_units": rec.get("prompt_units"),
                "n_choices": rec.get("n_choices"),
                "cont_tokens_gold": (
                    int(cont[int(gold)]) if gold is not None and 0 <= int(gold) < len(cont) else None
                ),
                # The join is by row_index; this is the check that it landed on
                # the same question. A false here means the dumps disagree about
                # what row N is, which no cross-cell number would survive.
                "question_match": (rec.get("question") or "") == (anchor_rec.get("question") or ""),
                "flags": {
                    name: bool(rec.get(name)) for name in
                    ("sentinel", "all_sentinel", "hit_cap", "near_tie", "disagree")
                },
            }
        out.append(row)
    return out


def _finite(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v != v else round(v, 6)   # NaN → None


def compare(repo_root: Path, cells: Sequence[str], task: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """The same benchmark rows answered by several cells, side by side.

    ``cells`` are cell directories relative to the repo root; the first, or
    ``params['anchor']``, is the cell every cross-cell axis is measured against.
    Only the ``row_index`` values **every** selected cell holds are shown.
    Filters are the single-cell ones combined under ``match``, plus the
    cross-cell ``agreement`` and ``clean_only`` (drop every row any cell
    truncated — the restriction ``mcq_compare.py`` calls the intersection, so
    the accuracy shown under it is that script's primary number).

    ``total`` counts the rows shown; ``n_selected`` the rows the filters select
    before ``agreement``, which is the population every per-cell ``summary``,
    the ``agreement_counts``, the ``pair`` 2x2 and the sub-config facets
    describe. That split is deliberate: a comparison restricted to the rows
    only one cell got right must still be able to say what the cells score
    overall, and every cell of the 2x2 has to keep a count you can click.
    """
    import numpy as np

    plan = _compare_plan(repo_root, cells, task, params)
    names, kept, selected = plan["names"], plan["kept"], plan["selected"]
    total = int(len(kept))
    n_selected = int(len(selected))

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
    window = kept[(page - 1) * page_size: (page - 1) * page_size + page_size]

    cell_blocks = []
    for cell, (rel, handle) in plan["opened"].items():
        pos = plan["positions"][cell][selected] if n_selected else np.array([], dtype="int64")
        cell_blocks.append({
            "cell": cell,
            "path": rel,
            "n_rows_file": handle.n_rows,
            "metadata": handle.metadata,
            "only_in": plan["only_in"][cell],
            "summary": handle.summary(pos, scoring=plan["scoring"]),
            **{k: v for k, v in _cell_context(Path(repo_root) / cell).items() if k != "reported"},
        })

    # Sub-config facets over the shown rows, from the anchor: the page's
    # sub-config filter is the same control in both modes.
    anchor_handle = plan["opened"][plan["anchor"]][1]
    anchor_pos = plan["positions"][plan["anchor"]][selected] if n_selected else np.array([], dtype="int64")
    subconfigs = _subconfig_facets(anchor_handle, anchor_pos)

    counts = {name: int(mask[selected].sum()) if n_selected else 0
              for name, mask in plan["patterns"].items()}
    hist = [int((plan["n_correct"][selected] == k).sum()) if n_selected else 0
            for k in range(len(names) + 1)]
    pair = None
    if len(names) == 2 and n_selected:
        a, b = plan["correct"][names.index(plan["anchor"])][selected], plan["correct"][
            1 - names.index(plan["anchor"])][selected]
        pair = {
            "anchor": plan["anchor"],
            "other": names[1 - names.index(plan["anchor"])],
            "both": int((a & b).sum()), "only_anchor": int((a & ~b).sum()),
            "only_other": int((~a & b).sum()), "neither": int((~a & ~b).sum()),
        }

    return {
        "task": plan["task"], "anchor": plan["anchor"], "scoring": plan["scoring"],
        "match": plan["match"], "agreement": plan["agreement"], "sort": plan["sort"],
        "clean_only": plan["clean_only"],
        "cells": cell_blocks, "unavailable": plan["unavailable"], "warnings": plan["warnings"],
        "n_common": plan["n_common"], "total": total, "n_selected": n_selected,
        "page": page, "pages": pages, "page_size": page_size,
        "agreement_counts": counts, "correct_hist": hist, "pair": pair,
        "subconfigs": subconfigs,
        "rows": _compare_rows(plan, window),
    }


def compare_row(repo_root: Path, cells: Sequence[str], task: Any, row_index: int) -> Dict[str, Any]:
    """One shared row in full, per cell: the prompt each model received and the
    scores it gave every choice. The payload the side-by-side detail renders."""
    import numpy as np

    task = _task_name(task)
    opened, unavailable = _open_cells(repo_root, cells, task)
    names = list(opened)
    anchor = names[0]
    try:
        wanted = int(row_index)
    except (TypeError, ValueError):
        raise EvalRowsError("row_index must be an integer") from None

    out: Dict[str, Any] = {}
    anchor_prompt = None
    for cell, (rel, handle) in opened.items():
        ids = handle.index()["row_index"]
        hit = np.nonzero(ids == wanted)[0]
        if not len(hit):
            out[cell] = {"missing": True, "reason": f"row {wanted} is not in this cell's dump"}
            continue
        rec = handle.rows([int(hit[0])])[0]
        rec["metadata"] = handle.metadata
        rec["path"] = rel
        if cell == anchor:
            anchor_prompt = rec.get("prompt")
        out[cell] = rec
    for cell, rec in out.items():
        if not rec.get("missing"):
            rec["prompt_match"] = (rec.get("prompt") or "") == (anchor_prompt or "")
    return {"task": task, "row_index": wanted, "anchor": anchor,
            "cells": out, "unavailable": unavailable}


#: Shared columns of a compare export, then one group per cell.
_COMPARE_EXPORT_SHARED: Sequence[str] = (
    "row_index", "source_config", "question", "gold_text",
    "n_correct", "agreement", "margin_spread", "units_ratio",
)
_COMPARE_EXPORT_PER_CELL: Sequence[str] = (
    "pred_text", "correct", "margin", "decision_margin", "prompt_units",
    "cont_tokens_gold", "all_sentinel", "hit_cap", "near_tie",
)


def export_compare_csv(repo_root: Path, cells: Sequence[str], task: Any,
                       params: Dict[str, Any]) -> Tuple[str, str]:
    """The current comparison as CSV: shared columns, then ``<cell>.<field>``."""
    plan = _compare_plan(repo_root, cells, task, params)
    kept = plan["kept"][:MAX_EXPORT_ROWS]
    names = plan["names"]
    # Column groups are prefixed with the cell's own name, not its whole path —
    # a spreadsheet header is read by a person. Two cells of different
    # experiments can share a name, so a collision keeps the full path.
    short = {cell: Path(cell).name for cell in names}
    if len(set(short.values())) != len(names):
        short = {cell: cell for cell in names}
    fields = list(_COMPARE_EXPORT_SHARED)
    for cell in names:
        fields += [f"{short[cell]}.{f}" for f in _COMPARE_EXPORT_PER_CELL]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in _compare_rows(plan, kept):
        flat = {k: row.get(k) for k in _COMPARE_EXPORT_SHARED}
        for cell in names:
            block = row["cells"].get(cell) or {}
            flags = block.get("flags") or {}
            for f in _COMPARE_EXPORT_PER_CELL:
                flat[f"{short[cell]}.{f}"] = flags[f] if f in flags else block.get(f)
        writer.writerow(flat)
    experiment = Path(names[0]).parent.name or "experiment"
    return f"{experiment}_{plan['task']}_compare.csv", buf.getvalue()
