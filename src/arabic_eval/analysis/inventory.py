"""What exists: experiments, cells, and the facts that decide whether two cells compare.

A *cell* is a folder under ``outputs/experiments`` that holds an
``all_metrics.json`` or eval dumps — one tokenizer × model × training run, or an
eval-only re-run of a checkpoint. Its id is its path relative to
``outputs/experiments`` (``qwen_native_vs_araroopat/araroopat_3phase_v5_distill``);
every helper also accepts the bare folder name when it is unique.
Folders under ``_superseded/`` are skipped (moved there when a rule changed).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from arabic_eval.analysis._root import experiments_root, repo_root

SUPERSEDED = "_superseded"
MCQ_TASKS: Tuple[str, ...] = ("acva", "alghafa", "arabic_exam", "culture_arabic_mmlu")
FREEFORM_TASK = "freeform_cidar"
PHASES: Tuple[str, ...] = ("embedding_alignment", "warmup", "sft")
_PHASE_SHORT = {"embedding_alignment": "P1", "warmup": "P2", "sft": "P3"}
#: Sub-folders of a cell that never hold cells (checkpoints, caches, dumps).
_PRUNE = {"training", "data", "eval_rows", "freeform_judge", "failure_reports", "unk_reports",
          "freeform_rating", "__pycache__", SUPERSEDED}
_CACHE_TTL = 30.0


class CellNotFound(LookupError):
    """No cell (or more than one) matches a reference."""


@dataclass(frozen=True)
class Cell:
    id: str            # path relative to outputs/experiments
    path: Path         # absolute folder
    experiment: str    # parent path ('' for a top-level single-run folder)
    name: str          # folder name

    def __str__(self) -> str:
        return self.id


# --------------------------------------------------------------------------
# small readers (mtime-keyed caches: the helpers are called many times per step)
# --------------------------------------------------------------------------

_JSON: Dict[Tuple[str, int], Any] = {}
_META: Dict[Tuple[str, int], Tuple[Dict[str, Any], int]] = {}


def read_json(path: Path) -> Optional[Any]:
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns)
    if key not in _JSON:
        try:
            _JSON[key] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _JSON[key] = None
    return _JSON[key]


def parquet_meta(path: Path) -> Tuple[Dict[str, Any], int]:
    """(``arabic_eval`` schema metadata, row count) from the footer only."""
    st = path.stat()
    key = (str(path), st.st_mtime_ns)
    if key not in _META:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        raw = (pf.schema_arrow.metadata or {}).get(b"arabic_eval")
        try:
            meta = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            meta = {}
        _META[key] = (meta, pf.metadata.num_rows)
    return _META[key]


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

_CELLS: Dict[str, Tuple[float, List[Cell]]] = {}


def clear_cache() -> None:
    _CELLS.clear()


def _walk(root: Path, include_superseded: bool) -> List[Cell]:
    out: List[Cell] = []
    if not root.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        is_cell = d != root and ("all_metrics.json" in filenames or "eval_rows" in dirnames
                                 or "freeform_judge" in dirnames)
        prune = set(_PRUNE)
        if include_superseded:
            prune.discard(SUPERSEDED)
        dirnames[:] = sorted(n for n in dirnames if n not in prune)
        if is_cell:
            rel = d.relative_to(root)
            parent = rel.parent.as_posix()
            out.append(Cell(id=rel.as_posix(), path=d, experiment="" if parent == "." else parent, name=rel.name))
    return sorted(out, key=lambda c: c.id)


def cells(include_superseded: bool = False) -> List[Cell]:
    """Every cell under the experiments root (cached for 30 s)."""
    root = experiments_root()
    key = f"{root}|{include_superseded}"
    hit = _CELLS.get(key)
    if hit and time.monotonic() - hit[0] < _CACHE_TTL:
        return hit[1]
    found = _walk(root, include_superseded)
    _CELLS[key] = (time.monotonic(), found)
    return found


def resolve(ref: Any) -> Cell:
    """A cell from its id, its unique folder name, or a path to its folder.

    Raises ``CellNotFound`` with the candidates when the name is ambiguous or unknown."""
    if isinstance(ref, Cell):
        return ref
    root = experiments_root()
    s = str(ref).strip().rstrip("/")
    if not s:
        raise CellNotFound("empty cell reference")
    # A path: absolute, repo-relative ("outputs/experiments/…") or relative to the root.
    for cand in (Path(s), repo_root() / s, root / s):
        try:
            p = cand.resolve()
            rel = p.relative_to(root.resolve())
        except (ValueError, OSError):
            continue
        if p.is_dir() and ((p / "all_metrics.json").exists() or (p / "eval_rows").is_dir()
                           or (p / "freeform_judge").is_dir()):
            parent = rel.parent.as_posix()
            return Cell(id=rel.as_posix(), path=p, experiment="" if parent == "." else parent, name=rel.name)
    matches = [c for c in cells() if c.name == s or c.id == s or c.id.endswith("/" + s)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        near = [c.id for c in cells() if s.lower() in c.id.lower()][:12]
        hint = f" — did you mean one of {near}?" if near else " — ae.catalog() lists every cell"
        raise CellNotFound(f"no cell {s!r}{hint}")
    raise CellNotFound(f"{s!r} is ambiguous: {[c.id for c in matches]} — pass the full id")


def resolve_many(refs: Any) -> List[Cell]:
    if isinstance(refs, (str, Cell, Path)):
        refs = [refs]
    return [resolve(r) for r in refs]


def short_names(cs: Sequence[Cell]) -> List[str]:
    """Folder names, or full ids where two cells share a folder name."""
    names = [c.name for c in cs]
    return [c.id if names.count(c.name) > 1 else c.name for c in cs]


# --------------------------------------------------------------------------
# facts per cell
# --------------------------------------------------------------------------

def _parent_checkpoint(name_or_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """(cell id, phase) when the model was loaded from another cell's phase checkpoint."""
    if not name_or_path or "/training/" not in name_or_path:
        return None, None
    head, _, phase = name_or_path.partition("/training/")
    marker = "outputs/experiments/"
    if marker in head:
        head = head.split(marker, 1)[1]
    return head.strip("/"), phase.strip("/") or None


def _mixture_text(phase_cfg: Dict[str, Any]) -> Optional[str]:
    mix = (phase_cfg or {}).get("mixture")
    if not mix:
        return None
    shares = mix.get("shares") or {}
    order = [k for k in ("extractive", "mcq", "free_form") if k in shares] + sorted(set(shares) - {"extractive", "mcq", "free_form"})
    pct = "/".join(f"{round(100 * float(shares[k]))}" for k in order)
    return f"{mix.get('total_examples')} @ {pct} ({'/'.join(order)})"


def _teacher_overlay(cfg: Dict[str, Any]) -> Optional[str]:
    params = ((cfg.get("training") or {}).get("corpus_params")) or {}
    files = sorted({Path(v["teacher_answers"]).name for v in params.values()
                    if isinstance(v, dict) and v.get("teacher_answers")})
    return ", ".join(files) or None


def _dumps(cell: Cell) -> Dict[str, Path]:
    d = cell.path / "eval_rows"
    return {p.stem: p for p in sorted(d.glob("*.parquet"))} if d.is_dir() else {}


def _judges(cell: Cell) -> List[str]:
    d = cell.path / "freeform_judge"
    return [p.stem for p in sorted(d.glob("*.parquet"))] if d.is_dir() else []


def _task_rule(task: str, meta: Dict[str, Any]) -> str:
    if task == FREEFORM_TASK:
        dec = meta.get("decoding") or {}
        hp = str(meta.get("heldout_path") or "")
        # The prompt set: the 250 held-out test prompts, or the teacher bake-off's cidar-dev prompts.
        pset = ("test " if "freeform_cidar_heldout" in hp else "dev " if "prompts_dev" in hp
                else (Path(hp).stem + " ") if hp else "")
        s = f"{pset}{dec.get('max_output_chars', meta.get('max_output_chars', '?'))}ch {dec.get('sampling', 'greedy')}"
        rp = dec.get("repetition_penalty")
        if rp not in (None, 1.0):
            s += f" rp{rp}"
        if dec.get("no_repeat_ngram_size"):
            s += f" nrng{dec['no_repeat_ngram_size']}"
        return s
    s = f"len{meta.get('max_length', '?')} {meta.get('num_fewshot', '?')}shot s{meta.get('schema_version', 1)}"
    if meta.get("label_rotation"):
        s += f" rot{meta['label_rotation']}"
    if meta.get("rows_file"):
        s += " subset"
    return s


def cell_facts(ref: Any) -> Dict[str, Any]:
    """Everything the catalog knows about one cell (a flat-ish dict)."""
    cell = resolve(ref)
    cfg = read_json(cell.path / "config.json") or {}
    am = read_json(cell.path / "all_metrics.json") or {}
    amc = am.get("config") or {}
    tok = cfg.get("tokenizer") or {}
    model = cfg.get("model") or {}
    base_model = model.get("name_or_path") or amc.get("model")
    parent, parent_phase = _parent_checkpoint(base_model)
    training = am.get("training") or {}
    phases_cfg = ((cfg.get("training") or {}).get("phases")) or {}
    trained = [_PHASE_SHORT[p] for p in PHASES if (training.get(p) or {}).get("status") == "ok"]
    downstream = am.get("downstream") or {}
    dumps = _dumps(cell)
    tasks: Dict[str, Dict[str, Any]] = {}
    for task, path in dumps.items():
        try:
            meta, n = parquet_meta(path)
        except Exception as e:  # noqa: BLE001 - a dump still being written has no footer
            tasks[task] = {"rows": None, "error": f"{type(e).__name__}: unreadable (still being written?)"}
            continue
        tasks[task] = {"rows": n, "rule": _task_rule(task, meta), "schema_version": meta.get("schema_version", 1)}
    ff = downstream.get(FREEFORM_TASK) or {}
    if ff.get("status") == "generation_unsupported" and FREEFORM_TASK not in tasks:
        tasks[FREEFORM_TASK] = {"rows": 0, "rule": "generation_unsupported"}
    judges = _judges(cell)
    judge_means = {j: ((ff.get("judge") or {}).get(j) or {}).get("score_mean") for j in judges}
    sft = training.get("sft") or {}
    mtime = None
    for f in ("all_metrics.json", "config.json"):
        p = cell.path / f
        if p.exists():
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            break
    return {
        "cell": cell.id,
        "experiment": cell.experiment,
        "name": cell.name,
        "tokenizer": tok.get("type") or amc.get("tokenizer"),
        "vocab_size": (am.get("intrinsic") or {}).get("vocab_size") or tok.get("vocab_size"),
        "model_type": model.get("type"),
        "base_model": base_model,
        "parent_cell": parent,
        "parent_phase": parent_phase,
        "trained": "+".join(trained) if trained else ("eval-only" if training or am else None),
        "sft_mixture": _mixture_text(phases_cfg.get("sft") or {}) if "P3" in trained else None,
        "sft_steps_done": sft.get("steps_completed"),
        "sft_early_stopped": sft.get("early_stopped"),
        "teacher_answers": _teacher_overlay(cfg) if "P3" in trained else None,
        "embedding_init": ((model.get("embedding_init") or {}).get("method")) if ("P1" in trained or "P2" in trained) else None,
        "mcq_tasks": [t for t in MCQ_TASKS if t in tasks] + sorted(t for t in tasks if t not in MCQ_TASKS and t != FREEFORM_TASK),
        "freeform": FREEFORM_TASK in tasks,
        "judges": judges,
        "judge_means": judge_means,
        "tasks": tasks,
        "score_normalization": (cfg.get("evaluation") or {}).get("score_normalization"),
        "has_metrics": bool(am),
        "updated": mtime,
        "description": cfg.get("description"),
    }


def catalog(experiment: Optional[str] = None, *, full: bool = False, include_superseded: bool = False):
    """One row per cell: tokenizer, what was trained, which evals and judges exist, and the eval rules.

    ``experiment`` keeps the cells whose id starts with it. ``full=True`` adds
    every fact (parent checkpoint, mixture, teacher file, embedding init, …).
    The ``rules`` column is what decides comparability: two free-form cells
    compare only under the same character budget and decoding; two MCQ cells
    under the same max_length / few-shot / label rotation."""
    import pandas as pd
    rows = []
    for c in cells(include_superseded):
        if experiment and not (c.id == experiment or c.id.startswith(experiment.rstrip("/") + "/")):
            continue
        f = cell_facts(c)
        rules = "; ".join(f"{t}: {v.get('rule')}" for t, v in f["tasks"].items() if v.get("rule"))
        row = {
            "cell": f["cell"], "tokenizer": f["tokenizer"], "vocab": f["vocab_size"], "trained": f["trained"],
            "parent": (f"{f['parent_cell']}:{f['parent_phase']}" if f["parent_cell"] else None),
            "mcq": ",".join(f["mcq_tasks"]) or None, "freeform": f["freeform"],
            "judges": ",".join(f["judges"]) or None, "rules": rules or None,
        }
        if full:
            row.update({k: f[k] for k in ("experiment", "model_type", "base_model", "sft_mixture", "sft_steps_done",
                                          "sft_early_stopped", "teacher_answers", "embedding_init",
                                          "score_normalization", "updated", "description")})
        rows.append(row)
    return pd.DataFrame(rows).set_index("cell") if rows else pd.DataFrame(columns=["cell"]).set_index("cell")


def experiments():
    """The experiment folders with their number of cells."""
    import pandas as pd
    counts: Dict[str, int] = {}
    for c in cells():
        top = c.id.split("/", 1)[0]
        counts[top] = counts.get(top, 0) + 1
    return pd.DataFrame(sorted(counts.items()), columns=["experiment", "cells"]).set_index("experiment")


def files(ref: Any, *, include_checkpoints: bool = False):
    """The files of a cell (relative path, size in KB), checkpoints left out by default."""
    import pandas as pd
    cell = resolve(ref)
    rows = []
    for dirpath, dirnames, filenames in os.walk(cell.path):
        d = Path(dirpath)
        if not include_checkpoints:
            dirnames[:] = [n for n in dirnames if not (d == cell.path and n == "training")]
        for fn in sorted(filenames):
            p = d / fn
            rows.append({"file": p.relative_to(cell.path).as_posix(), "kb": round(p.stat().st_size / 1024, 1)})
    return pd.DataFrame(rows)
