"""Free-form generations + judge verdicts for the console's *Free-form* tab.

A cell's ``eval_rows/freeform_cidar.parquet`` (250 rows: instruction,
reference, generation, per-row metrics) joined by ``id`` with every
``freeform_judge/<judge>.parquet`` next to it. Files are small, so a cell is
read whole and cached by (path, mtime). ``discover`` lists every cell that
has generations — and the cells whose task record says
``generation_unsupported``, so a sweep shows all its variants; ``query``
filters / sorts / pages one cell; ``row`` returns one full record plus the
same prompt's generation in every sibling cell of the experiment (the
cross-variant view a tokenizer comparison is read by).
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from arabic_eval.evaluation.eval_rows import is_superseded

TASK = "freeform_cidar"
GENERATIONS_REL = Path("eval_rows") / f"{TASK}.parquet"
JUDGE_DIR = "freeform_judge"
STOP_REASONS = ("eos", "marker", "loop", "cap")     # mirrors generation.STOP_REASONS (torch-free here)
SORTS = ("row", "score_asc", "score_desc", "chrf_asc", "chrf_desc", "disagree", "gen_len_desc", "gen_len_asc")
ROW_FLAGS = ("degenerate", "empty", "latin", "hit_cap", "hit_loop", "char_truncated")
CLIP = 240

_CACHE: Dict[Tuple[str, int], Any] = {}


class FreeformError(ValueError):
    pass


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _outputs(repo_root: Path) -> Path:
    return Path(repo_root) / "outputs" / "experiments"


def _resolve_cell(repo_root: Path, rel: str) -> Path:
    if not rel:
        raise FreeformError("cell required")
    p = (Path(repo_root) / rel).resolve()
    if _outputs(repo_root).resolve() not in p.parents:
        raise FreeformError("only cells under outputs/experiments can be read")
    if not p.is_dir():
        raise FreeformError(f"no such cell: {rel}")
    return p


def _read_table(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    key = (str(path), path.stat().st_mtime_ns)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    meta: Dict[str, Any] = {}
    if t.schema.metadata and b"arabic_eval" in t.schema.metadata:
        meta = json.loads(t.schema.metadata[b"arabic_eval"].decode("utf-8"))
    rows = t.to_pylist()
    if len(_CACHE) > 64:
        _CACHE.clear()
    _CACHE[key] = (rows, meta)
    return rows, meta


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def judge_files(cell_dir: Path) -> List[Path]:
    d = cell_dir / JUDGE_DIR
    return sorted(d.glob("*.parquet")) if d.is_dir() else []


def load_cell(repo_root: Path, rel: str) -> Dict[str, Any]:
    """Generations joined with every judge file: ``rows[i]["judges"][name]``."""
    cell = _resolve_cell(repo_root, rel)
    gpath = cell / GENERATIONS_REL
    if not gpath.exists():
        raise FreeformError(f"{rel} has no free-form generations")
    rows, meta = _read_table(gpath)
    rows = [dict(r) for r in rows]
    judges: List[Dict[str, Any]] = []
    for jf in judge_files(cell):
        vrows, jmeta = _read_table(jf)
        by_id = {v["id"]: v for v in vrows}
        name = jf.stem
        judges.append({"name": name, "model": ((jmeta.get("judge") or {}).get("model")), "n": len(vrows)})
        for r in rows:
            v = by_id.get(r["id"])
            r.setdefault("judges", {})[name] = None if v is None else {
                "score": v.get("score"), "correctness": v.get("correctness"), "fluency": v.get("fluency"),
                "instruction_following": v.get("instruction_following"),
                "flags": [f for f in (v.get("flags") or "").split(",") if f],
                "rationale": v.get("rationale"), "parse_ok": bool(v.get("parse_ok")),
            }
    for r in rows:
        r.setdefault("judges", {})
    return {"cell": rel, "dir": cell, "rows": rows, "metadata": meta, "judges": judges}


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def _cell_summary(cell_dir: Path) -> Dict[str, Any]:
    cfg = _read_json(cell_dir / "config.json") or {}
    am = _read_json(cell_dir / "all_metrics.json") or {}
    task = ((am.get("downstream") or {}).get(TASK)) or {}
    tok = cfg.get("tokenizer") or {}
    keep = ("chrf", "bertscore_f1", "degenerate_rate", "empty_rate", "latin_rate", "hit_cap_rate", "loop_stop_rate",
            "reference_roundtrip_chrf", "gen_chars_per_sec", "mean_gen_chars", "token_cap", "max_output_chars", "num_samples")
    return {
        "tokenizer": tok.get("type"), "vocab_size": tok.get("vocab_size"),
        "model": (cfg.get("model") or {}).get("name_or_path"),
        "status": task.get("status"), "reason": task.get("reason"),
        "metrics": {k: task.get(k) for k in keep},
        "judges": [{"name": n, "model": s.get("model"), "score_mean": s.get("score_mean"), "n": s.get("n"),
                    "any_flag_rate": s.get("any_flag_rate"), "vs_baseline": s.get("vs_baseline")}
                   for n, s in sorted((task.get("judge") or {}).items())],
        "judge_agreement": task.get("judge_agreement") or {},
    }


def discover(repo_root: Path) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    base = _outputs(repo_root)
    experiments: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    if not base.is_dir():
        return {"experiments": []}
    cells = {p.parent.parent for p in base.glob(f"**/eval_rows/{TASK}.parquet") if not is_superseded(p, base)}
    for am in base.glob("**/all_metrics.json"):           # unsupported cells (no dump) still listed
        if is_superseded(am, base):
            continue
        d = _read_json(am) or {}
        if ((d.get("downstream") or {}).get(TASK) or {}).get("status") == "generation_unsupported":
            cells.add(am.parent)
    for cell_dir in sorted(cells):
        rel_parts = cell_dir.relative_to(base).parts
        if len(rel_parts) == 1:
            experiment, cell, single = rel_parts[0], rel_parts[0], True
        else:
            experiment, cell, single = str(Path(*rel_parts[:-1])), rel_parts[-1], False
        gpath = cell_dir / GENERATIONS_REL
        entry: Dict[str, Any] = {"cell": cell, "dir": str(cell_dir.relative_to(repo_root)),
                                 "has_generations": gpath.exists(), "n_rows": None, **_cell_summary(cell_dir)}
        if gpath.exists():
            try:
                rows, meta = _read_table(gpath)
                entry["n_rows"] = len(rows)
                entry["decoding"] = meta.get("decoding")
                entry["token_cap"] = meta.get("token_cap")
            except Exception as e:  # noqa: BLE001
                entry["error"] = f"{type(e).__name__}: {e}"
        exp = experiments.setdefault(experiment, {"experiment": experiment, "single": single,
                                                  "dir": str((base / experiment).relative_to(repo_root)), "cells": []})
        exp["cells"].append(entry)
    out = list(experiments.values())
    for exp in out:
        exp["cells"].sort(key=lambda c: (not c["has_generations"], c["cell"]))
        exp["judges"] = sorted({j["name"] for c in exp["cells"] for j in c["judges"]})
    out.sort(key=lambda e: e["experiment"])
    return {"experiments": out}


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

def _to_int(v: Any) -> Optional[int]:
    try:
        return None if v in (None, "") else int(v)
    except (TypeError, ValueError):
        return None


def _truthy(v: Any) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on")


def _judge_score(r: Dict[str, Any], judge: Optional[str]) -> Optional[float]:
    if not judge:
        return None
    v = (r.get("judges") or {}).get(judge)
    return None if not v or v.get("score") is None else float(v["score"])


def _disagreement(r: Dict[str, Any]) -> Optional[float]:
    s = [v["score"] for v in (r.get("judges") or {}).values() if v and v.get("score") is not None]
    return None if len(s) < 2 else float(max(s) - min(s))


def _compact(r: Dict[str, Any]) -> Dict[str, Any]:
    gen = r.get("generation") or ""
    return {
        "id": r["id"], "stratum": r.get("stratum"), "instruction": r.get("instruction"),
        "generation": gen[:CLIP] + ("…" if len(gen) > CLIP else ""),
        "gen_chars": r.get("gen_chars"), "gen_tokens": r.get("gen_tokens"), "ref_chars": r.get("ref_chars"),
        "chrf": r.get("chrf"), "bertscore_f1": r.get("bertscore_f1"), "stop_reason": r.get("stop_reason"),
        **{f: bool(r.get(f)) for f in ROW_FLAGS},
        "judges": {n: (None if v is None else {"score": v["score"], "flags": v["flags"], "parse_ok": v["parse_ok"]})
                   for n, v in (r.get("judges") or {}).items()},
        "disagreement": _disagreement(r),
    }


def query(repo_root: Path, rel: str, params: Dict[str, Any]) -> Dict[str, Any]:
    data = load_cell(repo_root, rel)
    rows = data["rows"]
    judges = [j["name"] for j in data["judges"]]
    judge = params.get("judge") or (judges[0] if judges else None)
    if judge and judge not in judges:
        raise FreeformError(f"unknown judge {judge!r}; this cell has {judges}")
    sel = rows
    smin, smax = _to_int(params.get("score_min")), _to_int(params.get("score_max"))
    if judge and (smin is not None or smax is not None):
        sel = [r for r in sel if (s := _judge_score(r, judge)) is not None
               and (smin is None or s >= smin) and (smax is None or s <= smax)]
    jflag = params.get("jflag")
    if judge and jflag:
        sel = [r for r in sel if (v := (r.get("judges") or {}).get(judge)) and
               (v["flags"] if jflag == "any" else jflag in v["flags"])]
    if params.get("unparsed") and judge:
        sel = [r for r in sel if (v := (r.get("judges") or {}).get(judge)) is None or not v["parse_ok"]]
    stop = params.get("stop")
    if stop:
        if stop not in STOP_REASONS:
            raise FreeformError(f"stop must be one of {STOP_REASONS}")
        sel = [r for r in sel if r.get("stop_reason") == stop]
    for f in ROW_FLAGS:
        if f in params and params[f] != "":
            want = _truthy(params[f])
            sel = [r for r in sel if bool(r.get(f)) == want]
    if params.get("stratum"):
        sel = [r for r in sel if r.get("stratum") == params["stratum"]]
    q = (params.get("q") or "").strip()
    if q:
        sel = [r for r in sel if q in (r.get("instruction") or "") or q in (r.get("generation") or "")
               or q in (r.get("reference") or "") or q == r["id"]]
    sort = params.get("sort") or "row"
    if sort not in SORTS:
        raise FreeformError(f"sort must be one of {SORTS}")
    big = float("inf")
    if sort == "score_asc":
        sel = sorted(sel, key=lambda r: (_judge_score(r, judge) if _judge_score(r, judge) is not None else big, r["id"]))
    elif sort == "score_desc":
        sel = sorted(sel, key=lambda r: (-(_judge_score(r, judge) if _judge_score(r, judge) is not None else -big), r["id"]))
    elif sort == "chrf_asc":
        sel = sorted(sel, key=lambda r: (r.get("chrf") if r.get("chrf") is not None else big, r["id"]))
    elif sort == "chrf_desc":
        sel = sorted(sel, key=lambda r: (-(r.get("chrf") or 0.0), r["id"]))
    elif sort == "disagree":
        sel = sorted(sel, key=lambda r: (-(_disagreement(r) or 0.0), r["id"]))
    elif sort == "gen_len_desc":
        sel = sorted(sel, key=lambda r: (-(r.get("gen_chars") or 0), r["id"]))
    elif sort == "gen_len_asc":
        sel = sorted(sel, key=lambda r: (r.get("gen_chars") or 0, r["id"]))
    page_size = max(1, min(_to_int(params.get("page_size")) or 50, 500))
    total = len(sel)
    pages = max(1, math.ceil(total / page_size))
    page = max(1, min(_to_int(params.get("page")) or 1, pages))
    chunk = sel[(page - 1) * page_size: page * page_size]
    return {
        "cell": rel, "n_rows_file": len(rows), "total": total, "page": page, "pages": pages, "page_size": page_size,
        "judge": judge, "judges": data["judges"], "metadata": data["metadata"],
        "rows": [_compact(r) for r in chunk], "summary": summarize(sel, judges),
    }


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    v = [x for x in xs if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def summarize(rows: Sequence[Dict[str, Any]], judges: Sequence[str]) -> Dict[str, Any]:
    n = len(rows)
    out: Dict[str, Any] = {
        "n_rows": n,
        "chrf_mean": _mean([r.get("chrf") for r in rows]),
        "bertscore_mean": _mean([r.get("bertscore_f1") for r in rows]),
        "gen_chars_mean": _mean([r.get("gen_chars") for r in rows]),
        "stop": {s: sum(1 for r in rows if r.get("stop_reason") == s) for s in STOP_REASONS},
        **{f: sum(1 for r in rows if r.get(f)) for f in ROW_FLAGS},
        "judges": {},
    }
    for j in judges:
        scores = [s for r in rows if (s := _judge_score(r, j)) is not None]
        flagged = sum(1 for r in rows if (v := (r.get("judges") or {}).get(j)) and v["flags"])
        verdicts = [(r.get("judges") or {}).get(j) for r in rows]
        out["judges"][j] = {"mean": _mean(scores), "n": len(scores),
                            "hist": [sum(1 for s in scores if int(s) == k) for k in range(1, 6)],
                            "flagged": flagged,
                            "unparsed": sum(1 for v in verdicts if v is not None and not v["parse_ok"]),
                            "missing": sum(1 for v in verdicts if v is None)}     # no verdict (e.g. judged with --limit)
    return out


# ---------------------------------------------------------------------------
# one row, across cells
# ---------------------------------------------------------------------------

def row(repo_root: Path, rel: str, row_id: str) -> Dict[str, Any]:
    data = load_cell(repo_root, rel)
    rec = next((r for r in data["rows"] if r["id"] == row_id), None)
    if rec is None:
        raise FreeformError(f"no row {row_id!r} in {rel}")
    cell_dir: Path = data["dir"]
    across: List[Dict[str, Any]] = []
    for sib in sorted(p for p in cell_dir.parent.iterdir() if p.is_dir() and (p / GENERATIONS_REL).exists()):
        if sib == cell_dir:
            continue
        try:
            sd = load_cell(repo_root, str(sib.relative_to(Path(repo_root))))
        except FreeformError:
            continue
        srow = next((r for r in sd["rows"] if r["id"] == row_id), None)
        if srow is None:
            continue
        across.append({"cell": sib.name, "tokenizer": _cell_summary(sib).get("tokenizer"),
                       "generation": srow.get("generation"), "chrf": srow.get("chrf"),
                       "bertscore_f1": srow.get("bertscore_f1"), "stop_reason": srow.get("stop_reason"),
                       **{f: bool(srow.get(f)) for f in ROW_FLAGS},
                       "judges": {n: (None if v is None else v["score"]) for n, v in (srow.get("judges") or {}).items()}})
    return {"cell": rel, "row": rec, "across": across, "metadata": data["metadata"]}
