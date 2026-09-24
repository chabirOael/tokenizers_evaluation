"""Free-form generations + judge verdicts for the console's *Free-form* tab.

A cell's ``eval_rows/freeform_cidar.parquet`` (250 rows: instruction,
reference, generation, per-row metrics) joined by ``id`` with every
``freeform_judge/<judge>.parquet`` next to it. Files are small, so a cell is
read whole and cached by (path, mtime). ``discover`` lists every cell that
has generations — and the cells whose task record says
``generation_unsupported``, so a sweep shows all its variants; ``query``
filters / sorts / pages one cell; ``row`` returns one full record plus the
same prompt's generation in the sibling cells of the experiment; ``compare``
puts several cells' answers to the same prompts side by side (the
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
# compare mode adds the cross-cell orders: the signed difference to the anchor cell, the spread
# over every selected cell (max − min, the convention `disagree` uses over judges), and length.
COMPARE_SORTS = SORTS + ("delta_desc", "delta_asc", "spread_desc", "chrf_delta_desc", "chrf_delta_asc",
                         "len_ratio_desc", "len_ratio_asc")
MATCH_MODES = ("any", "all", "anchor")
MAX_COMPARE_CELLS = 6
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

def row(repo_root: Path, rel: str, row_id: str, cells: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """One full record, plus the same prompt in the sibling cells. ``cells``
    (names, not paths) restricts the walk — an experiment with 18 cells would
    otherwise read 18 dumps to fill a table nobody asked for."""
    data = load_cell(repo_root, rel)
    rec = next((r for r in data["rows"] if r["id"] == row_id), None)
    if rec is None:
        raise FreeformError(f"no row {row_id!r} in {rel}")
    cell_dir: Path = data["dir"]
    want = {str(c) for c in cells} if cells else None
    across: List[Dict[str, Any]] = []
    for sib in sorted(p for p in cell_dir.parent.iterdir() if p.is_dir() and (p / GENERATIONS_REL).exists()):
        if sib == cell_dir or (want is not None and sib.name not in want):
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
                       # the whole verdict, not just the score: the compare panes show every cell's
                       # rationale and flags, and a reader must not get more detail for one column
                       "judges": {n: v for n, v in (srow.get("judges") or {}).items()}})
    return {"cell": rel, "row": rec, "across": across, "metadata": data["metadata"]}


# ---------------------------------------------------------------------------
# several cells, same prompts (compare mode)
# ---------------------------------------------------------------------------

def _row_matches(r: Dict[str, Any], params: Dict[str, Any], judge: Optional[str]) -> bool:
    """Whether one cell's record passes the per-row filters (the same rules
    ``query`` applies, expressed per record so compare can ask them of one
    cell, of any, or of all)."""
    smin, smax = _to_int(params.get("score_min")), _to_int(params.get("score_max"))
    if smin is not None or smax is not None:
        s = _judge_score(r, judge)
        if s is None or (smin is not None and s < smin) or (smax is not None and s > smax):
            return False
    jflag = params.get("jflag")
    if jflag:
        v = (r.get("judges") or {}).get(judge)
        if not v or not (v["flags"] if jflag == "any" else jflag in v["flags"]):
            return False
    if params.get("unparsed"):
        v = (r.get("judges") or {}).get(judge)
        if v is not None and v["parse_ok"]:
            return False
    stop = params.get("stop")
    if stop and r.get("stop_reason") != stop:
        return False
    for f in ROW_FLAGS:
        if f in params and params[f] != "" and bool(r.get(f)) != _truthy(params[f]):
            return False
    q = (params.get("q") or "").strip()
    if q and not (q in (r.get("instruction") or "") or q in (r.get("generation") or "")
                  or q in (r.get("reference") or "") or q == r["id"]):
        return False
    return True


def _compact_with_verdict(r: Dict[str, Any], judge: Optional[str]) -> Dict[str, Any]:
    """A table record plus the selected judge's **whole** verdict for this cell
    — the rationale is the reason a reader opens a comparison at all, so it
    travels with the row instead of being fetched per cell on expand."""
    out = _compact(r)
    v = (r.get("judges") or {}).get(judge) if judge else None
    out["verdict"] = None if v is None else {
        "score": v.get("score"), "correctness": v.get("correctness"), "fluency": v.get("fluency"),
        "instruction_following": v.get("instruction_following"),
        "flags": v.get("flags") or [], "rationale": v.get("rationale"), "parse_ok": v.get("parse_ok"),
    }
    return out


def _compare_summary(rows: Sequence[Dict[str, Any]], judge: Optional[str]) -> Dict[str, Any]:
    """One cell's numbers over the prompts currently shown. Descriptive only —
    the inferential comparison (bootstrap CI, win/tie/loss) is the judge
    stage's ``vs_baseline`` and ``scripts/judge/paired_compare.py``."""
    n = len(rows)
    scores = [s for r in rows if (s := _judge_score(r, judge)) is not None]
    return {
        "n": n,
        "judge_mean": _mean(scores), "judge_n": len(scores),
        "chrf_mean": _mean([r.get("chrf") for r in rows]),
        "bertscore_mean": _mean([r.get("bertscore_f1") for r in rows]),
        "gen_chars_mean": _mean([r.get("gen_chars") for r in rows]),
        "stop": {s: sum(1 for r in rows if r.get("stop_reason") == s) for s in STOP_REASONS},
        **{f: sum(1 for r in rows if r.get(f)) for f in ROW_FLAGS},
    }


def compare(repo_root: Path, cells: Sequence[str], params: Dict[str, Any]) -> Dict[str, Any]:
    """The same prompts answered by several cells, side by side.

    ``cells`` are cell paths (relative to the repo root); the first one, or
    ``params['anchor']`` when given, is the **anchor** every Δ is measured
    against. Only the prompt ids **common to every selected cell** are shown —
    the ids that are not are reported per cell rather than dropped silently.
    Filters are the ones of :func:`query`, applied under ``match`` (``any`` /
    ``all`` / ``anchor``); sorts add the cross-cell orders of
    ``COMPARE_SORTS``. Every cell's dump is read through the same cache, so a
    compare of four cells costs four cached reads, not four scans.
    """
    names = list(dict.fromkeys(str(c) for c in cells if str(c).strip()))
    if len(names) < 2:
        raise FreeformError("compare needs at least two cells")
    if len(names) > MAX_COMPARE_CELLS:
        raise FreeformError(f"at most {MAX_COMPARE_CELLS} cells can be compared at once (got {len(names)})")
    loaded = [load_cell(repo_root, rel) for rel in names]
    anchor = str(params.get("anchor") or "") or names[0]
    if anchor not in names:
        raise FreeformError(f"anchor {anchor!r} is not among the compared cells")

    by_cell = {d["cell"]: {r["id"]: r for r in d["rows"]} for d in loaded}
    common = set.intersection(*(set(ids) for ids in by_cell.values()))
    order = [r["id"] for r in by_cell[anchor].values() if r["id"] in common] if anchor in by_cell else sorted(common)
    only_in = {c: sorted(set(ids) - common) for c, ids in by_cell.items()}

    judges_per_cell = {d["cell"]: [j["name"] for j in d["judges"]] for d in loaded}
    all_judges = sorted({j for js in judges_per_cell.values() for j in js})
    judge = params.get("judge") or (all_judges[0] if all_judges else None)
    if judge and judge not in all_judges:
        raise FreeformError(f"unknown judge {judge!r}; the selected cells have {all_judges}")

    match = params.get("match") or "any"
    if match not in MATCH_MODES:
        raise FreeformError(f"match must be one of {MATCH_MODES}")
    if params.get("stratum"):
        order = [i for i in order if by_cell[anchor][i].get("stratum") == params["stratum"]]
    keep = []
    for rid in order:
        hits = [_row_matches(by_cell[c][rid], params, judge) for c in names]
        ok = hits[names.index(anchor)] if match == "anchor" else (all(hits) if match == "all" else any(hits))
        if ok:
            keep.append(rid)

    def _score(rid: str, cell: str) -> Optional[float]:
        return _judge_score(by_cell[cell][rid], judge)

    def _delta(rid: str) -> Optional[float]:            # other − anchor (two cells: the signed difference)
        a = _score(rid, anchor)
        others = [s for c in names if c != anchor and (s := _score(rid, c)) is not None]
        return None if a is None or not others else float(sum(others) / len(others) - a)

    def _spread(rid: str) -> Optional[float]:           # max − min over every selected cell
        s = [v for c in names if (v := _score(rid, c)) is not None]
        return None if len(s) < 2 else float(max(s) - min(s))

    def _chrf_delta(rid: str) -> Optional[float]:
        a = by_cell[anchor][rid].get("chrf")
        others = [v for c in names if c != anchor and (v := by_cell[c][rid].get("chrf")) is not None]
        return None if a is None or not others else float(sum(others) / len(others) - a)

    def _len_ratio(rid: str) -> Optional[float]:
        a = by_cell[anchor][rid].get("gen_chars") or 0
        others = [by_cell[c][rid].get("gen_chars") or 0 for c in names if c != anchor]
        return None if not a or not others else (sum(others) / len(others)) / a

    sort = params.get("sort") or "row"
    if sort not in COMPARE_SORTS:
        raise FreeformError(f"sort must be one of {COMPARE_SORTS}")
    # one value per kept prompt, then one sort: descending flips the value, the id always
    # breaks ties ascending, and a prompt without the value sorts last in either direction.
    keyers = {
        "score": lambda i: _score(i, anchor),
        "chrf": lambda i: by_cell[anchor][i].get("chrf"),
        "gen_len": lambda i: by_cell[anchor][i].get("gen_chars"),
        "disagree": lambda i: _disagreement(by_cell[anchor][i]),
        "delta": _delta, "spread": _spread, "chrf_delta": _chrf_delta, "len_ratio": _len_ratio,
    }
    if sort != "row":
        base, _, direction = sort.rpartition("_")
        if not base:                                      # "disagree" / "spread_desc" style
            base, direction = sort, "desc"
        rev = direction == "desc"
        vals = {i: keyers[base](i) for i in keep}
        keep = sorted(keep, key=lambda i: (vals[i] is None,
                                           -(vals[i] or 0.0) if rev else (vals[i] or 0.0), i))

    page_size = max(1, min(_to_int(params.get("page_size")) or 25, 200))
    total = len(keep)
    pages = max(1, math.ceil(total / page_size))
    page = max(1, min(_to_int(params.get("page")) or 1, pages))
    chunk = keep[(page - 1) * page_size: page * page_size]

    prompts = []
    for rid in chunk:
        a = by_cell[anchor][rid]
        prompts.append({
            "id": rid, "stratum": a.get("stratum"), "instruction": a.get("instruction"),
            "context": a.get("context"), "reference": a.get("reference"), "ref_chars": a.get("ref_chars"),
            "delta": _delta(rid), "spread": _spread(rid), "chrf_delta": _chrf_delta(rid), "len_ratio": _len_ratio(rid),
            "cells": {c: _compact_with_verdict(by_cell[c][rid], judge) for c in names},
        })
    selected = {c: [by_cell[c][i] for i in keep] for c in names}
    summary = {c: _compare_summary(selected[c], judge) for c in names}
    for c in names:                                      # descriptive Δ over the shown prompts, never a CI
        a_mean, c_mean = summary[anchor]["judge_mean"], summary[c]["judge_mean"]
        summary[c]["judge_delta_vs_anchor"] = None if (a_mean is None or c_mean is None or c == anchor) \
            else round(c_mean - a_mean, 4)
    return {
        "cells": names, "anchor": anchor, "judge": judge, "judges": all_judges,
        "judges_per_cell": judges_per_cell, "match": match, "sort": sort,
        "n_common": len(common), "only_in": {c: len(ids) for c, ids in only_in.items()},
        "only_in_ids": {c: ids[:20] for c, ids in only_in.items() if ids},
        "total": total, "page": page, "pages": pages, "page_size": page_size,
        "prompts": prompts, "summary": summary,
        "tokenizers": {d["cell"]: _cell_summary(d["dir"]).get("tokenizer") for d in loaded},
        "metadata": {d["cell"]: d["metadata"] for d in loaded},
    }


MAX_COMPARE_EXPORT = 500


def export_compare_csv(repo_root: Path, cells: Sequence[str], params: Dict[str, Any]) -> Tuple[str, str]:
    """The compare view as CSV — one row per prompt, one group of columns per
    cell. The spreadsheet escape hatch for a view someone wants to annotate or
    paste into an appendix; the dumps themselves stay Parquet."""
    import csv
    import io
    p = dict(params)
    p["page"], p["page_size"] = 1, MAX_COMPARE_EXPORT
    data = compare(repo_root, cells, p)
    names, judge = data["cells"], data["judge"]
    short = [Path(c).name for c in names]
    fields = ["id", "stratum", "instruction", "reference", "delta", "spread"]
    for n in short:
        fields += [f"{n}.generation", f"{n}.chrf", f"{n}.judge_score", f"{n}.judge_rationale",
                   f"{n}.stop_reason", f"{n}.gen_chars"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for pr in data["prompts"]:
        rec = {"id": pr["id"], "stratum": pr["stratum"], "instruction": pr["instruction"],
               "reference": pr["reference"], "delta": pr["delta"], "spread": pr["spread"]}
        for cell, n in zip(names, short):
            c = pr["cells"][cell]
            v = c.get("verdict")
            rec.update({f"{n}.generation": c["generation"], f"{n}.chrf": c["chrf"],
                        f"{n}.judge_score": None if not v else v["score"],
                        f"{n}.judge_rationale": None if not v else v["rationale"],
                        f"{n}.stop_reason": c["stop_reason"], f"{n}.gen_chars": c["gen_chars"]})
        w.writerow(rec)
    name = f"freeform_compare_{'_vs_'.join(short)[:80]}.csv"
    return name, buf.getvalue()
