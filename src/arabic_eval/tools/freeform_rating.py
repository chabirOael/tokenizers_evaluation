"""Blind human check of the free-form generations (the console's *Rate* tab).

``build_set`` draws ``n_prompts`` held-out prompts from a finished sweep —
half at random, half where the judges disagree most (or, with one judge,
where its scores vary most across variants) — and for each one the baseline
cell's generation plus ``variants_per_prompt − 1`` other cells', balanced by
round-robin over the cells; the items are shuffled and the cell of every
item is kept in the set file only. A rater sees instruction, reference and
generation and gives the judge's 1–5 score plus flags; ratings land in one
JSON per rater. ``agreement`` then reveals the cells and computes rater vs
judge (Spearman, exact, quadratic-weighted kappa, mean |Δ|) on the rated
items, rater vs rater when there are two, and the per-cell human mean —
the calibration the judge scores are read with.
"""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from arabic_eval.judge.freeform_judge import FLAGS, exact_agreement, quadratic_weighted_kappa, spearman
from arabic_eval.tools.freeform_rows_browser import (
    GENERATIONS_REL, FreeformError, _cell_summary, _outputs, load_cell,
)

RATING_DIR = "freeform_rating"
_SAFE = re.compile(r"[A-Za-z0-9_.-]{1,40}")


def _experiment_dir(repo_root: Path, rel: str) -> Path:
    if not rel:
        raise FreeformError("experiment required")
    p = (Path(repo_root) / rel).resolve()
    if _outputs(repo_root).resolve() not in p.parents:
        raise FreeformError("only experiments under outputs/experiments")
    if not p.is_dir():
        raise FreeformError(f"no such experiment: {rel}")
    return p


def _safe(name: str, what: str) -> str:
    if not name or not _SAFE.fullmatch(name):
        raise FreeformError(f"{what} must match {_SAFE.pattern}")
    return name


def _cells_with_generations(exp_dir: Path) -> List[Path]:
    if (exp_dir / GENERATIONS_REL).exists():
        return [exp_dir]
    return sorted(p for p in exp_dir.iterdir() if p.is_dir() and (p / GENERATIONS_REL).exists())


def _set_path(exp_dir: Path, name: str) -> Path:
    return exp_dir / RATING_DIR / f"{name}.json"


def _ratings_dir(exp_dir: Path, name: str) -> Path:
    return exp_dir / RATING_DIR / f"{name}.ratings"


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------

def build_set(repo_root: Path, experiment: str, name: str = "v1", n_prompts: int = 50, variants_per_prompt: int = 3,
              seed: int = 42, baseline: Optional[str] = None, overwrite: bool = False) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    exp_dir = _experiment_dir(repo_root, experiment)
    name = _safe(name, "set name")
    out = _set_path(exp_dir, name)
    if out.exists() and not overwrite:
        raise FreeformError(f"rating set {name!r} exists; pass overwrite to rebuild (existing ratings stay)")
    cells = _cells_with_generations(exp_dir)
    if not cells:
        raise FreeformError("no cell with free-form generations")
    loaded = {c.name: load_cell(repo_root, str(c.relative_to(repo_root))) for c in cells}
    names = [c.name for c in cells]
    base = baseline or next((n for n in names if n.startswith("native_")), names[0])
    if base not in names:
        raise FreeformError(f"baseline {base!r} not among {names}")
    judges = sorted({j["name"] for d in loaded.values() for j in d["judges"]})
    ids = sorted(set.intersection(*[{r["id"] for r in d["rows"]} for d in loaded.values()]))
    by_cell = {n: {r["id"]: r for r in d["rows"]} for n, d in loaded.items()}

    def disagreement(pid: str) -> float:
        vals = []
        for n in names:
            js = by_cell[n][pid].get("judges") or {}
            s = [v["score"] for v in js.values() if v and v.get("score") is not None]
            if len(judges) >= 2 and len(s) >= 2:
                vals.append(max(s) - min(s))
            elif len(judges) == 1 and s:
                vals.append(s[0])
        if len(judges) >= 2:
            return sum(vals) / len(vals) if vals else 0.0
        return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0     # one judge: spread across variants

    rng = random.Random(seed)
    n_prompts = min(n_prompts, len(ids))
    n_dis = n_prompts // 2 if judges else 0
    ranked = sorted(ids, key=lambda i: (-disagreement(i), i))
    chosen_dis = [i for i in ranked if disagreement(i) > 0][:n_dis]
    rest = [i for i in ids if i not in set(chosen_dis)]
    rng.shuffle(rest)
    chosen_rand = rest[: n_prompts - len(chosen_dis)]
    prompts = chosen_dis + chosen_rand
    others = [n for n in names if n != base]
    k = max(0, min(variants_per_prompt - 1, len(others)))
    items: List[Dict[str, Any]] = []
    cursor = 0
    for pid in prompts:
        picked = [base]
        if k:
            order = others[cursor:] + others[:cursor]          # round-robin so every cell is rated evenly
            picked += order[:k]
            cursor = (cursor + k) % len(others)
        for cell in picked:
            r = by_cell[cell][pid]
            items.append({"item_id": f"{pid}::{cell}", "prompt_id": pid, "instruction": r.get("instruction"),
                          "context": r.get("context") or "", "reference": r.get("reference"),
                          "generation": r.get("generation"), "hidden": {"cell": cell, "tokenizer": _cell_summary(exp_dir / cell if (exp_dir / cell).is_dir() else exp_dir).get("tokenizer")}})
    rng.shuffle(items)
    for pos, it in enumerate(items):
        it["position"] = pos
    doc = {"schema": 1, "name": name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": seed,
           "experiment": experiment, "baseline": base, "cells": names, "judges": judges,
           "n_prompts": len(prompts), "variants_per_prompt": 1 + k,
           "selection": {"disagreement": chosen_dis, "random": chosen_rand}, "items": items}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return {k2: v for k2, v in doc.items() if k2 != "items"} | {"n_items": len(items), "path": str(out.relative_to(repo_root))}


# ---------------------------------------------------------------------------
# rating
# ---------------------------------------------------------------------------

def _load_set(exp_dir: Path, name: str) -> Dict[str, Any]:
    p = _set_path(exp_dir, _safe(name, "set name"))
    if not p.exists():
        raise FreeformError(f"no rating set {name!r}")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_ratings(exp_dir: Path, name: str, rater: str) -> Dict[str, Any]:
    p = _ratings_dir(exp_dir, name) / f"{_safe(rater, 'rater')}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def list_sets(repo_root: Path, experiment: str) -> Dict[str, Any]:
    exp_dir = _experiment_dir(Path(repo_root), experiment)
    out = []
    d = exp_dir / RATING_DIR
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        if p.name.endswith(".agreement.json"):
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        raters = []
        rd = _ratings_dir(exp_dir, doc["name"])
        for rp in sorted(rd.glob("*.json")) if rd.is_dir() else []:
            raters.append({"rater": rp.stem, "n_rated": len(json.loads(rp.read_text(encoding="utf-8")))})
        out.append({"name": doc["name"], "created_at": doc.get("created_at"), "n_items": len(doc["items"]),
                    "n_prompts": doc.get("n_prompts"), "variants_per_prompt": doc.get("variants_per_prompt"),
                    "baseline": doc.get("baseline"), "judges": doc.get("judges"), "raters": raters})
    return {"experiment": experiment, "sets": out, "cells": [c.name for c in _cells_with_generations(exp_dir)]}


def get_items(repo_root: Path, experiment: str, name: str, rater: str) -> Dict[str, Any]:
    """The items without their cell, plus this rater's ratings so far."""
    exp_dir = _experiment_dir(Path(repo_root), experiment)
    doc = _load_set(exp_dir, name)
    ratings = _load_ratings(exp_dir, name, rater)
    items = [{k: v for k, v in it.items() if k != "hidden"} | {"rating": ratings.get(it["item_id"])} for it in doc["items"]]
    return {"experiment": experiment, "set": name, "rater": rater, "items": items,
            "progress": {"rated": len(ratings), "total": len(items)}, "flags": list(FLAGS)}


def submit(repo_root: Path, experiment: str, name: str, rater: str, item_id: str, score: Any,
           flags: Optional[Sequence[str]] = None, note: str = "") -> Dict[str, Any]:
    exp_dir = _experiment_dir(Path(repo_root), experiment)
    doc = _load_set(exp_dir, name)
    if not any(it["item_id"] == item_id for it in doc["items"]):
        raise FreeformError(f"unknown item {item_id!r}")
    try:
        s = int(score)
    except (TypeError, ValueError):
        raise FreeformError("score must be an integer 1–5") from None
    if not 1 <= s <= 5:
        raise FreeformError("score must be 1–5")
    fl = [f for f in (flags or []) if f in FLAGS]
    rd = _ratings_dir(exp_dir, name)
    rd.mkdir(parents=True, exist_ok=True)
    ratings = _load_ratings(exp_dir, name, rater)
    ratings[item_id] = {"score": s, "flags": fl, "note": (note or "")[:500], "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (rd / f"{_safe(rater, 'rater')}.json").write_text(json.dumps(ratings, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"ok": True, "progress": {"rated": len(ratings), "total": len(doc["items"])}}


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------

def _pair_stats(x: Sequence[int], y: Sequence[int]) -> Dict[str, Any]:
    return {"n": len(x), "spearman": spearman(x, y), "exact": exact_agreement(x, y),
            "qwk": quadratic_weighted_kappa(x, y),
            "mean_abs_diff": round(sum(abs(a - b) for a, b in zip(x, y)) / len(x), 4) if x else None,
            "mean_x": round(sum(x) / len(x), 4) if x else None, "mean_y": round(sum(y) / len(y), 4) if y else None}


def agreement(repo_root: Path, experiment: str, name: str) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    exp_dir = _experiment_dir(repo_root, experiment)
    doc = _load_set(exp_dir, name)
    rd = _ratings_dir(exp_dir, name)
    raters = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (sorted(rd.glob("*.json")) if rd.is_dir() else [])}
    cells = {}
    for it in doc["items"]:
        cell = it["hidden"]["cell"]
        if cell not in cells:
            cdir = exp_dir / cell if (exp_dir / cell).is_dir() else exp_dir
            cells[cell] = load_cell(repo_root, str(cdir.relative_to(repo_root)))
    judge_score: Dict[str, Dict[str, Optional[int]]] = {}       # judge → item_id → score
    for it in doc["items"]:
        cell = it["hidden"]["cell"]
        r = next((x for x in cells[cell]["rows"] if x["id"] == it["prompt_id"]), None)
        for jn, v in ((r or {}).get("judges") or {}).items():
            judge_score.setdefault(jn, {})[it["item_id"]] = None if not v or v.get("score") is None else int(v["score"])
    out: Dict[str, Any] = {"set": name, "n_items": len(doc["items"]), "raters": {}, "rater_vs_rater": {},
                           "judge_vs_judge_on_items": {}, "per_cell": {}}
    for rn, ratings in raters.items():
        entry: Dict[str, Any] = {"n_rated": len(ratings), "vs_judge": {}}
        for jn, js in judge_score.items():
            ids = [i for i in ratings if js.get(i) is not None]
            entry["vs_judge"][jn] = _pair_stats([int(ratings[i]["score"]) for i in ids], [js[i] for i in ids])
        out["raters"][rn] = entry
    rnames = sorted(raters)
    for i in range(len(rnames)):
        for j in range(i + 1, len(rnames)):
            a, b = raters[rnames[i]], raters[rnames[j]]
            ids = sorted(set(a) & set(b))
            out["rater_vs_rater"][f"{rnames[i]}|{rnames[j]}"] = _pair_stats([int(a[k]["score"]) for k in ids], [int(b[k]["score"]) for k in ids])
    jnames = sorted(judge_score)
    for i in range(len(jnames)):
        for j in range(i + 1, len(jnames)):
            a, b = judge_score[jnames[i]], judge_score[jnames[j]]
            ids = [k for k in a if a.get(k) is not None and b.get(k) is not None]
            out["judge_vs_judge_on_items"][f"{jnames[i]}|{jnames[j]}"] = _pair_stats([a[k] for k in ids], [b[k] for k in ids])
    # per cell: human mean (all raters pooled) next to each judge's mean on the same items
    for cell in sorted(cells):
        items = [it for it in doc["items"] if it["hidden"]["cell"] == cell]
        human = [int(r[it["item_id"]]["score"]) for it in items for r in raters.values() if it["item_id"] in r]
        out["per_cell"][cell] = {"n_items": len(items), "human_n": len(human),
                                 "human_mean": round(sum(human) / len(human), 4) if human else None,
                                 "judges": {jn: (lambda v: round(sum(v) / len(v), 4) if v else None)(
                                     [js[it["item_id"]] for it in items if js.get(it["item_id"]) is not None])
                                     for jn, js in judge_score.items()}}
    (exp_dir / RATING_DIR / f"{name}.agreement.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
