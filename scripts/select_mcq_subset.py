#!/usr/bin/env python
"""Draw the seeded 1 000-row subsets of the letter-slot diagnostic (docs/report.md §3.11).

Per task (``arabic_exam``, ``culture_arabic_mmlu``): the rows present in the
dumps of every source cell, untruncated in every cell (``mcq_compare.py``'s
intersection rule: ``hit_cap`` or ``sentinel`` anywhere drops the row; it
implies ``all_sentinel`` false too) **and** with four choices; sorted by
``row_index``, then ``n`` of them drawn with ``numpy.random.default_rng(seed)``
without replacement — reproducible from the dumps alone. The prompts must
agree across the cells at every candidate row (the same benchmark rows).

Writes ``<out-dir>/rows_<task>.json`` = ``{"task", "row_index": [sorted],
"provenance": {...}}`` — the ``rows_file`` a LightEval task reads.

    .venv/bin/python scripts/select_mcq_subset.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from arabic_eval.tasks.lighteval.utils import ARABIC_CHOICE_LETTERS  # noqa: E402

EXPERIMENT = "outputs/experiments/qwen_native_vs_araroopat"
CELLS = (
    "araroopat_3phase_v5_distill_mcq4096",
    "bpe_16k_3phase_v5_distill_mcq4096",
    "native_qwen3_sft_v5_distill_mcq4096",
    "native_qwen3_base_mcq4096",
)
TASKS = ("arabic_exam", "culture_arabic_mmlu")
COLUMNS = ("row_index", "n_choices", "gold_idx", "hit_cap", "sentinel", "all_sentinel", "prompt")
RULE = ("present in every source cell's dump; hit_cap, sentinel and all_sentinel false in every cell "
        "(the mcq_compare.py intersection rule); n_choices == 4; sorted by row_index; "
        "numpy.random.default_rng(seed).choice(candidates, n, replace=False), then sorted")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    tbl = pq.read_table(path, columns=list(COLUMNS))
    ri = np.asarray(tbl.column("row_index").to_pylist(), dtype=np.int64)
    order = np.argsort(ri, kind="stable")
    out = {"row_index": ri[order]}
    for c in ("n_choices", "gold_idx"):
        out[c] = np.asarray(tbl.column(c).to_pylist(), dtype=np.int64)[order]
    for c in ("hit_cap", "sentinel", "all_sentinel"):
        out[c] = np.asarray([bool(v) for v in tbl.column(c).to_pylist()], dtype=bool)[order]
    out["prompt_sha"] = np.asarray(
        [hashlib.sha1(p.encode("utf-8")).hexdigest() for p in tbl.column("prompt").to_pylist()],
        dtype=object)[order]
    return out


def candidates(dumps: Sequence[Dict[str, np.ndarray]]) -> np.ndarray:
    """Row indices every dump holds, untruncated everywhere, with four choices (sorted)."""
    common = set(dumps[0]["row_index"].tolist())
    for d in dumps[1:]:
        common &= set(d["row_index"].tolist())
    keep = None
    for d in dumps:
        pos = {int(r): i for i, r in enumerate(d["row_index"])}
        ok = np.array([
            not (d["hit_cap"][pos[r]] or d["sentinel"][pos[r]] or d["all_sentinel"][pos[r]])
            and d["n_choices"][pos[r]] == 4
            for r in sorted(common)
        ], dtype=bool)
        keep = ok if keep is None else keep & ok
    rows = np.asarray(sorted(common), dtype=np.int64)
    rows = rows[keep] if keep is not None else rows
    # Same benchmark rows: the prompt of each candidate agrees across cells.
    ref = {int(r): s for r, s in zip(dumps[0]["row_index"], dumps[0]["prompt_sha"])}
    for d in dumps[1:]:
        other = {int(r): s for r, s in zip(d["row_index"], d["prompt_sha"])}
        bad = [int(r) for r in rows if other[int(r)] != ref[int(r)]]
        if bad:
            raise SystemExit(f"prompts differ across cells at row_index {bad[:5]}")
    return rows


def draw(rows: np.ndarray, n: int, seed: int) -> List[int]:
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    if n > len(rows):
        raise SystemExit(f"cannot draw {n} rows from {len(rows)} candidates")
    rng = np.random.default_rng(seed)
    return sorted(int(v) for v in rng.choice(rows, size=n, replace=False))


def gold_letter_counts(dump: Dict[str, np.ndarray], picked: Sequence[int]) -> Dict[str, int]:
    pos = {int(r): i for i, r in enumerate(dump["row_index"])}
    gold = [int(dump["gold_idx"][pos[r]]) for r in picked]
    return {ARABIC_CHOICE_LETTERS[k]: int(sum(1 for g in gold if g == k)) for k in range(4)}


def render_rows_file(task: str, picked: Sequence[int], prov: Dict[str, Any]) -> str:
    """The JSON text: the index list on one line, the provenance indented."""
    body = json.dumps(prov, ensure_ascii=False, indent=2).replace("\n", "\n ")
    return (f'{{\n "task": {json.dumps(task)},\n "row_index": {json.dumps(list(picked))},\n'
            f' "provenance": {body}\n}}\n')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiment", default=EXPERIMENT)
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="configs/experiments/mcq_letter_slot")
    args = ap.parse_args(argv)

    exp = REPO_ROOT / args.experiment
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        paths = [exp / c / "eval_rows" / f"{task}.parquet" for c in args.cells]
        dumps = [load(p) for p in paths]
        rows = candidates(dumps)
        picked = draw(rows, args.n, args.seed)
        counts = gold_letter_counts(dumps[0], picked)
        prov: Dict[str, Any] = {
            "created_by": "scripts/select_mcq_subset.py",
            "experiment": args.experiment,
            "source_cells": list(args.cells),
            "rule": RULE,
            "seed": args.seed,
            "n": args.n,
            "n_candidates": int(len(rows)),
            "source_dumps_sha256": {c: sha256_file(p) for c, p in zip(args.cells, paths)},
            "gold_letter_counts": counts,
        }
        path = out_dir / f"rows_{task}.json"
        path.write_text(render_rows_file(task, picked, prov), encoding="utf-8")
        shares = {k: round(v / len(picked), 3) for k, v in counts.items()}
        print(f"{task}: {len(rows)} candidates, {len(picked)} drawn, gold {counts} {shares} "
              f"-> {path.relative_to(REPO_ROOT)} sha256 {sha256_file(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
