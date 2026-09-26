#!/usr/bin/env python
"""Where does the Culture-MMLU residual sit? AraRooPat − BPE-16K by subject category and by prompt fragmentation.

§3.11 left a letter-invariant gap of −0.039 [−0.068, −0.011] between AraRooPat v5 and BPE-16K v5 on Culture-MMLU
(the rotation-averaged decision on the 1 000-row subset) against −0.001 [−0.030, +0.027] on Arabic-Exam. This
script asks whether that residual is a broad recognition gap, a few subjects, or the rows where AraRooPat
fragments the prompt most (docs/report.md §5 item 26(c), §3.13). Two inputs per benchmark, four arms each
(``araroopat_v5``, ``bpe16k_v5``, ``native_v5``, ``native_base``):

  * **standard decision, full dumps** — the ``*_mcq4096`` cells of the main experiment, joined on ``row_index``
    by ``mcq_compare.load_dump`` (the prompt of every row must agree across cells), on the rows **no cell
    truncates** (``hit_cap`` or ``sentinel`` in any cell drops the row for all: §3.10's intersection rule),
    ``correct_char`` and ``correct_pmi``;
  * **rotation-averaged decision, the 1 000-row subsets** — the sixteen ``_letter_slot/<arm>_rot{0..3}`` cells
    through ``mcq_letter_slot.build_panel`` + ``rotation_averaged_correct`` (char and PMI as that script defines
    them — they coincide, the letter prior cancels over the four rotations). The subset's partition labels are
    read from the full dumps by ``row_index`` (rotation 0 is byte-identical to the full dump, §3.11 *Checks*).

**Partitions.** *Category*: Culture-MMLU's 57 MMLU subjects in the four Hendrycks et al. categories
(``CULTURE_MMLU_CATEGORIES``); Arabic-Exam — the control — by the dataset's own ``Group`` column (and ``Level``),
read from the cached ``MBZUAI/ArabicMMLU`` CSVs and required to be one-to-one per subject config. *Fragmentation*:
terciles of ``prompt_units[araroopat] / prompt_units[bpe16k]`` and, separately, of
``prompt_units[araroopat] / prompt_units[native]`` (the same prompt string in every cell, counted in each cell's
own tokens), cut on the analysed rows. Per-subject numbers are written to the JSON only.

**Per group**, both benchmarks, both decisions, both normalizations: ``n``, each arm's accuracy, and for
AraRooPat − BPE-16K and AraRooPat − native v5 the paired gap with a cluster bootstrap over rows
(``mcq_letter_slot.boot_weights``, 5 000 resamples, seed 0 — one resample matrix per input, so every group and
pair shares it) and McNemar's discordant counts with the exact two-sided p.

**Heterogeneity test (pre-registered).** Statistic: the spread (max − min over groups) of the paired gap.
Null: ``--n-perm`` (5 000) random re-assignments of the group labels to the rows that preserve the group sizes
(``numpy.random.default_rng(seed).permutation`` of the label vector, the same permutations for every pair and
normalization of a partition); ``p`` = the share of permutations whose spread is ≥ the observed one.

**Reading rules** (pre-registered, on the Culture-MMLU rotation-averaged AraRooPat − BPE-16K residual): R1
*concentrated* — category-spread p < 0.05; R2 *broad* — p ≥ 0.05 while the pooled gap's CI excludes zero; R3
*fragmentation* — the tercile-spread p < 0.05 with the gap monotone across the terciles. The control: the same
tests on Arabic-Exam are expected not to fire. Nothing here is a gate; every result is reported.

    .venv/bin/python scripts/mcq_residual_by_subject.py
Writes ``<letter_slot>/residual_by_subject.{md,json}`` (``--name``, ``--out-dir`` override).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402


def _sibling(name: str):
    """Import a sibling script as a module (``scripts/`` is not a package)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ls = _sibling("mcq_letter_slot")
mc = _sibling("mcq_compare")

EXPERIMENT = "outputs/experiments/qwen_native_vs_araroopat"
LETTER_SLOT = "outputs/experiments/qwen_native_vs_araroopat/_letter_slot"
ROWS_DIR = ls.ROWS_DIR
TASKS = ("culture_arabic_mmlu", "arabic_exam")
ARMS = ("araroopat_v5", "bpe16k_v5", "native_v5", "native_base")
#: The full-benchmark cell of each arm (§3.10, ``max_length`` 4 096, rotation 0).
FULL_CELLS = {
    "araroopat_v5": "araroopat_3phase_v5_distill_mcq4096",
    "bpe16k_v5": "bpe_16k_3phase_v5_distill_mcq4096",
    "native_v5": "native_qwen3_sft_v5_distill_mcq4096",
    "native_base": "native_qwen3_base_mcq4096",
}
PAIRS = (("araroopat_v5", "bpe16k_v5"), ("araroopat_v5", "native_v5"))
FOCUS_PAIR = "araroopat_v5-bpe16k_v5"
NORMS = ("char", "pmi")
DECISIONS = ("standard", "rotation_averaged")
ALPHA = 0.05

#: Hendrycks et al. (2021) MMLU categories of the 57 subjects in ``OALL/Arabic_MMLU`` (``source_config``).
CULTURE_MMLU_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "stem": (
        "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics", "computer_security",
        "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science", "high_school_mathematics", "high_school_physics",
        "high_school_statistics", "machine_learning",
    ),
    "humanities": (
        "formal_logic", "high_school_european_history", "high_school_us_history", "high_school_world_history",
        "international_law", "jurisprudence", "logical_fallacies", "moral_disputes", "moral_scenarios",
        "philosophy", "prehistory", "professional_law", "world_religions",
    ),
    "social_sciences": (
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology", "human_sexuality",
        "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy",
    ),
    "other": (
        "business_ethics", "clinical_knowledge", "college_medicine", "global_facts", "human_aging", "management",
        "marketing", "medical_genetics", "miscellaneous", "nutrition", "professional_accounting",
        "professional_medicine", "virology",
    ),
}


def subject_category_map() -> Dict[str, str]:
    """subject → category; raises if a subject sits in two categories."""
    out: Dict[str, str] = {}
    for cat, subjects in CULTURE_MMLU_CATEGORIES.items():
        for s in subjects:
            if s in out:
                raise ValueError(f"subject {s!r} in both {out[s]!r} and {cat!r}")
            out[s] = cat
    return out


SUBJECT_CATEGORY = subject_category_map()
ARABIC_MMLU_REPO = "MBZUAI/ArabicMMLU"


def arabic_mmlu_group_map(snapshot_dir: Optional[Path] = None) -> Dict[str, Any]:
    """``{"group": {config: Group}, "level": {config: Level or "unspecified"}, "snapshot": …}`` from the cached
    ``MBZUAI/ArabicMMLU`` test CSVs (the ``All`` union excluded). Each config must carry exactly one Group and at
    most one Level — asserted, since the partition is only meaningful if it is a function of the config."""
    import pandas as pd

    if snapshot_dir is None:
        snapshot_dir = _hub_snapshot(ARABIC_MMLU_REPO)
    snapshot_dir = Path(snapshot_dir)
    group: Dict[str, str] = {}
    level: Dict[str, str] = {}
    for p in sorted(snapshot_dir.glob("*/test.csv")):
        cfg = p.parent.name
        if cfg == "All":
            continue
        df = pd.read_csv(p, usecols=lambda c: c in ("Group", "Level"))
        groups = sorted(set(df["Group"].dropna()))
        levels = sorted(set(df["Level"].dropna())) if "Level" in df else []
        if len(groups) != 1:
            raise ValueError(f"ArabicMMLU config {cfg!r}: Group is not one-to-one ({groups})")
        if len(levels) > 1:
            raise ValueError(f"ArabicMMLU config {cfg!r}: Level is not one-to-one ({levels})")
        group[cfg] = groups[0]
        level[cfg] = levels[0] if levels else "unspecified"
    if not group:
        raise FileNotFoundError(f"no */test.csv under {snapshot_dir}")
    return {"group": group, "level": level, "snapshot": snapshot_dir.name}


def _hub_snapshot(repo_id: str) -> Path:
    """The local snapshot of a cached HF dataset repo (no network)."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id, repo_type="dataset", local_files_only=True,
                                      allow_patterns=["*/test.csv"]))
    except Exception:            # noqa: BLE001 — fall back to the cache layout
        root = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
        snaps = sorted(glob.glob(str(root / f"datasets--{repo_id.replace('/', '--')}" / "snapshots" / "*")))
        if not snaps:
            raise FileNotFoundError(f"{repo_id} is not in the local HF cache ({root})")
        return Path(snaps[-1])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_full(cell_dir: Path, task: str) -> Dict[str, Any]:
    """``mcq_compare.load_dump`` (ordered by ``row_index``) plus ``prompt_units`` and ``n_choices``."""
    import pyarrow.parquet as pq

    d = mc.load_dump(cell_dir, task)
    if d is None:
        raise FileNotFoundError(f"no dump {cell_dir}/eval_rows/{task}.parquet")
    tbl = pq.read_table(d["path"], columns=["row_index", "prompt_units", "n_choices"])
    ri = np.asarray(tbl.column("row_index").to_pylist(), dtype=np.int64)
    order = np.argsort(ri, kind="stable")
    d["prompt_units"] = np.asarray(tbl.column("prompt_units").to_pylist(), dtype=np.float64)[order]
    d["n_choices"] = np.asarray(tbl.column("n_choices").to_pylist(), dtype=np.int64)[order]
    return d


def join_full(experiment: Path, task: str, cells: Dict[str, str] = FULL_CELLS) -> Dict[str, Any]:
    """The four arms' full dumps on the rows no cell truncates."""
    dumps = {arm: load_full(experiment / cell, task) for arm, cell in cells.items()}
    ref_arm = next(iter(dumps))
    ref = dumps[ref_arm]
    for arm, d in dumps.items():
        if not np.array_equal(d["row_index"], ref["row_index"]):
            raise ValueError(f"{task}: {arm} and {ref_arm} do not hold the same row_index set")
        bad = int((d["prompt_sha"] != ref["prompt_sha"]).sum())
        if bad:
            raise ValueError(f"{task}: {bad} prompts of {arm} differ from {ref_arm} at the same row_index")
        if not np.array_equal(d["source_config"], ref["source_config"]):
            raise ValueError(f"{task}: source_config differs between {arm} and {ref_arm}")
    n_total = len(ref["row_index"])
    keep = np.ones(n_total, dtype=bool)
    capped = {}
    for arm, d in dumps.items():
        m = d["hit_cap"] | d["sentinel"]
        capped[arm] = int(m.sum())
        keep &= ~m
    return {
        "row_index": ref["row_index"][keep],
        "prompt_sha": ref["prompt_sha"][keep],
        "source_config": ref["source_config"][keep].astype(str),
        "prompt_units": {arm: d["prompt_units"][keep] for arm, d in dumps.items()},
        "correct": {arm: {nm: d[f"correct_{nm}"][keep] for nm in NORMS} for arm, d in dumps.items()},
        "n_total": n_total, "n_intersection": int(keep.sum()), "capped": capped,
        "cells": dict(cells),
    }


def load_subset(letter_slot: Path, rows_dir: Path, task: str, full: Dict[str, Any],
                arms: Sequence[str] = ARMS) -> Dict[str, Any]:
    """Rotation-averaged correctness of every arm on the 1 000-row subset (rows-file order), plus the standard
    decision at rotation 0 on the same rows, and the partition fields from the full dumps."""
    spec = json.loads((rows_dir / f"rows_{task}.json").read_text(encoding="utf-8"))
    rows = [int(r) for r in spec["row_index"]]
    pos = {int(r): i for i, r in enumerate(full["row_index"])}
    missing = [r for r in rows if r not in pos]
    if missing:
        raise ValueError(f"{task}: {len(missing)} subset rows are not in the full dumps' untruncated intersection")
    idx = np.asarray([pos[r] for r in rows], dtype=np.int64)
    rav: Dict[str, Dict[str, np.ndarray]] = {}
    rot0: Dict[str, Dict[str, np.ndarray]] = {}
    for arm in arms:
        dumps = {k: ls.load_dump(letter_slot / ls.cell_name(arm, f"rot{k}") / "eval_rows" / f"{task}.parquet")
                 for k in ls.ROTATIONS}
        # rotation 0 is the full dump's prompt, row for row (the full dumps' prompts agree across arms)
        d0 = ls.align(dumps[0], rows)
        mism = sum(hashlib.sha1(p.encode("utf-8")).hexdigest() != full["prompt_sha"][i]
                   for p, i in zip(d0["prompt"], idx))
        if mism:
            raise ValueError(f"{task} {arm}: {mism} rotation-0 prompts differ from the full dump")
        panel = ls.build_panel(dumps, rows)
        r = ls.rotation_averaged_correct(panel)
        if set(r) != set(NORMS):
            raise ValueError(f"{task} {arm}: rotation-averaged decision missing a normalization ({sorted(r)})")
        rav[arm] = r
        rot0[arm] = {nm: panel["correct"][nm][:, 0].copy() for nm in NORMS}
    return {
        "row_index": np.asarray(rows, dtype=np.int64),
        "source_config": full["source_config"][idx],
        "prompt_units": {arm: full["prompt_units"][arm][idx] for arm in full["prompt_units"]},
        "correct": {"rotation_averaged": rav, "standard_rot0": rot0},
        "n_rows": len(rows),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def tercile_labels(ratio: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """``T1`` (lowest ratio) … ``T3`` by the 1/3 and 2/3 quantiles of the analysed rows."""
    q1, q2 = (float(v) for v in np.quantile(ratio, [1 / 3, 2 / 3]))
    lab = np.where(ratio <= q1, "T1", np.where(ratio <= q2, "T2", "T3"))
    return lab.astype(object), [round(q1, 6), round(q2, 6)]


def spread_permutation(labels: np.ndarray, D: np.ndarray, n_perm: int, seed: int) -> Dict[str, Any]:
    """Permutation test of the spread (max − min over groups) of the column means of ``D`` ``[n, m]``.

    The group labels are permuted over the rows (sizes preserved); the same ``n_perm`` permutations
    (``default_rng(seed)``) serve every column. Returns the observed spread per column, ``p`` (share of
    permutations with a spread ≥ the observed, ties within 1e-12 counted) and the null's 95th percentile."""
    groups, lab = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
    G = len(groups)
    D = np.asarray(D, dtype=np.float64).reshape(len(lab), -1)
    m = D.shape[1]
    sizes = np.bincount(lab, minlength=G).astype(np.float64)
    if G < 2:
        return {"groups": groups.tolist(), "observed": [None] * m, "p": [None] * m, "null_q95": [None] * m}

    def spreads(l: np.ndarray) -> np.ndarray:
        out = np.empty(m)
        for j in range(m):
            means = np.bincount(l, weights=D[:, j], minlength=G) / sizes
            out[j] = means.max() - means.min()
        return out

    obs = spreads(lab)
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, m))
    for b in range(n_perm):
        null[b] = spreads(rng.permutation(lab))
    ge = (null >= obs[None, :] - 1e-12).sum(0)
    return {"groups": groups.tolist(), "observed": [round(float(v), 6) for v in obs],
            "p": [round(float(v) / n_perm, 6) for v in ge],
            "null_q95": [round(float(v), 6) for v in np.percentile(null, 95, axis=0)]}


def group_block(labels: np.ndarray, correct: Dict[str, Dict[str, np.ndarray]], W: np.ndarray,
                pairs: Sequence[Tuple[str, str]], order: Optional[Sequence[str]] = None,
                n_perm: Optional[int] = None, seed: int = 0) -> Dict[str, Any]:
    """Per group: n, every arm's accuracy, and per pair the paired gap (bootstrap CI) and McNemar counts, under
    both normalizations; with ``n_perm`` also the spread permutation test per pair × normalization."""
    labels = np.asarray(labels, dtype=str)
    present = set(labels.tolist())
    order = list(order or [])
    groups = [g for g in order if g in present] + sorted(present - set(order))
    masks = np.stack([labels == g for g in groups], axis=1).astype(np.float64)        # [n, G]
    den = W @ masks                                                                   # [B+1, G]
    out: Dict[str, Any] = {"groups": {}}
    for gi, g in enumerate(groups):
        msk = masks[:, gi].astype(bool)
        out["groups"][g] = {"n": int(msk.sum()),
                            "acc": {arm: {nm: round(float(c[nm][msk].mean()), 6) for nm in NORMS}
                                    for arm, c in correct.items()},
                            "pairs": {}}
    diffs: List[np.ndarray] = []
    keys: List[str] = []
    for a, b in pairs:
        key = f"{a}-{b}"
        for nm in NORMS:
            ca, cb = correct[a][nm].astype(bool), correct[b][nm].astype(bool)
            d = ca.astype(np.float64) - cb.astype(np.float64)
            diffs.append(d)
            keys.append(f"{key}|{nm}")
            with np.errstate(invalid="ignore", divide="ignore"):
                stat = np.where(den > 0, (W @ (masks * d[:, None])) / np.where(den > 0, den, 1.0), np.nan)
            for gi, g in enumerate(groups):
                msk = masks[:, gi].astype(bool)
                only_a, only_b = int((ca & ~cb & msk).sum()), int((~ca & cb & msk).sum())
                out["groups"][g]["pairs"].setdefault(key, {})[nm] = {
                    **ls.summarize(stat[:, gi]), "only_a": only_a, "only_b": only_b,
                    "mcnemar_p": mc.mcnemar_exact_p(only_a, only_b)}
    if n_perm:
        perm = spread_permutation(labels, np.stack(diffs, axis=1), n_perm, seed)
        out["spread_test"] = {k: {"observed": perm["observed"][j], "p": perm["p"][j], "null_q95": perm["null_q95"][j]}
                              for j, k in enumerate(keys)}
        out["spread_test"]["_n_perm"] = n_perm
        out["spread_test"]["_seed"] = seed
    out["order"] = groups
    return out


def pooled_block(correct: Dict[str, Dict[str, np.ndarray]], W: np.ndarray,
                 pairs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    n = len(correct[next(iter(correct))]["char"])
    ones = np.ones(n, dtype=bool)
    out: Dict[str, Any] = {"n": n, "acc": {arm: {nm: round(float(c[nm].mean()), 6) for nm in NORMS}
                                           for arm, c in correct.items()}, "pairs": {}}
    for a, b in pairs:
        key = f"{a}-{b}"
        for nm in NORMS:
            ca, cb = correct[a][nm].astype(bool), correct[b][nm].astype(bool)
            d = ca.astype(np.float64) - cb.astype(np.float64)
            only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
            out["pairs"].setdefault(key, {})[nm] = {**ls.summarize(ls.wmean(W, d, ones)), "only_a": only_a,
                                                    "only_b": only_b, "mcnemar_p": mc.mcnemar_exact_p(only_a, only_b)}
    return out


def monotone(values: Sequence[Optional[float]]) -> Optional[str]:
    """``increasing`` / ``decreasing`` (weakly, not constant), ``none``; ``None`` if a value is missing."""
    if any(v is None for v in values):
        return None
    inc = all(x <= y for x, y in zip(values, values[1:]))
    dec = all(x >= y for x, y in zip(values, values[1:]))
    if inc and not dec:
        return "increasing"
    if dec and not inc:
        return "decreasing"
    return "none"


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------

def partitions_for(task: str, source_config: np.ndarray, units: Dict[str, np.ndarray],
                   exam_map: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{name: {"labels", "order", "test", "note"}}`` — the category / group partition(s), the two fragmentation
    tercile partitions and the subject partition (JSON only, no test)."""
    parts: Dict[str, Dict[str, Any]] = {}
    subjects = sorted(set(source_config.tolist()))
    if task == "culture_arabic_mmlu":
        unknown = [s for s in subjects if s not in SUBJECT_CATEGORY]
        if unknown:
            raise ValueError(f"culture_arabic_mmlu subjects without a category: {unknown}")
        parts["category"] = {"labels": np.asarray([SUBJECT_CATEGORY[s] for s in source_config], dtype=object),
                             "order": list(CULTURE_MMLU_CATEGORIES), "test": True}
    elif task == "arabic_exam":
        if exam_map is None:
            raise ValueError("arabic_exam needs the ArabicMMLU config → Group map")
        unknown = [s for s in subjects if s not in exam_map["group"]]
        if unknown:
            raise ValueError(f"arabic_exam configs without a Group: {unknown}")
        parts["group"] = {"labels": np.asarray([exam_map["group"][s] for s in source_config], dtype=object),
                          "order": ["STEM", "Social Science", "Humanities", "Language", "Other"], "test": True}
        parts["level"] = {"labels": np.asarray([exam_map["level"][s] for s in source_config], dtype=object),
                          "order": ["Primary", "Middle", "High", "Univ", "Prof", "unspecified"], "test": True}
    for name, other in (("frag_vs_bpe16k", "bpe16k_v5"), ("frag_vs_native", "native_v5")):
        ratio = units["araroopat_v5"] / units[other]
        lab, bounds = tercile_labels(ratio)
        parts[name] = {"labels": lab, "order": ["T1", "T2", "T3"], "test": True, "boundaries": bounds,
                       "ratio": f"prompt_units[araroopat_v5] / prompt_units[{other}]",
                       "ratio_range": [round(float(ratio.min()), 6), round(float(ratio.max()), 6)],
                       "ratio_mean_by_tercile": {t: round(float(ratio[lab == t].mean()), 6) for t in ("T1", "T2", "T3")}}
    parts["subject"] = {"labels": source_config.astype(object), "order": subjects, "test": False}
    return parts


def analyze(experiment: Path, letter_slot: Path, rows_dir: Path, tasks: Sequence[str] = TASKS, *,
            n_boot: int = 5000, n_perm: int = 5000, seed: int = 0, exam_map: Optional[Dict[str, Any]] = None,
            full_cells: Dict[str, str] = FULL_CELLS, arms: Sequence[str] = ARMS,
            pairs: Sequence[Tuple[str, str]] = PAIRS) -> Dict[str, Any]:
    res: Dict[str, Any] = {"experiment": str(experiment), "letter_slot": str(letter_slot), "n_boot": n_boot,
                           "n_perm": n_perm, "seed": seed, "arms": list(arms), "full_cells": dict(full_cells),
                           "pairs": [f"{a}-{b}" for a, b in pairs], "tasks": {}}
    if exam_map is not None:
        res["arabic_mmlu_snapshot"] = exam_map.get("snapshot")
    for task in tasks:
        full = join_full(experiment, task, full_cells)
        t: Dict[str, Any] = {"full": {"decision": "standard", "n_total": full["n_total"],
                                      "n_intersection": full["n_intersection"], "capped": full["capped"],
                                      "partitions": {}}}
        # --- standard decision, full dumps
        Wf = ls.boot_weights(full["n_intersection"], n_boot, seed)
        t["full"]["pooled"] = pooled_block(full["correct"], Wf, pairs)
        parts_f = partitions_for(task, full["source_config"], full["prompt_units"], exam_map)
        for name, p in parts_f.items():
            blk = group_block(p["labels"], full["correct"], Wf, pairs, p["order"],
                              n_perm if p["test"] else None, seed)
            blk.update({k: v for k, v in p.items() if k not in ("labels", "order", "test")})
            t["full"]["partitions"][name] = blk
        del Wf
        # --- rotation-averaged decision (and rotation 0), the 1 000-row subset
        sub = load_subset(letter_slot, rows_dir, task, full, arms)
        Ws = ls.boot_weights(sub["n_rows"], n_boot, seed)
        t["subset"] = {"decision": "rotation_averaged", "n_rows": sub["n_rows"], "partitions": {},
                       "partitions_standard_rot0": {}}
        t["subset"]["pooled"] = pooled_block(sub["correct"]["rotation_averaged"], Ws, pairs)
        t["subset"]["pooled_standard_rot0"] = pooled_block(sub["correct"]["standard_rot0"], Ws, pairs)
        parts_s = partitions_for(task, sub["source_config"], sub["prompt_units"], exam_map)
        for name, p in parts_s.items():
            for key, corr in (("partitions", sub["correct"]["rotation_averaged"]),
                              ("partitions_standard_rot0", sub["correct"]["standard_rot0"])):
                blk = group_block(p["labels"], corr, Ws, pairs, p["order"],
                                  n_perm if (p["test"] and key == "partitions") else None, seed)
                blk.update({k: v for k, v in p.items() if k not in ("labels", "order", "test")})
                t["subset"][key][name] = blk
        # composition of the subset against the full benchmark, per tested partition
        comp = {}
        for name in parts_f:
            if name == "subject" or name.startswith("frag"):
                continue
            nf = {g: v["n"] for g, v in t["full"]["partitions"][name]["groups"].items()}
            ns = {g: v["n"] for g, v in t["subset"]["partitions"][name]["groups"].items()}
            tf, ts = sum(nf.values()), sum(ns.values())
            comp[name] = {g: {"full_n": nf.get(g, 0), "full_share": round(nf.get(g, 0) / tf, 6),
                              "subset_n": ns.get(g, 0), "subset_share": round(ns.get(g, 0) / ts, 6)}
                          for g in nf}
        t["composition"] = comp
        if task == "arabic_exam" and exam_map is not None:
            t["group_map"] = {s: {"group": exam_map["group"][s], "level": exam_map["level"][s]}
                              for s in sorted(set(full["source_config"].tolist()))}
        res["tasks"][task] = t
    res["readings"] = readings(res)
    return res


def _pair_p(blk: Dict[str, Any], pair: str, nm: str) -> Optional[float]:
    s = (blk.get("spread_test") or {}).get(f"{pair}|{nm}")
    return None if s is None else s["p"]


def readings(res: Dict[str, Any]) -> Dict[str, Any]:
    """R1 / R2 / R3 on Culture-MMLU (rotation-averaged, AraRooPat − BPE-16K), the control on Arabic-Exam, and —
    supplementary — the same rules under the standard decision on the full dumps."""
    out: Dict[str, Any] = {}
    for task, t in res["tasks"].items():
        cat = "category" if task == "culture_arabic_mmlu" else "group"
        for dec, src in (("rotation_averaged", t["subset"]), ("standard_full", t["full"])):
            parts = src["partitions"]
            for pair in res["pairs"]:
                for nm in NORMS:
                    pooled = src["pooled"]["pairs"][pair][nm]
                    blk = parts.get(cat)
                    p_cat = _pair_p(blk, pair, nm) if blk else None
                    ex = [g for g, v in blk["groups"].items() if ls.excludes_zero(v["pairs"][pair][nm])] if blk else []
                    inc = [g for g, v in blk["groups"].items() if not ls.excludes_zero(v["pairs"][pair][nm])] if blk else []
                    r: Dict[str, Any] = {
                        "pooled": pooled,
                        "partition": cat, "category_p": p_cat,
                        "R1_concentrated": p_cat is not None and p_cat < ALPHA,
                        "R1_groups_ci_excludes_zero": ex, "R1_groups_ci_includes_zero": inc,
                        "R2_broad": p_cat is not None and p_cat >= ALPHA and ls.excludes_zero(pooled),
                        "R3": {},
                    }
                    if task == "arabic_exam" and "level" in parts:
                        r["level_p"] = _pair_p(parts["level"], pair, nm)
                    for fr in ("frag_vs_bpe16k", "frag_vs_native"):
                        fb = parts[fr]
                        gaps = [fb["groups"][g]["pairs"][pair][nm]["est"] for g in ("T1", "T2", "T3")]
                        p_fr = _pair_p(fb, pair, nm)
                        mono = monotone(gaps)
                        r["R3"][fr] = {"p": p_fr, "gaps": gaps, "monotone": mono,
                                       "fires": p_fr is not None and p_fr < ALPHA and mono in ("increasing", "decreasing")}
                    out[f"{task}|{dec}|{pair}|{nm}"] = r
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_ci = ls._ci


def _p(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.4f}"


def _acc_cell(g: Dict[str, Any], arm: str) -> str:
    a = g["acc"][arm]
    return f"{a['char']:.3f} / {a['pmi']:.3f}"


def to_markdown(res: Dict[str, Any]) -> str:
    L: List[str] = ["# The Culture-MMLU residual by subject category and by fragmentation", ""]
    L.append(f"Cluster bootstrap over rows ({res['n_boot']} resamples, seed {res['seed']}); heterogeneity = spread "
             f"(max − min over groups) of the paired gap against {res['n_perm']} label permutations that preserve "
             "the group sizes (seed {s}). Standard decision = the full `*_mcq4096` dumps on the rows no cell "
             "truncates; rotation-averaged = the 1 000-row subsets of `_letter_slot/` (char and PMI coincide "
             "there). Pairs: A − B.".format(s=res["seed"]))
    L.append("")
    for task, t in res["tasks"].items():
        cat = "category" if task == "culture_arabic_mmlu" else "group"
        L += [f"## {task}", ""]
        f = t["full"]
        L.append(f"Full dumps: {f['n_total']} rows, {f['n_intersection']} untruncated in every cell "
                 f"(capped per cell: {', '.join(f'{a} {n}' for a, n in f['capped'].items())}). "
                 f"Subset: {t['subset']['n_rows']} rows.")
        L.append("")
        L.append("### Pooled")
        L.append("")
        L.append("| input | decision | pair | char Δ [CI] (McNemar A/B) | PMI Δ [CI] (McNemar A/B) |")
        L.append("|---|---|---|---|---|")
        for lab, blk in (("full", f["pooled"]), ("subset", t["subset"]["pooled"]),
                         ("subset rot0", t["subset"]["pooled_standard_rot0"])):
            dec = "standard" if lab != "subset" else "rotation-averaged"
            for pair, v in blk["pairs"].items():
                L.append(f"| {lab} | {dec} | {pair} | {_ci(v['char'])} ({v['char']['only_a']}/{v['char']['only_b']}) | "
                         f"{_ci(v['pmi'])} ({v['pmi']['only_a']}/{v['pmi']['only_b']}) |")
        L.append("")
        for inp_lab, src, key in (("standard decision, full dumps", f, "partitions"),
                                  ("rotation-averaged decision, subset", t["subset"], "partitions"),
                                  ("standard decision (rotation 0), subset", t["subset"], "partitions_standard_rot0")):
            parts = src[key]
            for name in [n for n in parts if n != "subject"]:
                blk = parts[name]
                L.append(f"### {name} — {inp_lab}")
                if "boundaries" in blk:
                    L.append("")
                    L.append(f"`{blk['ratio']}`; tercile boundaries {blk['boundaries'][0]:.3f} / {blk['boundaries'][1]:.3f}; "
                             f"range {blk['ratio_range'][0]:.3f}–{blk['ratio_range'][1]:.3f}; mean per tercile "
                             + " / ".join(f"{v:.3f}" for v in blk["ratio_mean_by_tercile"].values()))
                L.append("")
                L.append("| group | n | AraRooPat | BPE-16K | native v5 | base | AraRooPat − BPE char | PMI | McNemar char | "
                         "AraRooPat − native char | PMI |")
                L.append("|---|---|---|---|---|---|---|---|---|---|---|")
                for g in blk["order"]:
                    v = blk["groups"][g]
                    pb, pn = v["pairs"]["araroopat_v5-bpe16k_v5"], v["pairs"]["araroopat_v5-native_v5"]
                    L.append(f"| {g} | {v['n']} | {_acc_cell(v, 'araroopat_v5')} | {_acc_cell(v, 'bpe16k_v5')} | "
                             f"{_acc_cell(v, 'native_v5')} | {_acc_cell(v, 'native_base')} | {_ci(pb['char'])} | "
                             f"{_ci(pb['pmi'])} | {pb['char']['only_a']}/{pb['char']['only_b']} | {_ci(pn['char'])} | "
                             f"{_ci(pn['pmi'])} |")
                st = blk.get("spread_test")
                if st:
                    L.append("")
                    L.append("Spread test (observed spread, p): " + "; ".join(
                        f"{k.replace('|', ' ')} {v['observed']:.3f}, p {v['p']:.4f}" for k, v in st.items()
                        if not k.startswith("_")))
                L.append("")
        comp = t.get("composition", {}).get(cat)
        if comp:
            L.append(f"### Composition ({cat}): subset vs full")
            L.append("")
            L.append("| group | full n | full share | subset n | subset share |")
            L.append("|---|---|---|---|---|")
            for g, v in comp.items():
                L.append(f"| {g} | {v['full_n']} | {v['full_share']:.3f} | {v['subset_n']} | {v['subset_share']:.3f} |")
            L.append("")
        if task == "culture_arabic_mmlu":
            subj = f["partitions"]["subject"]["groups"]
            top = sorted(subj.items(), key=lambda kv: -abs(kv[1]["pairs"]["araroopat_v5-bpe16k_v5"]["char"]["est"] or 0))[:5]
            L.append("### Five subjects with the largest |AraRooPat − BPE-16K| (full dumps, char)")
            L.append("")
            L.append("| subject | category | n | Δ char [CI] | Δ PMI [CI] |")
            L.append("|---|---|---|---|---|")
            for s, v in top:
                pb = v["pairs"]["araroopat_v5-bpe16k_v5"]
                L.append(f"| {s} | {SUBJECT_CATEGORY[s]} | {v['n']} | {_ci(pb['char'])} | {_ci(pb['pmi'])} |")
            L.append("")
    L.append("## Readings (pre-registered)")
    L.append("")
    L.append("| task | decision | pair | norm | pooled Δ [CI] | category/group p | R1 | R2 | R3 vs BPE (p, gaps T1/T2/T3, monotone) | R3 vs native |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, r in res["readings"].items():
        task, dec, pair, nm = key.split("|")
        r3 = r["R3"]
        fmt = lambda x: (f"{_p(x['p'])}, " + " / ".join("—" if g is None else f"{g:+.3f}" for g in x["gaps"])
                         + f", {x['monotone']}{' — fires' if x['fires'] else ''}")
        L.append(f"| {task} | {dec} | {pair} | {nm} | {_ci(r['pooled'])} | {_p(r['category_p'])} | "
                 f"{'fires' if r['R1_concentrated'] else '—'} | {'fires' if r['R2_broad'] else '—'} | "
                 f"{fmt(r3['frag_vs_bpe16k'])} | {fmt(r3['frag_vs_native'])} |")
    L.append("")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default=EXPERIMENT, help="folder of the full *_mcq4096 cells")
    ap.add_argument("--letter-slot", default=LETTER_SLOT, help="folder of the <arm>_rot<k> cells")
    ap.add_argument("--rows-dir", default=ROWS_DIR)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arabic-mmlu-dir", default=None, help="a local MBZUAI/ArabicMMLU snapshot (default: the HF cache)")
    ap.add_argument("--name", default="residual_by_subject")
    ap.add_argument("--out-dir", default=None, help="where to write (default: the letter-slot folder)")
    args = ap.parse_args(argv)
    rel = lambda p: Path(p) if Path(p).is_absolute() else REPO_ROOT / p
    exp, lsd, rows_dir = rel(args.experiment), rel(args.letter_slot), rel(args.rows_dir)
    exam_map = arabic_mmlu_group_map(Path(args.arabic_mmlu_dir) if args.arabic_mmlu_dir else None) \
        if "arabic_exam" in args.tasks else None
    res = analyze(exp, lsd, rows_dir, args.tasks, n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed,
                  exam_map=exam_map)
    out = Path(args.out_dir) if args.out_dir else lsd
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(res)
    (out / f"{args.name}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out / (args.name + '.md')} and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
