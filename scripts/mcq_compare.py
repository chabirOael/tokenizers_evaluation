#!/usr/bin/env python
"""Compare N cells of one experiment on the four MCQ benchmarks, row by row.

Every cell scores the same rows with the same prompts (the prompt string is
tokenizer-independent), so the comparison that means something is paired and
restricted to rows every cell actually scored. This script reads each cell's
``eval_rows/<task>.parquet``, joins the cells by ``row_index`` (and checks that
the prompts agree), and reports per task and per cell:

  * rows, ``hit_cap`` / ``all_sentinel`` / ``sentinel`` / ``near_tie`` shares,
    mean ``cont_tokens`` per scored choice (schema-2 dumps only);
  * accuracy under char-norm and PMI on **all rows** and on the
    **intersection** — the rows that hit the cap in *no* cell (``hit_cap`` or
    ``sentinel`` anywhere drops the row for everyone). The intersection is the
    primary comparison: a truncated MCQ prompt loses its question and choices
    (``encode(truncation=True)`` cuts from the right) whether or not the
    continuation still fit;
  * per sub-config (JSON; in the markdown only for Alghafa's nine), and Alghafa's two groups: the four **fixed-label**
    sub-configs (true/false + the three sentiment ones — the same 2–3 label
    phrases on every row, e.g. ``هو رأي ايجابي``) and the five **choice-text**
    MCQ sub-configs (per-row answer phrases). Every Alghafa sub-config scores
    the choice text (LightEval's official prompt; no letter scoring). The
    choice-text group is also split by the **MSA scope rule** (2026-09-25):
    ``choice_text_msa`` (the four MSA exam / reading sub-configs, in scope) and
    ``choice_text_dialect`` (``meta_ar_dialects``, reported, out of scope);
  * for the fixed-label groups (ACVA; Alghafa's true/false and sentiment
    sub-configs) the predicted-label shares under char and PMI against the gold
    shares, flagging a class collapse (one label on >= 90 % of rows) — Alghafa
    shuffles the choice order per row, so a position histogram cannot show it;
  * MEI from ``all_metrics.json``.

For every ``--pairs A:B``: the accuracy difference A − B on the intersection
(char and PMI) with a paired percentile bootstrap over rows
(``freeform_judge.paired_bootstrap`` — 10 000 resamples, seed 0, the
convention of ``scripts/judge/paired_compare.py``) and the McNemar discordant
counts (rows only A got right / only B got right) with the exact two-sided
binomial p.

A cell whose dumps were written at a different ``max_length`` than the
majority is flagged (``†``) — it still enters the intersection, so its
truncated rows leave the primary comparison like anyone's.

    .venv/bin/python scripts/mcq_compare.py \\
        --experiment outputs/experiments/qwen_native_vs_araroopat \\
        --cells araroopat_3phase_v5_distill_mcq4096 bpe_16k_3phase_v5_distill_mcq4096 \\
                native_qwen3_sft_v5_distill_mcq4096 native_qwen3_base_mcq4096 \\
        --pairs araroopat_3phase_v5_distill_mcq4096:bpe_16k_3phase_v5_distill_mcq4096

Writes ``<experiment>/_mcq_compare/<name>.{md,json}`` (``--name``, default
``mcq_compare``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from arabic_eval.judge.freeform_judge import paired_bootstrap  # noqa: E402

TASKS = ("acva", "alghafa", "arabic_exam", "culture_arabic_mmlu")

#: Alghafa sub-configs whose continuations are the same 2–3 label phrases on
#: every row (true/false, sentiment). The other five score per-row answer text.
ALGHAFA_FIXED_LABEL = frozenset({
    "multiple_choice_facts_truefalse_balanced_task",
    "multiple_choice_rating_sentiment_no_neutral_task",
    "multiple_choice_rating_sentiment_task",
    "multiple_choice_sentiment_task",
})

#: The MSA scope rule (2026-09-25, docs/report.md §3.11): the campaign is about MSA — every from-scratch arm's
#: tokenizer corpus, pretraining pool and morphology are MSA by construction — so Alghafa's dialect sub-config is
#: reported but out of the headline scope, for every arm alike; the four MSA exam / reading sub-configs form the
#: in-scope choice-text group. Both groups are subsets of ``choice_text``.
ALGHAFA_CHOICE_TEXT_SCOPE = {
    "choice_text_msa": frozenset({
        "mcq_exams_test_ar",
        "meta_ar_msa",
        "multiple_choice_grounded_statement_soqal_task",
        "multiple_choice_grounded_statement_xglue_mlqa_task",
    }),
    "choice_text_dialect": frozenset({"meta_ar_dialects"}),
}

MAX_MD_SUBCONFIGS = 12

#: A group of rows whose continuations are drawn from at most this many distinct
#: texts is a fixed-label group (ACVA's صح / خطأ, Alghafa's sentiment labels);
#: its predicted-label shares are reported, because a scorer that picks one label
#: on almost every row (class collapse) scores the label's base rate, and Alghafa
#: shuffles the choice order per row, so a position histogram cannot show it.
MAX_FIXED_LABELS = 4
#: A predicted-label share at or above this is flagged as a collapse.
COLLAPSE_SHARE = 0.9

COLUMNS = ("row_index", "source_config", "prompt", "correct_char", "correct_pmi",
           "hit_cap", "all_sentinel", "sentinel", "near_tie", "cont_tokens",
           "continuations", "gold_idx", "pred_idx_char", "pred_idx_pmi")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dump(cell_dir: Path, task: str) -> Optional[Dict[str, Any]]:
    """One cell's dump of one task as numpy columns ordered by ``row_index``."""
    import pyarrow.parquet as pq

    path = cell_dir / "eval_rows" / f"{task}.parquet"
    if not path.exists():
        return None
    pf = pq.ParquetFile(path)
    have = set(pf.schema_arrow.names)
    tbl = pf.read(columns=[c for c in COLUMNS if c in have])
    meta_raw = (pf.schema_arrow.metadata or {}).get(b"arabic_eval")
    meta = json.loads(meta_raw.decode("utf-8")) if meta_raw else {}
    ri = np.asarray(tbl.column("row_index").to_pylist(), dtype=np.int64)
    order = np.argsort(ri, kind="stable")

    def col(name, dtype=None, fill=None):
        if name not in have:
            return None
        vals = tbl.column(name).to_pylist()
        if fill is not None:
            vals = [fill if v is None else v for v in vals]
        arr = np.asarray(vals, dtype=dtype) if dtype is not None else np.asarray(vals, dtype=object)
        return arr[order]

    prompts = tbl.column("prompt").to_pylist()
    conts = tbl.column("continuations").to_pylist()

    def labels(idx_name):
        if idx_name not in have:
            return None
        idx = tbl.column(idx_name).to_pylist()
        return np.asarray([c[i].strip() if i is not None and 0 <= i < len(c) else None
                           for c, i in zip(conts, idx)], dtype=object)[order]

    return {
        "path": path,
        "meta": meta,
        "row_index": ri[order],
        "source_config": col("source_config"),
        "prompt_sha": np.asarray([hashlib.sha1(p.encode("utf-8")).hexdigest() for p in prompts],
                                 dtype=object)[order],
        "correct_char": col("correct_char", bool, fill=False),
        "correct_pmi": col("correct_pmi", bool, fill=False),
        "hit_cap": col("hit_cap", bool, fill=False),
        "all_sentinel": col("all_sentinel", bool, fill=False),
        "sentinel": col("sentinel", bool, fill=False),
        "near_tie": col("near_tie", bool, fill=False),
        "cont_tokens": col("cont_tokens"),
        "label_set": np.asarray([frozenset(c.strip() for c in cs) for cs in conts], dtype=object)[order],
        "gold_label": labels("gold_idx"),
        "pred_label_char": labels("pred_idx_char"),
        "pred_label_pmi": labels("pred_idx_pmi"),
    }


def load_mei(cell_dir: Path, task: str) -> Optional[Dict[str, Any]]:
    p = cell_dir / "all_metrics.json"
    if not p.exists():
        return None
    rec = (json.loads(p.read_text(encoding="utf-8")).get("mei") or {}).get(task)
    if not rec:
        return None
    return {"mei": rec.get("mei"), "status": rec.get("status"),
            "accuracy_source": (rec.get("inputs") or {}).get("accuracy_source")}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _acc(mask: np.ndarray, correct: np.ndarray) -> Optional[float]:
    n = int(mask.sum())
    return round(float(correct[mask].mean()), 6) if n else None


def mean_cont_tokens(d: Dict[str, Any], mask: np.ndarray) -> Optional[float]:
    """Mean number of scored tokens per choice, over the choices that were
    scored (a sentinel choice has 0). ``None`` for a pre-schema-2 dump."""
    ct = d.get("cont_tokens")
    if ct is None:
        return None
    vals = [t for row in ct[mask] if row is not None for t in row if t]
    return round(float(np.mean(vals)), 4) if vals else None


def fixed_label_groups(d: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Row masks of the fixed-label groups of a task: the whole task when all its
    rows draw their continuations from <= MAX_FIXED_LABELS texts (ACVA), else each
    sub-config that does (Alghafa's true/false and sentiment ones)."""
    labels_all = frozenset().union(*d["label_set"]) if len(d["label_set"]) else frozenset()
    if 0 < len(labels_all) <= MAX_FIXED_LABELS:
        return {"_all": np.ones(len(d["label_set"]), dtype=bool)}
    out = {}
    sub = d["source_config"]
    if sub is None:
        return out
    for cfg in sorted(set(sub.tolist())):
        m = sub == cfg
        labels = frozenset().union(*d["label_set"][m])
        if 0 < len(labels) <= MAX_FIXED_LABELS:
            out[cfg] = m
    return out


def label_shares(d: Dict[str, Any], mask: np.ndarray) -> Optional[Dict[str, Any]]:
    """Gold and predicted label shares (char, PMI) over ``mask``, with the top
    predicted label per normalization and a collapse flag."""
    n = int(mask.sum())
    if not n or d.get("gold_label") is None:
        return None

    def shares(arr):
        if arr is None:
            return None
        vals, counts = np.unique(np.asarray([str(v) for v in arr[mask]], dtype=object), return_counts=True)
        return {str(k): round(float(c) / n, 4) for k, c in zip(vals.tolist(), counts.tolist())}

    out: Dict[str, Any] = {"n": n, "gold": shares(d["gold_label"])}
    for norm in ("char", "pmi"):
        sh = shares(d.get(f"pred_label_{norm}"))
        out[norm] = sh
        if sh:
            top = max(sh.items(), key=lambda kv: kv[1])
            out[f"top_{norm}"] = {"label": top[0], "share": top[1],
                                  "gold_share": (out["gold"] or {}).get(top[0], 0.0),
                                  "collapse": top[1] >= COLLAPSE_SHARE}
    return out


def mcnemar_exact_p(only_a: int, only_b: int) -> Optional[float]:
    """Exact two-sided binomial test on the discordant pairs (p = 0.5)."""
    n = only_a + only_b
    if n == 0:
        return None
    k = min(only_a, only_b)
    log_half_n = n * math.log(0.5)
    tail = sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n)
               for i in range(k + 1))
    return round(min(1.0, 2.0 * tail), 6)


def pair_stats(a: np.ndarray, b: np.ndarray, *, n_boot: int, seed: int) -> Dict[str, Any]:
    """A − B on paired 0/1 correctness vectors (same rows, same order)."""
    ids = range(len(a))
    boot = paired_bootstrap({i: float(b[i]) for i in ids}, {i: float(a[i]) for i in ids},
                            n_boot=n_boot, seed=seed)
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    return {
        "n": int(len(a)),
        "acc_a": round(float(a.mean()), 6) if len(a) else None,
        "acc_b": round(float(b.mean()), 6) if len(b) else None,
        "delta": boot["delta_mean"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        "only_a": only_a, "only_b": only_b, "mcnemar_p": mcnemar_exact_p(only_a, only_b),
    }


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

def compare(experiment: Path, cells: Sequence[str], tasks: Sequence[str] = TASKS,
            pairs: Sequence[Tuple[str, str]] = (), *, n_boot: int = 10000, seed: int = 0) -> Dict[str, Any]:
    out: Dict[str, Any] = {"experiment": str(experiment), "cells": list(cells), "tasks": {},
                           "pairs": [list(p) for p in pairs], "n_boot": n_boot, "seed": seed}
    for a, b in pairs:
        for c in (a, b):
            if c not in cells:
                raise SystemExit(f"pair cell {c!r} is not in --cells")
    for task in tasks:
        dumps: Dict[str, Dict[str, Any]] = {}
        missing = []
        for c in cells:
            d = load_dump(experiment / c, task)
            if d is None:
                missing.append(c)
            else:
                dumps[c] = d
        if not dumps:
            out["tasks"][task] = {"missing": missing}
            continue
        ref_cell = next(iter(dumps))
        ref = dumps[ref_cell]
        for c, d in dumps.items():
            if not np.array_equal(d["row_index"], ref["row_index"]):
                raise SystemExit(f"{task}: {c} and {ref_cell} do not hold the same row_index set "
                                 f"({len(d['row_index'])} vs {len(ref['row_index'])} rows)")
            bad = int((d["prompt_sha"] != ref["prompt_sha"]).sum())
            if bad:
                raise SystemExit(f"{task}: {bad} prompts of {c} differ from {ref_cell} at the same "
                                 f"row_index — the dumps are not of the same benchmark rows")
        n = len(ref["row_index"])
        lengths = Counter(d["meta"].get("max_length") for d in dumps.values())
        majority_len = lengths.most_common(1)[0][0]
        capped = {c: (d["hit_cap"] | d["sentinel"]) for c, d in dumps.items()}
        keep = np.ones(n, dtype=bool)
        for m in capped.values():
            keep &= ~m
        # What the rule removed: rows capped in each cell, and rows capped in that cell *only*.
        removed = {}
        for c, m in capped.items():
            others = np.zeros(n, dtype=bool)
            for o, mo in capped.items():
                if o != c:
                    others |= mo
            removed[c] = {"capped": int(m.sum()), "capped_only_here": int((m & ~others).sum())}
        sub = ref["source_config"]
        groups: Dict[str, np.ndarray] = {}
        if task == "alghafa":
            fixed = np.isin(sub, list(ALGHAFA_FIXED_LABEL))
            groups = {"fixed_label": fixed, "choice_text": ~fixed}
            for g, cfgs in ALGHAFA_CHOICE_TEXT_SCOPE.items():
                groups[g] = np.isin(sub, list(cfgs))
        per_cell = {}
        for c, d in dumps.items():
            rec = {
                "max_length": d["meta"].get("max_length"),
                "num_fewshot": d["meta"].get("num_fewshot"),
                "schema_version": d["meta"].get("schema_version"),
                "flag_max_length": d["meta"].get("max_length") != majority_len,
                "rows": n,
                "hit_cap_share": round(float(d["hit_cap"].mean()), 6),
                "all_sentinel_share": round(float(d["all_sentinel"].mean()), 6),
                "sentinel_share": round(float(d["sentinel"].mean()), 6),
                "near_tie_share": round(float(d["near_tie"].mean()), 6),
                "near_tie_share_intersection": _acc(keep, d["near_tie"]),
                "mean_cont_tokens": mean_cont_tokens(d, np.ones(n, dtype=bool)),
                "acc_char_all": _acc(np.ones(n, dtype=bool), d["correct_char"]),
                "acc_pmi_all": _acc(np.ones(n, dtype=bool), d["correct_pmi"]),
                "acc_char_int": _acc(keep, d["correct_char"]),
                "acc_pmi_int": _acc(keep, d["correct_pmi"]),
                "mei": load_mei(experiment / c, task),
            }
            if sub is not None and len(set(sub.tolist())) > 1:
                by = {}
                for cfg in sorted(set(sub.tolist())):
                    m = sub == cfg
                    by[cfg] = {"rows": int(m.sum()), "rows_int": int((m & keep).sum()),
                               "acc_char_all": _acc(m, d["correct_char"]), "acc_pmi_all": _acc(m, d["correct_pmi"]),
                               "acc_char_int": _acc(m & keep, d["correct_char"]),
                               "acc_pmi_int": _acc(m & keep, d["correct_pmi"]),
                               "mean_cont_tokens": mean_cont_tokens(d, m)}
                rec["per_subconfig"] = by
            fl = fixed_label_groups(d)
            if fl:
                rec["label_shares"] = {g: label_shares(d, m & keep) for g, m in fl.items()}
            if groups:
                rec["groups"] = {
                    g: {"rows": int(m.sum()), "rows_int": int((m & keep).sum()),
                        "acc_char_int": _acc(m & keep, d["correct_char"]),
                        "acc_pmi_int": _acc(m & keep, d["correct_pmi"]),
                        "acc_char_all": _acc(m, d["correct_char"]), "acc_pmi_all": _acc(m, d["correct_pmi"])}
                    for g, m in groups.items()}
            per_cell[c] = rec
        task_out: Dict[str, Any] = {
            "rows": n, "intersection_rows": int(keep.sum()), "missing_cells": missing,
            "majority_max_length": majority_len, "removed_by_intersection": removed,
            "cells": per_cell, "pairs": [],
        }
        for a, b in pairs:
            if a not in dumps or b not in dumps:
                task_out["pairs"].append({"a": a, "b": b, "missing": True})
                continue
            entry = {"a": a, "b": b}
            for norm in ("char", "pmi"):
                entry[norm] = pair_stats(dumps[a][f"correct_{norm}"][keep], dumps[b][f"correct_{norm}"][keep],
                                         n_boot=n_boot, seed=seed)
            for g, m in groups.items():
                gm = m & keep
                entry[f"{g}_pmi"] = pair_stats(dumps[a]["correct_pmi"][gm], dumps[b]["correct_pmi"][gm],
                                               n_boot=n_boot, seed=seed)
                entry[f"{g}_char"] = pair_stats(dumps[a]["correct_char"][gm], dumps[b]["correct_char"][gm],
                                                n_boot=n_boot, seed=seed)
            task_out["pairs"].append(entry)
        out["tasks"][task] = task_out
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _f(v: Optional[float], nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{100 * v:.1f} %"


def to_markdown(res: Dict[str, Any]) -> str:
    lines: List[str] = [f"# MCQ comparison — {res['experiment']}", ""]
    lines.append("Primary numbers: the **intersection** of rows that hit the cap in no cell (`hit_cap` or "
                 "`sentinel` in any cell drops the row for all). `†` = dumps at a different `max_length` "
                 "than the majority. Pairs are A − B on the intersection, paired bootstrap 95 % CI "
                 f"({res['n_boot']} resamples, seed {res['seed']}), McNemar only-A / only-B and exact p.")
    lines.append("")
    for task, t in res["tasks"].items():
        if "cells" not in t:
            lines += [f"## {task}", "", f"no dumps (missing: {', '.join(t.get('missing', []))})", ""]
            continue
        lines += [f"## {task} — {t['rows']} rows, intersection {t['intersection_rows']}", ""]
        if t["missing_cells"]:
            lines += [f"missing dumps: {', '.join(t['missing_cells'])}", ""]
        lines.append("| cell | max_len | hit_cap | all_sentinel | sentinel | near_tie | cont_tok | char all | "
                     "PMI all | char ∩ | PMI ∩ | MEI |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c, r in t["cells"].items():
            mei = r["mei"]["mei"] if r.get("mei") else None
            lines.append(
                f"| {c}{' †' if r['flag_max_length'] else ''} | {r['max_length']} | {_pct(r['hit_cap_share'])} | "
                f"{_pct(r['all_sentinel_share'])} | {_pct(r['sentinel_share'])} | {_pct(r['near_tie_share'])} | "
                f"{_f(r['mean_cont_tokens'], 2)} | {_f(r['acc_char_all'])} | {_f(r['acc_pmi_all'])} | "
                f"{_f(r['acc_char_int'])} | {_f(r['acc_pmi_int'])} | {_f(mei, 4)} |")
        lines.append("")
        rem = t["removed_by_intersection"]
        lines.append("Removed by the intersection rule (capped rows / capped in this cell only): "
                     + "; ".join(f"{c} {v['capped']} / {v['capped_only_here']}" for c, v in rem.items()))
        lines.append("")
        first = next(iter(t["cells"].values()))
        # Per sub-config in markdown only when it is readable (Alghafa's 9); the JSON has every task's.
        if "per_subconfig" in first and len(first["per_subconfig"]) <= MAX_MD_SUBCONFIGS:
            cfgs = list(first["per_subconfig"])
            lines.append("Per sub-config, intersection rows — char / PMI:")
            lines.append("")
            lines.append("| cell | " + " | ".join(
                f"{k}{' (fixed)' if k in ALGHAFA_FIXED_LABEL else ''} n={first['per_subconfig'][k]['rows_int']}"
                for k in cfgs) + " |")
            lines.append("|---|" + "---|" * len(cfgs))
            for c, r in t["cells"].items():
                ps = r["per_subconfig"]
                lines.append(f"| {c} | " + " | ".join(
                    f"{_f(ps[k]['acc_char_int'])} / {_f(ps[k]['acc_pmi_int'])}" for k in cfgs) + " |")
            lines.append("")
        if "groups" in first:
            gnames = list(first["groups"])
            lines.append("Groups, intersection rows — char / PMI (`fixed_label` = the four fixed-label sub-configs, "
                         "decided by label priors; `choice_text_msa` = the in-scope MSA choice-text group; "
                         "`choice_text_dialect` = `meta_ar_dialects`, out of scope under the MSA scope rule):")
            lines.append("")
            lines.append("| cell | " + " | ".join(f"{g} n={first['groups'][g]['rows_int']}" for g in gnames) + " |")
            lines.append("|---|" + "---|" * len(gnames))
            for c, r in t["cells"].items():
                g = r["groups"]
                lines.append(f"| {c} | " + " | ".join(
                    f"{_f(g[k]['acc_char_int'])} / {_f(g[k]['acc_pmi_int'])}" for k in gnames) + " |")
            lines.append("")
        if any("label_shares" in r for r in t["cells"].values()):
            gnames = sorted({g for r in t["cells"].values() for g in r.get("label_shares", {})})
            lines.append("Fixed-label groups, intersection rows — the most-predicted label and its share under "
                         f"char / PMI (gold share of that label in brackets; ⚠ = share ≥ {COLLAPSE_SHARE:.0%}, "
                         "a class collapse that scores the label's base rate):")
            lines.append("")
            lines.append("| cell | " + " | ".join("all rows" if g == "_all" else g for g in gnames) + " |")
            lines.append("|---|" + "---|" * len(gnames))

            def _top(ls, norm):
                t_ = (ls or {}).get(f"top_{norm}")
                if not t_:
                    return "—"
                return f"{t_['label']} {t_['share']:.2f} [{t_['gold_share']:.2f}]{' ⚠' if t_['collapse'] else ''}"
            for c, r in t["cells"].items():
                ls = r.get("label_shares", {})
                lines.append(f"| {c} | " + " | ".join(
                    f"{_top(ls.get(g), 'char')} / {_top(ls.get(g), 'pmi')}" for g in gnames) + " |")
            lines.append("")
        if t["pairs"]:
            lines.append("| A − B | norm | n | acc A | acc B | Δ | 95 % CI | only A | only B | McNemar p |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for p in t["pairs"]:
                if p.get("missing"):
                    lines.append(f"| {p['a']} − {p['b']} | — | missing dump | | | | | | | |")
                    continue
                keys = ["char", "pmi"] + [k for k in p if k.endswith(("_char", "_pmi")) and k not in ("char", "pmi")]
                for k in keys:
                    s = p[k]
                    if s["delta"] is None:            # a group with no intersection rows
                        lines.append(f"| {p['a']} − {p['b']} | {k} | 0 | — | — | — | — | — | — | — |")
                        continue
                    lines.append(f"| {p['a']} − {p['b']} | {k} | {s['n']} | {_f(s['acc_a'])} | {_f(s['acc_b'])} | "
                                 f"{s['delta']:+.3f} | [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}] | {s['only_a']} | "
                                 f"{s['only_b']} | {_f(s['mcnemar_p'], 4)} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, help="experiment folder holding the cells")
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--pairs", nargs="*", default=[], help="A:B (reports A − B)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="mcq_compare", help="output file stem under <experiment>/_mcq_compare/")
    args = ap.parse_args(argv)
    exp = Path(args.experiment)
    pairs = []
    for p in args.pairs:
        a, sep, b = p.partition(":")
        if not sep:
            raise SystemExit(f"--pairs entries are A:B, got {p!r}")
        pairs.append((a, b))
    res = compare(exp, args.cells, args.tasks, pairs, n_boot=args.n_boot, seed=args.seed)
    out_dir = exp / "_mcq_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(res)
    (out_dir / f"{args.name}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out_dir / (args.name + '.md')} and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
