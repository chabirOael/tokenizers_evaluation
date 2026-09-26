"""Cells compared on the same items: MCQ rows joined on row_index, free-form prompts on their id.

The MCQ side delegates to ``scripts/mcq_compare.py`` (the report's numbers: the
rows no cell truncated, char and PMI, paired bootstrap + McNemar per pair); the
judge side to ``scripts/judge/paired_compare.py``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from arabic_eval.analysis._root import experiments_root, script_module
from arabic_eval.analysis.inventory import MCQ_TASKS, resolve, resolve_many, short_names


def _mcq_compare_mod():
    return script_module("scripts/mcq_compare.py")


def mcq_compare(cells: Sequence[Any], tasks: Sequence[str] = MCQ_TASKS, pairs: Sequence[Tuple[Any, Any]] = (), *,
                n_boot: int = 10000, seed: int = 0) -> Dict[str, Any]:
    """The report's MCQ comparison (``scripts/mcq_compare.py``) as a dict.

    Per task: ``rows``, ``intersection_rows`` (rows that hit the cap / a sentinel in
    NO cell — the primary comparison), per cell ``acc_char_all`` / ``acc_pmi_all`` /
    ``acc_char_int`` / ``acc_pmi_int``, per-sub-config accuracies, Alghafa groups
    (fixed_label, choice_text, choice_text_msa = in-scope, choice_text_dialect = out of
    scope), label shares with a collapse flag for fixed-label groups; per pair (a, b):
    ``char`` / ``pmi`` with delta = a − b, ci_low, ci_high, only_a, only_b, mcnemar_p.
    Cells may come from different experiments. Tables: ``mcq_table(res)``, ``mcq_pairs_table(res)``."""
    cs = resolve_many(cells)
    ids = [c.id for c in cs]
    pair_ids = [(resolve(a).id, resolve(b).id) for a, b in pairs]
    try:
        return _mcq_compare_mod().compare(experiments_root(), ids, tuple(tasks), pair_ids, n_boot=n_boot, seed=seed)
    except SystemExit as e:                      # the script reports misaligned dumps this way
        raise ValueError(str(e)) from None


def mcq_table(res: Dict[str, Any]):
    """Per (task, cell) accuracies of an ``mcq_compare`` result: all rows and the intersection."""
    import pandas as pd
    rows = []
    for task, t in (res.get("tasks") or {}).items():
        for cell, c in (t.get("cells") or {}).items():
            rows.append({"task": task, "cell": cell, "rows": c.get("rows"), "intersection": t.get("intersection_rows"),
                         "acc_char_all": c.get("acc_char_all"), "acc_pmi_all": c.get("acc_pmi_all"),
                         "acc_char_int": c.get("acc_char_int"), "acc_pmi_int": c.get("acc_pmi_int"),
                         "hit_cap_share": c.get("hit_cap_share"), "max_length": c.get("max_length"),
                         "num_fewshot": c.get("num_fewshot")})
    return pd.DataFrame(rows)


def mcq_pairs_table(res: Dict[str, Any]):
    """Per (task, pair, normalization[, group]) paired differences of an ``mcq_compare`` result (delta = a − b)."""
    import pandas as pd
    rows = []
    for task, t in (res.get("tasks") or {}).items():
        for p in t.get("pairs") or []:
            if p.get("missing"):
                continue
            for key, s in p.items():
                if key in ("a", "b") or not isinstance(s, dict):
                    continue
                group, _, norm = key.rpartition("_") if key not in ("char", "pmi") else ("all", "", key)
                rows.append({"task": task, "a": p["a"], "b": p["b"], "group": group, "norm": norm, **s})
    return pd.DataFrame(rows)


def mcq_paired(cells: Sequence[Any], task: str, *, clean_only: bool = True):
    """The same benchmark rows in several cells, one row per ``row_index``.

    Columns: ``subconfig``, ``gold`` (gold label text), ``clean`` (no cell hit the cap
    or a sentinel on this row); per cell (short name): ``<cell>.char`` / ``<cell>.pmi``
    (correct?), ``<cell>.pred_char`` / ``<cell>.pred_pmi`` (predicted label text),
    ``<cell>.capped``. ``clean_only=True`` keeps the intersection — the rows every
    report comparison uses. Raises when the cells do not hold the same rows / prompts."""
    import pandas as pd
    mod = _mcq_compare_mod()
    cs = resolve_many(cells)
    names = short_names(cs)
    dumps = []
    for c in cs:
        d = mod.load_dump(c.path, task)
        if d is None:
            raise FileNotFoundError(f"{c.id} has no eval_rows/{task}.parquet")
        dumps.append(d)
    ref = dumps[0]
    for c, d in zip(cs[1:], dumps[1:]):
        if not np.array_equal(d["row_index"], ref["row_index"]):
            raise ValueError(f"{task}: {c.id} and {cs[0].id} do not hold the same rows "
                             f"({len(d['row_index'])} vs {len(ref['row_index'])})")
        bad = int((d["prompt_sha"] != ref["prompt_sha"]).sum())
        if bad:
            raise ValueError(f"{task}: {bad} prompts of {c.id} differ from {cs[0].id} at the same row_index")
    capped = np.zeros(len(ref["row_index"]), dtype=bool)
    cols: Dict[str, Any] = {"subconfig": ref["source_config"], "gold": ref["gold_label"]}
    for n, d in zip(names, dumps):
        m = d["hit_cap"] | d["sentinel"]
        capped |= m
        cols[f"{n}.char"] = d["correct_char"]
        cols[f"{n}.pmi"] = d["correct_pmi"]
        cols[f"{n}.pred_char"] = d["pred_label_char"]
        cols[f"{n}.pred_pmi"] = d["pred_label_pmi"]
        cols[f"{n}.capped"] = m
    df = pd.DataFrame(cols, index=pd.Index(ref["row_index"], name="row_index"))
    df.insert(2, "clean", ~capped)
    return df[df["clean"]] if clean_only else df


def _judge_scores(cell_dir, judge: str, field: str) -> Dict[str, float]:
    mod = script_module("scripts/judge/paired_compare.py")
    p = cell_dir / "freeform_judge" / f"{judge}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{cell_dir} has no freeform_judge/{judge}.parquet")
    return mod.load_scores(p, field)


def freeform_paired(cells: Sequence[Any], judge: str = "gemma4_31b", field: str = "score"):
    """One judge's scores for several cells on the same prompts (index = prompt id).

    Columns: ``stratum`` (reference-length tercile: short / medium / long) and one column
    per cell (short name). ``field``: score | correctness | fluency | instruction_following.
    Unparsed verdicts are NaN (the scripts skip them the same way)."""
    import pandas as pd
    cs = resolve_many(cells)
    names = short_names(cs)
    data = {n: _judge_scores(c.path, judge, field) for n, c in zip(names, cs)}
    df = pd.DataFrame(data)
    df.index.name = "id"
    strata = pd.read_parquet(cs[0].path / "freeform_judge" / f"{judge}.parquet", columns=["id", "stratum"]).set_index("id")["stratum"]
    df.insert(0, "stratum", strata.reindex(df.index))
    return df.sort_index()


def judge_compare(baseline: Any, cell: Any, judge: str = "gemma4_31b", *, n_boot: int = 10000, seed: int = 0,
                  sub_scores: bool = False) -> Dict[str, Any]:
    """``scripts/judge/paired_compare.py``: Δ = **cell − baseline** on the prompts both answered.

    Returns mean_baseline, mean_cell, and ``score`` = {n, delta_mean, ci_low, ci_high,
    win_rate, tie_rate, loss_rate} (win = cell scored higher); ``sub_scores=True``
    adds the same for correctness / fluency / instruction_following."""
    mod = script_module("scripts/judge/paired_compare.py")
    a, b = resolve(baseline), resolve(cell)
    for c in (a, b):
        _judge_scores(c.path, judge, "score")    # a clear error when the judge file is missing
    res = mod.compare(None, str(a.path), str(b.path), judge, n_boot=n_boot, seed=seed, sub_scores=sub_scores)
    res["baseline"], res["cell"] = a.id, b.id
    return res
