"""Read a cell's numbers and rows into pandas."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from arabic_eval.analysis.inventory import (
    FREEFORM_TASK,
    MCQ_TASKS,
    PHASES,
    Cell,
    cells,
    read_json,
    resolve,
    resolve_many,
)

INTRINSIC_KEYS = ("fertility", "compression_ratio", "unk_rate", "vocab_size", "root_conservation_rate",
                  "root_conservation_attainable", "pattern_conservation_rate", "semantic_fragmentation_ratio",
                  "morph_alignment_coverage")
FREEFORM_KEYS = ("num_samples", "chrf", "chrf_corpus", "bertscore_f1", "empty_rate", "degenerate_rate", "latin_rate",
                 "loop_stop_rate", "eos_rate", "hit_cap_rate", "marker_stop_rate", "char_truncated_rate",
                 "mean_gen_chars", "mean_gen_tokens", "mean_ref_chars", "gen_chars_per_sec",
                 "reference_roundtrip_chrf", "chars_per_token", "token_cap", "max_output_chars")


def metrics_json(cell: Any) -> Dict[str, Any]:
    """The cell's ``all_metrics.json`` as a dict ({} when absent)."""
    return read_json(resolve(cell).path / "all_metrics.json") or {}


def config_json(cell: Any) -> Dict[str, Any]:
    """The resolved config the cell ran with (``config.json``)."""
    return read_json(resolve(cell).path / "config.json") or {}


def metric(cell: Any, dotted: str, default: Any = None) -> Any:
    """One value of ``all_metrics.json`` by dotted path, e.g. ``"downstream.arabic_exam.accuracy_pmi"``."""
    node: Any = metrics_json(cell)
    for key in dotted.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
        else:
            return default
    return node


def _selected(refs: Optional[Iterable[Any]], experiment: Optional[str]) -> List[Cell]:
    if refs is not None:
        return resolve_many(refs)
    out = cells()
    if experiment:
        out = [c for c in out if c.id == experiment or c.id.startswith(experiment.rstrip("/") + "/")]
    return out


def metrics(cells: Optional[Iterable[Any]] = None, *, experiment: Optional[str] = None):
    """Headline numbers, one row per cell (index = cell id), from ``all_metrics.json``.

    Columns: ``tokenizer``; ``intrinsic.<key>``; per MCQ task ``<task>.acc_char`` /
    ``<task>.acc_pmi`` / ``<task>.n`` / ``<task>.mei`` — accuracies over ALL rows of
    that cell (not the paired, untruncated intersection: use ``mcq_compare`` for
    comparisons); ``freeform.<key>`` (chrf, bertscore_f1, loop_stop_rate, eos_rate,
    mean_gen_chars, …); ``judge.<judge>.mean`` / ``.se`` / ``.n``. The bare
    ``accuracy`` field of all_metrics is deliberately not used — it follows the
    dump's primary normalization (PMI in the v5 cells), not char. Columns that are
    empty for every selected cell are dropped."""
    import pandas as pd
    rows = []
    for c in _selected(cells, experiment):
        am = metrics_json(c)
        if not am:
            continue
        cfg = config_json(c)
        rec: Dict[str, Any] = {"cell": c.id,
                               "tokenizer": ((cfg.get("tokenizer") or {}).get("type")) or (am.get("config") or {}).get("tokenizer")}
        intr = am.get("intrinsic") or {}
        for k in INTRINSIC_KEYS:
            rec[f"intrinsic.{k}"] = intr.get(k)
        down = am.get("downstream") or {}
        mei = am.get("mei") or {}
        for task in MCQ_TASKS:
            d = down.get(task)
            if not isinstance(d, dict):
                continue
            has_char = "accuracy_char_norm" in d
            rec[f"{task}.acc_char"] = d.get("accuracy_char_norm") if has_char else (
                d.get("accuracy") if "accuracy_pmi" not in d else None)
            rec[f"{task}.acc_pmi"] = d.get("accuracy_pmi")
            rec[f"{task}.n"] = d.get("num_samples")
            rec[f"{task}.mei"] = (mei.get(task) or {}).get("mei")
        ff = down.get(FREEFORM_TASK)
        if isinstance(ff, dict):
            rec["freeform.status"] = ff.get("status")
            for k in FREEFORM_KEYS:
                rec[f"freeform.{k}"] = ff.get(k)
            for j, s in sorted((ff.get("judge") or {}).items()):
                rec[f"judge.{j}.mean"] = s.get("score_mean")
                rec[f"judge.{j}.se"] = s.get("score_se")
                rec[f"judge.{j}.n"] = s.get("n")
        rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=["cell"]).set_index("cell")
    df = pd.DataFrame(rows).set_index("cell")
    return df.dropna(axis=1, how="all")


def _leaves(node: Any, prefix: str, out: List[tuple]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            _leaves(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(node, (list, tuple)):
        return                                   # histories and tails: use training_history / eval_history
    else:
        out.append((prefix, node))


def metrics_long(cells: Optional[Iterable[Any]] = None, *, experiment: Optional[str] = None, contains: Optional[str] = None):
    """Every scalar of ``all_metrics.json`` as rows (cell, key, value); lists are skipped.
    ``contains`` keeps the keys containing that substring (e.g. ``"judge.gemma4_31b.score_by_stratum"``)."""
    import pandas as pd
    rows = []
    for c in _selected(cells, experiment):
        leaves: List[tuple] = []
        _leaves(metrics_json(c), "", leaves)
        rows.extend({"cell": c.id, "key": k, "value": v} for k, v in leaves if not contains or contains in k)
    return pd.DataFrame(rows, columns=["cell", "key", "value"])


def dump_path(cell: Any, task: str) -> Path:
    c = resolve(cell)
    p = c.path / "eval_rows" / f"{task}.parquet"
    if not p.exists():
        have = sorted(q.stem for q in (c.path / "eval_rows").glob("*.parquet")) if (c.path / "eval_rows").is_dir() else []
        raise FileNotFoundError(f"{c.id} has no eval_rows/{task}.parquet (it has: {have or 'no dumps'})")
    return p


def dump_metadata(cell: Any, task: str) -> Dict[str, Any]:
    """The dump's own metadata (max_length, num_fewshot, schema_version, decoding, …)."""
    from arabic_eval.analysis.inventory import parquet_meta
    return parquet_meta(dump_path(cell, task))[0]


def mcq_rows(cell: Any, task: str, columns: Optional[Sequence[str]] = None):
    """Every scored row of one MCQ benchmark in one cell (``eval_rows/<task>.parquet``).

    Useful columns: row_index (the join key across cells), source_config (sub-config),
    prompt (exact scored prompt — long, avoid printing), question, choices, continuations,
    gold_idx, gold_text, pred_idx_char, pred_idx_pmi, correct_char, correct_pmi, ll,
    score_char, score_pmi, uncond_ll (per-choice lists), cont_tokens (schema ≥ 2),
    cont_token_ll (schema 3), decision_margin, prompt_units (tokenizer's own units),
    n_choices, and the flags sentinel / all_sentinel / hit_cap / near_tie / disagree.
    ``columns`` reads only those (much faster on the big benchmarks)."""
    import pandas as pd
    return pd.read_parquet(dump_path(cell, task), columns=list(columns) if columns else None)


def freeform_rows(cell: Any, *, judges: Any = True, rationale: bool = False):
    """The cell's 250 free-form generations (index = prompt id) joined with its judge verdicts.

    Generation columns: stratum (reference-length tercile), instruction, context, reference,
    generation (after the loop cut), generation_raw, gen_chars, gen_tokens, ref_chars,
    stop_reason (eos | loop | marker | cap), hit_loop, loop_rule, loop_period, hit_cap,
    empty, degenerate, latin, chrf, bertscore_f1, reference_roundtrip_chrf, gen_time_sec.
    Judge columns per judge J: ``J.score`` (1–5), ``J.correctness``, ``J.fluency``,
    ``J.instruction_following``, ``J.flags`` (comma-separated), ``J.parse_ok`` (+ ``J.rationale``).
    ``judges``: True = every judge file, a name or a list of names, False = none."""
    import pandas as pd
    c = resolve(cell)
    df = pd.read_parquet(dump_path(c, FREEFORM_TASK)).set_index("id")
    names = _judge_names(c, judges)
    for j in names:
        jd = judge_rows(c, j)
        keep = ["score", "correctness", "fluency", "instruction_following", "flags", "parse_ok"] + (["rationale"] if rationale else [])
        jd = jd[[k for k in keep if k in jd.columns]].add_prefix(f"{j}.")
        df = df.join(jd, how="left")
    return df


def _judge_names(c: Cell, judges: Any) -> List[str]:
    d = c.path / "freeform_judge"
    have = [p.stem for p in sorted(d.glob("*.parquet"))] if d.is_dir() else []
    if judges is True:
        return have
    if not judges:
        return []
    want = [judges] if isinstance(judges, str) else list(judges)
    missing = [j for j in want if j not in have]
    if missing:
        raise FileNotFoundError(f"{c.id} has no judge file for {missing} (it has: {have or 'none'})")
    return want


def judge_rows(cell: Any, judge: str):
    """One judge's verdicts on a cell (index = prompt id): score, sub-scores, flags, rationale, parse_ok, raw."""
    import pandas as pd
    c = resolve(cell)
    p = c.path / "freeform_judge" / f"{judge}.parquet"
    if not p.exists():
        _judge_names(c, [judge])                 # raises with the list of judges it has
    return pd.read_parquet(p).set_index("id")


def training_history(cell: Any, phase: str = "sft"):
    """The last training losses a phase recorded (step, loss) — all_metrics keeps a tail of ~200 steps."""
    import pandas as pd
    tr = (metrics_json(cell).get("training") or {}).get(phase) or {}
    tail = tr.get("train_losses_tail") or []
    return pd.DataFrame([{"step": s, "loss": l} for s, l in tail], columns=["step", "loss"])


def eval_history(cell: Any, phase: str = "sft"):
    """The phase's periodic early-stop evaluations: step, loss, tokens and, for an
    ``eval_mixture`` signal, one ``<category>.loss`` column per category.

    The *best* step the phase restored is ``phase_summary(cell).loc[phase, "best_eval_step"]``:
    early stopping counts an improvement only when it beats the best by more than ``min_delta``
    (5e-4), so it can differ from the plain minimum of this table (araroopat_3phase_v5_distill:
    restored 17 000 at 0.8195; the table's minimum is 19 500 at 0.8192)."""
    import pandas as pd
    tr = (metrics_json(cell).get("training") or {}).get(phase) or {}
    hist = tr.get("eval_history")
    if not hist:
        losses = tr.get("eval_losses") or []
        return pd.DataFrame([{"step": s, "loss": l} for s, l in losses] if losses and isinstance(losses[0], (list, tuple))
                            else [{"loss": l} for l in losses])
    rows = []
    for h in hist:
        r = {"step": h.get("step"), "loss": h.get("loss"), "tokens": h.get("tokens")}
        for cat, v in (h.get("per_category") or {}).items():
            r[f"{cat}.loss"] = v.get("loss")
            r[f"{cat}.tokens"] = v.get("tokens")
        rows.append(r)
    return pd.DataFrame(rows)


def phase_summary(cell: Any):
    """One row per training phase: status, steps, final / best losses, early stop, wall time."""
    import pandas as pd
    tr = metrics_json(cell).get("training") or {}
    rows = []
    for p in PHASES:
        t = tr.get(p) or {}
        rows.append({"phase": p, "status": t.get("status"), "steps_completed": t.get("steps_completed"),
                     "final_train_loss": t.get("final_train_loss"), "best_eval_loss": t.get("best_eval_loss"),
                     "best_eval_step": t.get("best_eval_step"), "early_stopped": t.get("early_stopped"),
                     "wall_time_min": round(t["wall_time_sec"] / 60, 1) if t.get("wall_time_sec") else None,
                     "eval_loss_definition": t.get("eval_loss_definition")})
    return pd.DataFrame(rows).set_index("phase")


def diag(cell: Any, name: Optional[str] = None) -> Any:
    """The cell's diagnostic JSONs (``diag_*.json``: held-out loss, uncertainty, …).
    Without ``name``: the list of file names; with one: that file's dict."""
    c = resolve(cell)
    names = sorted(p.name for p in c.path.glob("diag_*.json"))
    if name is None:
        return names
    fn = name if name.endswith(".json") else f"{name}.json"
    if fn not in names:
        raise FileNotFoundError(f"{c.id} has no {fn} (it has: {names or 'none'})")
    return read_json(c.path / fn)
