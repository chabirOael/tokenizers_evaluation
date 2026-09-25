#!/usr/bin/env python
"""Letter or slot: decompose the first-option deficit of the letter-scored MCQ tasks (docs/report.md §3.11).

On Arabic-Exam and Culture-MMLU the letter أ always labels slot 1, so a deficit on gold = أ rows cannot say
whether it belongs to the *letter* or to the *slot*. The twenty eval-only cells of ``_letter_slot/`` score the
same 1 000 rows per task (``configs/experiments/mcq_letter_slot/rows_<task>.json``) with the letters rotated
over the slots by k = 0..3 (``label_rotation``; prompt, demonstrations and continuations together), plus a 0-shot
pass at rotation 0. This script joins them on ``row_index`` and reports, per arm and task, under char-norm and PMI:

  1. the letter × slot accuracy table over the four rotations, its marginals, and the two effects
     ``E_letter = acc(gold letter ≠ أ) − acc(gold letter = أ)`` and ``E_slot = acc(gold slot ≠ 1) − acc(gold slot = 1)``
     with the interaction term (the أ-at-slot-1 cell against the additive prediction of the marginals);
  2. arm pairs (A − B) paired over (row, rotation), split by gold letter and by gold slot;
  3. 0-shot against 3-shot rotation 0 on the same rows, per arm and for the arm pairs;
  4. per-token means from ``cont_token_ll`` (schema 3) — for AraRooPat the three terms ``[LIT_BEGIN]``,
     ``[CHAR_x]``, ``[LIT_END]`` per letter, gold vs non-gold, by slot — and three post-hoc *diagnostic*
     scorings: (i) the sum of the terms (the official char-norm decision, reproduced), (ii) the letter term
     alone, (iii) letter term + terminal term; for a one-token letter the three coincide;
  5. predicted-letter and predicted-slot distributions per rotation, next to the gold's;
  6. a *proposal diagnostic*: the rotation-averaged decision — each option scored under its four letters and
     the four scores averaged before the argmax, so every letter's prior reaches every option equally
     (letter-invariant by construction; four forward passes per row). Computed from the dumps to show what the
     proposal would change; applied to nothing.

Rows are the unit; a row's four rotations are one cluster. Every interval is a cluster bootstrap over rows
(``--n-boot`` 5 000 resamples of the rows with replacement, keeping all their rotations,
``numpy.random.default_rng(seed)``, 2.5 / 97.5 percentiles); one resample matrix per task serves every arm and
pair, so pair intervals are paired. The pre-registered reading rules (the brief's P2.3) are evaluated and
recorded; nothing here is a gate, and the diagnostic scorings are diagnostics of the mechanism, not a protocol.

    .venv/bin/python scripts/mcq_letter_slot.py
Writes ``<experiment>/letter_slot.{md,json}``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from arabic_eval.tasks.lighteval.utils import ARABIC_CHOICE_LETTERS, choice_letters  # noqa: E402

EXPERIMENT = "outputs/experiments/qwen_native_vs_araroopat/_letter_slot"
ROWS_DIR = "configs/experiments/mcq_letter_slot"
TASKS = ("arabic_exam", "culture_arabic_mmlu")
ARMS = ("araroopat_v5", "bpe16k_v5", "native_v5", "native_base")
ROTATIONS = (0, 1, 2, 3)
ZERO_SHOT = "0shot"
PAIRS = (
    ("araroopat_v5", "bpe16k_v5"), ("araroopat_v5", "native_v5"), ("bpe16k_v5", "native_v5"),
    ("araroopat_v5", "native_base"), ("bpe16k_v5", "native_base"), ("native_v5", "native_base"),
)
#: The arm the pre-registered rules read (and the comparator of the terminal-token rule).
FOCUS, COMPARATOR = "araroopat_v5", "bpe16k_v5"
N_SLOTS = 4
LETTERS = ARABIC_CHOICE_LETTERS[:N_SLOTS]
NORMS = ("char", "pmi")
TERM_NAMES_3 = ("LIT_BEGIN", "CHAR", "LIT_END")
DIAG_SCORINGS = ("i_sum", "ii_letter", "iii_letter_end")

COLUMNS = ("row_index", "gold_idx", "pred_idx_char", "pred_idx_pmi", "correct_char", "correct_pmi",
           "continuations", "ll", "score_pmi", "cont_tokens", "cont_token_ll", "hit_cap", "all_sentinel", "sentinel",
           "n_choices", "prompt")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def cell_name(arm: str, cond: str) -> str:
    return f"{arm}_{cond}"


def load_dump(path: Path) -> Dict[str, Any]:
    """One dump as a dict of per-row lists keyed by column, plus the metadata."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    have = set(pf.schema_arrow.names)
    tbl = pf.read(columns=[c for c in COLUMNS if c in have])
    raw = (pf.schema_arrow.metadata or {}).get(b"arabic_eval")
    meta = json.loads(raw.decode("utf-8")) if raw else {}
    out = {c: tbl.column(c).to_pylist() for c in tbl.column_names}
    out["_meta"] = meta
    return out


def align(dump: Dict[str, Any], rows: Sequence[int]) -> Dict[str, Any]:
    """Reorder a dump to ``rows`` (the rows file's order); every listed row must be present exactly once."""
    pos = {int(r): i for i, r in enumerate(dump["row_index"])}
    if len(pos) != len(dump["row_index"]):
        raise ValueError("duplicate row_index in a dump")
    missing = [r for r in rows if r not in pos]
    if missing:
        raise ValueError(f"{len(missing)} listed rows missing from the dump (e.g. {missing[:5]})")
    order = [pos[r] for r in rows]
    return {k: ([v[i] for i in order] if k != "_meta" else v) for k, v in dump.items()}


def build_panel(dumps_by_rot: Dict[int, Dict[str, Any]], rows: Sequence[int]) -> Dict[str, Any]:
    """Per arm: arrays over (row, rotation) and, when the dumps carry them, per-token terms.

    ``gold_slot[r]`` is the gold index (the slot; the parser is untouched by the rotation) and must agree across
    rotations; the gold letter of (r, k) is ``(slot + k) % 4``. Each dump's continuations are checked against
    ``choice_letters(4, k)``, so a mislabelled cell cannot enter silently.
    """
    rots = sorted(dumps_by_rot)
    n, K = len(rows), len(rots)
    gold = np.zeros((n, K), dtype=np.int64)
    pred = {nm: np.zeros((n, K), dtype=np.int64) for nm in NORMS}
    correct = {nm: np.zeros((n, K), dtype=bool) for nm in NORMS}
    ll = np.zeros((n, K, N_SLOTS), dtype=np.float64)
    spmi = np.zeros((n, K, N_SLOTS), dtype=np.float64)
    flags = {"hit_cap": 0, "all_sentinel": 0, "sentinel": 0}
    terms: Optional[np.ndarray] = None
    n_terms: Optional[int] = None
    terms_complete = True                 # every rotation carries cont_token_ll (schema 3)
    token_sum_mismatch = 0.0
    for j, k in enumerate(rots):
        d = align(dumps_by_rot[k], rows)
        expected = [" " + l for l in choice_letters(N_SLOTS, k)]
        for r in range(n):
            if d["n_choices"][r] != N_SLOTS:
                raise ValueError(f"row {rows[r]}: {d['n_choices'][r]} choices, expected {N_SLOTS}")
            if list(d["continuations"][r]) != expected:
                raise ValueError(f"rotation {k}, row {rows[r]}: continuations {d['continuations'][r]} != {expected}")
        gold[:, j] = d["gold_idx"]
        for nm in NORMS:
            pred[nm][:, j] = d[f"pred_idx_{nm}"]
            correct[nm][:, j] = d[f"correct_{nm}"]
        ll[:, j, :] = np.asarray(d["ll"], dtype=np.float64)
        if d.get("score_pmi") is not None and all(v is not None for v in d["score_pmi"]):
            spmi[:, j, :] = np.asarray(d["score_pmi"], dtype=np.float64)
        else:
            spmi[:, j, :] = np.nan
        for f in flags:
            flags[f] += int(sum(bool(v) for v in d.get(f, [False] * n)))
        ctl = d.get("cont_token_ll")
        if ctl is not None and all(v is not None for v in ctl):
            lens = {len(t) for row in ctl for t in row}
            if len(lens) != 1:
                raise ValueError(f"rotation {k}: continuations of different token counts {sorted(lens)}")
            m = lens.pop()
            if n_terms is None:
                n_terms = m
                terms = np.zeros((n, K, N_SLOTS, m), dtype=np.float64)
            elif m != n_terms:
                raise ValueError(f"rotation {k}: {m} terms per letter, earlier rotations {n_terms}")
            terms[:, j, :, :] = np.asarray(ctl, dtype=np.float64)
            token_sum_mismatch = max(token_sum_mismatch, float(np.abs(
                terms[:, j].sum(-1) - np.asarray(d["ll"], dtype=np.float32).astype(np.float64)).max()))
        else:
            terms_complete = False
    if not terms_complete:
        terms, n_terms = None, None
    if (gold != gold[:, :1]).any():
        raise ValueError("the gold slot differs across rotations for some row — not the same rows")
    rot_arr = np.asarray(rots)[None, :]
    slot = gold[:, 0]
    return {
        "rows": list(rows), "rotations": rots,
        "gold_slot": slot,                                      # [n]
        "gold_letter": (gold + rot_arr) % N_SLOTS,               # [n, K]
        "pred_slot": pred,                                       # {norm: [n, K]}
        "pred_letter": {nm: (pred[nm] + rot_arr) % N_SLOTS for nm in NORMS},
        "correct": correct,                                      # {norm: [n, K]}
        "ll": ll,                                                # [n, K, 4]
        "score_pmi": spmi,                                       # [n, K, 4]
        "terms": terms,                                          # [n, K, 4, m] or None
        "n_terms": n_terms,
        "flags": flags,
        "max_abs_token_sum_minus_ll": token_sum_mismatch if terms is not None else None,
    }


# ---------------------------------------------------------------------------
# Cluster bootstrap over rows
# ---------------------------------------------------------------------------

def boot_weights(n: int, n_boot: int, seed: int) -> np.ndarray:
    """``[n_boot + 1, n]`` row weights: row 0 is the point estimate (all ones), rows 1.. are the resamples."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    W = np.empty((n_boot + 1, n), dtype=np.float64)
    W[0] = 1.0
    for b in range(n_boot):
        W[b + 1] = np.bincount(idx[b], minlength=n)
    return W


def _ratio(W: np.ndarray, num_rows: np.ndarray, den_rows: np.ndarray) -> np.ndarray:
    den = W @ den_rows
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (W @ num_rows) / np.where(den > 0, den, 1.0), np.nan)


def wmean(W: np.ndarray, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Weighted mean of ``values`` over ``mask``, both ``[n, K]`` (or ``[n]``), for every weight row."""
    v = values.reshape(len(values), -1).astype(np.float64)
    m = mask.reshape(len(mask), -1).astype(np.float64)
    return _ratio(W, (v * m).sum(1), m.sum(1))


def summarize(stat: np.ndarray, nd: int = 6) -> Dict[str, Any]:
    point = float(stat[0])
    boots = stat[1:][~np.isnan(stat[1:])]
    if np.isnan(point) or not len(boots):
        return {"est": None, "lo": None, "hi": None}
    return {"est": round(point, nd), "lo": round(float(np.percentile(boots, 2.5)), nd),
            "hi": round(float(np.percentile(boots, 97.5)), nd)}


def excludes_zero(s: Dict[str, Any]) -> bool:
    return s["lo"] is not None and (s["lo"] > 0 or s["hi"] < 0)


# ---------------------------------------------------------------------------
# 1. Letter × slot
# ---------------------------------------------------------------------------

def letter_slot_block(correct: np.ndarray, gold_letter: np.ndarray, gold_slot: np.ndarray,
                      W: np.ndarray) -> Dict[str, Any]:
    """The 4×4 table, the marginals, E_letter, E_slot and the interaction, for one 0/1 matrix ``[n, K]``."""
    slot2 = np.broadcast_to(gold_slot[:, None], correct.shape)
    c = correct.astype(np.float64)
    ones = np.ones_like(c, dtype=bool)
    table = [[None] * N_SLOTS for _ in range(N_SLOTS)]
    counts = [[0] * N_SLOTS for _ in range(N_SLOTS)]
    for x in range(N_SLOTS):
        for s in range(N_SLOTS):
            m = (gold_letter == x) & (slot2 == s)
            counts[x][s] = int(m.sum())
            v = wmean(W[:1], c, m)[0]
            table[x][s] = None if np.isnan(v) else round(float(v), 6)
    by_letter = {LETTERS[x]: summarize(wmean(W, c, gold_letter == x)) for x in range(N_SLOTS)}
    by_slot = {str(s + 1): summarize(wmean(W, c, slot2 == s)) for s in range(N_SLOTS)}
    acc_all = wmean(W, c, ones)
    a_letter0 = wmean(W, c, gold_letter == 0)
    a_slot0 = wmean(W, c, slot2 == 0)
    e_letter = wmean(W, c, gold_letter != 0) - a_letter0
    e_slot = wmean(W, c, slot2 != 0) - a_slot0
    cell00 = wmean(W, c, (gold_letter == 0) & (slot2 == 0))
    interaction = cell00 - (a_letter0 + a_slot0 - acc_all)
    return {
        "n_evals": int(c.size), "acc": summarize(acc_all),
        "table": table, "table_counts": counts,          # rows = gold letter أ ب ج د, cols = gold slot 1..4
        "by_letter": by_letter, "by_slot": by_slot,
        "E_letter": summarize(e_letter), "E_slot": summarize(e_slot),
        "interaction_alef_slot1": summarize(interaction),
    }


# ---------------------------------------------------------------------------
# 2. Pairs
# ---------------------------------------------------------------------------

def pair_block(ca: np.ndarray, cb: np.ndarray, gold_letter: np.ndarray, gold_slot: np.ndarray,
               W: np.ndarray) -> Dict[str, Any]:
    """A − B, paired over (row, rotation), overall, by gold letter and by gold slot, and the two contrasts
    of the gap (``G_letter`` = gap on letter ≠ أ minus gap on أ; ``G_slot`` likewise for slot 1)."""
    d = ca.astype(np.float64) - cb.astype(np.float64)
    slot2 = np.broadcast_to(gold_slot[:, None], d.shape)
    ones = np.ones_like(d, dtype=bool)
    return {
        "delta": summarize(wmean(W, d, ones)),
        "by_letter": {LETTERS[x]: summarize(wmean(W, d, gold_letter == x)) for x in range(N_SLOTS)},
        "by_slot": {str(s + 1): summarize(wmean(W, d, slot2 == s)) for s in range(N_SLOTS)},
        "letter_alef": summarize(wmean(W, d, gold_letter == 0)),
        "letter_other": summarize(wmean(W, d, gold_letter != 0)),
        "slot_1": summarize(wmean(W, d, slot2 == 0)),
        "slot_other": summarize(wmean(W, d, slot2 != 0)),
        "G_letter": summarize(wmean(W, d, gold_letter != 0) - wmean(W, d, gold_letter == 0)),
        "G_slot": summarize(wmean(W, d, slot2 != 0) - wmean(W, d, slot2 == 0)),
    }


# ---------------------------------------------------------------------------
# 3. 0-shot vs 3-shot rotation 0
# ---------------------------------------------------------------------------

def zero_shot_block(c0: np.ndarray, c3: np.ndarray, gold_slot: np.ndarray, W: np.ndarray) -> Dict[str, Any]:
    """Per arm: 0-shot and 3-shot (rotation 0) accuracy on the same rows and their paired difference, overall and
    by gold letter (= gold slot at rotation 0). ``c0`` / ``c3`` are ``[n]``."""
    ones = np.ones(len(c0), dtype=bool)
    out = {"acc_0shot": summarize(wmean(W, c0.astype(float), ones)),
           "acc_3shot_rot0": summarize(wmean(W, c3.astype(float), ones)),
           "delta_0_minus_3": summarize(wmean(W, c0.astype(float) - c3.astype(float), ones)),
           "by_letter": {}}
    for x in range(N_SLOTS):
        m = gold_slot == x
        out["by_letter"][LETTERS[x]] = {
            "n": int(m.sum()),
            "acc_0shot": summarize(wmean(W, c0.astype(float), m)),
            "acc_3shot_rot0": summarize(wmean(W, c3.astype(float), m)),
            "delta_0_minus_3": summarize(wmean(W, c0.astype(float) - c3.astype(float), m)),
        }
    return out


def zero_shot_pair_block(a0, b0, a3, b3, gold_slot, W) -> Dict[str, Any]:
    """A − B at 0-shot and at 3-shot rotation 0 on the gold-أ rows and on the rest, and the difference
    in differences (0-shot gap − 3-shot gap)."""
    out = {}
    for label, m in (("alef", gold_slot == 0), ("other", gold_slot != 0), ("all", np.ones(len(a0), bool))):
        g0 = a0.astype(float) - b0.astype(float)
        g3 = a3.astype(float) - b3.astype(float)
        out[label] = {"n": int(m.sum()), "gap_0shot": summarize(wmean(W, g0, m)),
                      "gap_3shot_rot0": summarize(wmean(W, g3, m)),
                      "gap_change_0_minus_3": summarize(wmean(W, g0 - g3, m))}
    return out


# ---------------------------------------------------------------------------
# 4. Per-token
# ---------------------------------------------------------------------------

def term_names(m: int) -> Tuple[str, ...]:
    return TERM_NAMES_3 if m == 3 else ("letter",) if m == 1 else tuple(f"t{j}" for j in range(m))


def per_token_block(panel: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Mean log-prob of each term per letter: gold vs non-gold choices, by slot, and by slot × gold."""
    T = panel["terms"]
    if T is None:
        return None
    n, K, S, m = T.shape
    names = term_names(m)
    rots = np.asarray(panel["rotations"])
    choice_letter = (np.arange(S)[None, :] + rots[:, None]) % S          # [K, S]
    letter = np.broadcast_to(choice_letter[None], (n, K, S))
    slot = np.broadcast_to(np.arange(S)[None, None, :], (n, K, S))
    is_gold = slot == panel["gold_slot"][:, None, None]

    def means(mask):
        cnt = int(mask.sum())
        return {"n": cnt, **{nm: (round(float(T[..., j][mask].mean()), 6) if cnt else None)
                             for j, nm in enumerate(names)}}

    out: Dict[str, Any] = {"n_terms": m, "terms": list(names), "by_letter": {}}
    for x in range(S):
        lx = letter == x
        rec = {"gold": means(lx & is_gold), "non_gold": means(lx & ~is_gold), "by_slot": {}}
        for s in range(S):
            ls = lx & (slot == s)
            rec["by_slot"][str(s + 1)] = {"all": means(ls), "gold": means(ls & is_gold),
                                          "non_gold": means(ls & ~is_gold)}
        out["by_letter"][LETTERS[x]] = rec
    return out


def diagnostic_scorings(panel: Dict[str, Any], W: np.ndarray) -> Optional[Dict[str, Any]]:
    """Argmax under (i) the sum of the terms, (ii) the letter term alone, (iii) letter + terminal term; the
    letter × slot effects under each. (i) must reproduce the dump's char-norm decision (checked: every
    4-choice letter has one character, so char-norm is the raw ll)."""
    T = panel["terms"]
    if T is None:
        return None
    m = T.shape[-1]
    variants = {"i_sum": T.sum(-1)}
    if m == 3:
        variants["ii_letter"] = T[..., 1]
        variants["iii_letter_end"] = T[..., 1] + T[..., 2]
    else:                                   # one-token letters: the three scorings coincide
        variants["ii_letter"] = variants["i_sum"]
        variants["iii_letter_end"] = variants["i_sum"]
    gold = panel["gold_slot"][:, None]
    out: Dict[str, Any] = {"coincide": m != 3}
    dump_ll_pred = panel["ll"].argmax(-1)
    out["i_reproduces_pred_char"] = {
        "argmax_of_dump_ll_vs_pred_idx_char_mismatches": int((dump_ll_pred != panel["pred_slot"]["char"]).sum()),
        "argmax_of_token_sum_vs_pred_idx_char_mismatches": int(
            (variants["i_sum"].argmax(-1) != panel["pred_slot"]["char"]).sum()),
        "n": int(dump_ll_pred.size),
    }
    for name, score in variants.items():
        pred = score.argmax(-1)
        correct = pred == gold
        blk = letter_slot_block(correct, panel["gold_letter"], panel["gold_slot"], W)
        out[name] = {k: blk[k] for k in ("acc", "by_letter", "by_slot", "E_letter", "E_slot",
                                          "interaction_alef_slot1")}
        out[name]["_correct"] = correct                    # kept for pairs; dropped before JSON
    return out


# ---------------------------------------------------------------------------
# 6. Proposal diagnostic: the rotation-averaged decision
# ---------------------------------------------------------------------------

def rotation_averaged_correct(panel: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Per row: argmax over slots of the option's score averaged over the four rotations (char = the raw ll —
    every 4-choice letter is one character — and PMI). ``[n]`` booleans per norm."""
    if len(panel["rotations"]) != N_SLOTS:
        raise ValueError("the rotation-averaged decision needs all four rotations")
    out = {}
    for nm, arr in (("char", panel["ll"]), ("pmi", panel["score_pmi"])):
        if np.isnan(arr).any():
            continue
        out[nm] = arr.mean(1).argmax(-1) == panel["gold_slot"]
    return out


def rotation_averaged_block(correct: np.ndarray, gold_slot: np.ndarray, W: np.ndarray) -> Dict[str, Any]:
    ones = np.ones(len(correct), dtype=bool)
    return {"acc": summarize(wmean(W, correct.astype(float), ones)),
            "by_gold_slot": {str(s + 1): summarize(wmean(W, correct.astype(float), gold_slot == s))
                             for s in range(N_SLOTS)}}


def rotation_averaged_pair(ca: np.ndarray, cb: np.ndarray, gold_slot: np.ndarray, W: np.ndarray) -> Dict[str, Any]:
    d = ca.astype(float) - cb.astype(float)
    ones = np.ones(len(d), dtype=bool)
    return {"delta": summarize(wmean(W, d, ones)), "slot_1": summarize(wmean(W, d, gold_slot == 0)),
            "slot_other": summarize(wmean(W, d, gold_slot != 0))}


# ---------------------------------------------------------------------------
# 5. Distributions
# ---------------------------------------------------------------------------

def distributions(panel: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = len(panel["rows"])
    for j, k in enumerate(panel["rotations"]):
        rec = {"gold_slot": [round(float((panel["gold_slot"] == s).mean()), 4) for s in range(N_SLOTS)],
               "gold_letter": [round(float((panel["gold_letter"][:, j] == x).mean()), 4) for x in range(N_SLOTS)]}
        for nm in NORMS:
            rec[f"pred_slot_{nm}"] = [round(float((panel["pred_slot"][nm][:, j] == s).mean()), 4)
                                      for s in range(N_SLOTS)]
            rec[f"pred_letter_{nm}"] = [round(float((panel["pred_letter"][nm][:, j] == x).mean()), 4)
                                        for x in range(N_SLOTS)]
        rec["n"] = n
        out[str(k)] = rec
    return out


# ---------------------------------------------------------------------------
# Reading rules (pre-registered, the brief's P2.3)
# ---------------------------------------------------------------------------

def reading_rules(block: Dict[str, Any]) -> Dict[str, bool]:
    el, es = block["E_letter"], block["E_slot"]
    if el["est"] is None or es["est"] is None:
        return {"slot": False, "letter": False, "both": False}
    slot = (excludes_zero(es) and not excludes_zero(el)) or abs(el["est"]) < es["est"] / 3
    letter = (excludes_zero(el) and not excludes_zero(es)) or abs(es["est"]) < el["est"] / 3
    return {"slot": bool(slot), "letter": bool(letter), "both": bool(excludes_zero(el) and excludes_zero(es))}


def terminal_token_rule(diag_focus: Optional[Dict[str, Any]], block_comparator: Dict[str, Any]) -> Optional[bool]:
    """Fires when the focus arm's E_letter under scoring (ii) falls inside the comparator's E_letter CI."""
    if not diag_focus or diag_focus.get("coincide"):
        return None
    e = diag_focus["ii_letter"]["E_letter"]["est"]
    ci = block_comparator["E_letter"]
    if e is None or ci["lo"] is None:
        return None
    return bool(ci["lo"] <= e <= ci["hi"])


def zero_shot_rule(zp: Dict[str, Any]) -> Dict[str, Any]:
    """On the gold-أ rows: does the 3-shot rotation-0 gap persist at 0-shot (the change 0 − 3 has a CI
    including 0), and does the 0-shot gap itself shrink to 0 (its CI includes 0)?"""
    a = zp["alef"]
    return {"gap_persists_within_ci": not excludes_zero(a["gap_change_0_minus_3"]),
            "gap_0shot_excludes_zero": excludes_zero(a["gap_0shot"])}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def analyze(experiment: Path, rows_dir: Path, tasks: Sequence[str] = TASKS, arms: Sequence[str] = ARMS,
            pairs: Sequence[Tuple[str, str]] = PAIRS, *, n_boot: int = 5000, seed: int = 0,
            focus: str = FOCUS, comparator: str = COMPARATOR) -> Dict[str, Any]:
    res: Dict[str, Any] = {"experiment": str(experiment), "n_boot": n_boot, "seed": seed, "arms": list(arms),
                           "pairs": [list(p) for p in pairs], "focus": focus, "comparator": comparator,
                           "tasks": {}}
    for task in tasks:
        spec = json.loads((rows_dir / f"rows_{task}.json").read_text(encoding="utf-8"))
        rows = [int(r) for r in spec["row_index"]]
        W = boot_weights(len(rows), n_boot, seed)
        panels: Dict[str, Dict[str, Any]] = {}
        zero: Dict[str, Dict[str, Any]] = {}
        t: Dict[str, Any] = {"n_rows": len(rows), "arms": {}, "pairs": {}, "zero_shot_pairs": {}}
        for arm in arms:
            dumps = {}
            for k in ROTATIONS:
                p = experiment / cell_name(arm, f"rot{k}") / "eval_rows" / f"{task}.parquet"
                if p.exists():
                    dumps[k] = load_dump(p)
            if len(dumps) != len(ROTATIONS):
                t["arms"][arm] = {"missing_rotations": [k for k in ROTATIONS if k not in dumps]}
                continue
            for k, d in dumps.items():
                meta = d["_meta"]
                if meta.get("label_rotation") not in (None, k) or meta.get("num_fewshot") not in (None, 3):
                    raise ValueError(f"{arm} rot{k} {task}: dump metadata says label_rotation="
                                     f"{meta.get('label_rotation')} num_fewshot={meta.get('num_fewshot')}")
            panel = build_panel(dumps, rows)
            panels[arm] = panel
            arm_out: Dict[str, Any] = {"flags": panel["flags"],
                                       "max_abs_token_sum_minus_ll": panel["max_abs_token_sum_minus_ll"],
                                       "n_terms": panel["n_terms"]}
            for nm in NORMS:
                blk = letter_slot_block(panel["correct"][nm], panel["gold_letter"], panel["gold_slot"], W)
                blk["rules"] = reading_rules(blk)
                arm_out[nm] = blk
            arm_out["per_token"] = per_token_block(panel)
            diag = diagnostic_scorings(panel, W)
            arm_out["diagnostic_scorings"] = diag
            arm_out["distributions"] = distributions(panel)
            arm_out["acc_by_rotation"] = {nm: {str(k): round(float(panel["correct"][nm][:, j].mean()), 6)
                                               for j, k in enumerate(panel["rotations"])} for nm in NORMS}
            rav = rotation_averaged_correct(panel)
            panel["_rav"] = rav
            arm_out["rotation_averaged"] = {nm: rotation_averaged_block(c, panel["gold_slot"], W)
                                            for nm, c in rav.items()}
            # 0-shot
            z = experiment / cell_name(arm, ZERO_SHOT) / "eval_rows" / f"{task}.parquet"
            if z.exists():
                zd = align(load_dump(z), rows)
                if zd["_meta"].get("num_fewshot") not in (None, 0):
                    raise ValueError(f"{arm} 0shot {task}: num_fewshot={zd['_meta'].get('num_fewshot')}")
                if list(zd["gold_idx"]) != panel["gold_slot"].tolist():
                    raise ValueError(f"{arm} 0shot {task}: gold differs from the rotation cells")
                zero[arm] = {nm: np.asarray(zd[f"correct_{nm}"], dtype=bool) for nm in NORMS}
                zero[arm]["flags"] = {f: int(sum(bool(v) for v in zd.get(f, []))) for f in panel["flags"]}
                arm_out["zero_shot"] = {"flags": zero[arm]["flags"], **{
                    nm: zero_shot_block(zero[arm][nm], panel["correct"][nm][:, 0], panel["gold_slot"], W)
                    for nm in NORMS}}
            t["arms"][arm] = arm_out
        for a, b in pairs:
            if a not in panels or b not in panels:
                continue
            key = f"{a}-{b}"
            pa, pb = panels[a], panels[b]
            t["pairs"][key] = {nm: pair_block(pa["correct"][nm], pb["correct"][nm], pa["gold_letter"],
                                              pa["gold_slot"], W) for nm in NORMS}
            da = t["arms"][a].get("diagnostic_scorings")
            db = t["arms"][b].get("diagnostic_scorings")
            if da and db:
                t["pairs"][key]["diagnostic_scorings"] = {
                    s: pair_block(da[s]["_correct"], db[s]["_correct"], pa["gold_letter"], pa["gold_slot"], W)
                    for s in DIAG_SCORINGS}
            ra, rb = pa.get("_rav", {}), pb.get("_rav", {})
            if ra and rb:
                t["pairs"][key]["rotation_averaged"] = {
                    nm: rotation_averaged_pair(ra[nm], rb[nm], pa["gold_slot"], W) for nm in ra if nm in rb}
            if a in zero and b in zero:
                t["zero_shot_pairs"][key] = {nm: zero_shot_pair_block(
                    zero[a][nm], zero[b][nm], pa["correct"][nm][:, 0], pb["correct"][nm][:, 0],
                    pa["gold_slot"], W) for nm in NORMS}
        # Pre-registered readings, on the focus arm (and, supplementary, on the focus − comparator gap).
        if focus in t["arms"] and "char" in t["arms"][focus]:
            fa = t["arms"][focus]
            readings: Dict[str, Any] = {nm: dict(fa[nm]["rules"]) for nm in NORMS}
            comp = t["arms"].get(comparator)
            if comp and "char" in comp:
                readings["terminal_token_char"] = terminal_token_rule(fa.get("diagnostic_scorings"), comp["char"])
            key = f"{focus}-{comparator}"
            if key in t["zero_shot_pairs"]:
                readings["zero_shot"] = {nm: zero_shot_rule(t["zero_shot_pairs"][key][nm]) for nm in NORMS}
            if key in t["pairs"]:
                readings["gap_contrasts"] = {nm: {"G_letter": t["pairs"][key][nm]["G_letter"],
                                                  "G_slot": t["pairs"][key][nm]["G_slot"]} for nm in NORMS}
            t["readings"] = readings
        res["tasks"][task] = t
    # Drop the per-(row, rotation) arrays kept for the pairs.
    for t in res["tasks"].values():
        for arm in t["arms"].values():
            for s in DIAG_SCORINGS:
                ((arm.get("diagnostic_scorings") or {}).get(s) or {}).pop("_correct", None)
    return res


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _ci(s: Optional[Dict[str, Any]], signed: bool = True) -> str:
    if not s or s.get("est") is None:
        return "—"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(s['est'])} [{f.format(s['lo'])}, {f.format(s['hi'])}]"


def _v(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.3f}"


def to_markdown(res: Dict[str, Any]) -> str:
    L: List[str] = ["# Letter or slot — the first-option deficit decomposed", ""]
    L.append(f"Cluster bootstrap over rows ({res['n_boot']} resamples, seed {res['seed']}); the four rotations of a "
             "row are one cluster. Tables are char-norm; the PMI line under each gives the same quantities. "
             "E_letter = acc(gold letter ≠ أ) − acc(gold letter = أ) (a within-row contrast: the rotation moves the "
             "letter, not the option); E_slot = acc(gold slot ≠ 1) − acc(gold slot = 1) (a between-row contrast: the "
             "rows whose gold is the first option are fixed). The diagnostic scorings are diagnostics of the "
             "mechanism, not a protocol.")
    L.append("")
    for task, t in res["tasks"].items():
        L += [f"## {task} — {t['n_rows']} rows × 4 rotations", ""]
        L.append("### 1. Letter × slot")
        L.append("")
        L.append("| arm | acc | by letter أ / ب / ج / د | by slot 1 / 2 / 3 / 4 | E_letter | E_slot | interaction | "
                 "rules (char) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for arm, a in t["arms"].items():
            if "char" not in a:
                L.append(f"| {arm} | missing rotations {a.get('missing_rotations')} |||||||")
                continue
            for nm in ("char", "pmi"):
                b = a[nm]
                fired = ", ".join(k for k, v in b["rules"].items() if v) or "none"
                L.append(f"| {arm}{'' if nm == 'char' else ' (PMI)'} | {_v(b['acc']['est'])} | "
                         + " / ".join(_v(b["by_letter"][l]["est"]) for l in LETTERS) + " | "
                         + " / ".join(_v(b["by_slot"][str(s)]["est"]) for s in range(1, 5)) + " | "
                         f"{_ci(b['E_letter'])} | {_ci(b['E_slot'])} | {_ci(b['interaction_alef_slot1'])} | {fired} |")
        L.append("")
        L.append("Letter × slot tables (char; rows = gold letter, columns = gold slot 1..4; n per cell in brackets):")
        L.append("")
        for arm, a in t["arms"].items():
            if "char" not in a:
                continue
            L.append(f"**{arm}**")
            L.append("")
            L.append("| gold letter | slot 1 | slot 2 | slot 3 | slot 4 |")
            L.append("|---|---|---|---|---|")
            tb, cn = a["char"]["table"], a["char"]["table_counts"]
            for x, l in enumerate(LETTERS):
                L.append(f"| {l} | " + " | ".join(f"{_v(tb[x][s])} ({cn[x][s]})" for s in range(N_SLOTS)) + " |")
            L.append("")
        L.append("### 2. Pairs (A − B, paired over row × rotation)")
        L.append("")
        L.append("| pair | norm | Δ | gold letter أ | letter ≠ أ | gold slot 1 | slot ≠ 1 | G_letter | G_slot |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for key, p in t["pairs"].items():
            for nm in NORMS:
                q = p[nm]
                L.append(f"| {key} | {nm} | {_ci(q['delta'])} | {_ci(q['letter_alef'])} | {_ci(q['letter_other'])} | "
                         f"{_ci(q['slot_1'])} | {_ci(q['slot_other'])} | {_ci(q['G_letter'])} | {_ci(q['G_slot'])} |")
        L.append("")
        L.append("By gold letter and by gold slot (char):")
        L.append("")
        L.append("| pair | letter أ | ب | ج | د | slot 1 | 2 | 3 | 4 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for key, p in t["pairs"].items():
            q = p["char"]
            L.append(f"| {key} | " + " | ".join(_ci(q["by_letter"][l]) for l in LETTERS) + " | "
                     + " | ".join(_ci(q["by_slot"][str(s)]) for s in range(1, 5)) + " |")
        L.append("")
        L.append("### 3. 0-shot vs 3-shot rotation 0 (same rows)")
        L.append("")
        L.append("| arm | norm | acc 0-shot | acc 3-shot rot0 | Δ (0 − 3) | Δ on gold أ | Δ on ب | Δ on ج | Δ on د |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for arm, a in t["arms"].items():
            z = a.get("zero_shot")
            if not z:
                continue
            for nm in NORMS:
                q = z[nm]
                L.append(f"| {arm} | {nm} | {_v(q['acc_0shot']['est'])} | {_v(q['acc_3shot_rot0']['est'])} | "
                         f"{_ci(q['delta_0_minus_3'])} | "
                         + " | ".join(_ci(q["by_letter"][l]["delta_0_minus_3"]) for l in LETTERS) + " |")
        L.append("")
        if t["zero_shot_pairs"]:
            L.append("| pair | norm | rows | gap 0-shot | gap 3-shot rot0 | change (0 − 3) |")
            L.append("|---|---|---|---|---|---|")
            for key, p in t["zero_shot_pairs"].items():
                for nm in NORMS:
                    for part in ("alef", "other"):
                        q = p[nm][part]
                        L.append(f"| {key} | {nm} | {'gold أ' if part == 'alef' else 'gold ≠ أ'} ({q['n']}) | "
                                 f"{_ci(q['gap_0shot'])} | {_ci(q['gap_3shot_rot0'])} | {_ci(q['gap_change_0_minus_3'])} |")
            L.append("")
        L.append("### 4. Per-token (char) and the diagnostic scorings")
        L.append("")
        for arm, a in t["arms"].items():
            pt = a.get("per_token")
            if not pt:
                continue
            L.append(f"**{arm}** — {pt['n_terms']} term(s) per letter ({', '.join(pt['terms'])}); max |Σ terms − ll| "
                     f"= {a['max_abs_token_sum_minus_ll']:.2e}")
            L.append("")
            L.append("| letter | choice | " + " | ".join(pt["terms"]) + " | n |")
            L.append("|---|---|" + "---|" * len(pt["terms"]) + "---|")
            for l in LETTERS:
                for part in ("gold", "non_gold"):
                    q = pt["by_letter"][l][part]
                    L.append(f"| {l} | {part} | " + " | ".join(_v(q[nm]) for nm in pt["terms"]) + f" | {q['n']} |")
            L.append("")
            if pt["n_terms"] > 1:
                L.append("By slot (all choices of that letter at that slot; mean of each term):")
                L.append("")
                L.append("| letter | " + " | ".join(f"slot {s}" for s in range(1, 5)) + " |")
                L.append("|---|---|---|---|---|")
                for l in LETTERS:
                    L.append(f"| {l} | " + " | ".join(
                        " / ".join(_v(pt["by_letter"][l]["by_slot"][str(s)]["all"][nm]) for nm in pt["terms"])
                        for s in range(1, 5)) + " |")
                L.append("")
            dg = a.get("diagnostic_scorings")
            if dg:
                chk = dg["i_reproduces_pred_char"]
                L.append(f"Diagnostic scorings{' (coincide: one-token letters)' if dg['coincide'] else ''} — "
                         f"(i) reproduces pred_idx_char: argmax(dump ll) mismatches {chk['argmax_of_dump_ll_vs_pred_idx_char_mismatches']}, "
                         f"argmax(Σ float32 terms) mismatches {chk['argmax_of_token_sum_vs_pred_idx_char_mismatches']} of {chk['n']}.")
                L.append("")
                L.append("| scoring | acc | by letter أ / ب / ج / د | by slot 1 / 2 / 3 / 4 | E_letter | E_slot |")
                L.append("|---|---|---|---|---|---|")
                for s in DIAG_SCORINGS:
                    b = dg[s]
                    L.append(f"| {s} | {_v(b['acc']['est'])} | "
                             + " / ".join(_v(b["by_letter"][l]["est"]) for l in LETTERS) + " | "
                             + " / ".join(_v(b["by_slot"][str(x)]["est"]) for x in range(1, 5)) + " | "
                             f"{_ci(b['E_letter'])} | {_ci(b['E_slot'])} |")
                L.append("")
        pd_ = {k: v for k, v in t["pairs"].items() if "diagnostic_scorings" in v}
        if pd_:
            L.append("Pairs under the diagnostic scorings (A − B on gold أ / on letter ≠ أ / G_letter):")
            L.append("")
            L.append("| pair | scoring | gold أ | letter ≠ أ | G_letter | G_slot |")
            L.append("|---|---|---|---|---|---|")
            for key, p in pd_.items():
                for s in DIAG_SCORINGS:
                    q = p["diagnostic_scorings"][s]
                    L.append(f"| {key} | {s} | {_ci(q['letter_alef'])} | {_ci(q['letter_other'])} | "
                             f"{_ci(q['G_letter'])} | {_ci(q['G_slot'])} |")
            L.append("")
        L.append("### 6. Proposal diagnostic — the rotation-averaged decision (each option's score averaged over its "
                 "four letters before the argmax; letter-invariant by construction; applied to nothing)")
        L.append("")
        L.append("| arm | char acc | by gold slot 1 / 2 / 3 / 4 | PMI acc | official rot0 char / PMI |")
        L.append("|---|---|---|---|---|")
        for arm, a in t["arms"].items():
            ra = a.get("rotation_averaged")
            if not ra:
                continue
            L.append(f"| {arm} | {_v(ra['char']['acc']['est'])} | "
                     + " / ".join(_v(ra["char"]["by_gold_slot"][str(s)]["est"]) for s in range(1, 5))
                     + f" | {_v(ra.get('pmi', {}).get('acc', {}).get('est'))} | "
                     f"{_v(a['acc_by_rotation']['char']['0'])} / {_v(a['acc_by_rotation']['pmi']['0'])} |")
        L.append("")
        L.append("| pair | norm | Δ | on gold slot 1 | on gold slot ≠ 1 |")
        L.append("|---|---|---|---|---|")
        for key, p in t["pairs"].items():
            for nm, q in (p.get("rotation_averaged") or {}).items():
                L.append(f"| {key} | {nm} | {_ci(q['delta'])} | {_ci(q['slot_1'])} | {_ci(q['slot_other'])} |")
        L.append("")
        L.append("### 5. Predicted slot and letter per rotation (char / PMI shares over slots 1..4 and letters أ ب ج د)")
        L.append("")
        L.append("| arm | rot | gold slot | gold letter | pred slot char | pred letter char | pred slot PMI | pred letter PMI |")
        L.append("|---|---|---|---|---|---|---|---|")
        fmt = lambda v: " ".join(f"{x:.2f}" for x in v)  # noqa: E731
        for arm, a in t["arms"].items():
            for k, q in (a.get("distributions") or {}).items():
                L.append(f"| {arm} | {k} | {fmt(q['gold_slot'])} | {fmt(q['gold_letter'])} | {fmt(q['pred_slot_char'])} | "
                         f"{fmt(q['pred_letter_char'])} | {fmt(q['pred_slot_pmi'])} | {fmt(q['pred_letter_pmi'])} |")
        L.append("")
        if "readings" in t:
            r = t["readings"]
            L.append(f"### Pre-registered readings ({res['focus']}; terminal-token rule against {res['comparator']})")
            L.append("")
            for nm in NORMS:
                L.append(f"- {nm}: " + ", ".join(f"{k} {'FIRES' if v else 'no'}" for k, v in r[nm].items()))
            if "terminal_token_char" in r:
                L.append(f"- terminal token (scoring (ii) E_letter inside the comparator's E_letter CI, char): "
                         f"{r['terminal_token_char']}")
            for nm, z in (r.get("zero_shot") or {}).items():
                L.append(f"- 0-shot, {nm}: gold-أ gap persists within its CI = {z['gap_persists_within_ci']}; "
                         f"0-shot gap CI excludes 0 = {z['gap_0shot_excludes_zero']}")
            for nm, g in (r.get("gap_contrasts") or {}).items():
                L.append(f"- the {res['focus']} − {res['comparator']} gap, {nm}: G_letter {_ci(g['G_letter'])}, "
                         f"G_slot {_ci(g['G_slot'])} (supplementary: row content cancels in the pair)")
            L.append("")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default=EXPERIMENT)
    ap.add_argument("--rows-dir", default=ROWS_DIR)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="letter_slot")
    args = ap.parse_args(argv)
    exp = REPO_ROOT / args.experiment if not Path(args.experiment).is_absolute() else Path(args.experiment)
    rows_dir = REPO_ROOT / args.rows_dir if not Path(args.rows_dir).is_absolute() else Path(args.rows_dir)
    res = analyze(exp, rows_dir, args.tasks, n_boot=args.n_boot, seed=args.seed)
    (exp / f"{args.name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(res)
    (exp / f"{args.name}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {exp / (args.name + '.md')} and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
