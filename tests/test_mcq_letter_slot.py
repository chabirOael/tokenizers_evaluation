"""scripts/mcq_letter_slot.py on synthetic dumps: planted effects are recovered.

  * a pure **letter** effect (wrong only when the gold letter is أ, at whatever slot) → E_letter > 0, E_slot ≈ 0;
  * a pure **slot** effect (wrong only when the gold is the first option, whatever its letter) → the converse;
  * the diagnostic scoring (i) reproduces the planted argmax; an effect planted in the terminal term alone
    vanishes under scoring (ii) and survives under (i) and (iii);
  * the pairs, the 0-shot comparison and the reading rules on the same fixtures.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from arabic_eval.evaluation.eval_rows import EvalRowWriter, build_row_record  # noqa: E402
from arabic_eval.tasks.lighteval.utils import choice_letters  # noqa: E402

_spec = importlib.util.spec_from_file_location("mcq_letter_slot", REPO / "scripts" / "mcq_letter_slot.py")
ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ls)

TASK = "arabic_exam"
N = 400


def _gold_slots(n: int) -> List[int]:
    return [int(v) for v in np.random.default_rng(1).integers(0, 4, size=n)]


def _flip(r: int, k: int) -> float:
    """A deterministic uniform in [0, 1) per (row, rotation)."""
    return float(np.random.default_rng(10_000 + 7 * r + k).random())


def _terms_for(correct: bool, gold: int, k: int, mode: str):
    """Three per-token terms per choice. ``plain``: the gold choice scores −1 when correct, the next slot −1
    when wrong, everything else −5 (split into three equal terms). ``terminal``: the letter term favours the
    gold always; the terminal term of the letter أ is −20 (so the sum loses أ whenever it is gold)."""
    out = []
    letters = choice_letters(4, k)
    for i in range(4):
        if mode == "plain":
            pick = gold if correct else (gold + 1) % 4
            total = -1.0 if i == pick else -5.0
            out.append([total / 3, total / 3, total / 3])
        else:
            letter_term = -0.5 if i == gold else -3.0
            end_term = -20.0 if letters[i] == "أ" else -0.1
            out.append([-0.2, letter_term, end_term])
    return out


def _write_cell(exp: Path, name: str, k: int, shots: int, golds: List[int],
                correct_fn: Callable[[int, int, int], bool], mode: str = "plain") -> None:
    path = exp / name / "eval_rows" / f"{TASK}.parquet"
    conts = [" " + l for l in choice_letters(4, k)]
    with EvalRowWriter(path, metadata={"task": TASK, "label_rotation": k, "num_fewshot": shots}) as w:
        for r, g in enumerate(golds):
            terms = _terms_for(correct_fn(r, g, k), g, k, mode)
            ll = [sum(t) for t in terms]
            pred = int(np.argmax(ll))
            w.write(build_row_record(
                row_index=r, example={"question": f"q{r}", "choices": ["a", "b", "c", "d"]},
                prompt=f"p{r}", continuations=conts, log_likelihoods=ll, scores_char=ll,
                scores_pmi=[v + 0.1 * i for i, v in enumerate(ll)], unconditioned_log_likelihoods=[0.0] * 4,
                gold_idx=g, pred_idx=pred, pred_idx_char=pred,
                pred_idx_pmi=int(np.argmax([v + 0.1 * i for i, v in enumerate(ll)])),
                prompt_units=10, max_length=4096, cont_tokens=[3] * 4, cont_truncated=[False] * 4,
                cont_token_ll=terms,
            ))


def _experiment(tmp: Path, arms: dict, mode: str = "plain") -> Path:
    golds = _gold_slots(N)
    exp = tmp / "exp"
    rows_dir = tmp / "rows"
    rows_dir.mkdir(parents=True)
    (rows_dir / f"rows_{TASK}.json").write_text(json.dumps({"task": TASK, "row_index": list(range(N))}))
    for arm, fn in arms.items():
        for k in range(4):
            _write_cell(exp, f"{arm}_rot{k}", k, 3, golds, fn, mode)
        _write_cell(exp, f"{arm}_0shot", 0, 0, golds, fn, mode)
    return exp


def letter_effect(r, gold, k):
    return (gold + k) % 4 != 0 or _flip(r, k) < 0.3


def slot_effect(r, gold, k):
    return gold != 0 or _flip(r, 0) < 0.3


def no_effect(r, gold, k):
    return _flip(r, k) < 0.6


def _run(exp, tmp, arms, pairs=(), **kw):
    return ls.analyze(exp, tmp / "rows", [TASK], list(arms), pairs, n_boot=300, seed=0,
                      focus=list(arms)[0], comparator=list(arms)[-1], **kw)


def test_pure_letter_effect_is_recovered(tmp_path):
    arms = {"let": letter_effect, "flat": no_effect}
    exp = _experiment(tmp_path, arms)
    res = _run(exp, tmp_path, arms, [("let", "flat")])
    b = res["tasks"][TASK]["arms"]["let"]["char"]
    assert b["E_letter"]["est"] == pytest.approx(0.7, abs=0.06) and ls.excludes_zero(b["E_letter"])
    assert abs(b["E_slot"]["est"]) < 0.05 and not ls.excludes_zero(b["E_slot"])
    assert b["by_letter"]["أ"]["est"] == pytest.approx(0.3, abs=0.06)
    assert b["rules"] == {"slot": False, "letter": True, "both": False}
    # Every row contributes one أ rotation, so each gold slot carries the same أ share.
    assert all(abs(b["by_slot"][str(s)]["est"] - b["acc"]["est"]) < 0.06 for s in range(1, 5))
    p = res["tasks"][TASK]["pairs"]["let-flat"]["char"]
    assert p["G_letter"]["est"] > 0.5 and abs(p["G_slot"]["est"]) < 0.08


def test_pure_slot_effect_is_recovered(tmp_path):
    arms = {"slt": slot_effect, "flat": no_effect}
    exp = _experiment(tmp_path, arms)
    res = _run(exp, tmp_path, arms)
    b = res["tasks"][TASK]["arms"]["slt"]["char"]
    assert b["E_slot"]["est"] == pytest.approx(0.7, abs=0.08) and ls.excludes_zero(b["E_slot"])
    assert abs(b["E_letter"]["est"]) < 0.05 and not ls.excludes_zero(b["E_letter"])
    assert b["rules"] == {"slot": True, "letter": False, "both": False}
    t = b["table"]
    assert all(t[x][0] < 0.45 for x in range(4)) and all(t[x][s] == 1.0 for x in range(4) for s in (1, 2, 3))


def test_diagnostic_scoring_i_reproduces_the_planted_argmax(tmp_path):
    arms = {"let": letter_effect, "flat": no_effect}
    exp = _experiment(tmp_path, arms)
    res = _run(exp, tmp_path, arms)
    a = res["tasks"][TASK]["arms"]["let"]
    dg = a["diagnostic_scorings"]
    assert dg["coincide"] is False
    chk = dg["i_reproduces_pred_char"]
    assert chk["argmax_of_dump_ll_vs_pred_idx_char_mismatches"] == 0
    assert chk["argmax_of_token_sum_vs_pred_idx_char_mismatches"] == 0
    assert dg["i_sum"]["acc"]["est"] == a["char"]["acc"]["est"]
    assert dg["i_sum"]["E_letter"] == a["char"]["E_letter"]
    assert a["max_abs_token_sum_minus_ll"] < 1e-5
    pt = a["per_token"]
    assert pt["terms"] == ["LIT_BEGIN", "CHAR", "LIT_END"] and pt["by_letter"]["أ"]["gold"]["n"] > 0


def test_terminal_term_effect_vanishes_under_the_letter_term(tmp_path):
    arms = {"term": letter_effect, "flat": no_effect}      # correct_fn unused in "terminal" mode
    exp = _experiment(tmp_path, arms, mode="terminal")
    res = _run(exp, tmp_path, arms)
    dg = res["tasks"][TASK]["arms"]["term"]["diagnostic_scorings"]
    assert dg["i_sum"]["by_letter"]["أ"]["est"] == 0.0 and dg["i_sum"]["E_letter"]["est"] == 1.0
    assert dg["iii_letter_end"]["E_letter"]["est"] == 1.0
    assert dg["ii_letter"]["acc"]["est"] == 1.0 and dg["ii_letter"]["E_letter"]["est"] == 0.0
    pt = res["tasks"][TASK]["arms"]["term"]["per_token"]
    assert pt["by_letter"]["أ"]["gold"]["LIT_END"] == pytest.approx(-20.0)
    assert pt["by_letter"]["ب"]["gold"]["LIT_END"] == pytest.approx(-0.1)


def test_zero_shot_and_distributions(tmp_path):
    arms = {"let": letter_effect, "flat": no_effect}
    exp = _experiment(tmp_path, arms)
    res = _run(exp, tmp_path, arms, [("let", "flat")])
    t = res["tasks"][TASK]
    z = t["arms"]["let"]["zero_shot"]["char"]
    # The 0-shot cells were written with the rotation-0 rule: identical to rot0 → Δ 0.
    assert z["delta_0_minus_3"]["est"] == 0.0
    zp = t["zero_shot_pairs"]["let-flat"]["char"]
    assert zp["alef"]["gap_change_0_minus_3"]["est"] == 0.0
    assert t["readings"]["zero_shot"]["char"]["gap_persists_within_ci"] is True
    d = t["arms"]["let"]["distributions"]
    assert set(d) == {"0", "1", "2", "3"} and sum(d["1"]["pred_slot_char"]) == pytest.approx(1.0, abs=1e-3)
    md = ls.to_markdown(res)
    assert "Letter × slot" in md and "Pre-registered readings" in md


def test_a_mislabelled_rotation_is_refused(tmp_path):
    arms = {"let": letter_effect}
    exp = _experiment(tmp_path, arms)
    golds = _gold_slots(N)
    _write_cell(exp, "let_rot2", 1, 3, golds, letter_effect)      # rot2's folder holds rotation-1 letters
    with pytest.raises(ValueError, match="continuations|label_rotation"):
        _run(exp, tmp_path, arms)


def test_rotation_averaged_decision_cancels_a_letter_prior(tmp_path):
    """A −20 penalty on the letter أ loses every gold-أ evaluation under the official sum, but averaged over the
    four rotations each option carries أ once, so the penalty cancels and every row is right."""
    arms = {"term": letter_effect, "flat": no_effect}
    exp = _experiment(tmp_path, arms, mode="terminal")
    res = _run(exp, tmp_path, arms, [("term", "flat")])
    a = res["tasks"][TASK]["arms"]["term"]
    assert a["char"]["acc"]["est"] == pytest.approx(0.75)
    assert a["rotation_averaged"]["char"]["acc"]["est"] == 1.0
    assert a["acc_by_rotation"]["char"]["0"] == pytest.approx(
        float(np.mean([g != 0 for g in _gold_slots(N)])), abs=1e-6)
    # The fixture writes every arm in the same mode, so the two arms' averaged decisions coincide.
    p = res["tasks"][TASK]["pairs"]["term-flat"]["rotation_averaged"]["char"]
    assert p["delta"]["est"] == 0.0 and res["tasks"][TASK]["pairs"]["term-flat"]["char"]["delta"]["est"] == 0.0
    assert "rotation-averaged decision" in ls.to_markdown(res)
