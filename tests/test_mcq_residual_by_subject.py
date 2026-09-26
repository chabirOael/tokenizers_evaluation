"""scripts/mcq_residual_by_subject.py on synthetic dumps.

  * the Culture-MMLU category constant maps exactly the 57 MMLU subjects, each once (and, when the real dump is on
    disk, exactly the subjects it holds);
  * the full-dump join keeps only the rows no cell truncates (``hit_cap`` or ``sentinel`` in any cell);
  * the spread permutation test: a planted between-group difference gives a small ``p``, a homogeneous gap a large
    one, and the same seed reproduces the same null;
  * end to end on a planted category effect: the rotation-averaged decision from four rotation cells, per-group
    gaps and McNemar counts, the readings;
  * the ArabicMMLU config → Group map refuses a config with two groups.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from arabic_eval.evaluation.eval_rows import SENTINEL_LL, EvalRowWriter, build_row_record  # noqa: E402
from arabic_eval.tasks.lighteval.utils import choice_letters  # noqa: E402

_spec = importlib.util.spec_from_file_location("mcq_residual_by_subject", REPO / "scripts" / "mcq_residual_by_subject.py")
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)

TASK = "culture_arabic_mmlu"
ARMS = rs.ARMS
CELLS = {arm: f"{arm}_full" for arm in ARMS}
#: four subjects per category is enough for the plumbing
SUBJECTS = [s for cat in rs.CULTURE_MMLU_CATEGORIES.values() for s in cat[:4]]


def _write_dump(path: Path, rows: List[Dict], k: int = 0) -> None:
    conts = [" " + l for l in choice_letters(4, k)]
    with EvalRowWriter(path, metadata={"task": TASK, "label_rotation": k, "num_fewshot": 3, "max_length": 4096}) as w:
        for r in rows:
            ll = list(r["ll"])
            if r.get("sentinel"):
                ll[3] = SENTINEL_LL
            pred = int(np.argmax(ll))
            w.write(build_row_record(
                row_index=r["row_index"], example={"question": f"q{r['row_index']}", "choices": ["a", "b", "c", "d"],
                                                   "_source_config": r["subject"]},
                prompt=f"p{r['row_index']}", continuations=conts, log_likelihoods=ll, scores_char=ll, scores_pmi=ll,
                unconditioned_log_likelihoods=[0.0] * 4, gold_idx=r["gold"], pred_idx=pred, pred_idx_char=pred,
                pred_idx_pmi=pred, prompt_units=r["units"], max_length=4096, cont_tokens=[1] * 4,
                cont_truncated=[bool(r.get("cap"))] * 4, cont_token_ll=[[v] for v in ll]))


def _ll(gold: int, correct: bool) -> List[float]:
    pick = gold if correct else (gold + 1) % 4
    return [-1.0 if i == pick else -5.0 for i in range(4)]


def _world(tmp: Path, n: int, correct_fn: Callable[[str, int, int], bool], caps: Dict[str, set] = None,
           sentinels: Dict[str, set] = None, subset: List[int] = None) -> Dict[str, Path]:
    """Four full cells + (for ``subset`` rows) four rotation cells per arm. ``correct_fn(arm, row, subject_idx)``."""
    caps, sentinels = caps or {}, sentinels or {}
    rng = np.random.default_rng(3)
    golds = rng.integers(0, 4, size=n)
    subj = [SUBJECTS[i % len(SUBJECTS)] for i in range(n)]
    units = {"araroopat_v5": 300 + (np.arange(n) % 90), "bpe16k_v5": np.full(n, 200), "native_v5": np.full(n, 240),
             "native_base": np.full(n, 240)}
    exp, lsd, rows_dir = tmp / "exp", tmp / "ls", tmp / "rows"
    for arm in ARMS:
        rows = [{"row_index": r, "subject": subj[r], "gold": int(golds[r]), "units": int(units[arm][r]),
                 "ll": _ll(int(golds[r]), correct_fn(arm, r, SUBJECTS.index(subj[r]))),
                 "cap": r in caps.get(arm, set()), "sentinel": r in sentinels.get(arm, set())} for r in range(n)]
        _write_dump(exp / CELLS[arm] / "eval_rows" / f"{TASK}.parquet", rows)
        if subset is not None:
            for k in range(4):     # the same option scores under every rotation → the averaged argmax is the slot's
                _write_dump(lsd / f"{arm}_rot{k}" / "eval_rows" / f"{TASK}.parquet",
                            [rows[r] for r in subset], k=k)
    if subset is not None:
        rows_dir.mkdir(parents=True)
        (rows_dir / f"rows_{TASK}.json").write_text(json.dumps({"task": TASK, "row_index": subset}))
    return {"exp": exp, "ls": lsd, "rows": rows_dir}


def test_category_constant_covers_exactly_the_57_subjects():
    subjects = [s for cat in rs.CULTURE_MMLU_CATEGORIES.values() for s in cat]
    assert len(subjects) == 57 and len(set(subjects)) == 57
    assert {c: len(v) for c, v in rs.CULTURE_MMLU_CATEGORIES.items()} == \
        {"stem": 19, "humanities": 13, "social_sciences": 12, "other": 13}
    assert set(rs.SUBJECT_CATEGORY) == set(subjects)
    real = REPO / rs.EXPERIMENT / rs.FULL_CELLS["araroopat_v5"] / "eval_rows" / f"{TASK}.parquet"
    if real.exists():
        import pyarrow.parquet as pq
        assert set(pq.read_table(real, columns=["source_config"]).column(0).to_pylist()) == set(subjects)


def test_join_keeps_only_rows_no_cell_truncates(tmp_path):
    w = _world(tmp_path, 40, lambda a, r, s: True, caps={"bpe16k_v5": {3, 5}, "araroopat_v5": {5}},
               sentinels={"native_base": {7}})
    full = rs.join_full(w["exp"], TASK, CELLS)
    assert full["n_total"] == 40 and full["n_intersection"] == 37
    assert not {3, 5, 7} & set(full["row_index"].tolist())
    assert full["capped"] == {"araroopat_v5": 1, "bpe16k_v5": 2, "native_v5": 0, "native_base": 1}
    assert all(len(full["correct"][a]["char"]) == 37 for a in ARMS)


def test_spread_permutation_heterogeneous_small_p_homogeneous_large_p():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.array(["a", "b", "c", "d"]), 250)
    homo = rng.choice([-1.0, 0.0, 0.0, 1.0], size=1000)                         # same gap distribution everywhere
    hetero = homo.copy()
    hetero[labels == "a"] = rng.choice([-1.0, -1.0, 0.0, 0.0], size=250)       # group a: gap −0.5, the rest 0
    out = rs.spread_permutation(labels, np.stack([hetero, homo], axis=1), n_perm=2000, seed=0)
    assert out["groups"] == ["a", "b", "c", "d"]
    assert out["p"][0] < 0.01 and out["observed"][0] > out["null_q95"][0]
    assert out["p"][1] > 0.2


def test_spread_permutation_is_seeded():
    rng = np.random.default_rng(1)
    labels = rng.choice(["x", "y", "z"], size=300)
    d = rng.choice([-1.0, 0.0, 1.0], size=300)
    a = rs.spread_permutation(labels, d[:, None], n_perm=500, seed=7)
    b = rs.spread_permutation(labels, d[:, None], n_perm=500, seed=7)
    c = rs.spread_permutation(labels, d[:, None], n_perm=500, seed=8)
    assert a == b
    assert a["observed"] == c["observed"] and (a["p"] != c["p"] or a["null_q95"] != c["null_q95"])
    # the permutation preserves the group sizes: a label vector with one group has no spread to test
    assert rs.spread_permutation(np.array(["x"] * 10), d[:10, None], 10, 0)["p"] == [None]


def test_group_labels_are_permuted_not_resampled():
    """Every permutation keeps the group sizes, so a constant gap has spread 0 under the null and p = 1."""
    labels = np.array(["a"] * 30 + ["b"] * 70)
    out = rs.spread_permutation(labels, np.full((100, 1), -1.0), n_perm=200, seed=0)
    assert out["observed"] == [0.0] and out["p"] == [1.0]


def test_end_to_end_planted_category_effect(tmp_path):
    """AraRooPat is wrong on every stem row and right elsewhere; the three other arms are always right. The
    AraRooPat − BPE-16K gap is −1 on stem and 0 on the other categories, in both decisions."""
    stem = set(range(4))                                                         # SUBJECTS[0:4] are stem

    def fn(arm, r, s):
        return not (arm == "araroopat_v5" and s in stem)
    n = 160
    subset = list(range(0, n, 2))
    w = _world(tmp_path, n, fn, subset=subset)
    res = rs.analyze(w["exp"], w["ls"], w["rows"], [TASK], n_boot=200, n_perm=300, seed=0, full_cells=CELLS)
    t = res["tasks"][TASK]
    assert t["full"]["n_intersection"] == n and t["subset"]["n_rows"] == len(subset)
    for src in (t["full"], t["subset"]):
        cat = src["partitions"]["category"]
        assert cat["order"] == ["stem", "humanities", "social_sciences", "other"]
        assert sum(g["n"] for g in cat["groups"].values()) == src["pooled"]["n"]
        g = cat["groups"]["stem"]["pairs"]["araroopat_v5-bpe16k_v5"]["char"]
        assert g["est"] == -1.0 and g["only_a"] == 0 and g["only_b"] == cat["groups"]["stem"]["n"]
        assert cat["groups"]["other"]["pairs"]["araroopat_v5-bpe16k_v5"]["char"]["est"] == 0.0
        assert cat["spread_test"]["araroopat_v5-bpe16k_v5|char"]["observed"] == 1.0
        assert cat["spread_test"]["araroopat_v5-bpe16k_v5|char"]["p"] < 0.01
        assert cat["groups"]["stem"]["acc"]["araroopat_v5"]["char"] == 0.0
    # the rotation-averaged decision equals the planted per-row correctness (same scores under every rotation)
    assert t["subset"]["pooled"]["acc"]["araroopat_v5"]["char"] == pytest.approx(
        1 - sum(1 for r in subset if (r % len(SUBJECTS)) in stem) / len(subset))
    r = res["readings"][f"{TASK}|rotation_averaged|araroopat_v5-bpe16k_v5|char"]
    assert r["R1_concentrated"] and r["R1_groups_ci_excludes_zero"] == ["stem"] and not r["R2_broad"]
    assert set(t["composition"]["category"]) == {"stem", "humanities", "social_sciences", "other"}
    # fragmentation terciles cut the analysed rows in three
    fr = t["full"]["partitions"]["frag_vs_bpe16k"]
    sizes = [fr["groups"][x]["n"] for x in ("T1", "T2", "T3")]
    assert sum(sizes) == n and min(sizes) >= n // 4 and len(fr["boundaries"]) == 2
    assert "subject" in t["full"]["partitions"] and "spread_test" not in t["full"]["partitions"]["subject"]
    md = rs.to_markdown(res)
    assert "## Readings (pre-registered)" in md and "stem" in md


def test_arabic_mmlu_group_map(tmp_path):
    snap = tmp_path / "snap"
    for cfg, rows in {"Math (Primary School)": [("STEM", "Primary")] * 3,
                      "Driving Test": [("Other", None)] * 2,
                      "All": [("STEM", "Primary"), ("Other", None)]}.items():
        (snap / cfg).mkdir(parents=True)
        lines = ["Group,Level"] + [f"{g},{'' if l is None else l}" for g, l in rows]
        (snap / cfg / "test.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    m = rs.arabic_mmlu_group_map(snap)
    assert m["group"] == {"Math (Primary School)": "STEM", "Driving Test": "Other"}          # "All" excluded
    assert m["level"] == {"Math (Primary School)": "Primary", "Driving Test": "unspecified"}
    (snap / "Mixed").mkdir()
    (snap / "Mixed" / "test.csv").write_text("Group,Level\nSTEM,High\nOther,High\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not one-to-one"):
        rs.arabic_mmlu_group_map(snap)


def test_monotone():
    assert rs.monotone([-0.1, -0.05, 0.0]) == "increasing"
    assert rs.monotone([0.0, -0.02, -0.02]) == "decreasing"
    assert rs.monotone([0.0, -0.1, 0.0]) == "none" and rs.monotone([0.1, 0.1, 0.1]) == "none"
    assert rs.monotone([None, 0.0, 0.1]) is None
