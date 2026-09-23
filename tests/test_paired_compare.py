"""``scripts/judge/paired_compare.py`` — paired judge comparison of two cells."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location("paired_compare", REPO / "scripts" / "judge" / "paired_compare.py")
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)


def _judge_file(cell_dir: Path, judge: str, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = cell_dir / "freeform_judge"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "id": [r[0] for r in rows],
        "score": [r[1] for r in rows],
        "correctness": [r[1] for r in rows],
        "fluency": [r[1] for r in rows],
        "instruction_following": [r[1] for r in rows],
        "parse_ok": [r[2] if len(r) > 2 else True for r in rows],
    }), d / f"{judge}.parquet")


@pytest.fixture
def experiment(tmp_path):
    _judge_file(tmp_path / "cell_a", "j1", [(f"p{i}", 2.0) for i in range(20)])
    _judge_file(tmp_path / "cell_b", "j1", [(f"p{i}", 3.0) for i in range(20)])
    return tmp_path


def test_delta_is_the_paired_mean_and_the_ci_brackets_it(experiment):
    res = pc.compare(experiment, "cell_a", "cell_b", "j1", n_boot=500, seed=0)
    assert res["score"]["n"] == 20
    assert res["score"]["delta_mean"] == pytest.approx(1.0)
    assert res["score"]["ci_low"] <= 1.0 <= res["score"]["ci_high"]
    assert res["score"]["win_rate"] == 1.0
    assert res["mean_baseline"] == pytest.approx(2.0) and res["mean_cell"] == pytest.approx(3.0)


def test_direction_is_cell_minus_baseline(experiment):
    a = pc.compare(experiment, "cell_a", "cell_b", "j1", n_boot=200, seed=0)
    b = pc.compare(experiment, "cell_b", "cell_a", "j1", n_boot=200, seed=0)
    assert a["score"]["delta_mean"] == pytest.approx(-b["score"]["delta_mean"])


def test_only_shared_ids_are_paired(tmp_path):
    _judge_file(tmp_path / "a", "j1", [("p1", 1.0), ("p2", 2.0), ("p3", 3.0)])
    _judge_file(tmp_path / "b", "j1", [("p2", 4.0), ("p3", 3.0), ("p9", 5.0)])
    res = pc.compare(tmp_path, "a", "b", "j1", n_boot=200, seed=0)
    assert res["score"]["n"] == 2                      # p2 and p3
    assert res["score"]["delta_mean"] == pytest.approx(1.0)
    assert res["score"]["win_rate"] == pytest.approx(0.5)
    assert res["score"]["tie_rate"] == pytest.approx(0.5)


def test_unparsed_verdicts_are_skipped(tmp_path):
    _judge_file(tmp_path / "a", "j1", [("p1", 1.0), ("p2", 1.0)])
    _judge_file(tmp_path / "b", "j1", [("p1", 5.0, False), ("p2", 2.0, True)])
    res = pc.compare(tmp_path, "a", "b", "j1", n_boot=200, seed=0)
    assert res["score"]["n"] == 1
    assert res["score"]["delta_mean"] == pytest.approx(1.0)


def test_sub_scores_are_reported_when_asked(experiment):
    res = pc.compare(experiment, "cell_a", "cell_b", "j1", n_boot=200, seed=0, sub_scores=True)
    assert set(res["sub_scores"]) == set(pc.SUB_SCORES)
    assert res["sub_scores"]["fluency"]["delta_mean"] == pytest.approx(1.0)


def test_a_missing_judge_file_names_the_path(experiment):
    with pytest.raises(SystemExit, match="no judge file at"):
        pc.compare(experiment, "cell_a", "cell_missing", "j1")


def test_seed_makes_the_ci_reproducible(experiment):
    a = pc.compare(experiment, "cell_a", "cell_b", "j1", n_boot=300, seed=5)
    b = pc.compare(experiment, "cell_a", "cell_b", "j1", n_boot=300, seed=5)
    assert a["score"] == b["score"]


def test_format_row_mentions_both_cells(experiment):
    line = pc.format_row(pc.compare(experiment, "cell_a", "cell_b", "j1", n_boot=200, seed=0))
    assert "cell_b" in line and "cell_a" in line and "win/tie/loss" in line


def test_two_experiments_pair_a_cell_with_its_twin(tmp_path):
    """--experiment twice: the baseline from the first folder, the cell from the second (the
    decoding ablation pairs an rp12 cell with its greedy twin in the main experiment)."""
    main_exp, abl = tmp_path / "main", tmp_path / "ablation"
    _judge_file(main_exp / "cell", "j1", [(f"p{i}", 2.0) for i in range(10)])
    _judge_file(abl / "cell_rp12", "j1", [(f"p{i}", 3.0 if i < 5 else 2.0) for i in range(10)])
    res = pc.compare(main_exp, "cell", "cell_rp12", "j1", experiment_b=abl)
    assert res["score"]["delta_mean"] == pytest.approx(0.5) and res["score"]["n"] == 10
    assert res["experiment"] == str(main_exp) and res["experiment_cell"] == str(abl)
    # the CLI form, and the bare-path form without --experiment
    assert pc.main(["--experiment", str(main_exp), "--experiment", str(abl), "--judge", "j1",
                    "--cells", "cell", "cell_rp12"]) == 0
    assert pc.main(["--judge", "j1", "--cells", str(main_exp / "cell"), str(abl / "cell_rp12")]) == 0
    res2 = pc.compare(None, str(main_exp / "cell"), str(abl / "cell_rp12"), "j1")
    assert res2["score"] == res["score"]
    with pytest.raises(SystemExit, match="no judge file"):
        pc.compare(main_exp, "cell", "cell_rp12", "j1")          # one folder: the rp12 cell is not in main
    with pytest.raises(SystemExit):
        pc.main(["--experiment", "a", "--experiment", "b", "--experiment", "c", "--cells", "x", "y"])
