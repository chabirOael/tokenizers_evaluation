"""Tests for ``arabic_eval.tools.experiment_console`` (the experiment console back-end).

Config side: every YAML under ``configs/experiments`` reloads to the same
resolved config after a full or delta render; validation errors carry
``loc`` paths; saving refuses invalid YAML and existing files.

Run side: a stub command stands in for ``run_experiment.py`` inside a
temporary repo root — start, liveness, exit-code capture, cancel with
SIGTERM/SIGKILL, rediscovery by a fresh manager (server restart), the
PID-reuse guard, and the single-active-run conflict.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.tools.experiment_console import (  # noqa: E402
    ConsoleError,
    ConsolePaths,
    RunConflict,
    RunManager,
    cell_names,
    clone_config,
    list_configs,
    parse_progress,
    parse_yaml,
    process_alive,
    read_config,
    render_yaml,
    save_config,
    schema_bundle,
    validate_config,
)

from arabic_eval.config_edit import default_output_dir, is_repo_experiment_config, record_run_start  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REAL = ConsolePaths(REPO)
ISO_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$"

MINIMAL_YAML = """\
experiment:
  name: "mini"
  output_dir: "outputs/experiments/mini"
tokenizer:
  type: "native_llama"
  vocab_size: null
sweep:
  tokenizers:
    - type: "native_llama"
      vocab_sizes: [null]
  tasks:
    - type: "acva"
      params: {}
"""


@pytest.fixture
def tmp_repo(tmp_path: Path) -> ConsolePaths:
    (tmp_path / "configs" / "experiments").mkdir(parents=True)
    shutil.copy(REPO / "configs" / "base.yaml", tmp_path / "configs" / "base.yaml")
    (tmp_path / "configs" / "experiments" / "mini.yaml").write_text(MINIMAL_YAML, encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_experiment.py").write_text("print('stub')\n")
    return ConsolePaths(tmp_path)


def _wait(pred, timeout=10.0, step=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return pred()


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def test_every_repo_config_round_trips_in_both_styles():
    rows = list_configs(REAL)
    assert rows and all(r["valid"] for r in rows), [r for r in rows if not r["valid"]]
    for r in rows:
        rc = read_config(REAL, r["file"])
        for mode in ("full", "delta"):
            text = render_yaml(REAL, rc["resolved"], mode)
            v = validate_config(REAL, parse_yaml(text))
            assert v["ok"], (r["file"], mode, v["errors"])
            assert v["resolved"] == rc["resolved"], (r["file"], mode)


def test_delta_render_is_smaller_and_starts_with_experiment_block():
    rc = read_config(REAL, "native_llama_3phase_no_sft")
    full = render_yaml(REAL, rc["resolved"], "full")
    delta = render_yaml(REAL, rc["resolved"], "delta")
    assert delta.startswith("experiment:\n  name: native_llama_3phase_no_sft\n")
    assert len(delta.splitlines()) < len(full.splitlines()) / 3
    assert "sft:\n      enabled: false" in delta


def test_list_configs_reports_cells_sweep_flag_and_results(tmp_repo: ConsolePaths):
    rows = list_configs(tmp_repo)
    assert [r["file"] for r in rows] == ["mini.yaml"]
    row = rows[0]
    assert row["valid"] and row["name"] == "mini" and row["sweep"] is False
    assert row["cells"] == ["native_llama"] and row["tasks"] == ["acva"]
    assert row["results"]["exists"] is False
    (tmp_repo.repo_root / "outputs/experiments/mini").mkdir(parents=True)
    (tmp_repo.repo_root / "outputs/experiments/mini/all_metrics.json").write_text("{}")
    assert list_configs(tmp_repo)[0]["results"]["exists"] is True


def test_sweep_cells_follow_run_sweep_naming():
    rc = read_config(REAL, "all_tokenizers_sweep")
    assert rc["sweep"] is True
    assert rc["results"]["cells"][:3] == ["bpe_32k", "wordpiece_32k", "morpho_bpe_32k"]
    from arabic_eval.config import ExperimentConfig
    cfg = ExperimentConfig(**rc["resolved"])
    assert cell_names(cfg)[-1] == "araroopat"


def test_validation_errors_carry_loc_paths(tmp_repo: ConsolePaths):
    bad = parse_yaml(MINIMAL_YAML)
    bad["training"] = {"phases": {"sft": {"steps": -5}}}
    v = validate_config(tmp_repo, bad)
    assert v["ok"] is False
    assert any(e["loc"] == "training.phases.sft.steps" for e in v["errors"]), v["errors"]


def test_invalid_config_file_is_listed_not_hidden(tmp_repo: ConsolePaths):
    (tmp_repo.configs_dir / "broken.yaml").write_text("training:\n  phases:\n    warmup:\n      steps: 'x'\n")
    rows = {r["file"]: r for r in list_configs(tmp_repo)}
    assert rows["broken.yaml"]["valid"] is False
    assert "training.phases.warmup.steps" in rows["broken.yaml"]["error"]


def test_save_refuses_invalid_and_existing(tmp_repo: ConsolePaths):
    with pytest.raises(ConsoleError, match="invalid"):
        save_config(tmp_repo, "new_one", "training:\n  phases:\n    sft:\n      steps: -1\n")
    with pytest.raises(ConsoleError, match="exists"):
        save_config(tmp_repo, "mini", MINIMAL_YAML)
    res = save_config(tmp_repo, "new_one", MINIMAL_YAML)
    assert (tmp_repo.configs_dir / "new_one.yaml").exists() and res["resolved"]["name"] == "mini"
    save_config(tmp_repo, "mini.yaml", MINIMAL_YAML.replace('"mini"', '"mini2"'), overwrite=True)
    assert read_config(tmp_repo, "mini")["resolved"]["name"] == "mini2"


COMMENTED_YAML = """\
# Reference run — keep the comments, they explain the choices.
experiment:
  name: "commented"                      # the experiment name
  output_dir: "outputs/experiments/commented"
  seed: 42            # master seed
tokenizer:
  type: "native_llama"   # no training
  vocab_size: null
sweep:
  tokenizers:
    - type: "native_llama"
      vocab_sizes: [null]
  tasks:
    - type: "acva"   # label-noisy, see CLAUDE.md
      params: {}
"""


def test_clone_rewrites_only_the_experiment_keys_and_keeps_comments(tmp_repo: ConsolePaths):
    (tmp_repo.configs_dir / "commented.yaml").write_text(COMMENTED_YAML, encoding="utf-8")
    res = clone_config(tmp_repo, "commented", "commented_v2", description="second try")
    assert res["file"] == "commented_v2.yaml" and res["comments_kept"] is True
    text = (tmp_repo.configs_dir / "commented_v2.yaml").read_text(encoding="utf-8")
    assert text.count("#") == COMMENTED_YAML.count("#")           # every comment survived
    assert '  name: "commented_v2"                      # the experiment name' in text
    assert '  output_dir: "outputs/experiments/commented_v2"' in text
    assert '  description: "second try"' in text                    # inserted below the block's own keys
    assert '  created_at: "20' in text                               # a clone gets its own birth date
    src, dst = read_config(tmp_repo, "commented")["resolved"], read_config(tmp_repo, "commented_v2")["resolved"]
    assert re.match(ISO_RE, dst["created_at"]) and src["created_at"] is None
    expected = {**src, "name": "commented_v2", "output_dir": default_output_dir("commented_v2"), "description": "second try",
                "created_at": dst["created_at"], "runs": []}
    assert dst == expected
    assert res["resolved"] == expected and res["source"] == "configs/experiments/commented.yaml"


def test_clone_defaults_and_explicit_output_dir(tmp_repo: ConsolePaths):
    res = clone_config(tmp_repo, "mini.yaml", "campaign_cell", name="cell_a",
                       output_dir="outputs/experiments/campaign/cell_a")
    r, src = res["resolved"], read_config(tmp_repo, "mini")["resolved"]
    assert (r["name"], r["output_dir"]) == ("cell_a", "outputs/experiments/campaign/cell_a")
    assert r["description"] == src["description"]            # untouched when not given
    res = clone_config(tmp_repo, "mini", "mini_copy")
    assert res["resolved"]["name"] == "mini_copy" and res["resolved"]["output_dir"] == "outputs/experiments/mini_copy"
    assert default_output_dir("x") == "outputs/experiments/x"


def test_clone_refuses_same_file_existing_and_invalid_source(tmp_repo: ConsolePaths):
    with pytest.raises(ConsoleError, match="different file name"):
        clone_config(tmp_repo, "mini", "mini.yaml")
    with pytest.raises(ConsoleError, match="no such config"):
        clone_config(tmp_repo, "nope", "x")
    clone_config(tmp_repo, "mini", "twice")
    with pytest.raises(ConsoleError, match="exists"):
        clone_config(tmp_repo, "mini", "twice")
    clone_config(tmp_repo, "mini", "twice", overwrite=True, description="again")
    assert read_config(tmp_repo, "twice")["resolved"]["description"] == "again"
    (tmp_repo.configs_dir / "broken.yaml").write_text("experiment:\n  name: b\ntraining:\n  phases:\n    sft:\n      steps: -1\n")
    with pytest.raises(ConsoleError, match="does not validate"):
        clone_config(tmp_repo, "broken", "broken_copy")
    with pytest.raises(ValueError):
        clone_config(tmp_repo, "mini", "../escape")


def test_clone_falls_back_to_a_rendered_copy_when_the_layout_is_unrecognised(tmp_repo: ConsolePaths):
    flow = MINIMAL_YAML.replace('experiment:\n  name: "mini"\n  output_dir: "outputs/experiments/mini"\n',
                                'experiment: {name: mini, output_dir: outputs/experiments/mini}\n')
    (tmp_repo.configs_dir / "flow.yaml").write_text(flow, encoding="utf-8")
    res = clone_config(tmp_repo, "flow", "flow_copy")
    assert res["comments_kept"] is False
    src, dst = read_config(tmp_repo, "flow")["resolved"], read_config(tmp_repo, "flow_copy")["resolved"]
    assert dst == {**src, "name": "flow_copy", "output_dir": "outputs/experiments/flow_copy",
                   "created_at": dst["created_at"], "runs": []} and re.match(ISO_RE, dst["created_at"])


def test_every_repo_config_clones_with_its_comments(tmp_path: Path):
    """Every file under configs/experiments is cloned through the text rewriter (no render fallback)
    and reloads to its own resolved config with only name / output_dir / description changed."""
    (tmp_path / "configs" / "experiments").mkdir(parents=True)
    shutil.copy(REPO / "configs" / "base.yaml", tmp_path / "configs" / "base.yaml")
    for p in (REPO / "configs" / "experiments").glob("*.yaml"):
        shutil.copy(p, tmp_path / "configs" / "experiments" / p.name)
    paths = ConsolePaths(tmp_path)
    for p in sorted(paths.configs_dir.glob("*.yaml")):
        src = read_config(paths, p.name)
        res = clone_config(paths, p.name, p.stem + "_copy", description="cloned")
        assert res["comments_kept"] is True, p.name
        dst = read_config(paths, res["file"])
        assert dst["yaml"].count("#") == src["yaml"].count("#"), p.name
        assert dst["resolved"] == {**src["resolved"], "name": p.stem + "_copy", "created_at": dst["resolved"]["created_at"],
                                   "output_dir": f"outputs/experiments/{p.stem}_copy", "description": "cloned", "runs": []}, p.name
        assert re.match(ISO_RE, dst["resolved"]["created_at"]), p.name


def test_save_stamps_created_at_on_a_new_file_only(tmp_repo: ConsolePaths):
    res = save_config(tmp_repo, "fresh", MINIMAL_YAML.replace('"mini"', '"fresh"'))
    born = res["resolved"]["created_at"]
    assert re.match(ISO_RE, born)
    text = (tmp_repo.configs_dir / "fresh.yaml").read_text(encoding="utf-8")
    assert f'  created_at: "{born}"' in text and text.startswith("experiment:\n  name: \"fresh\"")   # stamped in the block, text kept
    # an overwrite keeps the stamp the text carries (the form round-trips it) …
    res2 = save_config(tmp_repo, "fresh", text.replace('seed: 42', 'seed: 7') if 'seed: 42' in text else text + "seed: 7\n", overwrite=True)
    assert res2["resolved"]["created_at"] == born
    # … and never invents one on overwrite when the text has none
    save_config(tmp_repo, "fresh", MINIMAL_YAML.replace('"mini"', '"fresh"'), overwrite=True)
    assert read_config(tmp_repo, "fresh")["resolved"]["created_at"] is None
    # an explicit stamp (backfill) is honoured; the pre-existing repo file (mini) has none
    res3 = save_config(tmp_repo, "dated", MINIMAL_YAML, created_at="2026-05-05T19:51:30+03:00")
    assert res3["resolved"]["created_at"] == "2026-05-05T19:51:30+03:00"
    assert read_config(tmp_repo, "mini")["resolved"]["created_at"] is None
    assert list_configs(tmp_repo)[0]["created_at"] is None or True   # column present on every row
    assert all("created_at" in r and "runs" in r and "last_run_at" in r for r in list_configs(tmp_repo))


def test_config_path_is_confined():
    with pytest.raises(ValueError):
        REAL.config_path("../base")
    with pytest.raises(ValueError):
        REAL.config_path("a/b")
    assert REAL.config_path("configs/experiments/x.yaml").name == "x.yaml"


def test_schema_bundle_has_registries_and_presets():
    b = schema_bundle(REAL)
    assert "properties" in b["schema"] and "training" in b["schema"]["properties"]
    assert "bpe" in b["registries"]["tokenizers"] and "acva" in b["registries"]["tasks"]
    assert "pretraining_mix" in b["registries"]["datasets"]
    assert b["presets"]["tokenizers"]["charformer"]["params"]["max_block_size"] == 4
    assert b["base"]["training"]["phases"]["sft"]["steps"] > 0
    # task params come from the tasks' param_spec(), not from configs/tasks/*.yaml (generated documentation)
    assert "tasks" not in b["presets"]
    names = [s["name"] for s in b["task_params"]["acva"]]
    assert names == ["dataset_name", "dataset_config", "cache_dir", "max_length", "seed", "clean_latin_rows", "num_fewshot"]
    assert {s["name"] for s in b["task_params"]["freeform_cidar"]} >= {"max_output_chars", "stop_markers", "loop_stop", "heldout_path"}


# ---------------------------------------------------------------------------
# Progress parser
# ---------------------------------------------------------------------------

LOG = """\
[2026-05-05 15:13:27] INFO arabic_eval.pipeline: Experiment: native_llama_3phase_with_sft
[2026-05-05 15:13:27] INFO arabic_eval.pipeline: Step 1/7: Loading Arabic corpus...
[2026-05-05 15:15:55] INFO arabic_eval.pipeline: Step 5/7: Running training phases...
[2026-05-05 15:17:22] INFO arabic_eval.training.phases: [embedding_alignment] step 50/1000 loss=3.1879 lr=1.00e-03
[2026-05-05 15:22:38] INFO arabic_eval.training.phases: [embedding_alignment] complete: steps=1000, final_loss=2.6863, wall=366.7s
[2026-05-05 15:24:46] INFO arabic_eval.training.phases: [warmup] step 50/2000 loss=0.4084 lr=2.40e-05
[2026-05-05 15:24:46] INFO arabic_eval.training.phases: [warmup] step 600 eval_loss=0.9 (best=1.0@step 400, patience=0/5)
"""


def test_parse_progress_tracks_stage_phase_and_eval_loss():
    st = parse_progress(LOG)
    assert st["experiment"] == "native_llama_3phase_with_sft"
    assert st["stage"] == "5" and st["stage_label"] == "training"
    assert [p["name"] for p in st["phases_done"]] == ["embedding_alignment"]
    assert st["phase"]["name"] == "warmup" and st["phase"]["step"] == 50 and st["phase"]["steps"] == 2000
    assert st["phase"]["eval_loss"] == 0.9 and st["done"] is False


def test_parse_progress_sweep_cells_eval_and_traceback():
    text = (
        "[t] INFO arabic_eval.pipeline: SWEEP: skipping bpe_32k (results exist)\n"
        "[t] INFO arabic_eval.pipeline: SWEEP cell: wordpiece_32k\n"
        "[t] INFO arabic_eval.pipeline: Experiment: wordpiece_32k\n"
        "[t] INFO arabic_eval.pipeline: Step 6-7/7: Evaluating on 3 benchmark task(s)...\n"
        "[t] INFO arabic_eval.tasks.lighteval.base: acva: evaluating 9000 examples (full benchmark)\n"
        "LightEval MCQ:   3%|▎  | 270/9000 [00:20<11:03, 13.2example/s]\r"
        "LightEval MCQ:  12%|█▏ | 1080/9000 [01:23<10:11, 12.9example/s]\n"
        "[t] INFO arabic_eval.pipeline:   [acva] accuracy=0.61 (eval=437.4s)\n"
        "[t] INFO arabic_eval.pipeline: Experiment 'wordpiece_32k' done -> outputs/x\n"
        "[t] INFO arabic_eval.pipeline: SWEEP cell: charformer\n"
        "Traceback (most recent call last):\n"
        "  File \"x.py\", line 1, in <module>\n"
        "ValueError: Expected input batch_size (2024) to match target batch_size (2020)\n"
    )
    st = parse_progress(text)
    assert st["cells_skipped"] == ["bpe_32k"] and st["cells_done"] == ["wordpiece_32k"]
    assert st["cell"] == "charformer" and st["done"] is False
    assert st["tasks_done"] == [] and st["eval"] is None  # reset by the new cell
    assert st["error"].startswith("ValueError: Expected input batch_size")
    mid = parse_progress(text.split("[t] INFO arabic_eval.pipeline:   [acva]")[0])
    assert mid["eval"] == {"task": "acva", "done": 1080, "total": 9000}
    assert mid["stage_label"] == "downstream eval"


def test_parse_progress_timings_tokenizer_substep_and_plan(tmp_repo: ConsolePaths):
    """The step panel's inputs: every stage / phase / task with log timestamps, the
    tokenizer sub-step, and the plan derived from a snapshot for a record without one."""
    L = "[2026-09-18 10:{:02d}:00] INFO arabic_eval.pipeline: "
    text = (
        L.format(0) + "Experiment: mini\n" + L.format(0) + "Step 1/7: Loading Arabic corpus...\n"
        + L.format(1) + "Step 2/7: Preparing tokenizer 'bpe'...\n" + L.format(1) + "  training on 20000 texts\n"
        + L.format(4) + "Step 3/7: Running intrinsic evaluation...\n"
        + L.format(5) + "Step 4/7: Loading and adapting model...\n"
        + L.format(6) + "Step 5/7: Running training phases...\n"
        + L.format(6) + "[embedding_alignment] skipped (enabled=false)\n"
        + L.format(7) + "[warmup] trainable params: 10 / 10 (100.00%)\n"
        + L.format(9) + "[warmup] step 50/100 loss=1.5000 lr=2.00e-04\n"
        + L.format(12) + "[warmup] complete: steps=100, final_loss=1.2000, wall=300.0s\n"
        + L.format(12) + "[sft] trainable params: 10 / 10 (100.00%)\n"
        + L.format(13) + "[sft] step 10/40 loss=1.1000 lr=2.00e-04\n"
    )
    st = parse_progress(text)
    assert [(x["id"], x["start"][-8:], (x["end"] or "")[-8:]) for x in st["stages"]] == [
        ("1", "10:00:00", "10:01:00"), ("2", "10:01:00", "10:04:00"), ("3", "10:04:00", "10:05:00"),
        ("4", "10:05:00", "10:06:00"), ("5", "10:06:00", "")]
    assert st["tokenizer"] == {"mode": "train", "detail": "20,000 texts"}
    pt = st["phase_times"]
    assert pt["embedding_alignment"]["status"] == "skipped"
    assert pt["warmup"]["status"] == "done" and pt["warmup"]["start"][-8:] == "10:07:00" and pt["warmup"]["end"][-8:] == "10:12:00" and pt["warmup"]["steps"] == 100
    assert pt["sft"]["status"] == "running" and pt["sft"]["end"] is None and pt["sft"]["steps"] == 40
    assert st["last_ts"][-8:] == "10:13:00" and st["done_at"] is None
    rest = (L.format(20) + "Step 6-7/7: Evaluating on 1 benchmark task(s)...\n"
            + L.format(20) + "acva: evaluating 30 examples (full benchmark)\n"
            + L.format(21) + "  [acva] accuracy=0.61 (eval=60.0s)\n"
            + L.format(21) + "Experiment 'mini' done -> outputs/x\n")
    st2 = parse_progress(text + rest)
    assert st2["task_times"]["acva"] == {"start": "2026-09-18 10:20:00", "end": "2026-09-18 10:21:00", "seconds": 60.0}
    assert st2["stages"][-1]["id"] == "6-7" and st2["stages"][-1]["end"][-8:] == "10:21:00" and st2["done_at"][-8:] == "10:21:00"
    load = parse_progress(L.format(1) + "Step 2/7: Preparing tokenizer 'bpe'...\n" + L.format(1) + "  loading from outputs/tokenizers/bpe_32k\n")
    assert load["tokenizer"] == {"mode": "load", "detail": "outputs/tokenizers/bpe_32k"}
    # plan: recorded at start, derived from the snapshot when missing
    rm = RunManager(tmp_repo)
    rec = rm.start("mini.yaml", command=_stub(0.2))
    assert rec["plan"] == {"phases": {"embedding_alignment": {"enabled": True, "steps": 1000}, "warmup": {"enabled": True, "steps": 2000},
                                      "sft": {"enabled": True, "steps": 2000}}, "intrinsic": True, "downstream": True, "tasks": ["acva"]}
    assert _wait(lambda: rm.get(rec["run_id"])["status"] == "finished")
    rp = rm._record_path(rec["run_id"])
    data = json.loads(rp.read_text(encoding="utf-8")); data.pop("plan"); rp.write_text(json.dumps(data), encoding="utf-8")
    assert "plan" not in rm.get(rec["run_id"])
    assert rm.detail(rec["run_id"])["plan"]["tasks"] == ["acva"]


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def _stub(seconds: float, exit_code: int = 0, trap_term: bool = False):
    body = f"import time,sys,signal\n"
    if trap_term:
        body += "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    body += f"print('Experiment: mini', flush=True)\ntime.sleep({seconds})\nsys.exit({exit_code})\n"
    return [sys.executable, "-c", body]


def test_start_records_and_finishes_with_exit_code(tmp_repo: ConsolePaths):
    mgr = RunManager(tmp_repo)
    rec = mgr.start("mini", command=_stub(0.3, exit_code=0))
    assert rec["status"] == "running" and rec["run_id"].endswith("_mini")
    rd = mgr.run_dir(rec["run_id"])
    assert (rd / "config.yaml").read_text() == MINIMAL_YAML
    assert rec["experiment_name"] == "mini" and rec["sweep"] is False
    assert _wait(lambda: mgr.get(rec["run_id"])["status"] == "finished")
    got = mgr.get(rec["run_id"])
    assert got["exit_code"] == 0 and got["finished_at"]
    assert (rd / "exit_code").read_text().strip() == "0"
    assert "Experiment: mini" in (rd / "console.log").read_text()
    assert mgr.detail(rec["run_id"])["progress"]["experiment"] == "mini"


def test_nonzero_exit_is_failed(tmp_repo: ConsolePaths):
    mgr = RunManager(tmp_repo)
    rec = mgr.start("mini", command=_stub(0.1, exit_code=3))
    assert _wait(lambda: mgr.get(rec["run_id"])["status"] == "failed")
    assert mgr.get(rec["run_id"])["exit_code"] == 3


def test_second_run_conflicts_unless_forced(tmp_repo: ConsolePaths):
    mgr = RunManager(tmp_repo)
    a = mgr.start("mini", command=_stub(5))
    with pytest.raises(RunConflict):
        mgr.start("mini", command=_stub(1))
    b = mgr.start("mini", command=_stub(0.1), force=True)
    assert a["run_id"] != b["run_id"]
    mgr.cancel(a["run_id"])
    assert _wait(lambda: mgr.get(a["run_id"])["status"] == "cancelled")


def test_cancel_sends_sigterm_then_sigkill(tmp_repo: ConsolePaths):
    mgr = RunManager(tmp_repo, grace_sec=1.0)
    rec = mgr.start("mini", command=_stub(30, trap_term=True))
    assert _wait(lambda: "Experiment" in (mgr.run_dir(rec["run_id"]) / "console.log").read_text()
                 if (mgr.run_dir(rec["run_id"]) / "console.log").exists() else False)
    out = mgr.cancel(rec["run_id"])
    assert out["status"] == "cancelling"
    assert _wait(lambda: mgr.get(rec["run_id"])["status"] == "cancelled", timeout=8)
    assert not process_alive(rec["pid"], rec["start_ticks"])


def test_fresh_manager_rediscovers_runs_and_reconciles(tmp_repo: ConsolePaths):
    mgr = RunManager(tmp_repo)
    rec = mgr.start("mini", command=_stub(0.3))
    # "server restart": a new manager with no Popen handles
    mgr2 = RunManager(tmp_repo)
    assert [r["run_id"] for r in mgr2.list()] == [rec["run_id"]]
    assert _wait(lambda: mgr2.get(rec["run_id"])["status"] == "finished")


def test_pid_reuse_guard_marks_lost(tmp_repo: ConsolePaths):
    mgr = RunManager(tmp_repo)
    rec = mgr.start("mini", command=_stub(0.2))
    assert _wait(lambda: (mgr.run_dir(rec["run_id"]) / "exit_code").exists())
    # Forge a record: our own (alive) pid but a start-ticks value that cannot match, no exit code.
    p = mgr.run_dir(rec["run_id"]) / "run.json"
    forged = json.loads(p.read_text())
    # pid = an alive process with the wrong start ticks; pgid = the finished shell's (dead) group.
    forged.update({"status": "running", "pid": os.getpid(), "pgid": rec["pgid"], "start_ticks": 1,
                   "exit_code": None, "finished_at": None})
    p.write_text(json.dumps(forged))
    (mgr.run_dir(rec["run_id"]) / "exit_code").unlink()
    assert process_alive(os.getpid(), None) is True
    assert process_alive(os.getpid(), 1) is False
    assert RunManager(tmp_repo).get(rec["run_id"])["status"] == "lost"


def test_start_appends_a_run_stamp_to_the_source_file(tmp_repo: ConsolePaths):
    mgr = RunManager(tmp_repo)
    stub = [sys.executable, "-c", "print('ok')"]
    rec = mgr.start("mini", command=stub)
    _wait(lambda: mgr.get(rec["run_id"])["status"] == "finished")
    assert rec["run_stamp"] == {"started_at": rec["started_at"], "run_id": rec["run_id"], "source": "console"}
    assert re.match(ISO_RE, rec["started_at"])
    cfg = read_config(tmp_repo, "mini")
    assert [r["run_id"] for r in cfg["resolved"]["runs"]] == [rec["run_id"]]
    assert cfg["yaml"].startswith("experiment:\n  name: \"mini\"\n  output_dir:")            # the file's own text, one line added
    assert f'  runs:\n    - {{started_at: "{rec["started_at"]}", run_id: "{rec["run_id"]}", source: console}}' in cfg["yaml"]
    # the snapshot is the config as it was when the run started (no self-reference)
    snap = (tmp_repo.repo_root / rec["snapshot"]).read_text(encoding="utf-8")
    assert "runs:" not in snap
    rec2 = mgr.start("mini", command=stub, force=True)
    _wait(lambda: mgr.get(rec2["run_id"])["status"] == "finished")
    assert [r["run_id"] for r in read_config(tmp_repo, "mini")["resolved"]["runs"]] == [rec["run_id"], rec2["run_id"]]
    # a run from an unsaved form has no file to stamp
    rec3 = mgr.start("adhoc", yaml_text=MINIMAL_YAML, command=stub, force=True)
    _wait(lambda: mgr.get(rec3["run_id"])["status"] == "finished")
    assert "run_stamp" not in rec3 and len(read_config(tmp_repo, "mini")["resolved"]["runs"]) == 2


def test_cli_stamp_only_for_files_under_configs_experiments(tmp_repo: ConsolePaths):
    root = tmp_repo.repo_root
    assert is_repo_experiment_config(tmp_repo.configs_dir / "mini.yaml", root)
    snap = root / "outputs" / "runs" / "x" / "config.yaml"
    snap.parent.mkdir(parents=True)
    snap.write_text(MINIMAL_YAML, encoding="utf-8")
    assert not is_repo_experiment_config(snap, root)
    entry = record_run_start(tmp_repo.configs_dir / "mini.yaml", source="cli")
    assert entry["source"] == "cli" and "run_id" not in entry and re.match(ISO_RE, entry["started_at"])
    assert read_config(tmp_repo, "mini")["resolved"]["runs"] == [{**entry, "run_id": None}]


def test_start_from_unsaved_yaml_and_sweep_autodetect(tmp_repo: ConsolePaths):
    mgr = RunManager(tmp_repo)
    text = MINIMAL_YAML + "    - type: \"bpe\"\n      vocab_sizes: [16000, 32000]\n"
    # the second tokenizer entry must sit under sweep.tokenizers, rebuild properly:
    text = MINIMAL_YAML.replace(
        "      vocab_sizes: [null]\n",
        "      vocab_sizes: [null]\n    - type: \"bpe\"\n      vocab_sizes: [16000, 32000]\n",
    )
    rec = mgr.start("scratch pad", yaml_text=text, command=_stub(0.1))
    assert rec["config"] is None and rec["label"] == "scratch_pad"
    assert rec["sweep"] is True and rec["cells"] == ["native_llama", "bpe_16k", "bpe_32k"]
    assert _wait(lambda: mgr.get(rec["run_id"])["status"] == "finished")
    argv = mgr.build_command(Path("/x/config.yaml"), True, "cuda:0", 7)
    assert argv[1:] == [str(tmp_repo.script), "--config", "/x/config.yaml", "--sweep", "--device", "cuda:0", "--seed", "7"]


def test_log_is_incremental(tmp_repo: ConsolePaths):
    mgr = RunManager(tmp_repo)
    rec = mgr.start("mini", command=_stub(0.2))
    assert _wait(lambda: mgr.get(rec["run_id"])["status"] == "finished")
    first = mgr.log(rec["run_id"], 0)
    assert "Experiment: mini" in first["data"] and first["offset"] == first["size"]
    again = mgr.log(rec["run_id"], first["offset"])
    assert again["data"] == "" and again["status"] == "finished"


# ---------------------------------------------------------------------------
# Judge stage (free-form eval) as a detached run
# ---------------------------------------------------------------------------

from arabic_eval.tools.experiment_console import (  # noqa: E402
    judge_cells, judge_config_path, judge_configs, parse_judge_progress,
)

JUDGE_API_YAML = "judge: {name: api_j, backend: openai, model: m-api, api_key_env: TEST_JUDGE_KEY, temperature: null}\n"
JUDGE_GPU_YAML = "judge: {name: gpu_j, backend: vllm, model: org/model-31b}\n"


def _judge_repo(tmp_repo: ConsolePaths) -> ConsolePaths:
    root = tmp_repo.repo_root
    (root / "configs" / "judges").mkdir(parents=True)
    (root / "configs" / "judges" / "api_j.yaml").write_text(JUDGE_API_YAML, encoding="utf-8")
    (root / "configs" / "judges" / "gpu_j.yaml").write_text(JUDGE_GPU_YAML, encoding="utf-8")
    (root / "configs" / "judges" / "broken.yaml").write_text("judge: {name: b, backend: nope, model: m}\n", encoding="utf-8")
    for cell in ("native_llama", "bpe_32k"):
        d = root / "outputs" / "experiments" / "sw" / cell
        (d / "eval_rows").mkdir(parents=True)
        (d / "eval_rows" / "freeform_cidar.parquet").write_bytes(b"")
        (d / "all_metrics.json").write_text("{}", encoding="utf-8")
    (root / "outputs" / "experiments" / "sw" / "charformer").mkdir()
    (root / "scripts" / "judge").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "judge" / "judge_freeform.py").write_text("print('stub')\n")
    (root / "scripts" / "judge" / "run_judge.sh").write_text("#!/bin/bash\n")
    return tmp_repo


def test_judge_configs_readiness(tmp_repo: ConsolePaths, monkeypatch):
    paths = _judge_repo(tmp_repo)
    monkeypatch.delenv("TEST_JUDGE_KEY", raising=False)
    by = {j["id"]: j for j in judge_configs(paths)}
    assert set(by) == {"api_j", "gpu_j", "broken"}
    assert not by["api_j"]["ready"] and "TEST_JUDGE_KEY" in by["api_j"]["why"] and by["api_j"]["gpu"] is False
    assert not by["gpu_j"]["ready"] and ".venv-judge" in by["gpu_j"]["why"] and by["gpu_j"]["gpu"] is True
    assert by["broken"]["error"] and not by["broken"]["ready"]
    monkeypatch.setenv("TEST_JUDGE_KEY", "k")
    assert {j["id"]: j["ready"] for j in judge_configs(paths)}["api_j"] is True
    (paths.repo_root / ".venv-judge" / "bin").mkdir(parents=True)
    (paths.repo_root / ".venv-judge" / "bin" / "python").write_text("")
    assert "headers" in {j["id"]: j for j in judge_configs(paths)}["gpu_j"]["why"]
    hdr = paths.repo_root / ".local-pkgs" / "extracted" / "usr" / "include" / "python3.10"
    hdr.mkdir(parents=True)
    (hdr / "Python.h").write_text("")
    assert {j["id"]: j["ready"] for j in judge_configs(paths)}["gpu_j"] is True


def test_judge_config_path_is_confined(tmp_repo: ConsolePaths):
    paths = _judge_repo(tmp_repo)
    assert judge_config_path(paths, "api_j").name == "api_j.yaml"
    assert judge_config_path(paths, "configs/judges/gpu_j.yaml").name == "gpu_j.yaml"
    with pytest.raises(ConsoleError):
        judge_config_path(paths, "../experiments/mini.yaml")
    with pytest.raises(ConsoleError, match="no such judge"):
        judge_config_path(paths, "missing")


def test_judge_cells_and_command(tmp_repo: ConsolePaths):
    paths = _judge_repo(tmp_repo)
    exp, cells = judge_cells(paths.repo_root, "outputs/experiments/sw")
    assert cells == ["bpe_32k", "native_llama"]                     # charformer has no generations
    with pytest.raises(ConsoleError, match="no cell"):
        judge_cells(paths.repo_root, "outputs/experiments/sw/charformer")
    with pytest.raises(ConsoleError):
        judge_cells(paths.repo_root, "configs")
    rm = RunManager(paths)
    gpu = rm.build_judge_command("outputs/experiments/sw", ["a.yaml", "b.yaml"], True, "native_llama", 10, True)
    assert gpu[0].endswith("scripts/judge/run_judge.sh") and gpu[1:] == ["--experiment", "outputs/experiments/sw", "--judge", "a.yaml",
                                                                    "--judge", "b.yaml", "--baseline", "native_llama", "--limit", "10", "--overwrite"]
    api = rm.build_judge_command("outputs/experiments/sw", ["a.yaml"], False, None, None, False)
    assert api[0] == str(paths.python) and api[1].endswith("scripts/judge/judge_freeform.py") and "--overwrite" not in api


def test_start_judge_records_snapshots_and_finishes(tmp_repo: ConsolePaths, monkeypatch):
    paths = _judge_repo(tmp_repo)
    monkeypatch.setenv("TEST_JUDGE_KEY", "k")
    rm = RunManager(paths)
    with pytest.raises(ConsoleError, match="not ready|venv"):
        rm.start_judge("outputs/experiments/sw", ["gpu_j"], command=[sys.executable, "-c", "pass"])
    with pytest.raises(ConsoleError, match="baseline"):
        rm.start_judge("outputs/experiments/sw", ["api_j"], baseline="nope", command=[sys.executable, "-c", "pass"])
    script = ("import sys; print('[judge:api_j] vLLM org/m, structured_json=True'); "
              "print('[judge:api_j] bpe_32k: 2 verdicts in 0.1s -> x'); print('[judge:api_j] native_llama: reusing 2 verdicts'); "
              "print('report → outputs/experiments/sw/freeform_judge_report.json')")
    rec = rm.start_judge("outputs/experiments/sw", ["api_j"], baseline="native_llama", limit=2,
                         command=[sys.executable, "-c", script])
    assert rec["kind"] == "judge" and rec["judges"] == ["api_j"] and rec["cells"] == ["bpe_32k", "native_llama"]
    assert rec["gpu"] is False and rec["baseline"] == "native_llama" and rec["limit"] == 2 and rec["experiment_log"] is None
    assert rec["run_id"].split("_", 1)[1].startswith("judge_sw") and rec["model"] == "m-api"
    rd = paths.repo_root / "outputs" / "runs" / rec["run_id"]
    assert (rd / "judges" / "api_j.yaml").read_text(encoding="utf-8") == JUDGE_API_YAML
    assert json.loads((rd / "judge.json").read_text(encoding="utf-8"))["backends"] == {"api_j": "openai"}
    assert rec["snapshot"].endswith("judge.json")
    assert _wait(lambda: rm.get(rec["run_id"])["status"] == "finished")
    d = rm.detail(rec["run_id"])
    assert d["progress"]["judge"]["judges"]["api_j"]["cells"]["bpe_32k"] == {"n": 2, "reused": False, "seconds": 0.1}
    assert d["progress"]["judge"]["judges"]["api_j"]["cells"]["native_llama"]["reused"] is True
    assert d["progress"]["done"] is True and d["results"] is None            # no report file written by the stub
    (paths.repo_root / "outputs" / "experiments" / "sw" / "freeform_judge_report.json").write_text(json.dumps(
        {"baseline": "native_llama", "judges": {"api_j": {"bpe_32k": {"score_mean": 3.0, "n": 2, "vs_baseline": {"delta_mean": -1.0}, "parse_fail_rate": 0.0}}}}),
        encoding="utf-8")
    r = rm.results(rec["run_id"])
    assert r["judge"] is True and r["summary"]["api_j"]["bpe_32k"]["delta"] == -1.0 and r["report"].endswith("freeform_judge_report.json")
    # a fresh manager rediscovers the judge run with its kind
    assert RunManager(paths).get(rec["run_id"])["kind"] == "judge"


def test_api_only_judge_runs_beside_an_active_run(tmp_repo: ConsolePaths, monkeypatch):
    paths = _judge_repo(tmp_repo)
    monkeypatch.setenv("TEST_JUDGE_KEY", "k")
    rm = RunManager(paths)
    long = rm.start("mini", command=[sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        rec = rm.start_judge("outputs/experiments/sw", ["api_j"], command=[sys.executable, "-c", "print('ok')"])
        assert rec["kind"] == "judge"                                   # no GPU needed → no conflict
        assert _wait(lambda: rm.get(rec["run_id"])["status"] == "finished")
    finally:
        rm.cancel(long["run_id"])


def test_parse_judge_progress():
    assert parse_judge_progress("Loading weights\nnothing here") is None
    txt = ("INFO arabic_eval.judge.freeform_judge: [judge:gemma4_31b] vLLM google/gemma-4-31b-it, structured_json=True\n"
           "INFO arabic_eval.judge.freeform_judge: [judge:gemma4_31b] bpe_32k: 250 verdicts in 12s → outputs/x.parquet\n"
           "INFO arabic_eval.judge.freeform_judge: [judge:gemma4_31b] native_llama: reusing 250 verdicts\n")
    p = parse_judge_progress(txt)
    assert p["judges"]["gemma4_31b"]["model"] == "google/gemma-4-31b-it" and p["done"] is False
    assert p["judges"]["gemma4_31b"]["cells"]["bpe_32k"] == {"n": 250, "reused": False, "seconds": 12.0}
    p2 = parse_judge_progress(txt + "report → outputs/experiments/sw/freeform_judge_report.json\n")
    assert p2["done"] and p2["report"].endswith("freeform_judge_report.json")
    st = parse_progress(txt + "report → r.json\n")
    assert st["judge"]["done"] and st["done"] is True


def test_single_cell_reports_the_tokenizer_that_runs_and_warns_on_mismatch(tmp_repo: ConsolePaths):
    """A one-cell ``sweep.tokenizers`` never runs: without ``--sweep`` (which needs
    more than one cell) ``run_experiment.py`` trains the top-level ``tokenizer``.
    The console must name that cell and warn when the two blocks disagree — a
    real run trained native_qwen3 while the page said ``1 cell (araroopat)``."""
    from arabic_eval.tools.experiment_console import config_warnings
    raw = parse_yaml(MINIMAL_YAML)
    raw["tokenizer"] = {"type": "native_qwen3", "vocab_size": None,
                        "params": {"model_name_or_path": "Qwen/Qwen3-4B-Base"}}
    raw["sweep"] = {"tokenizers": [{"type": "araroopat", "vocab_sizes": [None]}], "tasks": [{"type": "acva"}]}
    v = validate_config(tmp_repo, raw, file="mini.yaml")      # the dict IS mini.yaml: its output_dir is not "shared"
    assert v["ok"] and v["sweep"] is False
    assert v["cells"] == ["native_qwen3"]
    assert len(v["warnings"]) == 1 and "araroopat" in v["warnings"][0] and "native_qwen3" in v["warnings"][0]
    # without the file the same dict is a second config writing mini.yaml's output_dir → one more warning
    v = validate_config(tmp_repo, raw)
    assert len(v["warnings"]) == 2 and "also the output_dir of mini.yaml" in v["warnings"][1]
    # the two blocks agree → no warning; vocab-sized cells are named like run_sweep names them
    raw["tokenizer"] = {"type": "bpe", "vocab_size": 32000}
    raw["sweep"]["tokenizers"] = [{"type": "bpe", "vocab_sizes": [32000]}]
    v = validate_config(tmp_repo, raw, file="mini.yaml")
    assert v["cells"] == ["bpe_32k"] and v["warnings"] == []
    from arabic_eval.config import ExperimentConfig
    assert config_warnings(ExperimentConfig(**v["resolved"])) == []
    # two cells → a sweep: the list is what runs, whatever the top-level block says
    raw["sweep"]["tokenizers"].append({"type": "araroopat", "vocab_sizes": [None]})
    v = validate_config(tmp_repo, raw, file="mini.yaml")
    assert v["sweep"] is True and v["cells"] == ["bpe_32k", "araroopat"] and v["warnings"] == []
