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

REPO = Path(__file__).resolve().parents[1]
REAL = ConsolePaths(REPO)

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
