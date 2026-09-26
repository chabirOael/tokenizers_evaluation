"""The Analysis tab's local model server (local_llm_service): lifecycle, GPU conflicts in both
directions, idle stop, rediscovery after a console restart, and a whole question answered
through it — against tests/data/fake_openai_server.py standing in for `vllm serve`."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from arabic_eval.tools.analysis_agent import AnalysisEndpoint, AnalysisService
from arabic_eval.tools.analysis_kernel import KernelPool
from arabic_eval.tools.experiment_console import ConsolePaths, RunConflict, RunManager, group_alive
from arabic_eval.tools.local_llm_service import LocalLLMService, LocalServerConflict, LocalServerError

REPO = Path(__file__).resolve().parents[1]
FAKE = REPO / "tests" / "data" / "fake_openai_server.py"


def _port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _ep(port: int, *extra: str, idle: float = 30, name: str = "fake_local") -> AnalysisEndpoint:
    return AnalysisEndpoint(name=name, model="fake-model", kind="local_vllm", base_url=f"http://127.0.0.1:{port}/v1",
                            api_key_env=None, temperature=0.0,
                            serve={"command": ["{python}", str(FAKE), "--port", "{port}", *extra], "port": port,
                                   "idle_stop_min": idle})


def _wait(svc, ep, states, timeout=30.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        st = svc.status(ep)
        if st["state"] in states:
            return st
        time.sleep(0.2)
    raise AssertionError(f"state stayed {svc.status(ep)['state']!r}, wanted {states}")


@pytest.fixture()
def svc(tmp_path):
    s = LocalLLMService(REPO, state_dir=tmp_path / "servers", gpu=lambda: [], stop_grace_sec=5, watchdog_sec=0.5)
    yield s
    for rec in s.running():                     # never leave a stand-in server behind
        try:
            os.killpg(int(rec["pgid"]), 9)
        except ProcessLookupError:
            pass


def test_start_ready_stop(svc):
    ep = _ep(_port(), "--delay", "1.5")
    assert svc.status(ep)["state"] == "stopped" and svc.gpu_guard() is None
    st = svc.start(ep)
    assert st["state"] == "starting" and "starting" in st["detail"]
    st = _wait(svc, ep, {"ready"})
    assert st["ready"] and st["ready_at"] and "idle" in st["detail"]
    assert "holds the GPU" in svc.gpu_guard()
    assert svc.start(ep)["state"] == "ready"                              # idempotent
    pgid = svc.running()[0]["pgid"]
    svc.stop(ep)
    st = _wait(svc, ep, {"stopped"})
    assert "stopped from the tab" in st["detail"] and not group_alive(pgid) and svc.gpu_guard() is None


def test_a_crashing_server_is_failed_with_its_log(svc):
    ep = _ep(_port(), "--delay", "0.2", "--fail")
    svc.start(ep)
    st = _wait(svc, ep, {"failed"})
    assert st["exit_code"] == 3 and "CUDA out of memory" in st["detail"]
    assert svc.status(ep)["state"] == "failed"                            # stable, not re-recorded
    assert any("fake weights" in l for l in svc.log_tail(ep)["lines"])
    assert svc.start(ep)["state"] == "starting"                           # a failed server can be restarted
    _wait(svc, ep, {"failed"})


def test_gpu_conflicts_block_a_start_unless_forced(tmp_path):
    busy = LocalLLMService(REPO, state_dir=tmp_path / "s1", gpu=lambda: [{"memory_used_mib": 30000, "memory_total_mib": 81559}])
    ep = _ep(_port())
    with pytest.raises(LocalServerConflict, match="29.3 of 80 GiB"):
        busy.start(ep)
    try:
        assert busy.start(ep, force=True)["state"] == "starting"
    finally:
        busy.stop(ep)
    runs = LocalLLMService(REPO, state_dir=tmp_path / "s2", gpu=lambda: [],
                           active_runs=lambda: [{"run_id": "r1", "status": "running", "kind": "experiment"}])
    assert "console run r1" in runs.gpu_conflict()
    api_judge = LocalLLMService(REPO, state_dir=tmp_path / "s3", gpu=lambda: [],
                                active_runs=lambda: [{"run_id": "j1", "status": "running", "kind": "judge", "gpu": False}])
    assert api_judge.gpu_conflict() is None                              # an API judge does not use the GPU


def test_runs_refuse_to_start_while_the_server_holds_the_gpu(svc, tmp_path):
    ep = _ep(_port())
    svc.start(ep)
    _wait(svc, ep, {"ready"})
    rm = RunManager(ConsolePaths(tmp_path / "repo"))
    rm.gpu_guards.append(svc.gpu_guard)
    with pytest.raises(RunConflict, match="local model fake_local"):
        rm._launch("x", ["true"], {"kind": "experiment"}, {}, snapshot_main="x.json", force=False)
    assert not (tmp_path / "repo" / "outputs" / "runs").exists()          # refused before anything was written
    svc.stop(ep)
    _wait(svc, ep, {"stopped"})


def test_idle_servers_are_stopped(svc):
    ep = _ep(_port(), idle=0.001)
    svc.start(ep)
    _wait(svc, ep, {"ready"})
    time.sleep(0.2)
    assert svc.reap_idle() == ["fake_local"]
    st = _wait(svc, ep, {"stopped"})
    assert "idle for" in st["detail"]


def test_touch_defers_the_idle_stop(svc):
    ep = _ep(_port(), idle=0.02)                                          # 1.2 s
    svc.start(ep)
    _wait(svc, ep, {"ready"})
    for _ in range(4):
        time.sleep(0.5)
        svc.touch(ep)
        assert svc.reap_idle() == []
    svc.stop(ep)


def test_a_restarted_console_rediscovers_the_server(svc, tmp_path):
    ep = _ep(_port())
    svc.start(ep)
    _wait(svc, ep, {"ready"})
    fresh = LocalLLMService(REPO, state_dir=tmp_path / "servers", gpu=lambda: [])
    fresh.adopt([ep])
    assert fresh.status(ep)["state"] == "ready" and "holds the GPU" in fresh.gpu_guard()
    fresh.stop(ep)
    _wait(fresh, ep, {"stopped"})


def test_a_foreign_server_on_the_port_is_not_adopted(svc):
    port = _port()
    p = subprocess.Popen([sys.executable, str(FAKE), "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ep = _ep(port)
        t0 = time.monotonic()
        while svc.status(ep)["state"] != "foreign" and time.monotonic() - t0 < 10:
            time.sleep(0.2)
        assert svc.status(ep)["state"] == "foreign"
        with pytest.raises(LocalServerError, match="not started by this console"):
            svc.start(ep)
    finally:
        p.kill()
        p.wait()


def test_the_vllm_command_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-reach-vllm")
    ep = AnalysisEndpoint.from_yaml(REPO / "configs" / "analysis" / "gemma4_31b_local.yaml")
    svc = LocalLLMService(REPO, state_dir=tmp_path)
    argv = svc.command(ep)
    assert argv[0].endswith(".venv-judge/bin/vllm") and argv[1:3] == ["serve", "google/gemma-4-31b-it"]
    joined = " ".join(argv)
    for flag in ("--port 8801", "--max-model-len 65536", "--kv-cache-dtype fp8", "--quantization fp8",
                 "--gpu-memory-utilization 0.92", "--served-model-name google/gemma-4-31b-it", "--host 127.0.0.1",
                 "--enable-auto-tool-choice", "--tool-call-parser gemma4", "--reasoning-parser gemma4"):
        assert flag in joined
    env = svc._env(ep)
    assert "OPENAI_API_KEY" not in env and env["VLLM_USE_FLASHINFER_SAMPLER"] == "0" and env["HF_HUB_OFFLINE"] == "1"
    assert ".local-pkgs/extracted/usr/include" in env["CPATH"] and env["PATH"].startswith(str(REPO / ".venv-judge" / "bin"))
    bare = LocalLLMService(tmp_path, state_dir=tmp_path / "s")
    assert "not found" in bare.readiness_problem(ep)
    assert bare.status(ep)["startable"] is False


def test_a_question_answered_through_the_local_server(svc):
    d = REPO / "outputs" / "analysis" / "_pytest" / ("local_" + uuid.uuid4().hex[:8])
    (d / "configs" / "analysis").mkdir(parents=True)
    port = _port()
    (d / "configs" / "analysis" / "fake_local.yaml").write_text(json.dumps({"analysis": {
        "name": "fake_local", "model": "fake-model", "kind": "local_vllm", "base_url": f"http://127.0.0.1:{port}/v1",
        "api_key_env": None, "temperature": 0.0, "max_tokens": 512, "context_tokens": 20000,
        "serve": {"command": ["{python}", str(FAKE), "--port", "{port}"], "port": port}}}))
    an = AnalysisService(d, pool=KernelPool(sandbox="off"), local_status=svc.status, activity_hook=svc.touch,
                         require_approval_unsandboxed=False)
    try:
        sid = an.new_session("fake_local")["id"]
        entry = next(e for e in an.endpoints()["endpoints"] if e["name"] == "fake_local")
        assert not entry["ready"] and entry["server"]["state"] == "stopped"
        ep = an.endpoint("fake_local")
        svc.start(ep)
        _wait(svc, ep, {"ready"})
        events = list(an.chat(sid, "what is six times seven?"))
        ex = next(e for e in events if e["type"] == "exec")
        ans = next(e for e in events if e["type"] == "answer")
        assert ex["ok"] and ex["stdout"] == "42\n" and ans["text"] == "The answer is 42."
        assert ans["provenance"]["unverified"] == []
        assert svc._read("fake_local")["last_activity"] is not None
    finally:
        an.pool.shutdown_all()
        svc.stop(an.endpoint("fake_local"))
        shutil.rmtree(d, ignore_errors=True)
