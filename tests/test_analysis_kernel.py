"""The Analysis tab's code runner: a persistent worker per session, time / memory / stop
limits, and the namespace sandbox (read-only filesystem, no network, own PID namespace,
no secrets in the environment). Sandboxed session folders cannot live under /tmp (the
sandbox mounts a private tmpfs there), so these tests work under outputs/analysis/_pytest."""
from __future__ import annotations

import json
import os
import shutil
import socket
import threading
import time
import uuid
from pathlib import Path

import pytest

from arabic_eval.tools.analysis_kernel import AnalysisKernel, KernelError, KernelLimits, KernelPool, sandbox_status

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "outputs" / "analysis" / "_pytest"
SANDBOX = sandbox_status()["available"]
needs_sandbox = pytest.mark.skipif(not SANDBOX, reason=f"namespace sandbox unavailable: {sandbox_status()['reason']}")


@pytest.fixture()
def session_dir():
    d = BASE / uuid.uuid4().hex[:10]
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(params=["off"] + (["on"] if SANDBOX else []))
def kernel(request, session_dir):
    k = AnalysisKernel(session_dir, sandbox=request.param,
                       limits=KernelLimits(timeout_sec=3, interrupt_grace_sec=2, mem_limit_mb=1500))
    yield k
    k.shutdown()


def test_state_persists_and_results_are_captured(kernel):
    r = kernel.execute("x = 20\nprint('hello', x)")
    assert r["ok"] and r["stdout"] == "hello 20\n" and r["result"] is None
    r = kernel.execute("x * 2 + 1")
    assert r["ok"] and r["result"] == "41"
    assert r["sandboxed"] is (kernel._sandbox_pref == "on")
    assert {"name": "x", "type": "int"} in kernel.vars()


def test_errors_show_the_step_not_the_worker(kernel):
    r = kernel.execute("def f(d):\n    return d['missing']\nf({})", step=7)
    assert not r["ok"] and r["error"]["ename"] == "KeyError"
    tb = r["error"]["traceback"]
    assert '"<step 7>", line 3' in tb and "analysis_worker" not in tb
    r = kernel.execute("def broken(:\n  pass")
    assert r["error"]["ename"] == "SyntaxError"
    r = kernel.execute("input('?')")
    assert "not available" in r["error"]["evalue"]
    r = kernel.execute("raise SystemExit(3)")
    assert r["error"]["ename"] == "SystemExit" and kernel.alive      # never fatal


def test_displays_tables_figures_and_open_matplotlib(kernel):
    r = kernel.execute(
        "import plotly.express as px\n"
        "ae.show(px.line(x=[1, 2, 3], y=[3, 1, 2]), title='curve')\n"
        "plt.plot([1, 2]); plt.title('left open')\n"
        "pd.DataFrame({'a': range(5)})")
    assert r["ok"], r["error"]
    kinds = [d["kind"] for d in r["displays"]]
    assert kinds == ["plotly", "image", "table"]                     # the trailing DataFrame is displayed
    for d in r["displays"]:
        assert (kernel.artifacts_dir / d["path"]).exists()
    assert r["displays"][2]["n_rows"] == 5


def test_output_is_capped(session_dir):
    k = AnalysisKernel(session_dir, sandbox="off", limits=KernelLimits(capture_chars=1000))
    try:
        r = k.execute("print('x' * 50000)")
        assert len(r["stdout"]) < 1200 and "more characters were printed" in r["stdout"]
    finally:
        k.shutdown()


def test_time_limit_interrupts_then_kills(kernel):
    kernel.execute("keep = 1")
    r = kernel.execute("while True:\n    pass")
    assert r["error"]["ename"] == "Interrupted" and "time limit" in r["error"]["evalue"] and not r["killed"]
    assert kernel.execute("keep")["result"] == "1"                   # interrupted, state intact
    r = kernel.execute("import time\ntry:\n    time.sleep(60)\nexcept KeyboardInterrupt:\n    time.sleep(60)")
    assert r["killed"] and r["error"]["ename"] == "Killed"
    r = kernel.execute("'keep' in globals()")
    assert r["restarted"] and r["result"] == "False"                 # a fresh worker, and it says so


def test_stop_button_interrupts(kernel):
    threading.Timer(0.8, kernel.interrupt).start()
    r = kernel.execute("import time; time.sleep(30)", timeout=60)
    assert r["error"]["ename"] == "Interrupted" and "stopped by the user" in r["error"]["evalue"]
    assert r["elapsed"] < 5
    assert kernel.interrupt() is False                               # nothing running


def test_memory_limit_kills_the_worker(kernel):
    r = kernel.execute("import numpy as np\nblocks = [np.ones(10**8) for _ in range(4)]", timeout=30)
    assert r["killed"] and "memory limit" in r["error"]["evalue"]
    assert kernel.execute("1 + 1")["result"] == "2"


def test_worker_sees_the_experiments_root_it_is_given(session_dir):
    exp = session_dir / "fixture_experiments" / "exp" / "cell1"
    exp.mkdir(parents=True)
    (exp / "all_metrics.json").write_text(json.dumps({"config": {"tokenizer": "bpe"}, "downstream": {}}))
    k = AnalysisKernel(session_dir / "s", sandbox="auto", experiments_root=session_dir / "fixture_experiments")
    try:
        r = k.execute("list(ae.catalog().index)")
        assert r["ok"], r["error"]
        assert r["result"] == "['exp/cell1']"
    finally:
        k.shutdown()


# --------------------------------------------------------------------------
# the sandbox
# --------------------------------------------------------------------------

@needs_sandbox
def test_sandbox_filesystem_is_read_only_except_the_session(session_dir, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
    monkeypatch.setenv("HF_TOKEN", "hf-must-not-leak")
    k = AnalysisKernel(session_dir, sandbox="on")
    try:
        target = REPO / "outputs" / f"_sandbox_escape_{uuid.uuid4().hex[:6]}.txt"
        r = k.execute(f"open({str(target)!r}, 'w').write('x')")
        assert r["error"]["ename"] == "OSError" and "Read-only" in r["error"]["evalue"]
        assert not target.exists()
        r = k.execute(f"open({str(Path.home() / '.sandbox_escape')!r}, 'w')")
        assert "Read-only" in r["error"]["evalue"]
        r = k.execute("open('mine.txt', 'w').write('ok'); open('/tmp/scratch', 'w').write('ok'); 'written'")
        assert r["ok"] and (session_dir / "mine.txt").read_text() == "ok"
        r = k.execute("import os; [os.environ.get('OPENAI_API_KEY'), os.environ.get('HF_TOKEN'), os.getpid(), "
                      "len([p for p in os.listdir('/proc') if p.isdigit()])]")
        assert r["result"] == "[None, None, 1, 1]"                     # no secrets, alone in its PID namespace
    finally:
        k.shutdown()


@needs_sandbox
def test_sandbox_has_no_network(session_dir):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    k = AnalysisKernel(session_dir, sandbox="on")
    try:
        r = k.execute(f"import socket\ns = socket.socket(); s.settimeout(2)\ns.connect(('127.0.0.1', {port}))")
        assert not r["ok"] and r["error"]["ename"] in ("ConnectionRefusedError", "OSError", "TimeoutError")
    finally:
        k.shutdown()
        srv.close()


@needs_sandbox
def test_sandboxed_session_under_tmp_is_refused(tmp_path):
    k = AnalysisKernel(tmp_path / "s", sandbox="on")
    with pytest.raises(KernelError, match="cannot live under /tmp"):
        k.start()


def test_pool_evicts_the_least_recently_used(session_dir):
    pool = KernelPool(max_kernels=2, sandbox="off")
    try:
        a = pool.get("a", session_dir / "a"); a.execute("1")
        time.sleep(0.01)
        b = pool.get("b", session_dir / "b"); b.execute("1")
        a.execute("2")                                               # b is now the oldest
        pool.get("c", session_dir / "c")
        assert pool.peek("b") is None and pool.peek("a") is a and not b.alive
        pool.idle_sec = 0
        pool.reap()
        assert pool.peek("a") is None and not a.alive
    finally:
        pool.shutdown_all()
