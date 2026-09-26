"""Sandboxed Python kernels for the console's Analysis tab.

``AnalysisKernel`` owns one ``analysis_worker`` process per analysis session:
it starts it, sends it code, enforces a wall-clock limit (SIGINT, then SIGKILL
after a grace period) and a memory limit (RSS of the process group), and
restarts it on the next step when it had to be killed (the variables are lost
and the result says so).

**Sandbox** (``sandbox_status()`` probes it once): the worker runs under
``unshare --user --map-root-user --mount --net --pid --fork --mount-proc`` with
the whole filesystem remounted read-only except the session folder, a private
tmpfs on /tmp and /dev/shm, no network (fresh network namespace), and its own
PID namespace (it cannot see or signal any other process). No root is needed:
unprivileged user namespaces. Measured on this machine 2026-09-25: writes to the
repo / home fail, the session folder is writable, the experiments' Parquet
files are readable, localhost is unreachable. The environment is rebuilt from
scratch — no API key or token of the console server reaches the worker.
When the probe fails, ``sandbox="auto"`` runs the worker unsandboxed and
reports it (the agent layer then asks before every step).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("analysis.kernel")

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Mount script run inside the new namespaces: keep the session folder writable,
#: make everything else read-only, give the process its own /tmp and /dev/shm.
_SANDBOX_SH = (
    'set -e; S="$1"; shift; '
    'mount --bind "$S" "$S"; mount --rbind / /; mount -o remount,bind,ro /; cd /; '
    'mount -t tmpfs -o size=2g,mode=1777 tmpfs /tmp; mount -t tmpfs -o size=1g,mode=1777 tmpfs /dev/shm; '
    'cd "$S"; exec "$@"'
)
_UNSHARE = ["--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "--mount-proc", "--kill-child"]


class KernelError(RuntimeError):
    pass


@dataclass
class KernelLimits:
    timeout_sec: float = 120.0          # per step; the agent may ask for more up to max_timeout_sec
    max_timeout_sec: float = 600.0
    interrupt_grace_sec: float = 5.0    # after SIGINT, before SIGKILL
    mem_limit_mb: int = 16384           # RSS of the worker's process group
    start_timeout_sec: float = 90.0
    capture_chars: int = 100_000


_SANDBOX: Optional[Dict[str, Any]] = None
_SANDBOX_LOCK = threading.Lock()


def sandbox_status(force: bool = False) -> Dict[str, Any]:
    """Whether the namespace sandbox works here: ``{"available", "reason"}`` (probed once)."""
    global _SANDBOX
    with _SANDBOX_LOCK:
        if _SANDBOX is not None and not force:
            return _SANDBOX
        unshare = shutil.which("unshare")
        if not unshare or not sys.platform.startswith("linux"):
            _SANDBOX = {"available": False, "reason": "unshare (util-linux) not found"}
            return _SANDBOX
        probe_dir = REPO_ROOT / "outputs" / "analysis" / ".sandbox_probe"
        try:
            probe_dir.mkdir(parents=True, exist_ok=True)
            outside = REPO_ROOT / "outputs" / "analysis" / ".sandbox_outside"   # gitignored, if the probe fails
            test = ('touch "$PWD/.w" && ! touch "' + str(outside) + '" 2>/dev/null && touch /tmp/.w && echo OK')
            r = subprocess.run([unshare, *_UNSHARE, "sh", "-c", _SANDBOX_SH, "sh", str(probe_dir), "sh", "-c", test],
                               capture_output=True, text=True, timeout=20)
            ok = r.returncode == 0 and r.stdout.strip().endswith("OK")
            _SANDBOX = {"available": ok, "reason": "" if ok else (r.stderr.strip() or r.stdout.strip() or f"exit {r.returncode}")[:300]}
        except (OSError, subprocess.SubprocessError) as e:
            _SANDBOX = {"available": False, "reason": f"{type(e).__name__}: {e}"}
        return _SANDBOX


def _worker_python() -> str:
    venv = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _group_rss_mb(pgid: int) -> float:
    """Resident memory (MB) of every process in a process group."""
    total = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                stat = fh.read().decode("ascii", "replace")
            fields = stat[stat.rfind(")") + 2:].split()
            if int(fields[2]) != pgid:           # field 5 (pgrp) of /proc/<pid>/stat
                continue
            with open(f"/proc/{pid}/statm") as fh:
                total += int(fh.read().split()[1])
        except (OSError, ValueError, IndexError):
            continue
    return total * os.sysconf("SC_PAGE_SIZE") / 1e6


class AnalysisKernel:
    """One worker process for one session folder. Thread-safe: one step at a time."""

    def __init__(self, session_dir: Path | str, *, sandbox: str = "auto", limits: Optional[KernelLimits] = None,
                 python: Optional[str] = None, experiments_root: Optional[Path | str] = None) -> None:
        if sandbox not in ("auto", "on", "off"):
            raise ValueError("sandbox must be auto | on | off")
        self.session_dir = Path(session_dir).resolve()
        self.artifacts_dir = self.session_dir / "artifacts"
        self.limits = limits or KernelLimits()
        self.python = python or _worker_python()
        self.experiments_root = Path(experiments_root).resolve() if experiments_root else None
        self._sandbox_pref = sandbox
        self.sandboxed = False
        self.proc: Optional[subprocess.Popen] = None
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 0
        self._stop = threading.Event()
        self.restarts = 0
        self.started_at: Optional[float] = None
        self.last_used = time.time()
        self.lost_state = False                  # set when a restart discarded the variables

    # ---- lifecycle ------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _env(self) -> Dict[str, str]:
        venv_bin = str(Path(self.python).parent)
        inner_tmp = "/tmp" if self.sandboxed else str(self.session_dir / ".tmp")
        env = {
            "PATH": f"{venv_bin}:/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "HOME": f"{inner_tmp}/home", "TMPDIR": inner_tmp, "MPLCONFIGDIR": f"{inner_tmp}/mpl",
            "XDG_CACHE_HOME": f"{inner_tmp}/cache", "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "NUMEXPR_MAX_THREADS": "8",
            "ARABIC_EVAL_REPO": str(REPO_ROOT),
        }
        if self.experiments_root is not None:
            env["ARABIC_EVAL_EXPERIMENTS"] = str(self.experiments_root)
        return env

    def start(self) -> Dict[str, Any]:
        if self.alive:
            return {"ok": True, "sandboxed": self.sandboxed}
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if self._sandbox_pref == "off":
            self.sandboxed = False
        else:
            st = sandbox_status()
            if not st["available"] and self._sandbox_pref == "on":
                raise KernelError(f"sandbox unavailable: {st['reason']}")
            self.sandboxed = bool(st["available"])
        if self.sandboxed and any(self.session_dir == Path(m) or Path(m) in self.session_dir.parents
                                  for m in ("/tmp", "/dev/shm")):
            raise KernelError(f"a sandboxed session folder cannot live under /tmp or /dev/shm (the sandbox mounts a "
                              f"private tmpfs there, which would hide it): {self.session_dir}")
        if not self.sandboxed:
            (self.session_dir / ".tmp").mkdir(exist_ok=True)
        worker = [self.python, "-m", "arabic_eval.tools.analysis_worker", "--out", str(self.artifacts_dir),
                  "--capture-chars", str(self.limits.capture_chars)]
        cmd = ([shutil.which("unshare") or "unshare", *_UNSHARE, "sh", "-c", _SANDBOX_SH, "sh", str(self.session_dir), *worker]
               if self.sandboxed else worker)
        self._q = queue.Queue()
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
            cwd=str(self.session_dir), env=self._env(), start_new_session=True,
            # The worker handles SIGINT itself; unshare / sh must ignore it (inherited through exec,
            # overridden by the worker's own handler) so a group SIGINT only interrupts the step.
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
        )
        # Each reader is bound to its own process and queue: a killed worker's late EOF must
        # never land in the queue of the worker that replaces it.
        threading.Thread(target=self._reader, args=(self.proc, self._q), daemon=True).start()
        threading.Thread(target=self._stderr_reader, args=(self.proc,), daemon=True).start()
        msg = self._wait_for(lambda m: m.get("op") in ("ready", "eof"), self.limits.start_timeout_sec)
        if msg is None or msg.get("op") != "ready" or not msg.get("ok"):
            err = (msg or {}).get("error") or self._stderr_tail()
            self.kill()
            raise KernelError(f"analysis worker failed to start: {err}")
        self.started_at = time.time()
        log.info("analysis kernel up: %s (sandboxed=%s, pid=%s)", self.session_dir.name, self.sandboxed, self.proc.pid)
        return {"ok": True, "sandboxed": self.sandboxed, "preloaded": msg.get("preloaded")}

    @staticmethod
    def _reader(proc: subprocess.Popen, q: "queue.Queue[Dict[str, Any]]") -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                q.put(json.loads(line))
            except ValueError:
                log.warning("analysis worker: non-JSON line %r", line[:200])
        q.put({"op": "eof"})

    _stderr_lines: List[str]

    def _stderr_reader(self, proc: subprocess.Popen) -> None:
        self._stderr_lines = []
        assert proc.stderr is not None
        for line in proc.stderr:
            self._stderr_lines.append(line.rstrip())
            del self._stderr_lines[:-40]

    def _stderr_tail(self) -> str:
        return "\n".join(getattr(self, "_stderr_lines", [])[-15:])

    def _wait_for(self, pred, timeout: float) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                msg = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            if pred(msg):
                return msg
        return None

    def kill(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            pass
        self.proc = None

    def shutdown(self) -> None:
        if self.alive:
            try:
                assert self.proc is not None and self.proc.stdin is not None
                self.proc.stdin.write(json.dumps({"id": -1, "op": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired, AssertionError):
                pass
        self.kill()

    def interrupt(self) -> bool:
        """Ask the running step to stop (the page's stop button). False when nothing runs."""
        if not self._lock.locked():
            return False
        self._stop.set()
        return True

    # ---- requests -------------------------------------------------------
    def _send(self, op: str, **payload: Any) -> int:
        assert self.proc is not None and self.proc.stdin is not None
        self._next_id += 1
        self.proc.stdin.write(json.dumps({"id": self._next_id, "op": op, **payload}, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return self._next_id

    def vars(self) -> List[Dict[str, Any]]:
        """The user variables alive in the worker (name, type, shape / len, columns)."""
        with self._lock:
            if not self.alive:
                return []
            rid = self._send("vars")
            msg = self._wait_for(lambda m: m.get("id") == rid or m.get("op") == "eof", 10)
            return (msg or {}).get("vars") or []

    def execute(self, code: str, *, step: Any = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Run one code step and return the worker's result, never raising for the code's own failures.

        Extra keys: ``restarted`` (the worker was (re)started for this step after a kill — the
        variables of earlier steps are gone), ``killed`` (this step was killed: time or memory),
        ``sandboxed``."""
        timeout = min(float(timeout or self.limits.timeout_sec), self.limits.max_timeout_sec)
        with self._lock:
            self._stop.clear()
            restarted = False
            if not self.alive:
                restarted = self.started_at is not None
                self.start()
                if restarted:
                    self.restarts += 1
            self.last_used = time.time()
            rid = self._send("exec", code=code, step=step if step is not None else self._next_id + 1)
            t0 = time.monotonic()
            interrupted_at: Optional[float] = None
            why = None
            next_mem = t0
            while True:
                try:
                    msg = self._q.get(timeout=0.1)
                except queue.Empty:
                    msg = None
                if msg is not None:
                    if msg.get("id") == rid:
                        res = {k: v for k, v in msg.items() if k not in ("id", "op")}
                        if why and not res.get("ok"):
                            res["error"] = {"ename": "Interrupted",
                                            "evalue": f"the step was interrupted: {why}", "traceback": ""}
                        res.update(restarted=restarted, killed=False, sandboxed=self.sandboxed)
                        # A fast allocation can finish between two polls: check once more, and do not
                        # keep a worker that holds more than the limit.
                        rss = _group_rss_mb(self.proc.pid) if self.proc is not None else 0.0
                        if rss > self.limits.mem_limit_mb:
                            self.kill()
                            self.lost_state = True
                            res.update(ok=False, killed=True, error={
                                "ename": "Killed", "traceback": "",
                                "evalue": f"memory limit: the step finished holding {rss:.0f} MB > "
                                          f"{self.limits.mem_limit_mb} MB — load fewer columns / rows; the "
                                          "variables are lost, the next step starts a fresh worker"})
                        self.last_used = time.time()
                        return res
                    if msg.get("op") == "eof":
                        why = why or f"the worker exited ({self._stderr_tail()[-400:] or 'no message'})"
                        self.kill()
                        return self._killed_result(why, restarted, t0)
                    continue
                now = time.monotonic()
                if now >= next_mem and self.proc is not None:
                    next_mem = now + 0.2
                    rss = _group_rss_mb(self.proc.pid)
                    if rss > self.limits.mem_limit_mb:
                        self.kill()
                        return self._killed_result(f"memory limit: {rss:.0f} MB > {self.limits.mem_limit_mb} MB — "
                                                   "load fewer columns / rows", restarted, t0)
                if interrupted_at is None and (self._stop.is_set() or now - t0 > timeout):
                    why = "stopped by the user" if self._stop.is_set() else f"time limit of {timeout:.0f} s"
                    interrupted_at = now
                    try:
                        os.killpg(self.proc.pid, signal.SIGINT)   # type: ignore[union-attr]
                    except (ProcessLookupError, AttributeError):
                        pass
                elif interrupted_at is not None and now - interrupted_at > self.limits.interrupt_grace_sec:
                    self.kill()
                    return self._killed_result(f"{why}; the step ignored the interrupt and the worker was killed",
                                               restarted, t0)

    def _killed_result(self, why: str, restarted: bool, t0: float) -> Dict[str, Any]:
        self.lost_state = True
        return {"ok": False, "stdout": "", "stderr": "", "result": None, "displays": [],
                "error": {"ename": "Killed", "evalue": why + " — the variables of earlier steps are lost; "
                          "the next step starts a fresh worker", "traceback": ""},
                "elapsed": round(time.monotonic() - t0, 3), "restarted": restarted, "killed": True,
                "sandboxed": self.sandboxed}


class KernelPool:
    """Kernels by session id, with idle reaping (the console holds one pool)."""

    def __init__(self, *, max_kernels: int = 4, idle_sec: float = 1800.0, **kernel_kwargs: Any) -> None:
        self.max_kernels = max_kernels
        self.idle_sec = idle_sec
        self.kernel_kwargs = kernel_kwargs
        self._kernels: Dict[str, AnalysisKernel] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str, session_dir: Path) -> AnalysisKernel:
        with self._lock:
            self._reap_locked()
            k = self._kernels.get(session_id)
            if k is None:
                if len(self._kernels) >= self.max_kernels:
                    oldest = min(self._kernels, key=lambda s: self._kernels[s].last_used)
                    self._kernels.pop(oldest).shutdown()
                k = AnalysisKernel(session_dir, **self.kernel_kwargs)
                self._kernels[session_id] = k
            return k

    def peek(self, session_id: str) -> Optional[AnalysisKernel]:
        return self._kernels.get(session_id)

    def close(self, session_id: str) -> None:
        with self._lock:
            k = self._kernels.pop(session_id, None)
        if k is not None:
            k.shutdown()

    def _reap_locked(self) -> None:
        now = time.time()
        for sid in [s for s, k in self._kernels.items() if now - k.last_used > self.idle_sec and not k._lock.locked()]:
            self._kernels.pop(sid).shutdown()

    def reap(self) -> None:
        with self._lock:
            self._reap_locked()

    def shutdown_all(self) -> None:
        with self._lock:
            ks = list(self._kernels.values())
            self._kernels.clear()
        for k in ks:
            k.shutdown()
