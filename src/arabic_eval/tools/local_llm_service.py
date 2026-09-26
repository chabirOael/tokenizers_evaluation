"""The local model behind a ``local_vllm`` Analysis endpoint: start, watch, stop.

``LocalLLMService`` runs ``vllm serve`` (from ``.venv-judge``, with the environment
``scripts/judge/run_judge.sh`` sets: the extracted Python headers on ``CPATH`` for
Triton, ``VLLM_USE_FLASHINFER_SAMPLER=0``) as a detached session leader — the same
launch as a console run: a bash wrapper records the exit code, the state lives in
``outputs/analysis/_servers/<endpoint>/server.json`` with the pid + ``/proc`` start
ticks, so a restarted console rediscovers a server that is still up.

States: ``stopped`` · ``starting`` (process alive, ``/v1/models`` not answering yet;
the last meaningful log line is the detail) · ``ready`` · ``stopping`` · ``failed``
(the process exited without a stop request; the log tail says why).

The GPU is shared with training / eval runs, so:
- ``start`` refuses while a console run that needs the GPU is active, or while the
  GPU already holds more than ``busy_mib`` of memory (another process — a peer
  session's run started from the shell counts too), unless ``force``;
- ``gpu_guard`` is registered with the console's ``RunManager``: a run refuses to
  start while this server is up (the same "run concurrently" override applies);
- a watchdog stops a server that answered no question for ``serve.idle_stop_min``
  minutes, so a forgotten tab does not hold the H100.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from arabic_eval.tools.experiment_console import _proc_start_ticks, gpu_snapshot, group_alive

log = logging.getLogger("analysis.local")

REPO_ROOT = Path(__file__).resolve().parents[3]

#: ``serve`` keys of an endpoint YAML and their defaults.
SERVE_DEFAULTS: Dict[str, Any] = {
    "venv": ".venv-judge", "port": 8801, "max_model_len": 24576, "kv_cache_dtype": "auto",
    "gpu_memory_utilization": 0.92, "dtype": "bfloat16", "quantization": None, "extra_args": [],
    "startup_timeout_sec": 900, "idle_stop_min": 30, "env": {}, "command": None,
}


class LocalServerError(ValueError):
    """A request about the local server that cannot be honoured."""


class LocalServerConflict(LocalServerError):
    """Starting would share the GPU with something else (the page may force it)."""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def serve_config(ep: Any) -> Dict[str, Any]:
    cfg = dict(SERVE_DEFAULTS)
    unknown = sorted(set(ep.serve or {}) - set(SERVE_DEFAULTS))
    if unknown:
        raise LocalServerError(f"{ep.name}: unknown serve keys {unknown}; known: {sorted(SERVE_DEFAULTS)}")
    cfg.update(ep.serve or {})
    return cfg


def _tail_lines(path: Path, n: int = 40, max_bytes: int = 96 * 1024) -> List[str]:
    """The last ``n`` lines of a log, progress-bar redraws (``\\r``) collapsed to their last state."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            fh.seek(max(0, size - max_bytes))
            data = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = []
    for raw in data.split("\n"):
        seg = raw.split("\r")
        last = next((s for s in reversed(seg) if s.strip()), "")
        if last.strip():
            lines.append(last.rstrip())
    return lines[-n:]


_NOISE = re.compile(r"^\s*$|Warning|warn\(|FutureWarning|DeprecationWarning", re.I)


def _detail(lines: List[str]) -> str:
    for line in reversed(lines):
        if not _NOISE.search(line):
            # vLLM: "(EngineCore pid=…) INFO 09-25 20:41:38 [model_runner.py:404] Model loading took …" → "INFO Model loading took …"
            s = re.sub(r"^(?:\([^)]*\)\s*)?(INFO|WARNING|ERROR|DEBUG)\s+\d\d-\d\d\s+[\d:]+\s+\[[^\]]*\]\s*", r"\1 ", line)
            return s[-220:]
    return ""


class LocalLLMService:
    """One state folder per endpoint; thread-safe; a background watchdog once something runs."""

    def __init__(self, repo_root: Path = REPO_ROOT, *, state_dir: Optional[Path] = None,
                 active_runs: Optional[Callable[[], List[dict]]] = None,
                 gpu: Callable[[], Optional[List[dict]]] = gpu_snapshot, busy_mib: int = 2048,
                 stop_grace_sec: float = 30.0, watchdog_sec: float = 30.0) -> None:
        self.repo_root = Path(repo_root)
        self.state_dir = Path(state_dir) if state_dir else self.repo_root / "outputs" / "analysis" / "_servers"
        self.active_runs = active_runs or (lambda: [])
        self.gpu = gpu
        self.busy_mib = busy_mib
        self.stop_grace_sec = stop_grace_sec
        self.watchdog_sec = watchdog_sec
        self._lock = threading.RLock()
        self._procs: Dict[str, subprocess.Popen] = {}
        self._watchdog: Optional[threading.Thread] = None
        self._eps: Dict[str, Any] = {}

    # ---- state files ------------------------------------------------------
    def _dir(self, name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name or ""):
            raise LocalServerError(f"bad endpoint name {name!r}")
        return self.state_dir / name

    def _read(self, name: str) -> Optional[Dict[str, Any]]:
        p = self._dir(name) / "server.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write(self, name: str, rec: Dict[str, Any]) -> None:
        d = self._dir(name)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "server.json.tmp"
        tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        os.replace(tmp, d / "server.json")

    # ---- the command --------------------------------------------------------
    def command(self, ep: Any) -> List[str]:
        s = serve_config(ep)
        if s["command"]:                         # a custom server (tests, llama.cpp, …)
            return [str(a).format(port=s["port"], model=ep.model, python=sys.executable,
                                  repo=os.fspath(self.repo_root)) for a in s["command"]]
        vllm = self.repo_root / s["venv"] / "bin" / "vllm"
        argv = [str(vllm), "serve", ep.model, "--host", "127.0.0.1", "--port", str(s["port"]),
                "--served-model-name", ep.model, "--max-model-len", str(s["max_model_len"]),
                "--gpu-memory-utilization", str(s["gpu_memory_utilization"]), "--dtype", str(s["dtype"])]
        if s["kv_cache_dtype"] and s["kv_cache_dtype"] != "auto":
            argv += ["--kv-cache-dtype", str(s["kv_cache_dtype"])]
        if s["quantization"]:
            argv += ["--quantization", str(s["quantization"])]
        return argv + [str(a) for a in s["extra_args"]]

    def _env(self, ep: Any) -> Dict[str, str]:
        s = serve_config(ep)
        env = {k: v for k, v in os.environ.items()
               if not re.search(r"(API_KEY|SECRET|PASSWORD)$", k)}          # the model server needs none of them
        if not s["command"]:
            hdr = self.repo_root / ".local-pkgs" / "extracted" / "usr" / "include"
            env["CPATH"] = f"{hdr}:{hdr / 'python3.10'}" + (f":{env['CPATH']}" if env.get("CPATH") else "")
            env["C_INCLUDE_PATH"] = env["CPATH"]
            env["PATH"] = f"{self.repo_root / s['venv'] / 'bin'}:{env.get('PATH', '')}"
            env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
            env.setdefault("HF_HUB_OFFLINE", "1")                           # the checkpoint is cached: no Hub calls at start
        env.update({str(k): str(v) for k, v in (s["env"] or {}).items()})
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def readiness_problem(self, ep: Any) -> Optional[str]:
        """Why this endpoint's server cannot be started on this machine (venv, headers), or None."""
        s = serve_config(ep)
        if s["command"]:
            return None
        if not (self.repo_root / s["venv"] / "bin" / "vllm").exists():
            return f"{s['venv']}/bin/vllm not found — set up the judge venv (scripts/judge/setup_judge_env.sh)"
        if not (self.repo_root / ".local-pkgs" / "extracted" / "usr" / "include" / "python3.10" / "Python.h").exists():
            return "Python headers not extracted under .local-pkgs/extracted — see scripts/judge/setup_judge_env.sh"
        return None

    # ---- probes -----------------------------------------------------------
    @staticmethod
    def _answers(port: int, timeout: float = 1.0) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as r:
                return r.status == 200
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return False

    def gpu_conflict(self) -> Optional[str]:
        """Why the GPU is not free for a model server: an active console run, or memory in use."""
        runs = [r for r in self.active_runs() if r.get("kind") != "judge" or r.get("gpu")]
        if runs:
            r = runs[0]
            return f"console run {r['run_id']} is {r.get('status', 'running')} and uses the GPU"
        snap = self.gpu() or []
        busy = [g for g in snap if g.get("memory_used_mib", 0) > self.busy_mib]
        if busy:
            g = busy[0]
            return (f"the GPU already holds {g['memory_used_mib'] / 1024:.1f} of {g['memory_total_mib'] / 1024:.0f} GiB "
                    "(another process — a run started from a shell or another session?)")
        return None

    # ---- lifecycle ----------------------------------------------------------
    def status(self, ep: Any) -> Dict[str, Any]:
        """The server's state, reconciled with the process table and a /v1/models probe."""
        with self._lock:
            self._eps[ep.name] = ep
            rec = self._read(ep.name)
            s = serve_config(ep)
            base = {"endpoint": ep.name, "model": ep.model, "port": s["port"], "idle_stop_min": s["idle_stop_min"]}
            problem = self.readiness_problem(ep)
            if not rec or rec.get("state") == "stopped":
                if self._answers(s["port"], 0.5):
                    return {**base, "state": "foreign", "ready": False,
                            "detail": f"port {s['port']} answers but was not started by this console — stop that server or change serve.port"}
                return {**base, "state": "stopped", "ready": False, "startable": problem is None,
                        "detail": problem or (f"stopped ({rec.get('stop_reason')})" if rec and rec.get("stop_reason") else "not running"),
                        "stopped_at": (rec or {}).get("stopped_at")}
            proc = self._procs.get(ep.name)
            if proc is not None:
                proc.poll()
            alive = group_alive(rec.get("pgid"))
            log_path = self._dir(ep.name) / "server.log"
            lines = _tail_lines(log_path, 30)
            out = {**base, "pid": rec.get("pid"), "started_at": rec.get("started_at"), "ready_at": rec.get("ready_at"),
                   "last_activity": rec.get("last_activity"), "argv": rec.get("argv"), "log": str(log_path.relative_to(self.repo_root))
                   if log_path.is_relative_to(self.repo_root) else str(log_path)}
            if rec.get("state") == "stopping":
                if alive:
                    return {**out, "state": "stopping", "ready": False, "detail": "stopping…"}
                rec.update(state="stopped", stopped_at=rec.get("stopped_at") or _now_iso())
                self._write(ep.name, rec)
                return {**base, "state": "stopped", "ready": False, "startable": problem is None,
                        "detail": f"stopped ({rec.get('stop_reason')})"}
            if not alive and rec.get("state") == "failed":
                return {**out, "state": "failed", "ready": False, "startable": problem is None, "exit_code": rec.get("exit_code"),
                        "detail": f"the server exited (code {rec.get('exit_code')}): {_detail(lines) or 'see its log'}",
                        "log_tail": lines[-15:]}
            if not alive:
                code = None
                ef = self._dir(ep.name) / "exit_code"
                if ef.exists():
                    try:
                        code = int(ef.read_text().strip())
                    except ValueError:
                        pass
                rec.update(state="failed", exit_code=code, stopped_at=rec.get("stopped_at") or _now_iso())
                self._write(ep.name, rec)
                return {**out, "state": "failed", "ready": False, "startable": problem is None, "exit_code": code,
                        "detail": f"the server exited (code {code}): {_detail(lines) or 'see its log'}", "log_tail": lines[-15:]}
            if self._answers(s["port"]):
                if not rec.get("ready_at"):
                    rec["ready_at"] = _now_iso()
                    rec["state"] = "ready"
                    self._write(ep.name, rec)
                    log.info("local model %s ready on :%s", ep.name, s["port"])
                idle = self._idle_min(rec)
                return {**out, "state": "ready", "ready": True, "ready_at": rec["ready_at"], "idle_min": round(idle, 1),
                        "detail": f"up on :{s['port']} · idle {idle:.0f} min (stops after {s['idle_stop_min']})"}
            started = datetime.fromisoformat(rec["started_at"])
            elapsed = (datetime.now().astimezone() - started).total_seconds()
            slow = elapsed > s["startup_timeout_sec"]
            return {**out, "state": "starting", "ready": False, "elapsed_sec": round(elapsed),
                    "detail": (f"{'still ' if slow else ''}starting · {elapsed / 60:.1f} min" + (f" — {_detail(lines)}" if lines else "")),
                    "slow": slow}

    def _idle_min(self, rec: Dict[str, Any]) -> float:
        ref = rec.get("last_activity") or rec.get("ready_at") or rec.get("started_at")
        try:
            return (datetime.now().astimezone() - datetime.fromisoformat(ref)).total_seconds() / 60.0
        except (TypeError, ValueError):
            return 0.0

    def start(self, ep: Any, *, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            st = self.status(ep)
            if st["state"] in ("starting", "ready", "stopping"):
                return st
            if st["state"] == "foreign":
                raise LocalServerError(st["detail"])
            problem = self.readiness_problem(ep)
            if problem:
                raise LocalServerError(problem)
            if not force:
                why = self.gpu_conflict()
                if why:
                    raise LocalServerConflict(f"{why} — the model server needs the whole GPU; start anyway?")
            d = self._dir(ep.name)
            d.mkdir(parents=True, exist_ok=True)
            for f in ("exit_code",):
                (d / f).unlink(missing_ok=True)
            argv = self.command(ep)
            log_path = d / "server.log"
            shell = (f"trap 'true' TERM; {' '.join(shlex.quote(a) for a in argv)} > {shlex.quote(str(log_path))} 2>&1; "
                     f"echo $? > {shlex.quote(str(d / 'exit_code'))}")
            proc = subprocess.Popen(["bash", "-c", shell], cwd=str(self.repo_root), env=self._env(ep),
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    start_new_session=True)
            self._procs[ep.name] = proc
            rec = {"endpoint": ep.name, "model": ep.model, "state": "starting", "pid": proc.pid, "pgid": proc.pid,
                   "start_ticks": _proc_start_ticks(proc.pid), "argv": argv, "port": serve_config(ep)["port"],
                   "started_at": _now_iso(), "ready_at": None, "last_activity": None, "stopped_at": None,
                   "stop_reason": None, "forced": bool(force)}
            self._write(ep.name, rec)
            log.info("local model %s starting (pid %s): %s", ep.name, proc.pid, " ".join(argv))
            self._ensure_watchdog()
            return self.status(ep)

    def stop(self, ep: Any, reason: str = "stopped from the tab") -> Dict[str, Any]:
        with self._lock:
            rec = self._read(ep.name)
            if not rec or rec.get("state") in ("stopped", "failed") or not group_alive(rec.get("pgid")):
                if rec and rec.get("state") not in ("stopped", "failed"):
                    rec.update(state="stopped", stopped_at=_now_iso(), stop_reason=reason)
                    self._write(ep.name, rec)
                return self.status(ep)
            rec.update(state="stopping", stop_reason=reason, stopped_at=_now_iso())
            self._write(ep.name, rec)
            pgid = int(rec["pgid"])
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            threading.Thread(target=self._kill_after_grace, args=(ep, pgid), daemon=True).start()
            log.info("local model %s stopping (%s)", ep.name, reason)
            return self.status(ep)

    def _kill_after_grace(self, ep: Any, pgid: int) -> None:
        deadline = time.monotonic() + self.stop_grace_sec
        while time.monotonic() < deadline:
            time.sleep(0.25)
            if not group_alive(pgid):
                break
        else:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc = self._procs.pop(ep.name, None)
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
        self.status(ep)

    def touch(self, ep: Any) -> None:
        """A question used the server: push its idle stop back."""
        with self._lock:
            rec = self._read(ep.name)
            if rec and rec.get("state") in ("starting", "ready"):
                rec["last_activity"] = _now_iso()
                self._write(ep.name, rec)

    def log_tail(self, ep: Any, n: int = 200) -> Dict[str, Any]:
        p = self._dir(ep.name) / "server.log"
        return {"endpoint": ep.name, "lines": _tail_lines(p, n), "exists": p.exists()}

    # ---- guard + watchdog -----------------------------------------------------
    def running(self) -> List[Dict[str, Any]]:
        out = []
        if not self.state_dir.is_dir():
            return out
        for p in sorted(self.state_dir.glob("*/server.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if rec.get("state") in ("starting", "ready", "stopping") and group_alive(rec.get("pgid")):
                out.append(rec)
        return out

    def gpu_guard(self) -> Optional[str]:
        """For RunManager: why a run should not start now (a model server holds the GPU)."""
        up = self.running()
        if up:
            r = up[0]
            return (f"the Analysis tab's local model {r['endpoint']} ({r['model']}) holds the GPU — stop it in the "
                    "Analysis tab first")
        return None

    def reap_idle(self) -> List[str]:
        stopped = []
        for rec in self.running():
            ep = self._eps.get(rec["endpoint"])
            if ep is None or rec.get("state") != "ready":
                continue
            limit = serve_config(ep)["idle_stop_min"]
            if limit and self._idle_min(rec) >= limit:
                self.stop(ep, reason=f"idle for {limit} min")
                stopped.append(ep.name)
        return stopped

    def _ensure_watchdog(self) -> None:
        if self._watchdog is not None and self._watchdog.is_alive():
            return

        def loop() -> None:
            while True:
                time.sleep(self.watchdog_sec)
                try:
                    for name, ep in list(self._eps.items()):
                        self.status(ep)          # records ready_at / failures even when no page polls
                    self.reap_idle()
                except Exception:  # noqa: BLE001 - the watchdog must survive a bad state file
                    log.exception("local model watchdog")
                if not self.running():
                    return

        self._watchdog = threading.Thread(target=loop, daemon=True, name="local-llm-watchdog")
        self._watchdog.start()

    def adopt(self, eps: List[Any]) -> None:
        """At console start: remember the endpoints and watch servers a previous console left running."""
        for ep in eps:
            self._eps[ep.name] = ep
        if self.running():
            self._ensure_watchdog()
