"""The Python process that runs model-written analysis code (console → Analysis tab).

One worker per analysis session, started by ``analysis_kernel.AnalysisKernel``
— normally inside a sandbox (read-only filesystem except the session folder,
private /tmp, no network, own PID namespace). Variables persist between steps,
like a notebook kernel. ``ae`` (``arabic_eval.analysis``), ``pd``, ``np``,
``px``, ``go`` and ``plt`` are preloaded.

Protocol: NDJSON on the *original* stdin / stdout (duplicated at start-up; fds
0 / 1 / 2 are then pointed at /dev/null and ``worker.log`` so nothing the user
code prints can corrupt the channel). Requests::

    {"id": 1, "op": "exec", "code": "...", "step": 3}
    {"id": 2, "op": "vars"}
    {"id": 3, "op": "shutdown"}

An ``exec`` answers ``{"id", "ok", "stdout", "stderr", "result", "displays",
"error", "elapsed"}``: ``result`` is the repr of a trailing expression (a
DataFrame / figure there is displayed instead, like Jupyter), ``displays`` the
objects handed to ``ae.show`` plus every matplotlib figure left open,
``error`` ``{ename, evalue, traceback}`` with the worker's own frames removed.
SIGINT interrupts a running step (time limit or stop button) and is ignored
between steps.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import io
import json
import linecache
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_IN_EXEC = False
PRELOADED = ("ae", "pd", "np", "px", "go", "plt", "json", "Path", "math", "re")


def _on_sigint(signum, frame):  # noqa: ARG001
    if _IN_EXEC:
        raise KeyboardInterrupt


class _Capture(io.TextIOBase):
    """A text sink that keeps the first ``limit`` characters and counts the rest."""

    def __init__(self, limit: int) -> None:
        self.parts: List[str] = []
        self.n = 0
        self.limit = limit
        self.dropped = 0

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            s = str(s)
        room = self.limit - self.n
        if room > 0:
            take = s[:room]
            self.parts.append(take)
            self.n += len(take)
            self.dropped += len(s) - len(take)
        else:
            self.dropped += len(s)
        return len(s)

    def getvalue(self) -> str:
        v = "".join(self.parts)
        if self.dropped:
            v += f"\n… [{self.dropped} more characters were printed and not kept — print less]"
        return v


def _no_input(*_a, **_k):
    raise RuntimeError("input() is not available in the analysis worker")


def _namespace(out_dir: Path) -> Dict[str, Any]:
    import math
    import re

    import numpy as np
    import pandas as pd

    import arabic_eval.analysis as ae
    from arabic_eval.analysis import display as _display

    sys.modules.setdefault("ae", ae)          # `import ae` works too (models write it although ae is preloaded)
    ns: Dict[str, Any] = {"__name__": "__analysis__", "__builtins__": builtins, "ae": ae, "pd": pd, "np": np,
                          "json": json, "Path": Path, "math": math, "re": re, "input": _no_input}
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.max_rows", 60)
    _display.install_style()
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        ns.update(px=px, go=go)
    except ImportError:
        pass
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ns["plt"] = plt
    except ImportError:
        pass
    return ns


def _displayable(v: Any) -> bool:
    mod = type(v).__module__
    return (mod.startswith("pandas.") and hasattr(v, "to_csv")) or (mod.startswith("plotly.") and hasattr(v, "to_json")) \
        or (mod.startswith("matplotlib.") and hasattr(v, "savefig"))


def _format_error(e: BaseException) -> Dict[str, Any]:
    te = traceback.TracebackException.from_exception(e, capture_locals=False)
    stack = list(te.stack)
    start = next((i for i, fr in enumerate(stack) if fr.filename.startswith("<step")), 0)
    stack = stack[start:]
    if len(stack) > 8:
        stack = stack[:2] + stack[-5:]
    te.stack = traceback.StackSummary.from_list(stack)
    text = "".join(te.format())
    if len(text) > 6000:
        text = text[:1500] + "\n…\n" + text[-4000:]
    return {"ename": type(e).__name__, "evalue": str(e)[:2000], "traceback": text}


def _repr(v: Any, limit: int = 20000) -> str:
    try:
        s = repr(v)
    except Exception as e:  # noqa: BLE001
        s = f"<repr failed: {type(e).__name__}: {e}>"
    return s if len(s) <= limit else s[:limit] + f" … [{len(s) - limit} more characters]"


def run_step(ns: Dict[str, Any], code: str, step: Any, out_dir: Path, capture_chars: int) -> Dict[str, Any]:
    global _IN_EXEC
    from arabic_eval.analysis import display as _display

    displays: List[Dict[str, Any]] = []
    filename = f"<step {step}>"
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    _display.set_sink(displays.append, out_dir, prefix=f"step{step}")
    out, err = _Capture(capture_chars), _Capture(capture_chars // 4)
    result: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    old_out, old_err = sys.stdout, sys.stderr
    t0 = time.perf_counter()
    try:
        sys.stdout, sys.stderr = out, err
        _IN_EXEC = True
        tree = ast.parse(code, filename)
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, filename, "exec"), ns)
        value = eval(compile(last, filename, "eval"), ns) if last is not None else None
        # Figures left open are shown first, the trailing value last (it is the step's "answer").
        plt = sys.modules.get("matplotlib.pyplot")
        if plt is not None:
            for num in list(plt.get_fignums())[:20]:
                fig = plt.figure(num)
                if fig is not value:
                    _display.show(fig)
            plt.close("all")
        if value is not None:
            if _displayable(value):
                _display.show(value)
            else:
                result = _repr(value)
    except KeyboardInterrupt:
        error = {"ename": "Interrupted", "evalue": "the step was interrupted (time limit or stop button)",
                 "traceback": ""}
    except BaseException as e:  # noqa: BLE001 - SystemExit / anything the code raises is reported, never fatal
        error = _format_error(e)
    finally:
        _IN_EXEC = False
        sys.stdout, sys.stderr = old_out, old_err
        _display.set_sink(None)
    return {"ok": error is None, "stdout": out.getvalue(), "stderr": err.getvalue(), "result": result,
            "displays": displays, "error": error, "elapsed": round(time.perf_counter() - t0, 3)}


def namespace_summary(ns: Dict[str, Any]) -> List[Dict[str, Any]]:
    import types
    out = []
    for k, v in ns.items():
        if k.startswith("_") or k in PRELOADED or k in ("input",) or isinstance(v, types.ModuleType):
            continue
        if callable(v) and not hasattr(v, "shape"):
            kind = "function" if isinstance(v, types.FunctionType) else type(v).__name__
            out.append({"name": k, "type": kind})
            continue
        entry: Dict[str, Any] = {"name": k, "type": type(v).__name__}
        shape = getattr(v, "shape", None)
        if shape is not None:
            entry["shape"] = list(shape)
        elif isinstance(v, (list, dict, tuple, set, str)):
            entry["len"] = len(v)
        cols = getattr(v, "columns", None)
        if cols is not None:
            entry["columns"] = [str(c) for c in list(cols)[:30]]
        out.append(entry)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="analysis worker (spawned by the console)")
    ap.add_argument("--out", required=True, help="where figures / tables go (inside the session folder)")
    ap.add_argument("--capture-chars", type=int, default=100_000)
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    proto_in = os.fdopen(os.dup(0), "r", encoding="utf-8")
    proto_out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    log = open(out_dir.parent / "worker.log", "a", encoding="utf-8", buffering=1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdin = io.StringIO("")
    sys.stdout = sys.stderr = log
    signal.signal(signal.SIGINT, _on_sigint)

    def send(msg: Dict[str, Any]) -> None:
        proto_out.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
        proto_out.flush()

    try:
        ns = _namespace(out_dir)
    except Exception as e:  # noqa: BLE001
        send({"op": "ready", "ok": False, "error": _format_error(e)})
        return 1
    send({"op": "ready", "ok": True, "pid": os.getpid(), "python": sys.version.split()[0],
          "preloaded": [k for k in PRELOADED if k in ns]})
    print(f"[worker] ready pid={os.getpid()} out={out_dir}", flush=True)

    for line in proto_in:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            send({"op": "error", "error": "request is not JSON"})
            continue
        rid, op = req.get("id"), req.get("op")
        if op == "exec":
            res = run_step(ns, str(req.get("code") or ""), req.get("step", "?"), out_dir, args.capture_chars)
            send({"id": rid, "op": "result", **res})
        elif op == "vars":
            send({"id": rid, "op": "vars", "vars": namespace_summary(ns)})
        elif op == "ping":
            send({"id": rid, "op": "pong"})
        elif op == "shutdown":
            send({"id": rid, "op": "bye"})
            break
        else:
            send({"id": rid, "op": "error", "error": f"unknown op {op!r}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
