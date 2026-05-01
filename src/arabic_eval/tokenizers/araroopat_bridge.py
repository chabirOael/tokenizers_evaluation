"""Subprocess bridge to the CAMeL Tools server.

Runs in the main `.venv`. Spawns `araroopat_camel_server.py` inside
`.venv-camel` (which has `camel-tools` installed) and exchanges NDJSON
over stdin/stdout. Single-threaded use only — one outstanding request
at a time, correlated by integer id.

Fail-loud policy:
- Missing `.venv-camel` interpreter → `CamelBridgeError` at first call.
- Server died (read returns "") → `CamelBridgeError`.
- Server returned `{"ok": false, ...}` → `CamelBridgeError`.
- Read timed out (`select()` returned no fd) → `CamelBridgeError`.
- Non-JSON line on stdout → `CamelBridgeError`.

Lifecycle:
- Lazy `Popen` on first call (avoids paying the ~2s init cost when
  araroopat isn't being used at all).
- `atexit` handler sends `shutdown` op and reaps. If the server is
  unresponsive, falls back to `terminate()` then `kill()`.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import select
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("arabic_eval.tokenizers.araroopat.bridge")


class CamelBridgeError(RuntimeError):
    """Any failure in the camel subprocess bridge."""


def _resolve_repo_root() -> Path:
    """Walk up from this file to find the repo root (where pyproject.toml lives)."""
    here = Path(__file__).resolve()
    for ancestor in (here.parent, *here.parents):
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    # Last resort: 4 levels up from src/arabic_eval/tokenizers/
    return here.parent.parent.parent.parent


def _resolve_camel_python() -> Path:
    """Pick the interpreter for the camel subprocess.

    Resolution order:
      1. $ARAROOPAT_CAMEL_PYTHON (if set and points to an executable file)
      2. <repo_root>/.venv-camel/bin/python

    Raises CamelBridgeError if neither exists. We do NOT silently fall
    back to the main interpreter — that would import camel-tools from
    the broken main env (or fail with ModuleNotFoundError), neither of
    which is a useful error message.
    """
    env_path = os.environ.get("ARAROOPAT_CAMEL_PYTHON")
    if env_path:
        p = Path(env_path)
        if p.is_file() and os.access(p, os.X_OK):
            return p
        raise CamelBridgeError(
            f"$ARAROOPAT_CAMEL_PYTHON={env_path!r} is not an executable file."
        )

    default = _resolve_repo_root() / ".venv-camel" / "bin" / "python"
    if default.is_file() and os.access(default, os.X_OK):
        return default

    raise CamelBridgeError(
        f"Camel subprocess interpreter not found at {default!s}. "
        "Set up the camel venv with:\n"
        "  python -m venv .venv-camel\n"
        '  .venv-camel/bin/pip install -e ".[araroopat-camel]"\n'
        "  .venv-camel/bin/camel_data -i light\n"
        "Or set $ARAROOPAT_CAMEL_PYTHON to an interpreter that has camel-tools."
    )


# Module path passed to `python -m`. Importable from .venv-camel as long
# as the repo's `src/` is on its PYTHONPATH (we set it explicitly).
_SERVER_MODULE = "arabic_eval.tokenizers.araroopat_camel_server"

# Default per-request read timeout. Generation calls were previously
# bounded by `generator_timeout_ms` (default 50ms) on a per-call basis.
# Bridge round-trip overhead is sub-ms on local pipes, so a 5s ceiling
# here covers the slowest realistic analyze() batch (256 words at
# ~10ms/word = 2.5s) without masking real hangs.
_DEFAULT_READ_TIMEOUT_S = 5.0

# Banner line the server emits after successful init. Reading it serves
# as a "ready" handshake so we surface init failures before the first
# real request is sent.
_BANNER_ID = 0


class CamelBridge:
    """Single-threaded NDJSON client for `araroopat_camel_server.py`.

    Use one instance per process. Methods are not safe to call from
    multiple threads concurrently (no request-id locking).
    """

    def __init__(
        self,
        camel_python: Optional[Path] = None,
        read_timeout_s: float = _DEFAULT_READ_TIMEOUT_S,
    ) -> None:
        self._camel_python = camel_python  # None → resolved lazily
        self._read_timeout_s = read_timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._next_id = 1
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_started(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        if self._closed:
            raise CamelBridgeError("CamelBridge already closed.")

        py = self._camel_python or _resolve_camel_python()
        repo_root = _resolve_repo_root()
        src_dir = repo_root / "src"

        # Inherit env, then prepend src/ to PYTHONPATH so the server
        # module is importable from .venv-camel (which doesn't have the
        # repo installed as a package).
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{src_dir}{os.pathsep}{existing_pp}" if existing_pp else str(src_dir)
        )
        # Force unbuffered I/O on the child so our reads see lines
        # immediately. The server already calls flush(), but -u is
        # belt-and-braces for stderr too.
        env["PYTHONUNBUFFERED"] = "1"

        logger.info("Spawning CAMeL bridge subprocess: %s -m %s", py, _SERVER_MODULE)
        try:
            self._proc = subprocess.Popen(
                [str(py), "-u", "-m", _SERVER_MODULE],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding="utf-8",
                bufsize=1,  # line-buffered
            )
        except (FileNotFoundError, OSError) as e:
            raise CamelBridgeError(f"Failed to spawn camel subprocess: {e}") from e

        # Read the banner. Long timeout — the disambiguator + DB load
        # take ~2-5s on the first call. If the process exits without a
        # banner, _read_response will surface the stderr.
        atexit.register(self._shutdown_quietly)
        try:
            banner = self._read_response(timeout_s=30.0)
        except CamelBridgeError as e:
            self._dump_stderr_and_die(e)
            raise  # unreachable, _dump_stderr_and_die raises
        if banner.get("id") != _BANNER_ID or banner.get("result") != "ready":
            raise CamelBridgeError(f"Unexpected banner from server: {banner!r}")
        return self._proc

    def _shutdown_quietly(self) -> None:
        """atexit hook — best-effort cleanup. Never raises."""
        if self._proc is None or self._proc.poll() is not None:
            self._closed = True
            return
        try:
            self._send({"id": -1, "op": "shutdown"})
        except Exception:  # noqa: BLE001
            pass
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._closed = True

    # ------------------------------------------------------------------
    # Public ops
    # ------------------------------------------------------------------

    def analyze(self, words: List[str]) -> List[List[Dict[str, str]]]:
        """Disambiguate a batch of words. Returns one sublist per word.

        Sublists are top-scored-first; empty sublist ⇒ no analysis.
        Each candidate is a trimmed dict with the ~11 fields araroopat
        consumes (root, pattern, stem, diac, lex, pos, prc0–prc3, enc0).
        """
        proc = self._ensure_started()  # noqa: F841
        req_id = self._take_id()
        self._send({"id": req_id, "op": "analyze", "words": list(words)})
        resp = self._read_response()
        self._check(resp, req_id)
        return resp["results"]

    def generate(self, root: str, pattern: str) -> Optional[str]:
        """Tier-2 reconstruction. Returns the bare stem or None on failure."""
        proc = self._ensure_started()  # noqa: F841
        req_id = self._take_id()
        self._send({"id": req_id, "op": "generate", "root": root, "pattern": pattern})
        resp = self._read_response()
        self._check(resp, req_id)
        return resp["result"]

    # ------------------------------------------------------------------
    # Wire I/O
    # ------------------------------------------------------------------

    def _take_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _send(self, payload: Dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False))
            self._proc.stdin.write("\n")
            self._proc.stdin.flush()
        except BrokenPipeError as e:
            self._dump_stderr_and_die(
                CamelBridgeError(f"Broken pipe writing to camel subprocess: {e}")
            )

    def _read_response(self, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        timeout = timeout_s if timeout_s is not None else self._read_timeout_s

        # Use select() so we can enforce a wall-clock timeout. readline()
        # alone would block indefinitely if the server hangs.
        rlist, _, _ = select.select([self._proc.stdout], [], [], timeout)
        if not rlist:
            self._dump_stderr_and_die(
                CamelBridgeError(f"Camel subprocess timed out after {timeout:.1f}s.")
            )

        line = self._proc.stdout.readline()
        if not line:
            # EOF — subprocess exited.
            self._dump_stderr_and_die(
                CamelBridgeError("Camel subprocess exited unexpectedly (EOF on stdout).")
            )

        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise CamelBridgeError(
                f"Non-JSON line from camel subprocess: {line!r} ({e})"
            ) from e

    @staticmethod
    def _check(resp: Dict[str, Any], expected_id: int) -> None:
        if resp.get("id") != expected_id:
            raise CamelBridgeError(
                f"Camel response id mismatch: got {resp.get('id')}, expected {expected_id}"
            )
        if not resp.get("ok"):
            raise CamelBridgeError(f"Camel server error: {resp.get('error')}")

    def _dump_stderr_and_die(self, err: CamelBridgeError) -> None:
        """Drain whatever stderr the subprocess wrote, attach it to err, raise.

        Always raises — never returns. Used when we've detected the
        subprocess is unrecoverable (EOF, broken pipe, timeout).
        """
        stderr_text = ""
        if self._proc is not None and self._proc.stderr is not None:
            try:
                # Non-blocking drain — we don't want to wait forever.
                # The subprocess is presumed dead/dying.
                if self._proc.poll() is None:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                stderr_text = self._proc.stderr.read() or ""
            except Exception:  # noqa: BLE001
                pass
        if stderr_text.strip():
            raise CamelBridgeError(
                f"{err}\n--- camel subprocess stderr ---\n{stderr_text.rstrip()}"
            ) from err
        raise err


# ---------------------------------------------------------------------------
# Module-level singleton accessor — the backend uses one bridge per process.
# ---------------------------------------------------------------------------

_SHARED_BRIDGE: Optional[CamelBridge] = None


def get_shared_bridge() -> CamelBridge:
    """Lazily build the process-wide CamelBridge instance."""
    global _SHARED_BRIDGE
    if _SHARED_BRIDGE is None:
        _SHARED_BRIDGE = CamelBridge()
    return _SHARED_BRIDGE
