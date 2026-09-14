#!/usr/bin/env python
"""Serve the AraRooPat *training* explorer locally, backed by the real CAMeL bridge.

    .venv/bin/python debugger/serve_araroopat_explorer.py            # http://127.0.0.1:8765
    .venv/bin/python debugger/serve_araroopat_explorer.py --open     # also open the browser
    .venv/bin/python debugger/serve_araroopat_explorer.py --port 9000

Routes
    GET  /               docs/araroopat_train_explorer.html (the new page)
    GET  /explainer      docs/araroopat_explainer.html      (the existing encode explainer, iframed)
    GET  /api/health     {"ok": true, "camel_python": "...", "warm_ms": ...}
    POST /api/trace      {"text": "...", "params": {...}}  →  full step trace (see araroopat_trace.py)
                         The response carries a ``trace_id``; the trained tokenizer is kept in
                         memory (one session — a new trace replaces it).
    POST /api/decode     {"trace_id": "...", "items": ["[ROOT_كتب]", 128, ...]}  →  real decode()
                         of that emission on the kept tokenizer + per-prefix streaming view
                         (see ``trace_decode``). 409 if the trace_id is not the current one.

Stdlib only (http.server) — no new dependencies. The CAMeL bridge is
single-threaded, so trace requests are serialized with a lock; the bridge
is warmed at startup so the first request doesn't pay the ~8 s spawn.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arabic_eval.tokenizers.araroopat_bridge import (  # noqa: E402
    CamelBridgeError,
    _resolve_camel_python,
    get_shared_bridge,
)
from arabic_eval.tokenizers.araroopat_trace import (  # noqa: E402
    MAX_PLAYGROUND_TOKENS,
    trace_decode,
    trace_training_with_tokenizer,
)

DOCS = REPO_ROOT / "docs"
PAGE = DOCS / "araroopat_train_explorer.html"
EXPLAINER = DOCS / "araroopat_explainer.html"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("araroopat.explorer")

_TRACE_LOCK = threading.Lock()
_STATE = {"camel_python": None, "warm_ms": None}
# The trained tokenizer of the most recent trace, kept for /api/decode.
_SESSION: dict = {"trace_id": None, "tokenizer": None}

# Explorer-side parameter allowlist (anything else in the POST body is ignored).
_INT_PARAMS = ("max_roots", "max_patterns", "min_root_freq", "min_pattern_freq")
_BOOL_PARAMS = ("use_diacritized_surface", "add_bos_eos")
_MAX_TEXT_CHARS = 20_000


class Handler(SimpleHTTPRequestHandler):
    server_version = "AraRooPatExplorer/1.0"

    # ---- helpers ------------------------------------------------------
    def _send_json(self, payload, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists():
            self._send_json({"error": f"missing file: {path.relative_to(REPO_ROOT)}"},
                            HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quieter than the default
        log.info("%s %s", self.address_string(), fmt % args)

    # ---- routes -------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route in ("/", "/index.html"):
            self._send_file(PAGE)
        elif route == "/explainer":
            self._send_file(EXPLAINER)
        elif route == "/api/health":
            self._send_json({"ok": True, **_STATE, "page": str(PAGE.relative_to(REPO_ROOT)),
                             "explainer_available": EXPLAINER.exists()})
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route not in ("/api/trace", "/api/decode"):
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except (ValueError, json.JSONDecodeError) as e:
            self._send_json({"error": f"bad JSON body: {e}"}, HTTPStatus.BAD_REQUEST)
            return
        if route == "/api/decode":
            self._handle_decode(req)
            return

        text = (req.get("text") or "").strip()
        if not text:
            self._send_json({"error": "text is empty"}, HTTPStatus.BAD_REQUEST)
            return
        if len(text) > _MAX_TEXT_CHARS:
            self._send_json({"error": f"text too long (> {_MAX_TEXT_CHARS} chars)"},
                            HTTPStatus.BAD_REQUEST)
            return

        raw_params = req.get("params") or {}
        params = {}
        for k in _INT_PARAMS:
            if k in raw_params and raw_params[k] not in (None, ""):
                try:
                    params[k] = max(0, int(raw_params[k]))
                except (TypeError, ValueError):
                    self._send_json({"error": f"{k} must be an integer"}, HTTPStatus.BAD_REQUEST)
                    return
        for k in _BOOL_PARAMS:
            if k in raw_params:
                params[k] = bool(raw_params[k])

        t0 = time.perf_counter()
        with _TRACE_LOCK:
            try:
                trace, tokenizer = trace_training_with_tokenizer(text, params)
                _SESSION["trace_id"] = uuid.uuid4().hex
                _SESSION["tokenizer"] = tokenizer
                trace["trace_id"] = _SESSION["trace_id"]
            except CamelBridgeError as e:
                log.error("CAMeL bridge error: %s", e)
                self._send_json({"error": f"CAMeL bridge error: {e}"},
                                HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            except Exception as e:  # noqa: BLE001
                log.exception("trace failed")
                self._send_json({"error": f"{type(e).__name__}: {e}"},
                                HTTPStatus.INTERNAL_SERVER_ERROR)
                return
        trace["server_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self._send_json(trace)

    def _handle_decode(self, req: dict) -> None:
        trace_id = req.get("trace_id")
        items = req.get("items")
        if not isinstance(items, list) or not items:
            self._send_json({"error": "items must be a non-empty list of ids / token strings"},
                            HTTPStatus.BAD_REQUEST)
            return
        if len(items) > MAX_PLAYGROUND_TOKENS:
            self._send_json({"error": f"too many items (> {MAX_PLAYGROUND_TOKENS})"},
                            HTTPStatus.BAD_REQUEST)
            return
        with _TRACE_LOCK:
            if _SESSION["tokenizer"] is None or trace_id != _SESSION["trace_id"]:
                self._send_json({"error": "no matching trained tokenizer in this server session — "
                                          "run the trace again, then decode"},
                                HTTPStatus.CONFLICT)
                return
            t0 = time.perf_counter()
            try:
                result = trace_decode(_SESSION["tokenizer"], items)
            except ValueError as e:
                self._send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                return
            except CamelBridgeError as e:
                log.error("CAMeL bridge error: %s", e)
                self._send_json({"error": f"CAMeL bridge error: {e}"},
                                HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            except Exception as e:  # noqa: BLE001
                log.exception("decode failed")
                self._send_json({"error": f"{type(e).__name__}: {e}"},
                                HTTPStatus.INTERNAL_SERVER_ERROR)
                return
        result["trace_id"] = trace_id
        result["server_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self._send_json(result)


def _warm_bridge() -> None:
    t0 = time.perf_counter()
    bridge = get_shared_bridge()
    bridge._ensure_started()
    bridge.analyze(["إحماء"])
    _STATE["camel_python"] = str(bridge._camel_python or _resolve_camel_python())
    _STATE["warm_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info("CAMeL bridge ready in %.1f s", _STATE["warm_ms"] / 1000)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    ap.add_argument("--no-warm", action="store_true", help="don't spawn CAMeL until the first request")
    args = ap.parse_args()

    if not PAGE.exists():
        log.error("page not found: %s", PAGE)
        return 1

    if not args.no_warm:
        try:
            _warm_bridge()
        except CamelBridgeError as e:
            log.error("CAMeL bridge failed to start:\n%s", e)
            return 2

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    log.info("AraRooPat train explorer → %s   (Ctrl-C to stop)", url)
    if args.open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
