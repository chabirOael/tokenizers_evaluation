#!/usr/bin/env python
"""Serve the experiment console: configure experiment YAMLs in a form, list the
existing ones, start a run, watch it, cancel it.

    .venv/bin/python debugger/serve_experiment_console.py            # http://127.0.0.1:8766
    .venv/bin/python debugger/serve_experiment_console.py --open     # also open the browser
    .venv/bin/python debugger/serve_experiment_console.py --port 9000

A started run is a detached session leader (see ``RunManager`` in
``arabic_eval.tools.experiment_console``): closing the page, the SSH session
or this server does not stop it. Runs live under ``outputs/runs/<run_id>/``
and are rediscovered when the server restarts.

Routes
    GET  /                          debugger/experiment_console.html
    GET  /api/health                {"ok": true, "python": ..., "hf_token_set": ...}
    GET  /api/schema                JSON schema of ExperimentConfig + base.yaml defaults + registries + presets
    GET  /api/configs               every configs/experiments/*.yaml with its cells / tasks / existing results
    GET  /api/configs/get?path=     one file: yaml text, raw dict, resolved (base-merged) dict, validation
    POST /api/config/validate       {"config": {...}}              → {"ok", "errors": [{loc, msg}], "resolved"}
    POST /api/config/render         {"config": {...}, "mode": "full"|"delta"}  → {"yaml": "..."}
    POST /api/config/parse          {"yaml": "..."}                → {"config": {...}}
    POST /api/configs/save          {"name", "yaml", "overwrite"}  → {"path"}
    GET  /api/configs/results?path= per-cell metrics summary of a config's output_dir
    GET  /api/runs                  every run (status reconciled from disk)
    GET  /api/runs/<id>             record + parsed progress + log tail + results
    GET  /api/runs/<id>/log?offset= incremental console.log chunk
    POST /api/runs/start            {"config": "<file>"} or {"yaml": "...", "label": "..."} plus
                                    optional "sweep" (auto), "device", "seed", "force"   → record (409 if a run is active)
    POST /api/runs/<id>/cancel      SIGTERM the process group, SIGKILL after the grace period
    GET  /api/gpu                   nvidia-smi snapshot (null when unavailable)
    GET  /api/text?path=&tail=      a text file under outputs/ (reports, experiment.log), tail-limited
    GET  /api/eval/tree             every eval-row dump: experiment → cell → benchmark
    GET  /api/eval/describe?path=   one dump's metadata, sub-configs and whole-file summary
    GET  /api/eval/rows?path=&…     filtered, paginated rows + the selection's aggregates
                                    (outcome, scoring, subconfig, flags, q, sort, page, page_size)
    GET  /api/eval/row?path=&position=   one full record
    GET  /api/eval/export?path=&…   the current selection as a CSV download
    GET  /api/judge/configs         configs/judges/*.yaml with backend, model and readiness (venv / API key)
    POST /api/judge/start           {"experiment", "judges": [names], "baseline", "limit", "overwrite", "force"}
                                    → a detached judge run (Runs tab); GPU judges conflict with active runs
    GET  /api/freeform/tree         every free-form generation dump: experiment → cell (+ unsupported cells)
    GET  /api/freeform/rows?cell=&… one cell's generations joined with its judge files, filtered / sorted / paged
    GET  /api/freeform/row?cell=&id= one full record + the same prompt in the sibling cells
    GET  /api/rating/sets?experiment=        blind rating sets of an experiment and who rated what
    POST /api/rating/build          {"experiment", "name", "n_prompts", "variants", "seed", "overwrite"} → the set (blind)
    GET  /api/rating/items?experiment=&set=&rater=   the items (cell hidden) with this rater's ratings
    POST /api/rating/submit         {"experiment", "set", "rater", "item_id", "score", "flags", "note"}
    GET  /api/rating/agreement?experiment=&set=      rater vs judge / rater vs rater / judge vs judge, cells revealed
    GET  /api/assistant/configs     configs/assistant/*.yaml with model, endpoint and readiness (key set / local server up)
    POST /api/assistant/chat        {"assistant", "config", "from_base", "file", "history", "message"} → an SSE stream
                                    of events (token / edits / repair / done / error); the model's edits are applied
                                    to the config and validated server-side, the page puts the result in the form
    POST /api/assistant/preview     the same body → the messages a turn would send, with their size

Stdlib only (http.server) — no new dependencies.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arabic_eval.tools import eval_rows_browser as eval_rows  # noqa: E402
from arabic_eval.tools import freeform_rating  # noqa: E402
from arabic_eval.tools import config_assistant as assistant  # noqa: E402
from arabic_eval.tools import freeform_rows_browser as freeform_rows  # noqa: E402
from arabic_eval.tools.experiment_console import (  # noqa: E402
    judge_configs,
    ConsoleError,
    ConsolePaths,
    RunConflict,
    RunManager,
    config_results,
    gpu_snapshot,
    list_configs,
    parse_yaml,
    read_config,
    render_yaml,
    save_config,
    schema_bundle,
    validate_config,
)

PAGE = REPO_ROOT / "debugger" / "experiment_console.html"
PATHS = ConsolePaths(REPO_ROOT)
RUNS = RunManager(PATHS)
ASSISTANT_CTX = assistant.ContextBuilder(PATHS)
_MAX_BODY = 4 * 1024 * 1024
_TEXT_ROOT = REPO_ROOT / "outputs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("experiment.console")


class Handler(SimpleHTTPRequestHandler):
    server_version = "ExperimentConsole/1.0"

    # ---- helpers ------------------------------------------------------
    def _send_json(self, payload, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_page(self) -> None:
        if not PAGE.exists():
            self._send_json({"error": f"missing page: {PAGE.relative_to(REPO_ROOT)}"}, HTTPStatus.NOT_FOUND)
            return
        body = PAGE.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_csv(self, filename: str, text: str) -> None:
        body = text.encode("utf-8-sig")   # BOM so Excel renders Arabic
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_events(self, events) -> None:
        """Relay an event iterator as a server-sent-event stream (``data: <json>``
        per event). HTTP/1.0: no Content-Length, the connection closes at the
        end, which is what the page's stream reader waits for. Failures after
        the headers are sent become an ``error`` event."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def _emit(ev) -> None:
            self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()
        try:
            for ev in events:
                _emit(ev)
        except (BrokenPipeError, ConnectionResetError):
            log.info("assistant stream: client went away")
        except Exception as e:  # noqa: BLE001
            log.exception("assistant stream failed")
            try:
                _emit({"type": "error", "message": f"{type(e).__name__}: {e}" if not isinstance(e, ConsoleError) else str(e)})
            except OSError:
                pass

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > _MAX_BODY:
            raise ConsoleError("request body too large")
        raw = self.rfile.read(n) if n else b""
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError as e:
            raise ConsoleError(f"invalid JSON body: {e}") from e
        return data if isinstance(data, dict) else {}

    def log_message(self, fmt, *args):  # quieter than the default
        if "/api/runs" in (args[0] if args else "") and "GET" in (args[0] if args else ""):
            return
        log.info("%s " + fmt, self.address_string(), *args)

    def _dispatch(self, method: str) -> None:
        parts = urlsplit(self.path)
        route = parts.path.rstrip("/") or "/"
        q = {k: v[-1] for k, v in parse_qs(parts.query).items()}
        try:
            if method == "GET":
                self._get(route, q)
            else:
                self._post(route, self._read_body())
        except RunConflict as e:
            self._send_json({"error": str(e)}, HTTPStatus.CONFLICT)
        except ConsoleError as e:
            self._send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        except ValueError as e:
            self._send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        except Exception as e:  # noqa: BLE001
            log.exception("%s %s failed", method, route)
            self._send_json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    # ---- routes -------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def _get(self, route: str, q: dict) -> None:
        if route == "/":
            self._send_page()
        elif route == "/api/health":
            b = schema_bundle(PATHS)["env"]
            self._send_json({"ok": True, **b, "page": PATHS.rel(PAGE)})
        elif route == "/api/schema":
            self._send_json(schema_bundle(PATHS))
        elif route == "/api/configs":
            self._send_json({"configs": list_configs(PATHS)})
        elif route == "/api/configs/get":
            self._send_json(read_config(PATHS, q.get("path", "")))
        elif route == "/api/configs/results":
            self._send_json(config_results(PATHS, q.get("path", "")))
        elif route == "/api/runs":
            self._send_json({"runs": RUNS.list(), "gpu": gpu_snapshot()})
        elif route.startswith("/api/runs/") and route.endswith("/log"):
            run_id = route[len("/api/runs/"):-len("/log")]
            self._send_json(RUNS.log(run_id, int(q.get("offset") or 0)))
        elif route.startswith("/api/runs/"):
            self._send_json(RUNS.detail(route[len("/api/runs/"):]))
        elif route == "/api/gpu":
            self._send_json({"gpu": gpu_snapshot()})
        elif route == "/api/text":
            self._send_json(_read_text(q.get("path", ""), int(q.get("tail") or 256 * 1024)))
        elif route == "/api/eval/tree":
            self._send_json(eval_rows.discover(REPO_ROOT))
        elif route == "/api/eval/describe":
            self._send_json(eval_rows.describe(REPO_ROOT, q.get("path", "")))
        elif route == "/api/eval/rows":
            self._send_json(eval_rows.query(REPO_ROOT, q.get("path", ""), q))
        elif route == "/api/eval/row":
            self._send_json(eval_rows.row(REPO_ROOT, q.get("path", ""), int(q.get("position") or 0)))
        elif route == "/api/eval/export":
            name, text = eval_rows.export_csv(REPO_ROOT, q.get("path", ""), q)
            self._send_csv(name, text)
        elif route == "/api/judge/configs":
            self._send_json({"judges": judge_configs(PATHS)})
        elif route == "/api/freeform/tree":
            self._send_json(freeform_rows.discover(REPO_ROOT))
        elif route == "/api/freeform/rows":
            self._send_json(freeform_rows.query(REPO_ROOT, q.get("cell", ""), q))
        elif route == "/api/freeform/row":
            self._send_json(freeform_rows.row(REPO_ROOT, q.get("cell", ""), q.get("id", "")))
        elif route == "/api/rating/sets":
            self._send_json(freeform_rating.list_sets(REPO_ROOT, q.get("experiment", "")))
        elif route == "/api/rating/items":
            self._send_json(freeform_rating.get_items(REPO_ROOT, q.get("experiment", ""), q.get("set", ""), q.get("rater", "")))
        elif route == "/api/assistant/configs":
            self._send_json({"assistants": assistant.assistant_configs(PATHS)})
        elif route == "/api/rating/agreement":
            self._send_json(freeform_rating.agreement(REPO_ROOT, q.get("experiment", ""), q.get("set", "")))
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _post(self, route: str, req: dict) -> None:
        if route == "/api/config/validate":
            self._send_json(validate_config(PATHS, req.get("config") or {}))
        elif route == "/api/config/render":
            self._send_json({"yaml": render_yaml(PATHS, req.get("config") or {}, req.get("mode") or "full")})
        elif route == "/api/config/parse":
            cfg = parse_yaml(req.get("yaml") or "")
            self._send_json({"config": cfg, **validate_config(PATHS, cfg)})
        elif route == "/api/configs/save":
            self._send_json(save_config(PATHS, str(req.get("name") or ""), str(req.get("yaml") or ""),
                                        bool(req.get("overwrite"))))
        elif route == "/api/runs/start":
            seed = req.get("seed")
            rec = RUNS.start(
                str(req.get("config") or req.get("label") or "unsaved"),
                yaml_text=req.get("yaml") if req.get("yaml") is not None else None,
                sweep=req.get("sweep") if req.get("sweep") in (True, False) else None,
                device=(str(req["device"]).strip() or None) if req.get("device") else None,
                seed=int(seed) if seed not in (None, "") else None,
                force=bool(req.get("force")),
            )
            log.info("started run %s (pid %s): %s", rec["run_id"], rec["pid"], " ".join(rec["argv"]))
            self._send_json(rec, HTTPStatus.ACCEPTED)
        elif route == "/api/judge/start":
            lim = req.get("limit")
            rec = RUNS.start_judge(
                str(req.get("experiment") or ""), list(req.get("judges") or []),
                baseline=(str(req["baseline"]) if req.get("baseline") else None),
                limit=int(lim) if lim not in (None, "", 0, "0") else None,
                overwrite=bool(req.get("overwrite")), force=bool(req.get("force")))
            log.info("started judge run %s (pid %s): %s", rec["run_id"], rec["pid"], " ".join(rec["argv"]))
            self._send_json(rec, HTTPStatus.ACCEPTED)
        elif route == "/api/rating/build":
            self._send_json(freeform_rating.build_set(
                REPO_ROOT, str(req.get("experiment") or ""), name=str(req.get("name") or "v1"),
                n_prompts=int(req.get("n_prompts") or 50), variants_per_prompt=int(req.get("variants") or 3),
                seed=int(req.get("seed") or 42), baseline=(str(req["baseline"]) if req.get("baseline") else None),
                overwrite=bool(req.get("overwrite"))))
        elif route == "/api/rating/submit":
            self._send_json(freeform_rating.submit(
                REPO_ROOT, str(req.get("experiment") or ""), str(req.get("set") or ""), str(req.get("rater") or ""),
                str(req.get("item_id") or ""), req.get("score"), req.get("flags") or [], str(req.get("note") or "")))
        elif route == "/api/assistant/chat":
            acfg, kw = _assistant_request(req)
            client = assistant.ChatClient(acfg)        # raises before any header is sent (key missing)
            self._send_events(assistant.chat_turn(PATHS, ASSISTANT_CTX, acfg, client, **kw))
        elif route == "/api/assistant/preview":
            acfg, kw = _assistant_request(req)
            self._send_json(assistant.preview_messages(PATHS, ASSISTANT_CTX, acfg, **kw))
        elif route.startswith("/api/runs/") and route.endswith("/cancel"):
            run_id = route[len("/api/runs/"):-len("/cancel")]
            rec = RUNS.cancel(run_id)
            log.info("cancel requested for %s (pgid %s)", run_id, rec.get("pgid"))
            self._send_json(rec)
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def _assistant_request(req: dict):
    """The assistant config and the keyword arguments of a chat / preview call."""
    acfg = assistant.AssistantConfig.from_yaml(assistant.assistant_config_path(PATHS, str(req.get("assistant") or "")))
    cfg = req.get("config")
    history = req.get("history") or []
    if not isinstance(history, list):
        raise ConsoleError("history must be a list of {role, content}")
    return acfg, dict(cfg=cfg if isinstance(cfg, dict) else None, history=history,
                      message=str(req.get("message") or ""), from_base=bool(req.get("from_base")),
                      file=(str(req["file"]) if req.get("file") else None))


def _read_text(rel: str, tail: int) -> dict:
    """A text file under ``outputs/`` (reports, experiment.log), last ``tail`` bytes."""
    if not rel:
        raise ConsoleError("path required")
    path = (REPO_ROOT / rel).resolve()
    if _TEXT_ROOT.resolve() not in path.parents:
        raise ConsoleError("only files under outputs/ can be read")
    if not path.is_file():
        raise ConsoleError(f"no such file: {rel}")
    size = path.stat().st_size
    with open(path, "rb") as fh:
        if size > tail:
            fh.seek(size - tail)
        data = fh.read().decode("utf-8", errors="replace")
    return {"path": rel, "size": size, "truncated": size > tail, "text": data}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = ap.parse_args()

    if not PAGE.exists():
        log.error("page not found: %s", PAGE)
        return 1
    active = RUNS.active()
    if active:
        log.info("rediscovered %d active run(s): %s", len(active), ", ".join(r["run_id"] for r in active))

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    log.info("Experiment console → %s   (Ctrl-C to stop; running experiments are NOT stopped)", url)
    if args.open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down (detached runs keep going)")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
