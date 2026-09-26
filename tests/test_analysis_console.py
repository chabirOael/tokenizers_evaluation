"""The Analysis tab's wiring into the experiment console: the page hooks, the script's
pinned libraries and API calls, and the server routes that serve them (a real server on
an ephemeral port)."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "debugger" / "experiment_console.html"
JS = REPO / "debugger" / "assets" / "console_analysis.js"
SERVER = REPO / "debugger" / "serve_experiment_console.py"


def test_the_page_wires_the_tab():
    page = PAGE.read_text(encoding="utf-8")
    assert '<button class="tab" data-view="analysis"' in page
    assert '<section class="view" id="view-analysis"></section>' in page
    assert "if (v==='analysis' && window.AnalysisTab) window.AnalysisTab.open();" in page
    assert page.index('<script src="/assets/console_analysis.js"></script>') > page.rindex("'use strict';")   # after the helpers


def test_every_library_is_pinned_on_cdnjs():
    js = JS.read_text(encoding="utf-8")
    libs = re.findall(r"src: '([^']+)',\s*\n?\s*sri: '([^']+)'", js)
    assert len(libs) == 4
    for src, sri in libs:
        assert src.startswith("https://cdnjs.cloudflare.com/ajax/libs/") and re.fullmatch(r"sha512-[A-Za-z0-9+/=]{88}", sri)
    assert "plotly.js/4.1.1/" in js                                   # the version plotly.py 7.1 writes JSON for
    assert "DOMPurify.sanitize(window.marked.parse" in js             # model text is sanitized before it becomes HTML


def test_the_script_only_calls_routes_the_server_has():
    js = JS.read_text(encoding="utf-8")
    server = SERVER.read_text(encoding="utf-8")
    called = set(re.findall(r"/api/analysis/([a-z/]+)", js))
    assert {"endpoints", "scope", "sessions", "session", "chat", "stop", "approve", "exec", "preview", "key",
            "export", "kernel/restart", "session/update"} <= called
    assert "local/log" in called and "/api/analysis/local/${what}" in js          # start / stop by a template
    assert 'sub == "local/start"' in server and 'sub == "local/stop"' in server
    for sub in called:
        if sub in ("artifact", "local/"):
            continue
        assert f'sub == "{sub}"' in server, sub


@pytest.fixture(scope="module")
def server():
    spec = importlib.util.spec_from_file_location("_console_server_under_test", SERVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.LOCAL.gpu_guard in mod.RUNS.gpu_guards        # runs refuse to start beside the local model server
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), mod.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def test_assets_are_served_and_confined(server):
    status, ctype, body = _get(server + "/assets/console_analysis.js")
    assert status == 200 and ctype.startswith("text/javascript") and b"window.AnalysisTab" in body
    for bad in ("/assets/../serve_experiment_console.py", "/assets/%2e%2e/serve_experiment_console.py",
                "/assets/nope.js", "/assets/../experiment_console.html"):
        assert _get(server + bad)[0] == 404, bad


def test_analysis_routes_answer(server):
    status, _, body = _get(server + "/api/analysis/endpoints")
    d = json.loads(body)
    assert status == 200 and {e["name"] for e in d["endpoints"]} >= {"gpt56_terra", "gemma4_31b_local"} and "sandbox" in d
    assert json.loads(_get(server + "/api/analysis/scope")[2])["experiments"] is not None
    assert "ae.catalog(" in json.loads(_get(server + "/api/analysis/api")[2])["reference"]
    status, _, body = _get(server + "/api/analysis/session?id=../../etc")
    assert status == 400 and b"bad session id" in body
    req = urllib.request.Request(server + "/api/analysis/chat", data=b'{"session": "20990101-000000-abcdef", "message": "x"}',
                                 method="POST", headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=20)
    assert e.value.code == 400
