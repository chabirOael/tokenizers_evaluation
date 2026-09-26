"""A stand-in for ``vllm serve`` in the Analysis tab's tests: an OpenAI-compatible
``/v1/models`` + streaming ``/v1/chat/completions`` on 127.0.0.1.

    python tests/data/fake_openai_server.py --port 8899 [--delay 2] [--fail]

The reply follows the Analysis protocol: a python step first, then — once the last
message carries that step's output — a final answer quoting it. ``--delay`` holds the
port closed for that many seconds (a model still loading); ``--fail`` exits 3 after it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STEP = "Let me compute it.\n```python\nprint(6 * 7)\n```"
ANSWER = "The answer is 42."


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        sys.stderr.write("fake: " + (fmt % args) + "\n")

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._json({"object": "list", "data": [{"id": "fake-model", "object": "model"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json({"error": "not found"}, 404)
            return
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        last = (req.get("messages") or [{}])[-1].get("content") or ""
        text = ANSWER if "[output of step" in last else STEP
        if not req.get("stream"):
            self._json({"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(0, len(text), 6):
            chunk = {"choices": [{"index": 0, "delta": {"content": text[i:i + 6]}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(("data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n").encode())
        self.wfile.write(("data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                                                          "total_tokens": 15}}) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--fail", action="store_true")
    a = ap.parse_args()
    print(f"INFO loading fake weights ({a.delay:.0f} s)…", flush=True)
    time.sleep(a.delay)
    if a.fail:
        print("ERROR CUDA out of memory (fake)", flush=True)
        return 3
    print(f"INFO serving on 127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
