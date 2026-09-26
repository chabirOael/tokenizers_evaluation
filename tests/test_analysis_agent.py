"""The Analysis tab's agent loop (analysis_agent) against a scripted model, plus the number
provenance check (analysis_provenance). Sessions live under outputs/analysis/_pytest (a
sandboxed worker cannot use a folder under /tmp); most tests run the worker unsandboxed for
speed, one runs the whole loop in the sandbox."""
from __future__ import annotations

import base64
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import arabic_eval.analysis as ae
from arabic_eval.tools.analysis_agent import (
    AnalysisEndpoint,
    AnalysisError,
    AnalysisService,
    KeyStore,
    SessionBusy,
    SessionStore,
    build_messages,
    export_markdown,
)
from arabic_eval.tools.analysis_kernel import KernelPool, sandbox_status
from arabic_eval.tools.analysis_provenance import answer_numbers, check_numbers, plotly_numbers

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "outputs" / "analysis" / "_pytest"

Reply = Union[str, Callable[[List[Dict[str, str]]], str]]


class FakeClient:
    """Plays a script of replies; records the messages of every call."""

    def __init__(self, replies: Sequence[Reply], *, delay: float = 0.0, finish: Sequence[str] = ()) -> None:
        self.replies = list(replies)
        self.calls: List[List[Dict[str, str]]] = []
        self.delay = delay
        self.finish = list(finish)
        self.last_usage = None
        self.last_finish_reason = None
        self.last_tool_calls = []

    def stream(self, messages):
        self.calls.append([dict(m) for m in messages])
        i = len(self.calls) - 1
        r = self.replies[min(i, len(self.replies) - 1)]
        r = r(messages) if callable(r) else r
        self.last_tool_calls = []
        if isinstance(r, dict):                   # native protocol: {"content": ..., "tool_calls": [code, ...]}
            self.last_tool_calls = [{"id": f"call_{i}_{k}", "name": "run_python", "arguments": json.dumps({"code": c})}
                                    for k, c in enumerate(r.get("tool_calls") or [])]
            r = r.get("content") or ""
        text = r
        self.last_finish_reason = self.finish[i] if i < len(self.finish) else "stop"
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": len(text) // 4, "total_tokens": 100 + len(text) // 4}
        for k in range(0, len(text), 7):
            if self.delay:
                time.sleep(self.delay)
            yield text[k:k + 7]


def _experiments(root: Path) -> None:
    cell = root / "exp" / "cellA"
    ids = [f"cidar-{i}" for i in range(6)]
    t = pa.table({"id": ids, "stratum": ["short", "medium", "long"] * 2, "generation": ["جواب"] * 6,
                  "reference": ["مرجع"] * 6, "stop_reason": ["eos"] * 6})
    (cell / "eval_rows").mkdir(parents=True)
    pq.write_table(t.replace_schema_metadata({b"arabic_eval": json.dumps({"schema_version": 3}).encode()}),
                   cell / "eval_rows" / "freeform_cidar.parquet")
    (cell / "all_metrics.json").write_text(json.dumps({
        "config": {"tokenizer": "bpe"}, "intrinsic": {"fertility": 1.51},
        "downstream": {"freeform_cidar": {"status": "ok", "chrf": 17.7, "loop_stop_rate": 0.196,
                                          "judge": {"gemma4_31b": {"score_mean": 2.132, "score_se": 0.08, "n": 250}}}}}))


@pytest.fixture()
def env():
    d = BASE / ("agent_" + uuid.uuid4().hex[:8])
    (d / "configs" / "analysis").mkdir(parents=True)
    (d / "configs" / "analysis" / "fake.yaml").write_text(
        "analysis:\n  name: fake\n  model: fake-model\n  api_key_env: null\n  max_steps: 3\n"
        "  step_timeout_sec: 20\n  context_tokens: 60000\n")
    _experiments(d / "experiments")
    ae.set_experiments_root(d / "experiments")
    yield d
    ae.set_experiments_root(None)
    shutil.rmtree(d, ignore_errors=True)


def _service(env: Path, client: FakeClient, *, sandbox: str = "off", approval: bool = False) -> AnalysisService:
    pool = KernelPool(sandbox=sandbox, experiments_root=env / "experiments")
    return AnalysisService(env, pool=pool, client_factory=lambda ep, key, opts: client,
                           require_approval_unsandboxed=approval, approval_timeout_sec=20)


def _run(svc: AnalysisService, sid: str, message: str, on_event=None) -> List[Dict[str, Any]]:
    events = []
    for ev in svc.chat(sid, message):
        events.append(ev)
        if on_event:
            on_event(ev)
    return events


def _types(events):
    return [e["type"] for e in events if e["type"] != "token"]


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def test_a_turn_runs_code_feeds_the_output_back_and_checks_the_answer(env):
    client = FakeClient([
        "Let me look.\n```python\nm = ae.metrics()\nprint(m['judge.gemma4_31b.mean'].round(3).to_dict())\n```",
        "The judge mean of cellA is **2.132** (not 2.5).",
    ])
    svc = _service(env, client)
    try:
        sid = svc.new_session("fake")["id"]
        ev = _run(svc, sid, "What is the judge mean?")
        # the first answer quotes 2.5, which no output printed: it is held once as a draft, the model is
        # asked to compute it, repeats itself (the script has no third reply) and is then shown, flagged
        assert _types(ev) == ["turn", "model_start", "reply", "exec_start", "exec", "model_start", "reply",
                              "model_start", "answer", "done"]
        assert next(e for e in ev if e["type"] == "reply" and e.get("draft"))["unverified"] == ["2.5"]
        assert "these numbers of your draft appear in no output of this session: 2.5" in client.calls[2][-1]["content"]
        ex = next(e for e in ev if e["type"] == "exec")
        assert ex["ok"] and "2.132" in ex["stdout"]
        fb = client.calls[1][-1]["content"]
        assert fb.startswith("[output of step 1 — ok") and "2.132" in fb
        assert client.calls[0][0]["role"] == "system" and "## The ae helpers" in client.calls[0][0]["content"]
        assert "exp/cellA" in client.calls[0][0]["content"]                      # the scope lists the cells
        ans = next(e for e in ev if e["type"] == "answer")
        assert [u["text"] for u in ans["provenance"]["unverified"]] == ["2.5"]
        sess = svc.store.load(sid)
        t = sess["turns"][0]
        assert t["status"] == "ok" and t["steps"][0]["code"].startswith("m = ae.metrics()") and t["answer"].startswith("The judge")
        assert sess["title"] == "What is the judge mean?" and sess["usage"]["total_tokens"] > 0
        assert sess["next_step"] == 2
    finally:
        svc.pool.shutdown_all()


def test_an_answer_with_invented_numbers_is_held_until_they_are_computed(env):
    client = FakeClient([
        "The judge difference is +0.116 [0.024, 0.208].",                       # no code at all: invented
        "```python\nprint(round(3.5 - 1.25, 3))\n```",
        "The difference is 2.25.",
    ])
    svc = _service(env, client)
    try:
        sid = svc.new_session("fake")["id"]
        ev = _run(svc, sid, "what is the difference?")
        draft = next(e for e in ev if e["type"] == "reply" and e.get("draft"))
        assert draft["unverified"] == ["+0.116", "0.024", "0.208"]
        assert "run the code that prints or computes them" in client.calls[1][-1]["content"]
        ans = next(e for e in ev if e["type"] == "answer")
        assert ans["text"] == "The difference is 2.25." and ans["provenance"]["unverified"] == []
        t = svc.store.load(sid)["turns"][0]
        assert t["steps"][0]["draft_answer"] and t["answer"] == "The difference is 2.25."
    finally:
        svc.pool.shutdown_all()


def test_errors_are_fed_back_and_the_state_persists_across_turns(env):
    client = FakeClient([
        "```python\nfoo = 41\nundefined_name + 1\n```",
        "Fixed.\n```python\nprint(foo + 1)\n```",
        "Done: 42.",
        "```python\nprint(foo * 2)\n```",
        "82 it is.",
    ])
    svc = _service(env, client)
    try:
        sid = svc.new_session("fake")["id"]
        ev = _run(svc, sid, "compute")
        errs = [e for e in ev if e["type"] == "exec" and not e["ok"]]
        assert errs and errs[0]["error"]["ename"] == "NameError"
        assert "error: NameError" in client.calls[1][-1]["content"]
        _run(svc, sid, "double it")
        assert "82" in client.calls[4][-1]["content"]                             # foo survived the turn
        second_turn_msgs = client.calls[3]
        assert any(m["role"] == "assistant" and m["content"] == "Done: 42." for m in second_turn_msgs)
    finally:
        svc.pool.shutdown_all()


def test_the_step_limit_forces_an_answer_without_running_code(env):
    client = FakeClient(["```python\nprint('again')\n```"])                   # never answers on its own
    svc = _service(env, client)
    try:
        sid = svc.new_session("fake")["id"]
        ev = _run(svc, sid, "loop forever")
        assert sum(1 for e in ev if e["type"] == "exec") == 3                 # max_steps = 3
        assert "Step limit reached" in client.calls[3][-1]["content"]
        assert ev[-1]["status"] == "step_limit" and any(e["type"] == "answer" for e in ev)
    finally:
        svc.pool.shutdown_all()


def test_a_cut_off_code_block_is_not_run(env):
    client = FakeClient(["```python\nx = [1, 2,", "No code needed: 12 cells."], finish=["length", "stop"])
    svc = _service(env, client)
    try:
        sid = svc.new_session("fake")["id"]
        ev = _run(svc, sid, "q")
        assert not any(e["type"] == "exec" for e in ev)
        assert "cut off" in client.calls[1][-1]["content"] and "length limit" in client.calls[1][-1]["content"]
        assert ev[-1]["status"] == "ok"
    finally:
        svc.pool.shutdown_all()


def test_stop_ends_the_turn_during_the_stream_and_during_execution(env):
    client = FakeClient(["```python\nimport time\ntime.sleep(30)\n```", "never reached"], delay=0.01)
    svc = _service(env, client)
    try:
        sid = svc.new_session("fake")["id"]

        def on_event(e):
            if e["type"] == "exec_start":
                threading.Timer(0.5, svc.stop, args=(sid,)).start()
        t0 = time.monotonic()
        ev = _run(svc, sid, "sleep", on_event)
        assert time.monotonic() - t0 < 10
        ex = next(e for e in ev if e["type"] == "exec")
        assert ex["error"]["ename"] == "Interrupted"
        assert ev[-1]["status"] == "stopped"
        client2 = FakeClient(["a long reply " * 50], delay=0.02)
        svc.client_factory = lambda ep, key, opts: client2
        ev = _run(svc, sid, "talk", lambda e: e["type"] == "model_start" and threading.Timer(0.2, svc.stop, args=(sid,)).start())
        assert ev[-1]["status"] == "stopped"
        assert svc.stop(sid) is False                                         # nothing running any more
    finally:
        svc.pool.shutdown_all()


def test_unsandboxed_steps_wait_for_approval(env):
    client = FakeClient(["```python\nprint('ran 1')\n```", "```python\nprint('ran 2')\n```", "ok, 2 steps."])
    svc = _service(env, client, approval=True)
    decisions = iter([True, False])
    try:
        sid = svc.new_session("fake")["id"]

        def on_event(e):
            if e["type"] == "approval":
                assert svc.session(sid)["runtime"]["awaiting_approval"] == e["step"]
                svc.approve(sid, e["step"], next(decisions))
        ev = _run(svc, sid, "q", on_event)
        execs = [e for e in ev if e["type"] == "exec"]
        assert execs[0]["ok"] and "ran 1" in execs[0]["stdout"]
        assert execs[1].get("declined") is True
        assert "declined" in client.calls[2][-1]["content"]
        with pytest.raises(AnalysisError, match="not waiting"):
            svc.approve(sid, 99, True)
    finally:
        svc.pool.shutdown_all()


def test_one_turn_at_a_time_per_session(env):
    client = FakeClient(["```python\nimport time; time.sleep(1)\n```", "done after 1 step."])
    svc = _service(env, client)
    try:
        sid = svc.new_session("fake")["id"]
        gen = svc.chat(sid, "first")
        next(gen)                                                             # the turn holds the session
        with pytest.raises(SessionBusy):
            svc.chat(sid, "second")
        with pytest.raises(SessionBusy):
            svc.exec_code(sid, "1")
        for _ in gen:
            pass
        assert svc.chat(sid, "third") is not None                            # released
    finally:
        svc.pool.shutdown_all()


def test_code_run_by_hand_is_part_of_the_context(env):
    client = FakeClient(["The variable holds 7 items? It holds 12."])
    svc = _service(env, client)
    try:
        sid = svc.new_session("fake")["id"]
        r = svc.exec_code(sid, "items = list(range(12))\nlen(items)")
        assert r["ok"] and r["result"] == "12" and r["step"] == 1
        _run(svc, sid, "how many items?")
        joined = "\n".join(m["content"] for m in client.calls[0])
        assert "I ran this code myself" in joined and "len(items)" in joined and "12" in joined
        assert svc.store.load(sid)["turns"][0]["kind"] == "code"
    finally:
        svc.pool.shutdown_all()


def test_displays_get_urls_and_artifacts_are_confined(env):
    client = FakeClient(["```python\nae.show(px.bar(x=['a', 'b'], y=[2.43, 2.13], title='judge'))\n"
                         "ae.show(pd.DataFrame({'cell': ['a'], 'score': [2.43]}))\n```",
                         "Shown: a (2.43) beats b (2.13)."])
    svc = _service(env, client)
    try:
        sid = svc.new_session("fake")["id"]
        ev = _run(svc, sid, "plot")
        ex = next(e for e in ev if e["type"] == "exec")
        assert [d["kind"] for d in ex["displays"]] == ["plotly", "table"]
        assert all(d["url"].startswith(f"/api/analysis/artifact?session={sid}&name=") for d in ex["displays"])
        assert svc.store.artifact(sid, ex["displays"][0]["path"]).exists()
        with pytest.raises(AnalysisError):
            svc.store.artifact(sid, "../session.json")
        ans = next(e for e in ev if e["type"] == "answer")
        assert ans["provenance"]["unverified"] == []                          # both numbers are in the outputs
    finally:
        svc.pool.shutdown_all()


@pytest.mark.skipif(not sandbox_status()["available"], reason="namespace sandbox unavailable")
def test_the_whole_loop_in_the_sandbox(env):
    client = FakeClient(["```python\nprint(ae.metrics().loc['exp/cellA', 'freeform.chrf'])\n"
                         "open('/etc/should_fail', 'w')\n```", "chrF is 17.7."])
    svc = _service(env, client, sandbox="auto")
    try:
        sid = svc.new_session("fake")["id"]
        ev = _run(svc, sid, "chrf?")
        assert ev[0]["sandboxed"] is True
        ex = next(e for e in ev if e["type"] == "exec")
        assert "17.7" in ex["stdout"] and "Read-only" in ex["error"]["evalue"]
    finally:
        svc.pool.shutdown_all()


def test_a_restarted_worker_is_announced_to_the_model(env):
    client = FakeClient(["```python\nz = 5\n```", "set.", "```python\nprint('z' in globals())\n```", "gone: False."])
    svc = _service(env, client)
    try:
        sid = svc.new_session("fake")["id"]
        _run(svc, sid, "set z")
        svc.close_kernel(sid)                                                 # e.g. a console restart
        _run(svc, sid, "is z there?")
        assert "was restarted since the previous question" in client.calls[2][-1]["content"]
    finally:
        svc.pool.shutdown_all()


# --------------------------------------------------------------------------
# endpoints, keys, context budget, export
# --------------------------------------------------------------------------

def test_endpoint_yaml_validation_and_the_shipped_configs(tmp_path):
    for p in sorted((REPO / "configs" / "analysis").glob("*.yaml")):
        AnalysisEndpoint.from_yaml(p)
    bad = tmp_path / "bad.yaml"
    bad.write_text("analysis:\n  name: x\n  model: m\n  nope: 1\n")
    with pytest.raises(ValueError, match="unknown keys"):
        AnalysisEndpoint.from_yaml(bad)
    ep = AnalysisEndpoint(name="e", model="m", reasoning_efforts=["low", "high"], extra_body={"reasoning_effort": "low"})
    assert ep.assistant_config(reasoning_effort="high").extra_body["reasoning_effort"] == "high"
    with pytest.raises(AnalysisError):
        ep.assistant_config(reasoning_effort="max")


def test_keys_live_in_memory_and_never_leave_it(env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (env / "configs" / "analysis" / "api.yaml").write_text("analysis:\n  name: api\n  model: m\n")
    svc = AnalysisService(env, pool=KernelPool(sandbox="off"))
    entry = next(e for e in svc.endpoints()["endpoints"] if e["name"] == "api")
    assert not entry["ready"] and "no API key" in entry["why"]
    with pytest.raises(AnalysisError, match="no API key"):
        svc.chat(svc.new_session("api")["id"], "hi")
    key = "sk-test-" + "x" * 30 + "ABCD"
    svc.keys.set("OPENAI_API_KEY", key)
    listing = json.dumps(svc.endpoints(), ensure_ascii=False)
    assert key not in listing
    assert "…ABCD" in listing
    entry = next(e for e in svc.endpoints()["endpoints"] if e["name"] == "api")
    assert entry["ready"] and entry["key"]["source"] == "tab"
    with pytest.raises(AnalysisError):
        svc.keys.set("OPENAI_API_KEY", "short")
    svc.keys.set("OPENAI_API_KEY", None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-" + "y" * 30)
    assert KeyStore().status("OPENAI_API_KEY")["source"] == "environment"
    for p in (env / "outputs").rglob("*") if (env / "outputs").exists() else []:     # nothing on disk holds it
        if p.is_file():
            assert key not in p.read_text(errors="ignore")


def test_the_context_budget_elides_then_drops_earlier_turns():
    big = "x" * 5000
    turns = [{"n": i, "kind": "ask", "user": f"q{i}", "status": "ok", "answer": f"a{i}",
              "steps": [{"n": i, "reply": "```python\n1\n```", "code": "1", "exec": {"ok": True, "stdout": big, "elapsed": 0.1}}]}
             for i in range(1, 5)]
    turns.append({"n": 5, "kind": "ask", "user": "q5", "status": "running", "steps": []})
    sess = {"turns": turns}
    roomy = AnalysisEndpoint(name="r", model="m", context_tokens=100000)
    msgs, c = build_messages(sess, roomy, "SYSTEM")
    assert c["elided_turns"] == 0 and any(big[:100] in m["content"] for m in msgs)
    tight = AnalysisEndpoint(name="t", model="m", context_tokens=4096, output_chars_to_model=6000)
    msgs, c = build_messages(sess, tight, "SYSTEM")
    assert c["elided_turns"] >= 1
    assert all(m["role"] != msgs[i + 1]["role"] for i, m in enumerate(msgs[:-1]))   # alternation kept
    assert msgs[-1]["content"].endswith("q5")
    tiny = AnalysisEndpoint(name="u", model="m", context_tokens=4096)
    sess2 = {"turns": [dict(t, answer="y" * 6000) for t in turns[:-1]] + [turns[-1]]}
    msgs, c = build_messages(sess2, tiny, "SYSTEM")
    assert c["dropped_turns"] >= 1 and "were dropped from the context" in msgs[1]["content"]
    nosys = AnalysisEndpoint(name="g", model="m", system_role=False)
    msgs, _ = build_messages(sess, nosys, "SYSTEM")
    assert msgs[0]["role"] == "user" and msgs[0]["content"].startswith("SYSTEM")


def test_exports(env):
    import nbformat
    client = FakeClient(["```python\nae.show(px.line(x=[1, 2], y=[3, 4], title='t'))\nprint(3.5)\n```",
                         "It is 3.5, maybe 9.99."])
    svc = _service(env, client)
    try:
        sid = svc.new_session("fake")["id"]
        _run(svc, sid, "q")
        name, data, ctype = svc.export(sid, "ipynb")
        nb = nbformat.reads(data.decode(), as_version=4)
        nbformat.validate(nb)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert "import arabic_eval.analysis as ae" in code_cells[0].source
        outs = code_cells[1].outputs
        assert any("application/vnd.plotly.v1+json" in (o.get("data") or {}) for o in outs)
        assert any(o.get("text") == "3.5\n" for o in outs)
        md = svc.export(sid, "md")[1].decode()
        assert "## Q1. q" in md and "9.99" in md and "Numbers no output of the session supports: 9.99" in md
        with pytest.raises(AnalysisError):
            svc.export(sid, "pdf")
    finally:
        svc.pool.shutdown_all()


def test_preview_shows_the_messages_a_turn_would_send(env):
    svc = _service(env, FakeClient(["x"]))
    sid = svc.new_session("fake", scope={"experiments": ["exp"]})["id"]
    p = svc.preview(sid, "a question")
    assert p["messages"][-1]["content"] == "a question" and p["approx_tokens"] > 1000
    assert "Scope chosen by the user: exp" in p["messages"][0]["content"]


# --------------------------------------------------------------------------
# the number check
# --------------------------------------------------------------------------

def test_answer_numbers_skip_identifiers_dates_sections_and_counts():
    text = ("AraRooPat v5 (`bpe_16k_3phase_v5`) with gemma4_31b on 2026-09-25, report §3.11, 3-shot, "
            "5 cells, 16k vocab, mcq4096: 0.551 vs 62.1 % on 14 114 rows, Δ −0.070, 250 prompts, in 2025.")
    assert [n["text"] for n in answer_numbers(text)] == ["0.551", "62.1 %", "14 114", "−0.070", "250"]
    n = answer_numbers("the gap is −0.070")[0]
    assert n["value"] == -0.07 and n["decimals"] == 3


def test_check_numbers_rounding_percent_and_sign():
    ev = ["acc 0.551297 0.62144 n 14114 delta -0.0701"]
    ok = check_numbers("0.55, 0.551, 55.1 %, 62 %, 14 114, −0.07 and 0.07 lower, 7.01 points", ev)
    assert ok["unverified"] == [] and ok["checked"] == 8
    bad = check_numbers("0.56 and 55.2 % and 14 115", ev)
    assert [u["text"] for u in bad["unverified"]] == ["0.56", "55.2 %", "14 115"]
    assert check_numbers("value 1.25", [], extra_numbers=[1.25])["unverified"] == []


def test_plotly_numbers_reads_typed_arrays(tmp_path):
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(x=["a", "b"], y=np.array([2.43, 2.13])))
    p = tmp_path / "f.json"
    p.write_text(fig.to_json())
    assert sorted(plotly_numbers(p)) == [2.13, 2.43]
    raw = {"data": [{"y": {"dtype": "f8", "bdata": base64.b64encode(np.array([1.5, 2.5]).tobytes()).decode()}}]}
    p.write_text(json.dumps(raw))
    assert plotly_numbers(p) == [1.5, 2.5]


# --------------------------------------------------------------------------
# the native protocol (function calling: the local Gemma endpoint)
# --------------------------------------------------------------------------

def _native_env(env):
    (env / "configs" / "analysis" / "native.yaml").write_text(
        "analysis:\n  name: native\n  model: m\n  api_key_env: null\n  tool_protocol: native\n  max_steps: 3\n"
        "  step_timeout_sec: 20\n  context_tokens: 60000\n")


def test_native_tool_calls_run_and_come_back_as_tool_messages(env):
    _native_env(env)
    client = FakeClient([{"content": "", "tool_calls": ["print(6 * 7)"]}, "The answer is 42."])
    svc = _service(env, client)
    try:
        sid = svc.new_session("native")["id"]
        ev = _run(svc, sid, "six times seven?")
        ex = next(e for e in ev if e["type"] == "exec")
        assert ex["ok"] and ex["stdout"] == "42\n"
        assert next(e for e in ev if e["type"] == "answer")["text"] == "The answer is 42."
        second = client.calls[1]
        call_msg = next(m for m in second if m.get("tool_calls"))
        assert call_msg["role"] == "assistant" and call_msg["tool_calls"][0]["function"]["name"] == "run_python"
        tool_msg = second[-1]
        assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == call_msg["tool_calls"][0]["id"]
        assert tool_msg["content"].startswith("[output of step 1 — ok")
        system = client.calls[0][0]["content"]
        assert "call the `run_python` tool" in system and "fenced block tagged" not in system
        t = svc.store.load(sid)["turns"][0]
        assert t["steps"][0]["tool_calls"][0]["name"] == "run_python" and t["steps"][0]["code"] == "print(6 * 7)"
    finally:
        svc.pool.shutdown_all()


def test_native_endpoint_still_runs_a_fenced_block_and_reports_an_empty_call(env):
    _native_env(env)
    client = FakeClient([{"content": "", "tool_calls": [""]}, "```python\nprint(2 + 2)\n```", "It is 4."])
    svc = _service(env, client)
    try:
        sid = svc.new_session("native")["id"]
        ev = _run(svc, sid, "two plus two?")
        assert client.calls[1][-1]["role"] == "tool" and "carried no code" in client.calls[1][-1]["content"]
        ex = next(e for e in ev if e["type"] == "exec")
        assert ex["stdout"] == "4\n" and client.calls[2][-1]["role"] == "user"       # a fenced step: output as user text
    finally:
        svc.pool.shutdown_all()


def test_tool_chat_client_assembles_streamed_tool_calls(monkeypatch):
    from arabic_eval.tools.analysis_agent import RUN_PYTHON_TOOL, ToolChatClient, tool_call_code
    ep = AnalysisEndpoint(name="n", model="m", api_key_env=None, tool_protocol="native")
    client = ToolChatClient(ep.assistant_config())
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "Let me run it."}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "run_python", "arguments": '{"co'}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'de": "print(1)"}'}}]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
    ]

    class Resp:
        def __iter__(self):
            for c in chunks:
                yield f"data: {json.dumps(c)}\n".encode()
            yield b"data: [DONE]\n"
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    sent = {}
    monkeypatch.setattr(client, "_request", lambda body: sent.setdefault("body", body) and Resp())
    assert "".join(client.stream([{"role": "user", "content": "x"}])) == "Let me run it."
    assert client.last_tool_calls == [{"id": "call_a", "name": "run_python", "arguments": '{"code": "print(1)"}'}]
    assert client.last_finish_reason == "tool_calls" and client.last_usage["total_tokens"] == 8
    assert sent["body"]["tools"] == [RUN_PYTHON_TOOL] and sent["body"]["tool_choice"] == "auto"
    assert tool_call_code(client.last_tool_calls[0]) == "print(1)"
    assert tool_call_code({"arguments": "print(2)"}) == "print(2)"                     # a lenient parser's bare string


def test_the_worker_lets_models_import_ae(env):
    client = FakeClient(["```python\nimport ae\nprint(type(ae.catalog).__name__)\n```", "It is a function."])
    svc = _service(env, client)
    try:
        sid = svc.new_session("fake")["id"]
        ev = _run(svc, sid, "q")
        ex = next(e for e in ev if e["type"] == "exec")
        assert ex["ok"] and ex["stdout"] == "function\n"
    finally:
        svc.pool.shutdown_all()
