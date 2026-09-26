"""The Analysis tab of the experiment console: an LLM that answers questions about the
experiments by running Python against their results.

A turn: the user asks; the model replies with prose and (usually) one fenced
``python`` block; the console runs it in the session's sandboxed worker
(``analysis_kernel``, where ``ae`` = ``arabic_eval.analysis`` is preloaded) and
sends the output back; the model continues — more code, or a reply without code,
which is the answer. The answer's numbers are checked against everything the
session printed (``analysis_provenance``) and the ones no output supports are
flagged. Figures and tables the code passes to ``ae.show`` go to the page.

Protocol with the model: free text plus a fenced ``python`` block — the one tool
is Python, identical for every backend (OpenAI-compatible API or a local
``vllm serve``), no function-calling parser involved.

Endpoints are ``configs/analysis/*.yaml`` (``AnalysisEndpoint``); the API key can
come from the console server's environment or be typed into the tab
(``KeyStore``: server memory only, never written anywhere, never sent back).
Sessions live under ``outputs/analysis/<id>/`` (``session.json`` + the worker's
``artifacts/``) and export as a Jupyter notebook or Markdown.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import yaml

from arabic_eval.tools.analysis_kernel import KernelPool, sandbox_status
from arabic_eval.tools.analysis_provenance import check_numbers, plotly_numbers
from arabic_eval.tools.config_assistant import AssistantConfig, AssistantError, ChatClient

log = logging.getLogger("analysis.agent")

REPO_ROOT = Path(__file__).resolve().parents[3]
_SESSION_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
_PY_BLOCK = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.S | re.I)
#: A reply that only announces work ("I'll compute …", "Let me check:") — not an answer.
_INTENT = re.compile(r"^\s*(?:I['’]ll|I will|I am going to|I'm going to|Let me|Let's|First,?|Next,?|Now,? I)\b|:\s*$", re.I)
_OPEN_PY = re.compile(r"```(?:python|py)[ \t]*\n(?!.*```)", re.S | re.I)


class AnalysisError(ValueError):
    """User-facing failure of the Analysis tab (bad request, endpoint not ready, …)."""


class SessionBusy(AnalysisError):
    """A turn is already running in this session."""


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------

@dataclass
class AnalysisEndpoint:
    """One model the tab can talk to (``configs/analysis/<name>.yaml``, top-level key ``analysis``)."""
    name: str
    model: str
    kind: str = "openai"                     # openai (any OpenAI-compatible API) | local_vllm (the console starts it)
    base_url: Optional[str] = None           # null = https://api.openai.com/v1
    api_key_env: Optional[str] = "OPENAI_API_KEY"   # null = no key (a local server)
    temperature: Optional[float] = None      # null = omitted (reasoning models accept only the default)
    max_tokens: int = 16000                  # per reply; reasoning tokens count against it
    timeout_sec: float = 300.0
    extra_body: Dict[str, Any] = field(default_factory=dict)
    reasoning_efforts: List[str] = field(default_factory=list)   # choices the tab offers (extra_body.reasoning_effort)
    context_tokens: int = 128000             # what the prompt + history may use (the reply is reserved on top)
    chars_per_token: float = 3.0             # for the budget (3.0 measured on Gemma 4 for this mix of English / Arabic / code)
    max_steps: int = 12                      # code steps per turn before the model is asked to answer
    step_timeout_sec: float = 120.0
    output_chars_to_model: int = 6000        # per step output sent back to the model
    api_reference: str = "full"              # full | short (docstrings of ae in the system prompt)
    system_role: bool = True                 # false: the system prompt is folded into the first user message
    tool_protocol: str = "fenced"            # fenced: code in a ```python block | native: the run_python tool (function calling)
    serve: Dict[str, Any] = field(default_factory=dict)          # local_vllm: how the console starts the server
    description: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name or ""):
            raise ValueError(f"endpoint name {self.name!r} must be a file-safe identifier")
        if self.kind not in ("openai", "local_vllm"):
            raise ValueError(f"endpoint {self.name!r}: kind must be openai | local_vllm")
        if self.tool_protocol not in ("fenced", "native"):
            raise ValueError(f"endpoint {self.name!r}: tool_protocol must be fenced | native")
        if self.api_reference not in ("full", "short"):
            raise ValueError(f"endpoint {self.name!r}: api_reference must be full | short")
        if self.max_steps < 1 or self.max_tokens < 256 or self.context_tokens < 4096:
            raise ValueError(f"endpoint {self.name!r}: max_steps ≥ 1, max_tokens ≥ 256, context_tokens ≥ 4096")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AnalysisEndpoint":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        d = raw.get("analysis", raw)
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"{path}: unknown keys {unknown}; known: {sorted(known)}")
        return cls(**d)

    @property
    def url(self) -> str:
        return (self.base_url or "https://api.openai.com/v1").rstrip("/")

    def assistant_config(self, *, model: Optional[str] = None, reasoning_effort: Optional[str] = None) -> AssistantConfig:
        extra = dict(self.extra_body)
        if reasoning_effort:
            if self.reasoning_efforts and reasoning_effort not in self.reasoning_efforts:
                raise AnalysisError(f"reasoning_effort must be one of {self.reasoning_efforts}")
            extra["reasoning_effort"] = reasoning_effort
        return AssistantConfig(name=self.name, model=model or self.model, base_url=self.base_url, api_key_env=None,
                               temperature=self.temperature, max_tokens=self.max_tokens, timeout_sec=self.timeout_sec,
                               history_messages=0, extra_body=extra, max_retries=2)


class KeyStore:
    """API keys typed into the tab, by environment-variable name, in server memory only.

    A key set here wins over the server's environment; nothing is written to disk, logged
    or returned to the page (the status shows the last four characters)."""

    def __init__(self) -> None:
        self._keys: Dict[str, str] = {}
        self._lock = threading.Lock()

    def set(self, env_name: str, key: Optional[str]) -> None:
        if not env_name or not re.fullmatch(r"[A-Z0-9_]+", env_name):
            raise AnalysisError("key name must be an environment-variable name such as OPENAI_API_KEY")
        with self._lock:
            if key is None or not str(key).strip():
                self._keys.pop(env_name, None)
                return
            key = str(key).strip()
            if len(key) < 20 or any(c.isspace() for c in key):
                raise AnalysisError("that does not look like an API key")
            self._keys[env_name] = key

    def get(self, env_name: Optional[str]) -> Optional[str]:
        if not env_name:
            return None
        with self._lock:
            return self._keys.get(env_name) or os.environ.get(env_name) or None

    def status(self, env_name: Optional[str]) -> Dict[str, Any]:
        if not env_name:
            return {"needed": False}
        with self._lock:
            mem = self._keys.get(env_name)
        env = os.environ.get(env_name)
        key = mem or env
        return {"needed": True, "name": env_name, "set": bool(key),
                "source": "tab" if mem else ("environment" if env else None),
                "hint": ("…" + key[-4:]) if key else None}


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


class SessionStore:
    """``outputs/analysis/<id>/session.json`` (+ the worker's ``artifacts/``)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def dir(self, sid: str) -> Path:
        if not _SESSION_ID.match(sid or ""):
            raise AnalysisError(f"bad session id {sid!r}")
        return self.root / sid

    def new(self, *, endpoint: str, model: str, scope: Optional[Dict[str, Any]] = None,
            options: Optional[Dict[str, Any]] = None, title: Optional[str] = None) -> Dict[str, Any]:
        sid = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        sess = {"id": sid, "title": title or "", "created": _now(), "updated": _now(), "endpoint": endpoint,
                "model": model, "options": options or {}, "scope": scope or {}, "next_step": 1, "turns": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        self.dir(sid).mkdir(parents=True, exist_ok=False)
        self.save(sess)
        return sess

    def save(self, sess: Dict[str, Any]) -> None:
        sess["updated"] = _now()
        d = self.dir(sess["id"])
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "session.json.tmp"
        tmp.write_text(json.dumps(sess, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, d / "session.json")

    def load(self, sid: str) -> Dict[str, Any]:
        p = self.dir(sid) / "session.json"
        if not p.exists():
            raise AnalysisError(f"no analysis session {sid!r}")
        return json.loads(p.read_text(encoding="utf-8"))

    def list(self) -> List[Dict[str, Any]]:
        out = []
        if not self.root.is_dir():
            return out
        for p in sorted(self.root.glob("*/session.json"), reverse=True):
            try:
                s = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"id": s["id"], "title": s.get("title") or "(untitled)", "created": s.get("created"),
                        "updated": s.get("updated"), "endpoint": s.get("endpoint"), "model": s.get("model"),
                        "turns": sum(1 for t in s.get("turns", []) if t.get("kind", "ask") == "ask")})
        return out

    def artifact(self, sid: str, name: str) -> Path:
        base = (self.dir(sid) / "artifacts").resolve()
        p = (base / name).resolve()
        if base not in p.parents or not p.is_file():
            raise AnalysisError(f"no artifact {name!r} in session {sid}")
        return p


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------

PRIMER = """You are the data-analysis assistant of the experiment console of the Arabic Tokenizers Evaluation Platform. The user asks questions about finished experiments; you answer by running Python against their results, then you state the answer.

## How you work
{PROTOCOL}
- Lead your final answer with the finding, then the numbers that support it, then the caveats that apply. Short markdown (no heading larger than ###); name the cells you compared.
- Never write what a step's output will be or "assume" an output: the console runs the code and sends you the real output.
- Already defined in the session (no import needed): `ae` (analysis helpers, reference below), `pd`, `np`, `px` (plotly.express), `go` (plotly.graph_objects), `plt`, `json`, `math`, `re`, `Path`. No network, no pip; the filesystem is read-only except the current folder.
- Print compactly: select columns, `.round(3)`, `.head(20)`. Long outputs are cut before you see them. Never dump prompts or generations — print two or three examples when the user needs to see some.
- Charts: build them with plotly (`px`) and pass them to `ae.show(fig)`; always a title and axis titles; ≤ 3 series on a scatter, more series only with direct labels or a table next to them. Label only the few points your answer is about (hover shows the rest) — a text label on every point overlaps; leave colours to the default palette. Plotly renders Arabic correctly, matplotlib does not. Tables the user should read: `ae.show(df)`. What you show is displayed to the user; you receive a short summary of it. Mention in your answer what you showed.
- Every number in your final answer must appear in an output of this session (numbers that do not are highlighted to the user as unverified). If the data cannot answer the question, say so; never guess.
- When a step fails, read the error, fix it and continue; do not repeat a failing call. `ae.catalog()` resolves cell names; a bare folder name works when unique.

## The data
- A cell = one tokenizer × model × training run, or an eval-only re-run of a checkpoint; a folder under outputs/experiments whose id is its path there (qwen_native_vs_araroopat/araroopat_3phase_v5_distill). `ae.catalog()` lists the cells; its `rules` column shows the eval settings of each dump.
- MCQ benchmarks (acva, alghafa, arabic_exam, culture_arabic_mmlu): log-likelihood multiple choice; accuracy under two normalizations, char and PMI. Free-form (freeform_cidar): greedy answers to 250 held-out CIDAR instructions, judged 1–5 by LLM judges (gemma4_31b is the primary judge, gpt56_terra a second one), plus chrF / BERTScore (secondary) and loop / EOS / length rates.

## Comparison rules — the report follows them, so do you
- Compare cells paired on the same items: `ae.mcq_compare` / `ae.mcq_paired` (MCQ rows no cell truncated: the "intersection"), `ae.judge_compare` / `ae.freeform_paired` (same prompts). Give differences with the 95 % bootstrap CI (10 000 resamples, seed 0) and, for MCQ, the McNemar counts. A difference whose CI contains 0 is "not separable", not a win.
- Compare only cells evaluated under the same rules (catalog `rules`): free-form on the same prompt set (test = the 250 held-out prompts; dev = the teacher bake-off's cidar-dev prompts, easier) with the same character budget and decoding (test 2400ch greedy is current; 1200ch and rp1.2 cells are other rules), MCQ under the same max_length / few-shot / label rotation / row subset. Say so when they differ.
- In all_metrics.json, `accuracy` follows each run's primary normalization (PMI in the v5 cells): use acc_char / acc_pmi (`ae.metrics`) or the dumps.
- ACVA and Alghafa's four fixed-label sub-configs (true/false, sentiment) mostly measure a label prior (one label takes most predictions), not the tokenizer: quote them with that caveat. Alghafa's in-scope group is choice_text_msa; meta_ar_dialects is out of scope.
- On the letter-scored tasks (arabic_exam, culture_arabic_mmlu) a letter prior is part of every difference; AraRooPat scores a letter as [LIT_BEGIN] [CHAR_x] [LIT_END].
- Answer NLL per character = Σ NLL ÷ Σ reference characters; per-token losses do not compare across tokenizers.
- Treat every arm alike: a rule, filter or correction applied to one cell applies to all compared cells.
- For what a cell is and why it was run, `ae.report_search("...")` searches the campaign report (cite "report §x"); compute the numbers from the data anyway."""


PROTOCOL = {
    "fenced": (
        "- To run code, write ONE fenced block tagged `python` in your reply. The console executes it in a persistent "
        "Python session (variables survive between steps and turns) and sends you its output; then you continue with more "
        "code or with the answer. Keep each step focused; several small steps beat one long one. Do not use function-call "
        "syntax (`call:`); there is no tool other than the python block.\n"
        "- Never announce work without doing it: a reply that says what you will compute must contain the python block "
        "that computes it, in the same reply.\n"
        "- A reply WITHOUT a python block is your final answer."),
    "native": (
        "- To run code, call the `run_python` tool with the code. The console executes it in a persistent Python session "
        "(variables survive between steps and turns) and returns its output as the tool result; then you continue with "
        "another call or with the answer. Keep each call focused; several small steps beat one long one.\n"
        "- Never announce work without doing it: when you need data, call `run_python` in the same reply.\n"
        "- A reply WITHOUT a tool call is your final answer."),
}

#: The one tool of the native protocol.
RUN_PYTHON_TOOL: Dict[str, Any] = {"type": "function", "function": {
    "name": "run_python",
    "description": ("Run Python in the session's persistent worker (ae, pd, np, px, go, plt already defined) and return "
                    "its stdout, the value of the last expression, what ae.show displayed, or the error."),
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to run"}},
                   "required": ["code"]}}}


class ToolChatClient(ChatClient):
    """``ChatClient`` for the native protocol: sends ``RUN_PYTHON_TOOL`` and collects the streamed tool calls
    (``last_tool_calls``: ``[{id, name, arguments}]``) next to the content deltas it yields."""

    def __init__(self, cfg: AssistantConfig, api_key: Optional[str] = None) -> None:
        super().__init__(cfg, api_key=api_key)
        self.last_tool_calls: List[Dict[str, Any]] = []

    def body(self, messages, stream: bool) -> Dict[str, Any]:
        b = super().body(messages, stream)
        b["tools"] = [RUN_PYTHON_TOOL]
        b["tool_choice"] = "auto"
        return b

    def stream(self, messages) -> Iterator[str]:
        from arabic_eval.tools.config_assistant import parse_sse
        self.last_usage = None
        self.last_finish_reason = None
        calls: Dict[int, Dict[str, Any]] = {}
        resp = self._request(self.body(messages, stream=True))
        with resp:
            for ev in parse_sse(resp):
                if ev.get("usage"):
                    self.last_usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    if ch.get("finish_reason"):
                        self.last_finish_reason = ch["finish_reason"]
                    d = ch.get("delta") or {}
                    for tc in d.get("tool_calls") or []:
                        c = calls.setdefault(int(tc.get("index") or 0), {"id": None, "name": "", "arguments": ""})
                        if tc.get("id"):
                            c["id"] = tc["id"]
                        f = tc.get("function") or {}
                        c["name"] += f.get("name") or ""
                        c["arguments"] += f.get("arguments") or ""
                    if d.get("content"):
                        yield d["content"]
        self.last_tool_calls = [calls[k] for k in sorted(calls)]


def tool_call_code(call: Dict[str, Any]) -> Optional[str]:
    """The code of a ``run_python`` call (JSON arguments, or a bare string from a lenient parser)."""
    args = call.get("arguments") or ""
    try:
        parsed = json.loads(args) if args.strip() else {}
    except ValueError:
        return args.strip() or None
    if isinstance(parsed, dict):
        code = parsed.get("code")
        return str(code) if code else None
    return str(parsed) if parsed else None


def scope_text(scope: Optional[Dict[str, Any]], budget_chars: int) -> str:
    """The experiments with their cell counts, then the cells of the scope (or of everything) one line each."""
    from arabic_eval.analysis import inventory
    cells = inventory.cells()
    exps: Dict[str, int] = {}
    for c in cells:
        exps[c.id.split("/", 1)[0]] = exps.get(c.id.split("/", 1)[0], 0) + 1
    want = [e for e in ((scope or {}).get("experiments") or []) if e]
    lines = ["Experiments (cells): " + ", ".join(f"{e} ({n})" for e, n in sorted(exps.items()))]
    chosen = [c for c in cells if not want or any(c.id == e or c.id.startswith(e.rstrip("/") + "/") for e in want)]
    lines.append(("Scope chosen by the user: " + ", ".join(want)) if want else "Scope: all experiments.")
    lines.append("Cells (id | tokenizer | trained phases | loaded from | judges | evals with their rules):")
    used = sum(len(l) + 1 for l in lines)
    shown = 0
    for c in chosen:
        f = inventory.cell_facts(c)
        groups: Dict[str, List[str]] = {}
        for t, v in f["tasks"].items():
            if v.get("rule"):
                groups.setdefault(v["rule"], []).append(t)       # task names exactly as ae.catalog()'s rules column
        rules = "; ".join(f"{','.join(ts)}: {r}" for r, ts in groups.items()) or "-"
        parent = f"{f['parent_cell'].rsplit('/', 1)[-1]}:{f['parent_phase']}" if f["parent_cell"] else "-"
        line = f"- {c.id} | {f['tokenizer']} | {f['trained'] or '-'} | {parent} | {','.join(f['judges']) or '-'} | {rules}"
        if used + len(line) + 1 > budget_chars:
            lines.append(f"- … {len(chosen) - shown} more cells: call ae.catalog() for the full list")
            break
        lines.append(line)
        used += len(line) + 1
        shown += 1
    return "\n".join(lines)


def system_prompt(endpoint: AnalysisEndpoint, scope: Optional[Dict[str, Any]]) -> str:
    import arabic_eval.analysis as ae
    budget = int(endpoint.context_tokens * endpoint.chars_per_token)
    ref = ae.api_reference(full=endpoint.api_reference == "full")
    primer = PRIMER.replace("{PROTOCOL}", PROTOCOL[endpoint.tool_protocol])
    head = f"{primer}\n\n## The ae helpers\n{ref}\n\n## What is on disk\n"
    return head + scope_text(scope, max(2000, min(12000, budget // 6)))


def _trim_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = max(200, limit - 80)
    return text[: keep * 2 // 3] + f"\n… [{len(text) - keep} characters cut] …\n" + text[-(keep // 3):]


def exec_feedback(step: Dict[str, Any], limit: int) -> str:
    """What the model is told about one executed step."""
    ex = step.get("exec") or {}
    if ex.get("declined"):
        return f"[step {step['n']}: the user declined to run this code]"
    if ex.get("skipped"):
        return f"[step {step['n']}: not run — {ex['skipped']}]"
    status = "ok" if ex.get("ok") else "failed"
    parts = [f"[output of step {step['n']} — {status} in {ex.get('elapsed', 0):.1f} s]"]
    if ex.get("restarted"):
        parts.append("(The Python session was restarted before this step: variables from earlier steps are gone.)")
    if ex.get("stdout"):
        parts.append("stdout:\n" + ex["stdout"].rstrip())
    if ex.get("result"):
        parts.append("value:\n" + ex["result"])
    shown = ex.get("displays") or []
    if shown:
        lines = []
        for d in shown:
            if d.get("kind") == "table":
                lines.append(f"- table{' ' + repr(d['title']) if d.get('title') else ''} ({d.get('n_rows')} rows × {d.get('n_cols')} cols):\n{d.get('text', '')}")
            else:
                lines.append(f"- {d.get('text') or d.get('kind')}")
        parts.append("shown to the user:\n" + "\n".join(lines))
    if ex.get("stderr") and not ex.get("ok"):
        parts.append("stderr:\n" + ex["stderr"].rstrip()[-1500:])
    err = ex.get("error")
    if err:
        parts.append(f"error: {err.get('ename')}: {err.get('evalue')}" + (f"\n{err['traceback']}" if err.get("traceback") else ""))
    if not ex.get("stdout") and not ex.get("result") and not shown and not err:
        parts.append("(no output)")
    return _trim_middle("\n".join(parts), limit)


def _code_of(reply: str) -> Tuple[Optional[str], bool]:
    """(the python code of a reply — every python block joined —, whether a block was left unclosed)."""
    blocks = [b.strip("\n") for b in _PY_BLOCK.findall(reply)]
    unclosed = bool(_OPEN_PY.search(reply)) and reply.count("```") % 2 == 1
    return ("\n\n".join(b for b in blocks if b.strip()) or None), unclosed


def prose_of(reply: str) -> str:
    return _PY_BLOCK.sub("", reply).strip()


def build_messages(sess: Dict[str, Any], endpoint: AnalysisEndpoint, system: str) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """The chat messages for the next model call, compacted to the endpoint's context budget.

    Oldest first to go: the step outputs of earlier turns (the question and the answer stay),
    then whole earlier turns; within the current turn the outputs of earlier steps are
    shortened. Returns the messages and what was compacted."""
    limit = endpoint.output_chars_to_model
    budget = int(endpoint.context_tokens * endpoint.chars_per_token)
    turns = sess.get("turns") or []

    def render(turn: Dict[str, Any], mode: str, step_limit: int) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if turn.get("kind") == "code":           # a cell the user ran by hand
            fb = exec_feedback({"n": turn["n"], "exec": turn.get("exec")}, step_limit)
            msgs.append({"role": "user", "content": f"(I ran this code myself in the session)\n```python\n{turn['code']}\n```\n{fb}"})
            return msgs
        msgs.append({"role": "user", "content": turn["user"] + (f"\n\n{turn['note']}" if turn.get("note") else "")})
        if mode == "full":
            for st in turn.get("steps") or []:
                calls = st.get("tool_calls") or []
                if calls:                        # native protocol: assistant(tool_calls) → one tool message per call
                    msgs.append({"role": "assistant", "content": st.get("reply") or "",
                                 "tool_calls": [{"id": c["id"], "type": "function",
                                                 "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
                    fb = exec_feedback(st, step_limit) if st.get("exec") is not None else (st.get("feedback") or "(not run)")
                    for i, c in enumerate(calls):
                        msgs.append({"role": "tool", "tool_call_id": c["id"],
                                     "content": fb if i == 0 else "(run together with the first call of this reply; see its output)"})
                    continue
                msgs.append({"role": "assistant", "content": st["reply"]})
                if st.get("exec") is not None:
                    msgs.append({"role": "user", "content": exec_feedback(st, step_limit)})
                elif st.get("feedback"):
                    msgs.append({"role": "user", "content": st["feedback"]})
        else:
            n_steps = len(turn.get("steps") or [])
            if n_steps:
                msgs.append({"role": "assistant", "content": f"(I ran {n_steps} code step(s) for this question; their outputs are elided.)"})
                msgs.append({"role": "user", "content": "(continue)"})
        if turn.get("answer") and turn.get("status") != "running":
            msgs.append({"role": "assistant", "content": turn["answer"]})
        return msgs

    current = turns[-1] if turns and turns[-1].get("status") == "running" else None
    earlier = turns[:-1] if current else turns
    modes = ["full"] * len(earlier)
    step_limit = limit
    compacted = {"elided_turns": 0, "dropped_turns": 0, "short_steps": False}

    def assemble() -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        for t, m in zip(earlier, modes):
            if m != "drop":
                msgs.extend(render(t, m, limit))
        if current is not None:
            msgs.extend(render(current, "full", step_limit))
        return msgs

    def size(msgs: List[Dict[str, Any]]) -> int:
        return len(system) + sum(len(m.get("content") or "") + 16
                                 + sum(len(c["function"]["arguments"]) for c in m.get("tool_calls") or []) for m in msgs)

    msgs = assemble()
    i = 0
    while size(msgs) > budget and i < len(earlier):                 # 1. elide earlier turns' steps
        if earlier[i].get("kind") != "code" and (earlier[i].get("steps") or []):
            modes[i] = "elided"
            compacted["elided_turns"] += 1
            msgs = assemble()
        i += 1
    i = 0
    while size(msgs) > budget and i < len(earlier):                 # 2. drop earlier turns
        modes[i] = "drop"
        compacted["dropped_turns"] += 1
        msgs = assemble()
        i += 1
    if size(msgs) > budget and current is not None:                 # 3. shorten the current turn's outputs
        step_limit = max(600, limit // 4)
        compacted["short_steps"] = True
        msgs = assemble()
    if compacted["dropped_turns"]:
        msgs.insert(0, {"role": "user", "content": f"({compacted['dropped_turns']} earlier question(s) of this session were dropped from the context; the Python variables they created still exist.)"})
        msgs.insert(1, {"role": "assistant", "content": "Understood."})
    # merge consecutive same-role messages (some chat templates require strict alternation)
    merged: List[Dict[str, str]] = []
    for m in msgs:
        plain = m["role"] in ("user", "assistant") and "tool_calls" not in m
        if merged and plain and merged[-1]["role"] == m["role"] and "tool_calls" not in merged[-1]:
            merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + "\n\n" + m["content"]}
        else:
            merged.append(dict(m))
    if endpoint.system_role:
        out = [{"role": "system", "content": system}] + merged
    else:
        first = merged[0] if merged else {"role": "user", "content": ""}
        out = [{"role": "user", "content": system + "\n\n---\n\n" + first["content"]}] + merged[1:]
    compacted["approx_tokens"] = int(size(msgs) / endpoint.chars_per_token)
    return out, compacted


# --------------------------------------------------------------------------
# evidence for the number check
# --------------------------------------------------------------------------

def _table_text(d: Dict[str, Any]) -> str:
    rows = d.get("rows") or []
    return "\n".join(" ".join("" if v is None else str(v) for v in r) for r in rows)


def session_evidence(sess: Dict[str, Any], store: SessionStore, system: str) -> Tuple[List[str], List[float]]:
    texts = [system]
    extra: List[float] = []
    art = store.dir(sess["id"]) / "artifacts"
    for t in sess.get("turns") or []:
        texts.append(t.get("user") or "")
        texts.append(t.get("code") or "")
        steps = t.get("steps") or []
        execs = [s.get("exec") for s in steps] + ([t.get("exec")] if t.get("kind") == "code" else [])
        texts.extend(s.get("code") or "" for s in steps)
        for ex in execs:
            if not ex:
                continue
            texts += [ex.get("stdout") or "", ex.get("result") or ""]
            for d in ex.get("displays") or []:
                texts.append(d.get("text") or "")
                if d.get("kind") == "table":
                    texts.append(_table_text(d))
                if d.get("kind") == "plotly" and d.get("path"):
                    extra.extend(plotly_numbers(art / d["path"]))
    return texts, extra


# --------------------------------------------------------------------------
# the service
# --------------------------------------------------------------------------

ClientFactory = Callable[[AnalysisEndpoint, Optional[str], Dict[str, Any]], Any]


def default_client_factory(ep: AnalysisEndpoint, key: Optional[str], options: Dict[str, Any]) -> ChatClient:
    cfg = ep.assistant_config(model=options.get("model") or None, reasoning_effort=options.get("reasoning_effort") or None)
    return ToolChatClient(cfg, api_key=key) if ep.tool_protocol == "native" else ChatClient(cfg, api_key=key)


@dataclass
class _Runtime:
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop: threading.Event = field(default_factory=threading.Event)
    approval: Optional[Dict[str, Any]] = None


class AnalysisService:
    """Everything the console's Analysis routes need; one instance per server."""

    def __init__(self, repo_root: Path = REPO_ROOT, *, store: Optional[SessionStore] = None,
                 keys: Optional[KeyStore] = None, pool: Optional[KernelPool] = None,
                 client_factory: ClientFactory = default_client_factory,
                 require_approval_unsandboxed: bool = True, approval_timeout_sec: float = 900.0,
                 local_status: Optional[Callable[[AnalysisEndpoint], Dict[str, Any]]] = None,
                 activity_hook: Optional[Callable[[AnalysisEndpoint], None]] = None) -> None:
        self.repo_root = Path(repo_root)
        self.store = store or SessionStore(self.repo_root / "outputs" / "analysis")
        self.keys = keys or KeyStore()
        # ARABIC_EVAL_ANALYSIS_SANDBOX=off|on|auto: force the worker's sandbox (off = every step asks for approval)
        self.pool = pool or KernelPool(sandbox=os.environ.get("ARABIC_EVAL_ANALYSIS_SANDBOX", "auto"))
        self.client_factory = client_factory
        self.require_approval_unsandboxed = require_approval_unsandboxed
        self.approval_timeout_sec = approval_timeout_sec
        self.local_status = local_status
        self.activity_hook = activity_hook        # a question used this endpoint (the local server's idle clock)
        self._rt: Dict[str, _Runtime] = {}
        self._rt_lock = threading.Lock()

    # ---- endpoints -------------------------------------------------------
    def endpoint_dir(self) -> Path:
        """``configs/analysis`` (``ARABIC_EVAL_ANALYSIS_CONFIGS`` points a test console elsewhere)."""
        env = os.environ.get("ARABIC_EVAL_ANALYSIS_CONFIGS")
        return Path(env) if env else self.repo_root / "configs" / "analysis"

    def endpoint(self, name: str) -> AnalysisEndpoint:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name or ""):
            raise AnalysisError(f"bad endpoint name {name!r}")
        for p in sorted(self.endpoint_dir().glob("*.yaml")):
            try:
                ep = AnalysisEndpoint.from_yaml(p)
            except Exception:  # noqa: BLE001 - listed as broken by endpoints()
                continue
            if ep.name == name or p.stem == name:
                return ep
        raise AnalysisError(f"no analysis endpoint {name!r} under configs/analysis/")

    def ready(self, ep: AnalysisEndpoint) -> Tuple[bool, str]:
        if ep.api_key_env and not self.keys.get(ep.api_key_env):
            return False, f"no API key — enter it in the tab or export {ep.api_key_env} before starting the console"
        if ep.kind == "local_vllm":
            st = self.local_status(ep) if self.local_status else _probe_models(ep)
            if not st.get("ready"):
                return False, st.get("why") or st.get("detail") or "the local server is not running — start it from the tab"
        return True, ""

    def endpoints(self) -> Dict[str, Any]:
        out = []
        for p in sorted(self.endpoint_dir().glob("*.yaml")):
            entry: Dict[str, Any] = {"file": str(p.relative_to(self.repo_root)) if p.is_relative_to(self.repo_root) else str(p),
                                     "id": p.stem}
            try:
                ep = AnalysisEndpoint.from_yaml(p)
                ok, why = self.ready(ep)
                entry.update(name=ep.name, model=ep.model, kind=ep.kind, base_url=ep.url, description=ep.description,
                             reasoning_efforts=ep.reasoning_efforts,
                             reasoning_effort=ep.extra_body.get("reasoning_effort"), context_tokens=ep.context_tokens,
                             max_steps=ep.max_steps, key=self.keys.status(ep.api_key_env), ready=ok, why=why)
                if ep.kind == "local_vllm" and self.local_status:
                    entry["server"] = self.local_status(ep)
            except Exception as e:  # noqa: BLE001 - a broken YAML is listed, not hidden
                entry.update(name=p.stem, error=f"{type(e).__name__}: {e}", ready=False, why="invalid config")
            out.append(entry)
        sb = dict(sandbox_status())
        if getattr(self.pool, "kernel_kwargs", {}).get("sandbox") == "off":
            sb = {"available": False, "reason": "turned off for this console (ARABIC_EVAL_ANALYSIS_SANDBOX=off): every step asks for approval"}
        return {"endpoints": out, "sandbox": sb}

    # ---- sessions --------------------------------------------------------
    def _runtime(self, sid: str) -> _Runtime:
        with self._rt_lock:
            return self._rt.setdefault(sid, _Runtime())

    def new_session(self, endpoint: str, *, scope: Optional[Dict[str, Any]] = None,
                    options: Optional[Dict[str, Any]] = None, title: Optional[str] = None) -> Dict[str, Any]:
        ep = self.endpoint(endpoint)
        opts = {k: v for k, v in (options or {}).items() if k in ("model", "reasoning_effort") and v}
        return self.store.new(endpoint=ep.name, model=opts.get("model") or ep.model, scope=_clean_scope(scope),
                              options=opts, title=title)

    def session(self, sid: str) -> Dict[str, Any]:
        sess = self.store.load(sid)
        rt = self._rt.get(sid)
        k = self.pool.peek(sid)
        sess["runtime"] = {"busy": bool(rt and rt.lock.locked()), "kernel_alive": bool(k and k.alive),
                           "sandboxed": bool(k and k.sandboxed) if k else None,
                           "awaiting_approval": (rt.approval or {}).get("step") if rt else None}
        return sess

    def update_session(self, sid: str, *, endpoint: Optional[str] = None, options: Optional[Dict[str, Any]] = None,
                       scope: Optional[Dict[str, Any]] = None, title: Optional[str] = None) -> Dict[str, Any]:
        rt = self._runtime(sid)
        if rt.lock.locked():
            raise SessionBusy("a turn is running in this session")
        sess = self.store.load(sid)
        if endpoint:
            ep = self.endpoint(endpoint)
            sess["endpoint"], sess["model"] = ep.name, ep.model
        if options is not None:
            sess["options"] = {k: v for k, v in options.items() if k in ("model", "reasoning_effort") and v}
            if sess["options"].get("model"):
                sess["model"] = sess["options"]["model"]
        if scope is not None:
            sess["scope"] = _clean_scope(scope)
        if title is not None:
            sess["title"] = str(title)[:200]
        self.store.save(sess)
        return sess

    # ---- control ---------------------------------------------------------
    def stop(self, sid: str) -> bool:
        rt = self._runtime(sid)
        if not rt.lock.locked():
            return False
        rt.stop.set()
        if rt.approval:
            rt.approval["decision"] = False
            rt.approval["event"].set()
        k = self.pool.peek(sid)
        if k is not None:
            k.interrupt()
        return True

    def approve(self, sid: str, step: int, approve: bool) -> bool:
        rt = self._runtime(sid)
        pending = rt.approval
        if not pending or pending.get("step") != step:
            raise AnalysisError(f"step {step} is not waiting for approval")
        pending["decision"] = bool(approve)
        pending["event"].set()
        return True

    def close_kernel(self, sid: str) -> None:
        self.pool.close(sid)

    # ---- a turn ----------------------------------------------------------
    def chat(self, sid: str, message: str) -> Iterator[Dict[str, Any]]:
        """Validate eagerly (so the route can answer 4xx), then return the turn's event stream."""
        message = (message or "").strip()
        if not message:
            raise AnalysisError("empty message")
        sess = self.store.load(sid)
        ep = self.endpoint(sess["endpoint"])
        ok, why = self.ready(ep)
        if not ok:
            raise AnalysisError(f"{ep.name}: {why}")
        rt = self._runtime(sid)
        if not rt.lock.acquire(blocking=False):
            raise SessionBusy("a turn is already running in this session")
        try:
            client = self.client_factory(ep, self.keys.get(ep.api_key_env), sess.get("options") or {})
        except Exception:
            rt.lock.release()
            raise
        rt.stop.clear()
        return self._turn(sess, ep, client, message, rt)

    def _turn(self, sess: Dict[str, Any], ep: AnalysisEndpoint, client: Any, message: str,
              rt: _Runtime) -> Iterator[Dict[str, Any]]:
        sid = sess["id"]
        t0 = time.perf_counter()
        turn: Dict[str, Any] = {"n": len(sess["turns"]) + 1, "kind": "ask", "user": message, "started": _now(),
                                "status": "running", "steps": [], "answer": None, "model": sess.get("model"),
                                "endpoint": ep.name, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        if not sess.get("title"):
            sess["title"] = message[:120]
        try:
            kernel = self.pool.get(sid, self.store.dir(sid))
            had_state = any(t.get("steps") or t.get("kind") == "code" for t in sess["turns"])
            if not kernel.alive:
                kernel.start()
                if had_state:
                    turn["note"] = ("(The Python session was restarted since the previous question: variables "
                                    "from earlier steps no longer exist — recompute what you need.)")
            sess["turns"].append(turn)
            self.store.save(sess)
            yield {"type": "turn", "turn": turn["n"], "session": sid, "sandboxed": kernel.sandboxed,
                   "model": sess.get("model"), "endpoint": ep.name}
            system = system_prompt(ep, sess.get("scope"))
            forced_final = False
            while True:
                if rt.stop.is_set():
                    turn["status"] = "stopped"
                    break
                n_code_steps = sum(1 for s in turn["steps"] if s.get("code"))
                if n_code_steps >= ep.max_steps and not forced_final:
                    forced_final = True
                    turn["steps"].append({"n": None, "reply": "", "code": None, "exec": None,
                                          "feedback": f"(Step limit reached: {ep.max_steps} code steps. Give your final "
                                                      "answer now from the outputs above, without code.)"})
                messages, compacted = build_messages(sess, ep, system)
                step_n = sess["next_step"]
                if self.activity_hook is not None and ep.kind == "local_vllm":
                    self.activity_hook(ep)
                yield {"type": "model_start", "step": step_n, "approx_tokens": compacted["approx_tokens"],
                       "compacted": compacted}
                buf: List[str] = []
                stopped = False
                for tok in client.stream(messages):
                    buf.append(tok)
                    yield {"type": "token", "text": tok}
                    if rt.stop.is_set():
                        stopped = True
                        break
                reply = "".join(buf)
                usage = dict(getattr(client, "last_usage", None) or {})
                for k in turn["usage"]:
                    turn["usage"][k] += int(usage.get(k) or 0)
                finish = getattr(client, "last_finish_reason", None)
                if stopped:
                    turn["steps"].append({"n": None, "reply": reply, "code": None, "exec": None, "usage": usage})
                    turn["status"] = "stopped"
                    break
                code, unclosed = _code_of(reply)
                # native protocol: the run_python tool calls of this reply (a fenced block still works as a fallback)
                tool_calls = [dict(c) for c in (getattr(client, "last_tool_calls", None) or [])
                              if (c.get("name") or "run_python") == "run_python"]
                for i, c in enumerate(tool_calls):
                    c["id"] = c.get("id") or f"call_{step_n}_{i}"
                    c["name"] = c.get("name") or "run_python"
                if tool_calls:
                    code = "\n\n".join(x for x in (tool_call_code(c) for c in tool_calls) if x) or None
                    unclosed = False
                if forced_final:
                    code, tool_calls = None, []
                if tool_calls and not code:
                    turn["steps"].append({"n": None, "reply": reply, "code": None, "exec": None, "usage": usage,
                                          "tool_calls": tool_calls,
                                          "feedback": "(The run_python call carried no code: put the code in its `code` argument.)"})
                    continue
                if not reply.strip() and not tool_calls:
                    if turn["steps"] and turn["steps"][-1].get("feedback", "").startswith("(Your reply was empty"):
                        raise AnalysisError("the model returned two empty replies in a row")
                    turn["steps"].append({"n": None, "reply": "(empty reply)", "code": None, "exec": None, "usage": usage,
                                          "feedback": "(Your reply was empty. Continue: run code, or give your final answer.)"})
                    continue
                if unclosed:
                    turn["steps"].append({"n": None, "reply": reply, "code": None, "exec": None, "usage": usage,
                                          "finish_reason": finish,
                                          "feedback": ("(Your reply was cut off before the python block closed"
                                                       + (" — you hit the reply length limit" if finish == "length" else "")
                                                       + ". Write a shorter step.)")})
                    yield {"type": "reply", "step": None, "text": reply, "code": None, "truncated": True}
                    continue
                ran_code = any(st.get("code") for st in turn["steps"])
                if (code is None and not ran_code and not forced_final and len(reply) < 600 and _INTENT.search(reply.strip())
                        and not any(st.get("feedback", "").startswith("(Your reply announced") for st in turn["steps"])):
                    turn["steps"].append({"n": None, "reply": reply, "code": None, "exec": None, "usage": usage,
                                          "feedback": ("(Your reply announced work but contained no code, so nothing "
                                                       "ran. Run the code now — or, if no code is needed, give your "
                                                       "final answer.)")})
                    yield {"type": "reply", "step": None, "text": reply, "code": None, "nudged": True}
                    continue
                if code is None:                                  # the answer
                    evidence, extra = session_evidence(sess, self.store, system)
                    prov = check_numbers(prose_of(reply), evidence, extra)
                    if prov["unverified"] and not forced_final and not turn.get("verify_round"):
                        # Hold a draft whose numbers no output supports: once per turn the model is told which
                        # ones and asked to compute them (or drop them) before anything is shown as the answer.
                        # Measured 2026-09-25: gpt-5.6-terra answered a judge question from nowhere, with no
                        # code and six invented numbers; the flag alone would still have shown that answer.
                        turn["verify_round"] = True
                        nums = ", ".join(u["text"] for u in prov["unverified"][:12])
                        turn["steps"].append({"n": None, "reply": reply, "code": None, "exec": None, "usage": usage,
                                              "draft_answer": True, "provenance": prov,
                                              "feedback": ("(Check before your answer is shown: these numbers of your draft "
                                                           f"appear in no output of this session: {nums}. Every number must "
                                                           "come from an output — run the code that prints or computes them, "
                                                           "then give the final answer; drop any number you cannot compute.)")})
                        yield {"type": "reply", "step": None, "text": reply, "code": None, "draft": True,
                               "unverified": [u["text"] for u in prov["unverified"]]}
                        continue
                    turn["answer"] = reply
                    turn["provenance"] = prov
                    turn["status"] = "ok" if not forced_final else "step_limit"
                    yield {"type": "answer", "text": reply, "provenance": prov}
                    break
                step = {"n": step_n, "reply": reply, "code": code, "exec": None, "usage": usage, "finish_reason": finish}
                if tool_calls:
                    step["tool_calls"] = tool_calls
                sess["next_step"] = step_n + 1
                turn["steps"].append(step)
                self.store.save(sess)
                yield {"type": "reply", "step": step_n, "text": reply, "code": code}
                if not kernel.sandboxed and self.require_approval_unsandboxed:
                    decision = yield from self._await_approval(rt, step_n, code)
                    if not decision:
                        step["exec"] = {"declined": True}
                        yield {"type": "exec", "step": step_n, "declined": True}
                        if rt.stop.is_set():
                            turn["status"] = "stopped"
                            break
                        continue
                yield {"type": "exec_start", "step": step_n}
                res = kernel.execute(code, step=step_n, timeout=ep.step_timeout_sec)
                for d in res.get("displays") or []:
                    if d.get("path"):
                        d["url"] = f"/api/analysis/artifact?session={sid}&name={d['path']}"
                step["exec"] = res
                self.store.save(sess)
                yield {"type": "exec", "step": step_n, **res}
            if turn["status"] == "running":
                turn["status"] = "ok"
        except AssistantError as e:
            turn["status"] = "error"
            turn["error"] = str(e)
            yield {"type": "error", "message": str(e)}
        except AnalysisError as e:
            turn["status"] = "error"
            turn["error"] = str(e)
            yield {"type": "error", "message": str(e)}
        except GeneratorExit:
            turn["status"] = "stopped"
            turn["error"] = "the page disconnected"
            raise
        except Exception as e:  # noqa: BLE001 - reported to the page, logged here
            log.exception("analysis turn failed")
            turn["status"] = "error"
            turn["error"] = f"{type(e).__name__}: {e}"
            yield {"type": "error", "message": turn["error"]}
        finally:
            turn["ended"] = _now()
            turn["elapsed_sec"] = round(time.perf_counter() - t0, 2)
            for k in sess["usage"]:
                sess["usage"][k] += turn["usage"].get(k, 0)
            if turn not in sess["turns"]:
                sess["turns"].append(turn)
            try:
                self.store.save(sess)
            finally:
                rt.approval = None
                rt.stop.clear()
                rt.lock.release()
        yield {"type": "done", "status": turn["status"], "turn": turn["n"], "usage": turn["usage"],
               "session_usage": sess["usage"], "elapsed_sec": turn["elapsed_sec"]}

    def _await_approval(self, rt: _Runtime, step: int, code: str):
        ev = threading.Event()
        rt.approval = {"step": step, "event": ev, "decision": None}
        yield {"type": "approval", "step": step, "code": code,
               "reason": "this console runs code without the sandbox: approve each step before it runs"}
        deadline = time.monotonic() + self.approval_timeout_sec
        while not ev.wait(0.5):
            if rt.stop.is_set() or time.monotonic() > deadline:
                break
        decision = bool(rt.approval and rt.approval.get("decision"))
        rt.approval = None
        return decision

    # ---- a code cell the user runs by hand --------------------------------
    def exec_code(self, sid: str, code: str) -> Dict[str, Any]:
        code = (code or "").strip("\n")
        if not code.strip():
            raise AnalysisError("empty code")
        rt = self._runtime(sid)
        if not rt.lock.acquire(blocking=False):
            raise SessionBusy("a turn is running in this session")
        try:
            sess = self.store.load(sid)
            ep = self.endpoint(sess["endpoint"])
            kernel = self.pool.get(sid, self.store.dir(sid))
            n = sess["next_step"]
            sess["next_step"] = n + 1
            res = kernel.execute(code, step=n, timeout=ep.step_timeout_sec)
            for d in res.get("displays") or []:
                if d.get("path"):
                    d["url"] = f"/api/analysis/artifact?session={sid}&name={d['path']}"
            sess["turns"].append({"n": len(sess["turns"]) + 1, "kind": "code", "code": code, "exec": res,
                                  "started": _now(), "ended": _now(), "status": "ok" if res.get("ok") else "error"})
            self.store.save(sess)
            return {"step": n, **res}
        finally:
            rt.lock.release()

    # ---- inspection / export ---------------------------------------------
    def preview(self, sid: str, message: str = "") -> Dict[str, Any]:
        sess = self.store.load(sid)
        ep = self.endpoint(sess["endpoint"])
        system = system_prompt(ep, sess.get("scope"))
        if message:
            sess["turns"].append({"n": 0, "kind": "ask", "user": message, "status": "running", "steps": []})
        msgs, compacted = build_messages(sess, ep, system)
        chars = sum(len(m["content"]) for m in msgs)
        return {"messages": msgs, "chars": chars, "approx_tokens": int(chars / ep.chars_per_token),
                "context_tokens": ep.context_tokens, "compacted": compacted, "endpoint": ep.name, "model": sess.get("model")}

    def export(self, sid: str, fmt: str) -> Tuple[str, bytes, str]:
        sess = self.store.load(sid)
        if fmt == "ipynb":
            return f"analysis_{sid}.ipynb", export_notebook(sess, self.store, self.repo_root).encode("utf-8"), "application/x-ipynb+json"
        if fmt == "md":
            return f"analysis_{sid}.md", export_markdown(sess).encode("utf-8"), "text/markdown; charset=utf-8"
        raise AnalysisError("format must be ipynb or md")


def _clean_scope(scope: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    exps = [str(e).strip().strip("/") for e in ((scope or {}).get("experiments") or []) if str(e).strip()]
    return {"experiments": exps} if exps else {}


def _probe_models(ep: AnalysisEndpoint, timeout: float = 2.0) -> Dict[str, Any]:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(urllib.request.Request(ep.url + "/models"), timeout=timeout):
            return {"ready": True}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"ready": False, "why": f"{ep.url} is not reachable ({getattr(e, 'reason', e)})"}


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def _table_html(d: Dict[str, Any]) -> str:
    import html
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.get("columns") or [])
    body = "".join("<tr>" + "".join(f"<td>{html.escape('' if v is None else str(v))}</td>" for v in r) + "</tr>"
                   for r in d.get("rows") or [])
    more = f"<p>{d.get('n_rows')} rows in total</p>" if (d.get("n_rows") or 0) > len(d.get("rows") or []) else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{more}"


def _nb_outputs(ex: Optional[Dict[str, Any]], art: Path) -> List[Dict[str, Any]]:
    import nbformat
    outs: List[Any] = []
    if not ex:
        return outs
    if ex.get("declined"):
        return [nbformat.v4.new_output("stream", name="stderr", text="(not run: declined)\n")]
    if ex.get("stdout"):
        outs.append(nbformat.v4.new_output("stream", name="stdout", text=ex["stdout"]))
    for d in ex.get("displays") or []:
        data: Dict[str, Any] = {"text/plain": d.get("text") or d.get("kind")}
        p = art / d["path"] if d.get("path") else None
        if d.get("kind") == "table":
            data["text/html"] = _table_html(d)
        elif d.get("kind") == "plotly" and p and p.exists():
            data["application/vnd.plotly.v1+json"] = json.loads(p.read_text(encoding="utf-8"))
        elif d.get("kind") == "image" and p and p.exists():
            data["image/png"] = base64.b64encode(p.read_bytes()).decode("ascii")
        elif d.get("kind") == "markdown":
            data["text/markdown"] = d.get("markdown") or ""
        outs.append(nbformat.v4.new_output("display_data", data=data))
    if ex.get("result"):
        outs.append(nbformat.v4.new_output("execute_result", data={"text/plain": ex["result"]}, execution_count=None))
    err = ex.get("error")
    if err:
        outs.append(nbformat.v4.new_output("error", ename=err.get("ename") or "Error", evalue=err.get("evalue") or "",
                                           traceback=(err.get("traceback") or "").splitlines()))
    return outs


def export_notebook(sess: Dict[str, Any], store: SessionStore, repo_root: Path) -> str:
    """The session as a Jupyter notebook: the questions and answers as markdown, every step as a code
    cell with its outputs; a first cell makes it runnable from the repo's main venv."""
    import nbformat
    art = store.dir(sess["id"]) / "artifacts"
    nb = nbformat.v4.new_notebook()
    cells = [nbformat.v4.new_markdown_cell(
        f"# {sess.get('title') or 'Analysis session'}\n\nSession `{sess['id']}` · {sess.get('model')} "
        f"({sess.get('endpoint')}) · created {sess.get('created')}\n\nExported from the experiment console's "
        "Analysis tab. Figures and tables were produced by the code cells below; the answers were written by the model."),
        nbformat.v4.new_code_cell(
            "import sys, json, math, re\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {str(repo_root / 'src')!r})\n"
            "import numpy as np, pandas as pd\nimport plotly.express as px, plotly.graph_objects as go\n"
            "import matplotlib.pyplot as plt\nimport arabic_eval.analysis as ae\nae.install_style()")]
    for t in sess.get("turns") or []:
        if t.get("kind") == "code":
            c = nbformat.v4.new_code_cell(t["code"])
            c.outputs = _nb_outputs(t.get("exec"), art)
            cells += [nbformat.v4.new_markdown_cell("*(code run by hand)*"), c]
            continue
        cells.append(nbformat.v4.new_markdown_cell(f"## Q{t['n']}. {t.get('user', '')}"))
        for st in t.get("steps") or []:
            prose = prose_of(st.get("reply") or "")
            if st.get("code"):
                if prose:
                    cells.append(nbformat.v4.new_markdown_cell(prose))
                c = nbformat.v4.new_code_cell(st["code"])
                c.outputs = _nb_outputs(st.get("exec"), art)
                cells.append(c)
        if t.get("answer"):
            cells.append(nbformat.v4.new_markdown_cell("**Answer.**\n\n" + t["answer"]))
        elif t.get("status") not in (None, "ok"):
            cells.append(nbformat.v4.new_markdown_cell(f"*(turn ended: {t.get('status')}{' — ' + t['error'] if t.get('error') else ''})*"))
    nb.cells = cells
    return nbformat.writes(nb)


def export_markdown(sess: Dict[str, Any]) -> str:
    out = [f"# {sess.get('title') or 'Analysis session'}", "",
           f"Session `{sess['id']}` · {sess.get('model')} ({sess.get('endpoint')}) · created {sess.get('created')}", ""]
    for t in sess.get("turns") or []:
        if t.get("kind") == "code":
            out += ["*(code run by hand)*", "", "```python", t["code"], "```", ""]
            continue
        out += [f"## Q{t['n']}. {t.get('user', '')}", ""]
        for st in t.get("steps") or []:
            if not st.get("code"):
                continue
            prose = prose_of(st.get("reply") or "")
            if prose:
                out += [prose, ""]
            out += ["```python", st["code"], "```", ""]
            ex = st.get("exec") or {}
            txt = (ex.get("stdout") or "") + (("\n" + ex["result"]) if ex.get("result") else "")
            if txt.strip():
                out += ["```text", txt.strip()[:4000], "```", ""]
            for d in ex.get("displays") or []:
                if d.get("path"):
                    out.append(f"- shown: {d.get('kind')} `artifacts/{d['path']}`" + (f" — {d['title']}" if d.get("title") else ""))
            if ex.get("error"):
                out += [f"> error: {ex['error'].get('ename')}: {ex['error'].get('evalue')}", ""]
        if t.get("answer"):
            out += ["**Answer.**", "", t["answer"], ""]
            unv = (t.get("provenance") or {}).get("unverified") or []
            if unv:
                out += [f"*Numbers no output of the session supports: {', '.join(u['text'] for u in unv)}*", ""]
    return "\n".join(out)
