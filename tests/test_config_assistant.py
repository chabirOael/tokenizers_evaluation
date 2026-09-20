"""Tests of the config assistant (``arabic_eval.tools.config_assistant``): endpoint
configs and readiness, the SSE / chat client on a fake ``urlopen``, the context
builder on the real repo, the reply parser, the dotted-path edits, and whole chat
turns with a scripted client — including the repair round and the from-base
create — checked through the real validation.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.config import ExperimentConfig  # noqa: E402
from arabic_eval.tools import config_assistant as ca  # noqa: E402
from arabic_eval.tools.config_hints import FIELD_HINTS, hint_for, norm_path  # noqa: E402
from arabic_eval.tools.experiment_console import ConsoleError, ConsolePaths, schema_bundle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REAL = ConsolePaths(REPO)

MINI_YAML = """\
experiment:
  name: "mini"
  output_dir: "outputs/experiments/mini"
tokenizer:
  type: "native_llama"
  vocab_size: null
training:
  phases:
    embedding_alignment: {enabled: false}
sweep:
  tokenizers:
    - type: "native_llama"
      vocab_sizes: [null]
  tasks:
    - type: "acva"
      params: {num_fewshot: 0}
"""

API_YAML = """\
assistant:
  name: "api_a"
  model: "m-api"
  api_key_env: "TEST_ASSISTANT_KEY"
  temperature: null
  max_tokens: 512
  history_messages: 4
  extra_body: {reasoning_effort: "low"}
"""
LOCAL_YAML = """\
assistant:
  name: "local_a"
  model: "gemma4"
  base_url: "http://127.0.0.1:1/v1"
  api_key_env: null
  temperature: 0.0
"""


@pytest.fixture
def tmp_repo(tmp_path: Path) -> ConsolePaths:
    (tmp_path / "configs" / "experiments").mkdir(parents=True)
    (tmp_path / "configs" / "assistant").mkdir()
    shutil.copy(REPO / "configs" / "base.yaml", tmp_path / "configs" / "base.yaml")
    (tmp_path / "configs" / "experiments" / "mini.yaml").write_text(MINI_YAML, encoding="utf-8")
    (tmp_path / "configs" / "assistant" / "api_a.yaml").write_text(API_YAML, encoding="utf-8")
    (tmp_path / "configs" / "assistant" / "local_a.yaml").write_text(LOCAL_YAML, encoding="utf-8")
    return ConsolePaths(tmp_path)


def _mini_cfg(paths: ConsolePaths) -> dict:
    from arabic_eval.tools.experiment_console import read_config
    return read_config(paths, "mini.yaml")["resolved"]


class FakeClient:
    """Scripted replies, streamed in small chunks; records every message list."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls = []
        self.last_usage = None

    def stream(self, messages):
        self.calls.append(list(messages))
        text = self.replies.pop(0)
        self.last_usage = None
        for i in range(0, len(text), 7):
            yield text[i:i + 7]
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": max(1, len(text) // 4), "total_tokens": 100 + max(1, len(text) // 4)}


def _events(gen):
    evs = list(gen)
    return evs, evs[-1]


# ---------------------------------------------------------------------------
# hints ↔ schema
# ---------------------------------------------------------------------------

def _schema_paths() -> set:
    """Every ``a.b.c`` path of ExperimentConfig, with ``*`` for list items and phase names."""
    schema = ExperimentConfig.model_json_schema()
    defs = schema.get("$defs", {})

    def resolve(node):
        if "$ref" in node:
            return defs[node["$ref"].split("/")[-1]]
        return node

    out = set()

    def walk(node, prefix):
        node = resolve(node)
        alts = [resolve(a) for a in node.get("anyOf", [])] or [node]
        for alt in alts:
            for k, sub in (alt.get("properties") or {}).items():
                p = f"{prefix}.{k}" if prefix else k
                out.add(p)
                walk(sub, p)
            if alt.get("type") == "array" and isinstance(alt.get("items"), dict):
                walk(alt["items"], f"{prefix}.*")
            add = alt.get("additionalProperties")
            if isinstance(add, dict) and add:
                walk(add, f"{prefix}.*")
    walk(schema, "")
    return {norm_path(p) for p in out}


def test_every_hint_key_names_a_schema_path_or_a_known_widget():
    paths = _schema_paths()
    widgets = {"params", "model.preset"}       # bare fallback key; a form-only preset picker
    stale = sorted(k for k in FIELD_HINTS if k not in paths and k not in widgets)
    assert not stale, f"hints without a schema field: {stale}"
    assert hint_for("training.phases.warmup.steps")[0].startswith("Number of micro-steps")
    # steps / warmup_steps are micro-steps; the hints say so and name the update count
    assert "micro-steps" in hint_for("training.phases.sft.warmup_steps")[0]
    assert "updates" in hint_for("training.phases.sft.gradient_accumulation_steps")[0]
    assert hint_for("sweep.tokenizers.2.params") == FIELD_HINTS["sweep.tokenizers.*.params"]
    assert hint_for("no.such.field") is None


def test_schema_bundle_serves_the_hints():
    b = schema_bundle(REAL)
    assert b["hints"]["training.phases.*.mix_tokens"][0].startswith("Only for datasets = pretraining_mix")
    assert len(b["hints"]) == len(FIELD_HINTS)


# ---------------------------------------------------------------------------
# config + readiness
# ---------------------------------------------------------------------------

def test_assistant_config_from_yaml_and_validation(tmp_repo: ConsolePaths, tmp_path: Path):
    api = ca.AssistantConfig.from_yaml(tmp_repo.repo_root / "configs" / "assistant" / "api_a.yaml")
    assert api.model == "m-api" and api.temperature is None and api.url == "https://api.openai.com/v1"
    assert api.is_local is False and api.extra_body == {"reasoning_effort": "low"} and api.history_messages == 4
    local = ca.AssistantConfig.from_yaml(tmp_repo.repo_root / "configs" / "assistant" / "local_a.yaml")
    assert local.is_local is True and local.api_key_env is None and local.url == "http://127.0.0.1:1/v1"
    bad = tmp_path / "bad.yaml"
    bad.write_text("assistant: {name: x, model: m, backend: vllm}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown assistant keys"):
        ca.AssistantConfig.from_yaml(bad)
    with pytest.raises(ValueError, match="file-safe"):
        ca.AssistantConfig(name="a b", model="m")
    with pytest.raises(ValueError, match="model is required"):
        ca.AssistantConfig(name="a", model="")


def test_repo_assistant_configs_load():
    for p in (REPO / "configs" / "assistant").glob("*.yaml"):
        cfg = ca.AssistantConfig.from_yaml(p)
        assert cfg.name and cfg.model


def test_readiness_key_and_local_probe(tmp_repo: ConsolePaths, monkeypatch):
    monkeypatch.delenv("TEST_ASSISTANT_KEY", raising=False)
    rows = {r["id"]: r for r in ca.assistant_configs(tmp_repo)}
    assert rows["api_a"]["ready"] is False and "TEST_ASSISTANT_KEY" in rows["api_a"]["why"]
    assert rows["local_a"]["ready"] is False and "not reachable" in rows["local_a"]["why"] and rows["local_a"]["local"] is True
    monkeypatch.setenv("TEST_ASSISTANT_KEY", "k")
    rows = {r["id"]: r for r in ca.assistant_configs(tmp_repo)}
    assert rows["api_a"]["ready"] is True and rows["api_a"]["why"] == ""
    (tmp_repo.repo_root / "configs" / "assistant" / "broken.yaml").write_text("assistant: {name: b}\n", encoding="utf-8")
    rows = {r["id"]: r for r in ca.assistant_configs(tmp_repo)}
    assert rows["broken"]["ready"] is False and "error" in rows["broken"]


def test_assistant_config_path_is_confined(tmp_repo: ConsolePaths):
    p = ca.assistant_config_path(tmp_repo, "api_a")
    assert p.name == "api_a.yaml"
    assert ca.assistant_config_path(tmp_repo, "configs/assistant/local_a.yaml").name == "local_a.yaml"
    with pytest.raises(ConsoleError, match="live under"):
        ca.assistant_config_path(tmp_repo, "configs/experiments/mini.yaml")
    with pytest.raises(ConsoleError, match="no such"):
        ca.assistant_config_path(tmp_repo, "nope")


# ---------------------------------------------------------------------------
# SSE + client
# ---------------------------------------------------------------------------

def test_parse_sse_skips_noise_and_stops_at_done():
    lines = [b": keep-alive\n", b"\n", b'data: {"a": 1}\n', b"event: x\n", b"data: not json\n",
             b'data: {"b": 2}\r\n', b"data: [DONE]\n", b'data: {"c": 3}\n']
    assert list(ca.parse_sse(lines)) == [{"a": 1}, {"b": 2}]


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _sse_bytes(*deltas: str, usage=None) -> bytes:
    out = []
    for d in deltas:
        out.append("data: " + json.dumps({"choices": [{"delta": {"content": d}}]}) + "\n\n")
    out.append("data: " + json.dumps({"choices": [], "usage": usage or {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}}) + "\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode("utf-8")


def test_chat_client_body_and_stream(monkeypatch):
    monkeypatch.setenv("TEST_ASSISTANT_KEY", "secret")
    cfg = ca.AssistantConfig(name="a", model="m", api_key_env="TEST_ASSISTANT_KEY", temperature=None,
                             max_tokens=99, extra_body={"reasoning_effort": "low"})
    client = ca.ChatClient(cfg)
    body = client.body([{"role": "user", "content": "hi"}], stream=True)
    assert "temperature" not in body and body["max_completion_tokens"] == 99 and body["stream"] is True
    assert body["stream_options"] == {"include_usage": True} and body["reasoning_effort"] == "low"
    cfg2 = ca.AssistantConfig(name="b", model="m", api_key_env=None, temperature=0.0)
    assert ca.ChatClient(cfg2).body([], stream=False)["temperature"] == 0.0

    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(_sse_bytes("Hel", "lo ", "world"))
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert "".join(client.stream([{"role": "user", "content": "hi"}])) == "Hello world"
    assert client.last_usage["completion_tokens"] == 3
    assert seen["url"] == "https://api.openai.com/v1/chat/completions" and seen["auth"] == "Bearer secret"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]

    def fake_complete(req, timeout=None):
        return _FakeResp(json.dumps({"choices": [{"message": {"content": "done"}}], "usage": {"total_tokens": 5}}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", fake_complete)
    assert client.complete([]) == "done" and client.last_usage == {"total_tokens": 5}


def test_chat_client_errors(monkeypatch):
    cfg = ca.AssistantConfig(name="a", model="m", api_key_env=None, max_retries=0)
    client = ca.ChatClient(cfg)

    def http_400(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b'{"error": "no such model"}'))
    monkeypatch.setattr(urllib.request, "urlopen", http_400)
    with pytest.raises(ca.AssistantError, match="HTTP 400.*no such model"):
        list(client.stream([]))

    def refused(req, timeout=None):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen", refused)
    with pytest.raises(ca.AssistantError, match="unreachable"):
        client.complete([])
    monkeypatch.delenv("NO_SUCH_KEY_X", raising=False)
    with pytest.raises(ca.AssistantError, match="NO_SUCH_KEY_X"):
        ca.ChatClient(ca.AssistantConfig(name="c", model="m", api_key_env="NO_SUCH_KEY_X"))


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------

def test_field_reference_lists_every_hint():
    ref = ca.field_reference()
    for path in FIELD_HINTS:
        assert f"- {path} — " in ref
    assert "### training" in ref and "\n" not in ref.split("- data.preprocessing — ")[1].split("\n")[0][:-1]


def test_system_prompt_on_the_real_repo_and_caching(tmp_repo: ConsolePaths):
    b = ca.ContextBuilder(REAL)
    s = b.system_prompt()
    assert s.startswith(ca.PRIMER.strip()[:60]) and s.rstrip().endswith(ca.PROTOCOL.strip()[-40:])
    for needle in ("- training.phases.*.mix_tokens — ", "native_qwen3", "contamination_exclusions",
                   "native_llama_3phase_with_sft.yaml", "all_tokenizers_sweep_pretrain_mix.yaml",
                   "### dataset names", "pretraining_mix (raw text)", "cidar (free_form)", "```edits"):
        assert needle in s, needle
    assert "# آ/أ/إ/ٱ → ا" not in s                        # base.yaml comments stripped
    assert b.system_prompt() is s                          # cached
    # a config file change invalidates the cache
    b2 = ca.ContextBuilder(tmp_repo)
    s1 = b2.system_prompt()
    assert "mini.yaml" in s1 and "cells: native_llama" in s1
    (tmp_repo.configs_dir / "other.yaml").write_text(MINI_YAML.replace('"mini"', '"other"'), encoding="utf-8")
    s2 = b2.system_prompt()
    assert s2 is not s1 and "other.yaml" in s2


def test_working_context_delta_and_validation(tmp_repo: ConsolePaths):
    b = ca.ContextBuilder(tmp_repo)
    cfg = _mini_cfg(tmp_repo)
    w = b.working_context(cfg, from_base=False, file="mini.yaml")
    assert "configs/experiments/mini.yaml" in w and "Validation: valid" in w and "cells: native_llama" in w
    assert "embedding_alignment:\n    enabled: false" in w or "enabled: false" in w
    assert "logging_steps" not in w                        # only the delta over base.yaml is sent
    base = ca.base_resolved(tmp_repo)
    w0 = b.working_context(base, from_base=True, file=None)
    assert "NEW config" in w0 and "identical to base.yaml" in w0 and "no sweep block yet" in w0 and "cells: bpe_32k" in w0
    bad = json.loads(json.dumps(cfg))
    bad["training"]["phases"]["sft"]["trainable_parameters"] = ["*", "x"]
    wb = b.working_context(bad, from_base=False, file=None)
    assert "INVALID" in wb and "training.phases.sft.trainable_parameters" in wb and "unsaved config" in wb


def test_build_messages_bounds_history(tmp_repo: ConsolePaths):
    b = ca.ContextBuilder(tmp_repo)
    acfg = ca.AssistantConfig(name="a", model="m", api_key_env=None, history_messages=2, max_message_chars=60)
    hist = [{"role": "user", "content": "one"}, {"role": "system", "content": "ignored"},
            {"role": "assistant", "content": "x" * 200}, {"role": "user", "content": "three"}]
    msgs = ca.build_messages(b, acfg, cfg=_mini_cfg(tmp_repo), history=hist, message="now", from_base=False, file="mini.yaml")
    assert [m["role"] for m in msgs] == ["system", "system", "assistant", "user", "user"]
    assert "…[cut]…" in msgs[2]["content"] and len(msgs[2]["content"]) < 100
    assert msgs[-1]["content"] == "now" and msgs[3]["content"] == "three"
    none = ca.build_messages(b, ca.AssistantConfig(name="a", model="m", api_key_env=None, history_messages=0),
                             cfg=_mini_cfg(tmp_repo), history=hist, message="q", from_base=False, file=None)
    assert [m["role"] for m in none] == ["system", "system", "user"]


# ---------------------------------------------------------------------------
# reply parsing + edits
# ---------------------------------------------------------------------------

def test_parse_reply_variants():
    r = ca.parse_reply("Just an answer.")
    assert r.edits is None and r.error is None and r.prose == "Just an answer."
    r = ca.parse_reply("Phase 3 off.\n\n```edits\ntraining.phases.sft.enabled: false\nname: x\n```\n")
    assert r.edits == {"training.phases.sft.enabled": False, "name": "x"} and r.prose == "Phase 3 off."
    r = ca.parse_reply("a\n```edits yaml\nsweep.tokenizers:\n  - type: bpe\n    vocab_sizes: [16000, 32000]\n```")
    assert r.edits["sweep.tokenizers"] == [{"type": "bpe", "vocab_sizes": [16000, 32000]}]
    r = ca.parse_reply("first\n```edits\nseed: 1\n```\nthen\n```edits\nseed: 2\n```")
    assert r.edits == {"seed": 2} and r.prose == "first\n```edits\nseed: 1\n```\nthen"   # last block wins
    r = ca.parse_reply("```edits\nseed: [1, 2\n```")
    assert r.edits is None and "not valid YAML" in r.error
    r = ca.parse_reply("```edits\n- a\n- b\n```")
    assert r.edits is None and "mapping" in r.error
    r = ca.parse_reply("```edits\n1: x\n```")
    assert r.edits is None and "non-string keys" in r.error
    r = ca.parse_reply("nothing to change\n```edits\n```")
    assert r.edits == {} and r.error is None
    r = ca.parse_reply("```yaml\nseed: 5\n```")                 # untagged YAML is not an edit
    assert r.edits is None and r.error is None


def test_apply_edits_paths_lists_and_prefix():
    cfg = {"name": "a", "sweep": None, "training": {"phases": {"sft": {"enabled": True, "steps": 10}}},
           "evaluation": {"num_eval_samples": None}}
    new, changes = ca.apply_edits(cfg, {
        "experiment.name": "b",
        "training.phases.sft.enabled": False,
        "sweep.tokenizers": [{"type": "bpe", "vocab_sizes": [32000]}],
        "sweep.tokenizers.1": {"type": "charformer", "vocab_sizes": [None]},
        "sweep.tokenizers.0.vocab_sizes": [16000],
        "sweep.tasks.0.type": "acva",
        "evaluation.num_eval_samples": 300,
        "training.phases.sft.mixture": None,
    })
    assert cfg["name"] == "a" and cfg["sweep"] is None                      # input untouched
    assert new["name"] == "b" and new["training"]["phases"]["sft"]["enabled"] is False
    assert new["sweep"]["tokenizers"] == [{"type": "bpe", "vocab_sizes": [16000]}, {"type": "charformer", "vocab_sizes": [None]}]
    assert new["sweep"]["tasks"] == [{"type": "acva"}] and new["training"]["phases"]["sft"]["mixture"] is None
    by = {c["path"]: c for c in changes}
    assert by["name"]["old"] == "a" and by["name"]["new"] == "b"
    assert by["sweep.tokenizers"]["old"] == "<unset>" and by["evaluation.num_eval_samples"]["old"] is None
    with pytest.raises(ca.AssistantError, match="invalid edit path"):
        ca.apply_edits(cfg, {"training..steps": 1})
    with pytest.raises(ca.AssistantError, match="out of range"):
        ca.apply_edits(new, {"sweep.tokenizers.5.type": "bpe"})
    with pytest.raises(ca.AssistantError, match="use a number"):
        ca.apply_edits(new, {"sweep.tokenizers.first": "bpe"})


# ---------------------------------------------------------------------------
# chat turns
# ---------------------------------------------------------------------------

def _turn(paths, client, **kw):
    b = ca.ContextBuilder(paths)
    acfg = ca.AssistantConfig(name="t", model="m", api_key_env=None)
    return _events(ca.chat_turn(paths, b, acfg, client, **kw))


def test_turn_question_has_no_edits(tmp_repo: ConsolePaths):
    answer = "mix_tokens must equal steps × batch_size × block_size."
    client = FakeClient(answer)
    evs, done = _turn(tmp_repo, client, cfg=_mini_cfg(tmp_repo), history=[], message="what must mix_tokens equal?",
                      from_base=False, file="mini.yaml")
    assert evs[0]["type"] == "context" and evs[0]["messages"] == 3
    assert "".join(e["text"] for e in evs if e["type"] == "token") == answer
    assert done["type"] == "done" and done["applied"] is None and done["parse_error"] is None
    assert done["reply"].startswith("mix_tokens must") and done["usage"]["rounds"] == 1 and done["elapsed_sec"] >= 0
    assert client.calls[0][-1] == {"role": "user", "content": "what must mix_tokens equal?"}
    assert client.calls[0][1]["content"].startswith("## Working config: configs/experiments/mini.yaml")


def test_turn_valid_edits_are_applied_and_resolved(tmp_repo: ConsolePaths):
    client = FakeClient("Phase 3 off, renamed.\n\n```edits\nname: mini_no_sft\noutput_dir: outputs/experiments/mini_no_sft\n"
                        "training.phases.sft.enabled: false\n```")
    evs, done = _turn(tmp_repo, client, cfg=_mini_cfg(tmp_repo), history=[], message="disable phase 3",
                      from_base=False, file="mini.yaml")
    ed = [e for e in evs if e["type"] == "edits"]
    assert len(ed) == 1 and ed[0]["ok"] is True and ed[0]["errors"] == []
    assert {c["path"] for c in ed[0]["changes"]} == {"name", "output_dir", "training.phases.sft.enabled"}
    a = done["applied"]
    assert a["ok"] is True and a["config"]["training"]["phases"]["sft"]["enabled"] is False
    assert a["config"]["name"] == "mini_no_sft" and a["cells"] == ["native_llama"] and a["sweep"] is False
    assert a["config"]["training"]["phases"]["warmup"]["steps"] == 2000    # resolved: base defaults present
    assert done["reply"] == "Phase 3 off, renamed." and "```edits" in done["raw"]
    assert not [e for e in evs if e["type"] == "repair"]


def test_turn_repairs_once_after_validation_failure(tmp_repo: ConsolePaths):
    bad = ("Switching Phase 2 to the mix.\n\n```edits\ntraining.phases.warmup.datasets: [pretraining_mix]\n"
           "training.phases.warmup.mix_tokens: 4096000\n```")                       # loss_target still answer_only
    good = ("Fixed: full-sequence loss.\n\n```edits\ntraining.phases.warmup.datasets: [pretraining_mix]\n"
            "training.phases.warmup.mix_tokens: 4096000\ntraining.phases.warmup.loss_target: full_sequence\n```")
    client = FakeClient(bad, good)
    evs, done = _turn(tmp_repo, client, cfg=_mini_cfg(tmp_repo), history=[], message="phase 2 on the mix",
                      from_base=False, file="mini.yaml")
    kinds = [e["type"] for e in evs if e["type"] != "token"]
    assert kinds == ["context", "edits", "repair", "edits", "done"]
    first, second = [e for e in evs if e["type"] == "edits"]
    assert first["ok"] is False and any("loss_target" in x["msg"] or "full_sequence" in x["msg"] for x in first["errors"])
    assert second["ok"] is True
    rep = [e for e in evs if e["type"] == "repair"][0]
    assert rep["errors"] == first["errors"] and rep["parse_error"] is None
    # the repair request carries the model's own reply and the errors
    assert client.calls[1][-2] == {"role": "assistant", "content": bad}
    assert "fails validation" in client.calls[1][-1]["content"] and first["errors"][0]["loc"] in client.calls[1][-1]["content"]
    assert done["applied"]["ok"] is True and done["usage"]["rounds"] == 2
    ph = done["applied"]["config"]["training"]["phases"]["warmup"]
    assert ph["datasets"] == ["pretraining_mix"] and ph["steps"] == 2000 and ph["loss_target"] == "full_sequence"


def test_turn_gives_up_after_the_repair_round(tmp_repo: ConsolePaths):
    bad = "```edits\ntraining.phases.sft.trainable_parameters: ['*', 'x']\n```"
    client = FakeClient(bad, bad)
    evs, done = _turn(tmp_repo, client, cfg=_mini_cfg(tmp_repo), history=[], message="x", from_base=False, file=None)
    assert [e["type"] for e in evs if e["type"] != "token"] == ["context", "edits", "repair", "edits", "done"]
    assert done["applied"]["ok"] is False and done["applied"]["errors"]
    assert done["applied"]["config"]["training"]["phases"]["sft"]["trainable_parameters"] == ["*", "x"]  # raw, for "apply anyway"


def test_turn_repairs_a_broken_edits_block(tmp_repo: ConsolePaths):
    client = FakeClient("```edits\nseed: [1, 2\n```", "ok\n```edits\nseed: 7\n```")
    evs, done = _turn(tmp_repo, client, cfg=_mini_cfg(tmp_repo), history=[], message="seed 7", from_base=False, file=None)
    rep = [e for e in evs if e["type"] == "repair"][0]
    assert rep["parse_error"] and "not valid YAML" in rep["parse_error"] and rep["errors"] == []
    assert "could not be used" in client.calls[1][-1]["content"]
    assert done["applied"]["ok"] is True and done["applied"]["config"]["seed"] == 7 and done["parse_error"] is None


def test_turn_bad_path_is_reported_and_repaired(tmp_repo: ConsolePaths):
    client = FakeClient("```edits\nsweep.tokenizers.first.type: bpe\n```", "```edits\nseed: 3\n```")
    evs, done = _turn(tmp_repo, client, cfg=_mini_cfg(tmp_repo), history=[], message="x", from_base=False, file=None)
    rep = [e for e in evs if e["type"] == "repair"][0]
    assert "use a number" in rep["parse_error"]
    assert done["applied"]["ok"] is True and done["applied"]["config"]["seed"] == 3


def test_turn_creates_from_base(tmp_repo: ConsolePaths):
    reply = ("A BPE-vs-native smoke sweep.\n\n```edits\nname: _smoke_bpe_vs_native\n"
             "output_dir: outputs/experiments/_smoke_bpe_vs_native\ntokenizer.type: bpe\ntokenizer.vocab_size: 16000\n"
             "sweep.tokenizers:\n  - {type: bpe, vocab_sizes: [16000]}\n  - {type: native_llama, vocab_sizes: [null]}\n"
             "sweep.tasks:\n  - {type: acva, params: {num_fewshot: 0}}\n  - {type: arabic_exam, params: {}}\n"
             "training.phases.embedding_alignment.steps: 30\ntraining.phases.warmup.steps: 30\n"
             "training.phases.sft.steps: 30\nevaluation.num_eval_samples: 30\nevaluation.morphological_metrics: false\n```")
    client = FakeClient(reply)
    evs, done = _turn(tmp_repo, client, cfg=_mini_cfg(tmp_repo), history=[], message="bpe 16k vs native llama, smoke",
                      from_base=True, file="mini.yaml")
    assert client.calls[0][1]["content"].startswith("## Working config: a NEW config")
    a = done["applied"]
    assert a["ok"] is True and a["cells"] == ["bpe_16k", "native_llama"] and a["sweep"] is True
    assert done["from_base"] is True and a["config"]["training"]["phases"]["sft"]["steps"] == 30
    assert a["config"]["training"]["phases"]["embedding_alignment"]["enabled"] is True   # base default, not mini's


def test_turn_rejects_empty_message(tmp_repo: ConsolePaths):
    with pytest.raises(ca.AssistantError, match="empty"):
        _turn(tmp_repo, FakeClient("x"), cfg=None, history=[], message="  ", from_base=False, file=None)


def test_preview_messages(tmp_repo: ConsolePaths):
    b = ca.ContextBuilder(tmp_repo)
    acfg = ca.AssistantConfig(name="t", model="m", api_key_env=None)
    p = ca.preview_messages(tmp_repo, b, acfg, cfg=None, history=[], message="", from_base=False, file=None)
    assert p["assistant"] == "t" and p["model"] == "m" and p["approx_tokens"] > 1000
    assert [m["role"] for m in p["messages"]] == ["system", "system", "user"] and p["messages"][-1]["content"] == "(your message)"
    assert "NEW config" not in p["messages"][1]["content"]      # cfg None without from_base = base, unsaved
