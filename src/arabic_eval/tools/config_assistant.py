"""Config assistant of the experiment console: an LLM that fills the form.

The Configs tab gets a chat drawer. Each turn the page sends the config that
is in the form (or asks to start from ``base.yaml``), the last few chat
messages and the new message; this module builds the prompt, streams the
model's reply, parses the **edits** it proposes, applies them to the config,
runs the real validation (``validate_config`` — merge over ``base.yaml`` +
Pydantic) and, when the result does not validate, sends the errors back to the
model once for a repair. The page then puts the resulting config into the
form. Nothing here saves or starts anything.

Protocol with the model (see ``PROTOCOL``): a short markdown reply and, when
the request changes the config, one fenced block tagged ``edits`` — a YAML
mapping from *dotted config paths* to their new values (``training.phases.sft.enabled:
false``). A value replaces the node it names (lists and mappings whole,
``null`` allowed). Questions get no block. Dotted edits are ~10× cheaper than a
whole config and exact for a modification; free text plus a fence rather than
constrained JSON, because constrained decoding biases some models.

Context (``ContextBuilder``): a hand-written primer of the platform's rules,
the field reference rendered from ``config_hints.FIELD_HINTS`` (the same table
behind the form tooltips), ``base.yaml`` with its comments stripped, the
registries and tokenizer presets, and every ``configs/experiments`` file as a
delta over ``base.yaml`` — static first so an endpoint's prompt cache hits —
then the working config as a delta, its validation state and the chat history.

The endpoint is any OpenAI-compatible ``/chat/completions`` (``AssistantConfig``,
YAMLs under ``configs/assistant/``): the API model, or a hand-started local
``vllm serve``. Stdlib only (urllib + SSE), no torch.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import yaml

from arabic_eval.config import EXPERIMENT_KEYS, CORPUS_CATEGORY, ExperimentConfig
from arabic_eval.tools.config_hints import FIELD_HINTS
from arabic_eval.tools.experiment_console import (
    ConsoleError,
    ConsolePaths,
    _deep_diff,
    _SAME,
    base_resolved,
    cell_names,
    flatten_experiment,
    read_config,
    render_yaml,
    schema_bundle,
    validate_config,
)

_EXPERIMENT_KEYS = EXPERIMENT_KEYS


class AssistantError(ConsoleError):
    """User-facing assistant failure (endpoint unreachable, bad reply, …)."""


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class AssistantConfig:
    """One OpenAI-compatible chat endpoint (``configs/assistant/<name>.yaml``)."""
    name: str
    model: str
    base_url: Optional[str] = None            # null = https://api.openai.com/v1
    api_key_env: Optional[str] = "OPENAI_API_KEY"   # null = no Authorization header (a local vllm serve)
    temperature: Optional[float] = None       # null = field omitted (reasoning models accept only the default)
    max_tokens: int = 4096                    # sent as max_completion_tokens
    timeout_sec: float = 180.0
    history_messages: int = 8                 # chat messages re-sent per request
    max_message_chars: int = 6000             # longer history messages are cut in the middle
    max_retries: int = 2                      # transient HTTP errors before the first byte
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name or ""):
            raise ValueError(f"assistant name {self.name!r} must be a file-safe identifier")
        if not self.model:
            raise ValueError(f"assistant {self.name!r}: model is required")
        if self.history_messages < 0 or self.max_tokens <= 0:
            raise ValueError(f"assistant {self.name!r}: history_messages must be ≥ 0 and max_tokens > 0")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AssistantConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        d = raw.get("assistant", raw)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"{path}: unknown assistant keys {unknown}; known: {sorted(known)}")
        return cls(**d)

    @property
    def url(self) -> str:
        return (self.base_url or "https://api.openai.com/v1").rstrip("/")

    @property
    def is_local(self) -> bool:
        return bool(self.base_url) and re.match(r"https?://(127\.0\.0\.1|localhost|0\.0\.0\.0)[:/]", self.base_url) is not None

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def assistant_config_path(paths: ConsolePaths, ref: str) -> Path:
    """``<name>`` or ``configs/assistant/<name>.yaml`` → the file, confined to configs/assistant."""
    d = (paths.repo_root / "configs" / "assistant").resolve()
    ref = str(ref or "").strip()
    if not ref:
        raise ConsoleError("assistant required")
    cand = (paths.repo_root / ref).resolve() if ref.endswith(".yaml") else (d / f"{ref}.yaml").resolve()
    if d not in cand.parents:
        raise ConsoleError("assistant configs live under configs/assistant/")
    if not cand.is_file():
        raise ConsoleError(f"no such assistant config: {ref}")
    return cand


def assistant_ready(cfg: AssistantConfig, probe_timeout: float = 2.0) -> Tuple[bool, str]:
    """Whether the endpoint can be used right now, and why not: the API key must be
    in the console server's environment; a local endpoint must answer ``/models``."""
    if cfg.api_key_env and not os.environ.get(cfg.api_key_env):
        return False, f"{cfg.api_key_env} is not set in the console server's environment (export it before starting the server)"
    if cfg.is_local:
        try:
            req = urllib.request.Request(cfg.url + "/models", method="GET")
            with urllib.request.urlopen(req, timeout=probe_timeout):
                pass
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return False, f"{cfg.url} is not reachable ({getattr(e, 'reason', e)}) — start the server (see the YAML's comments)"
    return True, ""


def assistant_configs(paths: ConsolePaths) -> List[dict]:
    """Every ``configs/assistant/*.yaml`` with its essentials and readiness."""
    d = paths.repo_root / "configs" / "assistant"
    out: List[dict] = []
    for p in sorted(d.glob("*.yaml")) if d.is_dir() else []:
        entry: Dict[str, Any] = {"file": paths.rel(p), "id": p.stem}
        try:
            cfg = AssistantConfig.from_yaml(p)
            ready, why = assistant_ready(cfg)
            entry.update({"name": cfg.name, "model": cfg.model, "base_url": cfg.url, "local": cfg.is_local,
                          "api_key_env": cfg.api_key_env, "ready": ready, "why": why})
        except Exception as e:  # noqa: BLE001 — a broken YAML is listed, not hidden
            entry.update({"name": p.stem, "error": f"{type(e).__name__}: {e}", "ready": False, "why": "invalid config"})
        out.append(entry)
    return out


# --------------------------------------------------------------------------
# chat client (OpenAI-compatible, streaming over urllib)
# --------------------------------------------------------------------------

def parse_sse(lines: Iterable[bytes]) -> Iterator[dict]:
    """The JSON payload of every ``data:`` line of a chat-completions stream, until
    ``[DONE]``. Blank lines and comments are skipped; a ``data:`` that is not JSON
    is ignored."""
    for raw in lines:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except ValueError:
            continue


class ChatClient:
    """``/chat/completions`` over plain urllib. ``stream`` yields content deltas
    and records ``last_usage`` (``stream_options.include_usage``); ``complete``
    returns the whole content."""

    def __init__(self, cfg: AssistantConfig) -> None:
        self.cfg = cfg
        self._key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        if cfg.api_key_env and not self._key:
            raise AssistantError(f"assistant {cfg.name!r}: environment variable {cfg.api_key_env} is not set")
        self.url = cfg.url + "/chat/completions"
        self.last_usage: Optional[Dict[str, Any]] = None

    def body(self, messages: Sequence[Dict[str, str]], stream: bool) -> Dict[str, Any]:
        b: Dict[str, Any] = {"model": self.cfg.model, "messages": list(messages),
                             "max_completion_tokens": self.cfg.max_tokens, "stream": stream}
        if self.cfg.temperature is not None:
            b["temperature"] = self.cfg.temperature
        if stream:
            b["stream_options"] = {"include_usage": True}
        b.update(self.cfg.extra_body)
        return b

    def _request(self, body: Dict[str, Any]):
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if body.get("stream") else "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        delay = 1.0
        for attempt in range(self.cfg.max_retries + 1):
            req = urllib.request.Request(self.url, data=data, method="POST", headers=headers)
            try:
                return urllib.request.urlopen(req, timeout=self.cfg.timeout_sec)
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace")[:800]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.cfg.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 10.0)
                    continue
                raise AssistantError(f"assistant {self.cfg.name!r}: HTTP {e.code}: {text}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < self.cfg.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 10.0)
                    continue
                raise AssistantError(f"assistant {self.cfg.name!r}: {self.url} unreachable ({getattr(e, 'reason', e)})") from e
        raise AssistantError("unreachable")

    def stream(self, messages: Sequence[Dict[str, str]]) -> Iterator[str]:
        self.last_usage = None
        resp = self._request(self.body(messages, stream=True))
        with resp:
            for ev in parse_sse(resp):
                if ev.get("usage"):
                    self.last_usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        yield delta

    def complete(self, messages: Sequence[Dict[str, str]]) -> str:
        self.last_usage = None
        resp = self._request(self.body(messages, stream=False))
        with resp:
            payload = json.loads(resp.read().decode("utf-8"))
        self.last_usage = payload.get("usage")
        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise AssistantError(f"assistant {self.cfg.name!r}: malformed response: {str(payload)[:300]}") from e


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------

PRIMER = """You are the configuration assistant of the experiment console of the Arabic Tokenizers Evaluation Platform. You help the user create and modify experiment configs (YAML files validated by a Pydantic schema) and answer questions about their fields. You never run, save or start anything: the user reviews the form and does that. Be concise and concrete; when you change something, say what and why in a few lines.

## How an experiment works
- One config runs a fixed pipeline: train (or load) a tokenizer on the tokenizer corpus (data.*), compute intrinsic metrics, load the base LLM (model.*), adapt its embeddings to the tokenizer, run three training phases in fixed order — training.phases.embedding_alignment (Phase 1) → warmup (Phase 2) → sft (Phase 3), each with its own `enabled` — then evaluate every task in sweep.tasks on the full benchmark. The only experimental variable across conditions is the tokenizer.
- sweep.tokenizers lists the tokenizer *cells*: one cell per (type, vocab size). Cells are named `<type>_<size/1000>k` (bpe_32k) or just `<type>` when the vocab size is null. More than one cell → the run is a sweep (a sub-folder of output_dir per cell + a comparison report). With ONE cell the run is NOT a sweep: run_experiment.py then trains the top-level tokenizer.* block and ignores sweep.tokenizers entirely — so for a single-cell config tokenizer.type / vocab_size / params MUST equal that cell (the console warns on a mismatch). In a sweep the top-level block is overridden per cell; keep tokenizer.type equal to the first cell's type. A new config must set sweep.tokenizers and sweep.tasks (base.yaml has no sweep block).
- Tokenizer types and their vocab size: bpe / wordpiece / morpho_bpe take one (16000, 32000, 50000 are the studied sizes); character_bert, farasa_character_bert, char_jaber, charformer, araroopat, native_llama, native_qwen3 take [null] (fixed or derived vocab). Families: subword (bpe, wordpiece, morpho_bpe, araroopat — standard embeddings); word/morpheme + CharCNN (character_bert, farasa_character_bert); character (char_jaber — sequences 4–6× longer, use max_length 2048 in every phase); byte (charformer — ~2× char_jaber, max_length 2048). morpho_bpe and farasa_character_bert need Java (Farasa); araroopat needs the CAMeL bridge (.venv-camel). native_llama wraps Llama's own tokenizer (model.type llama), native_qwen3 wraps Qwen3's (model.type qwen3 + Qwen/Qwen3-4B-Base); the phase `enabled` flags are config-wide, not per cell: in a config whose cells are ALL native_* set training.phases.embedding_alignment.enabled false (Phase 1 only damages pretrained embeddings), but in a sweep that mixes native_* with from-scratch tokenizers keep Phase 1 on — the from-scratch cells need it and the native cell pays some embedding drift (that is what all_tokenizers_sweep.yaml does); never claim Phase 1 is off for one cell only. character_bert, farasa_character_bert and charformer cannot generate: the freeform_cidar task reports generation_unsupported for them (not an error).
- Models: llama = meta-llama/Llama-3.2-1B (default) or meta-llama/Llama-3.2-3B; qwen3 = Qwen/Qwen3-4B-Base (~3× the per-step cost). Gated meta-llama repos need HF_TOKEN in the environment.
- Eval tasks (sweep.tasks, each `{type, params}`): acva (True/False cultural claims, label-noisy), alghafa (9 sub-configs), arabic_exam (MBZUAI ArabicMMLU), culture_arabic_mmlu (OALL Arabic_MMLU) — all log-likelihood MCQ scored by accuracy; freeform_cidar = greedy generation on 250 held-out CIDAR instructions (chrF, BERTScore, later an LLM judge) — only meaningful when Phase 3 trained on free-form corpora. Each task DECLARES its parameters (the "task params" list under Registries and presets: name, type, default, meaning). `params` is the ONLY place a run reads task parameters from (configs/tasks/<type>.yaml is generated documentation, never read by a run); write only keys whose value differs from the declared default — an undeclared key is ignored at run time and flagged. Conventions: acva gets {num_fewshot: 0} (word-scored true/false); the MCQ tasks get {max_length: 1024} under long-sequence tokenizers; an eval-only re-run pins the decoding rule it must reproduce (freeform_cidar {max_output_chars: 2400}). Default task list of the reference configs: acva, alghafa, arabic_exam (+ culture_arabic_mmlu in the full sweeps).
- Phases: Phase 1 trains embed_tokens + lm_head only on arabic_squad, full-sequence loss, constant LR; Phase 2 trains everything on arabic_squad, answer-only loss; Phase 3 trains everything on tydiqa_arabic + arcd + arabic_squad_mcq, answer-only, with early stopping on the dev slices. Dataset categories: extractive = arabic_squad, tydiqa_arabic, arcd; mcq = arabic_squad_mcq; free_form = cidar, bactrian_x_ar, aya_ar; raw text = pretraining_mix (packed 70 % FineWeb-2 / 20 % Wikipedia / 10 % ArabicWeb24 mix for Phases 1/2).

## Rules the validator enforces — get them right the first time
- sft.enabled true requires sft.early_stopping (present in base.yaml: keep it, set early_stopping.enabled false to skip the periodic eval).
- A phase on the raw-text mix: datasets [pretraining_mix] alone, loss_target full_sequence, max_length == training.pretraining_mix.block_size (512), and mix_tokens == steps × batch_size × block_size — set mix_tokens and leave steps null (derived), or make them agree. Only such a phase may carry qa_blend {datasets: [tydiqa_arabic, arcd], share: 0.07, split: train, seed: 42} (the share comes out of mix_tokens; split must be train).
- A mixture on a QA phase (never on a pretraining_mix phase): total_examples divisible by batch_size, steps null (derived = total_examples / batch_size) or equal to it, shares summing to 1 and naming exactly the categories of the listed datasets (a listed corpus without a share, or a share without a corpus, is an error). Free-form at 30 % of examples is ~87 % of the loss tokens.
- trainable_parameters: ["*"] alone, or a list of substrings — never mixed. Task types, tokenizer types, model types and dataset names must be registry keys (listed below).
- Conventions: output_dir = outputs/experiments/<name>; seed 42; evaluation.score_normalization "char+pmi"; evaluation.num_eval_samples null for a real run. Wall time on the H100 per cell: ~5 h with all three phases at default steps (Llama-3.2-1B), ~1.5 h without Phase 3.
- Smoke test recipe (wiring only, name prefixed `_smoke_`): data.max_train_samples 200–20000, data.max_eval_samples 100, every phase steps 30–50 with batch_size 2–4, sft.early_stopping {eval_every_n_steps: 25, min_steps_before_stop: 10, patience: 3}, evaluation.morphological_metrics false, evaluation.num_eval_samples 30, one or two tasks.
"""

PROTOCOL = """## How to answer
- Reply in short markdown (no headings larger than ###). Answer questions directly from the reference above; say so when something is not covered.
- When the request changes the config, end the reply with exactly one fenced block tagged `edits`: a YAML mapping from dotted config paths to their new values. A value REPLACES the node at that path — give lists whole (`sweep.tokenizers: [...]`), `null` is allowed, list indexes are allowed in paths (`sweep.tokenizers.0.vocab_sizes`). A mapping value also replaces its node and the keys you omit take base.yaml's defaults (NOT the working config's values), so to change one key inside a mapping address it by its full path (`training.phases.sft.early_stopping.patience: 3`). Include only the paths that change. Paths are the ones of the field reference (`training.phases.sft.enabled`, `evaluation.num_eval_samples`, top-level `name`, `output_dir`, `description`, `seed`); never wrap them in `experiment:`.
- Creating a new config: the working config below is base.yaml; set at least name, output_dir (outputs/experiments/<name>), tokenizer.type, sweep.tokenizers and sweep.tasks, plus whatever the request implies. Modifying: edit the working config below; keep everything else as it is.
- `created_at` and `runs` are provenance the tooling writes (first save, every start): never include them in an `edits` block.
- Never emit a config in any other form (no full YAML, no JSON). A question gets no edits block. If the request cannot be carried out without a decision from the user, ask the one question that unblocks it and emit no edits.

Example of a reply that changes the config:

Phase 3 off — the run becomes the Phase 1 + 2 baseline; name and output_dir renamed to match.

```edits
name: bpe_32k_no_sft
output_dir: outputs/experiments/bpe_32k_no_sft
training.phases.sft.enabled: false
```
"""

_TOKENIZER_NOTES: Dict[str, str] = {
    "bpe": "subword, HF BpeTrainer, ByteLevel; vocab_size required",
    "wordpiece": "subword, HF WordPieceTrainer, Whitespace; vocab_size required",
    "morpho_bpe": "Farasa morphological segmentation then BPE; needs Java; vocab_size required",
    "character_bert": "whole word → CharCNN; vocab_sizes [null]; cannot generate",
    "farasa_character_bert": "Farasa morpheme → CharCNN; needs Java; vocab_sizes [null]; cannot generate",
    "char_jaber": "one token per character; vocab_sizes [null]; max_length 2048",
    "charformer": "bytes + GBST inside the model; vocab_sizes [null]; max_length 2048; cannot generate",
    "araroopat": "Arabic root + pattern tokens via CAMeL Tools (.venv-camel); vocab_sizes [null]",
    "native_llama": "Llama-3.2's own tokenizer (128256); vocab_sizes [null]; Phase 1 off",
    "native_qwen3": "Qwen3-4B-Base's own tokenizer; model.type qwen3; vocab_sizes [null]; Phase 1 off",
}


def _strip_yaml_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        line = re.sub(r"\s+#.*$", "", line).rstrip()
        if line.strip():
            out.append(line)
    return "\n".join(out)


def _one_line(text: str) -> str:
    return " / ".join(part.strip() for part in text.split("\n") if part.strip())


def field_reference() -> str:
    """The hint table as ``- path — meaning · e.g. example`` lines, grouped by section."""
    groups: Dict[str, List[str]] = {}
    for path, (meaning, example) in FIELD_HINTS.items():
        head = path.split(".")[0] if "." in path else ("experiment" if path in _EXPERIMENT_KEYS else "other")
        groups.setdefault(head, []).append(f"- {path} — {meaning} · e.g. {_one_line(example)}")
    order = ["experiment", "sweep", "tokenizer", "model", "training", "evaluation", "data", "tracking", "task", "other"]
    parts = []
    for g in order + [k for k in groups if k not in order]:
        if g in groups:
            parts.append(f"### {g}\n" + "\n".join(groups[g]))
    return "\n".join(parts)


def _presets_text(bundle: dict) -> str:
    lines = ["### tokenizer types (registry) — `preset` = the recommended values of configs/tokenizers/<type>.yaml (what the form's "
             "fill button copies; these are what experiments use), `declares` = every key the tokenizer reads with its type, "
             "CODE default and meaning (an undeclared key is ignored at run time and flagged). Write preset values, not defaults, "
             "unless the request says otherwise."]
    presets = bundle["presets"]["tokenizers"]
    tspecs = bundle.get("tokenizer_params") or {}
    for t in bundle["registries"]["tokenizers"]:
        p = dict(presets.get(t) or {})
        p.pop("file", None)
        p.pop("type", None)
        params = p.get("params") if isinstance(p.get("params"), dict) else {}
        note = _TOKENIZER_NOTES.get(t, "")
        ptxt = ", ".join(f"{k}: {_short(v)}" for k, v in params.items()) if params else "no params"
        lines.append(f"- {t}: {note}; preset: {ptxt}")
        for s in tspecs.get(t) or []:
            meta = s["type"] + (" | null" if s.get("nullable") else "")
            if s.get("choices"):
                meta += "; one of " + ", ".join(json.dumps(c, ensure_ascii=False) for c in s["choices"])
            lines.append(f"  - declares {s['name']} ({meta}, code default {_short(s['default'])}): {s['help']}")
    lines.append("### model types (registry): " + ", ".join(bundle["registries"]["models"]))
    for name, p in bundle["presets"]["models"].items():
        lines.append(f"- {name}: name_or_path {p.get('name_or_path')}, dtype {p.get('dtype')} ({p.get('file')})")
    lines.append("### task types (registry): " + ", ".join(bundle["registries"]["tasks"]))
    lines.append(_task_params_text(bundle))
    lines.append("### dataset names (training.phases.*.datasets): "
                 + ", ".join(f"{d} ({CORPUS_CATEGORY.get(d, 'raw text')})" for d in bundle["registries"]["datasets"]))
    return "\n".join(lines)


def _short(v: Any) -> str:
    """A value for the prompt: long string lists (the AraRooPat inventories) by their size."""
    if isinstance(v, list) and len(v) > 3:
        return f"a list of {len(v)} strings"
    return json.dumps(v, ensure_ascii=False)


def _task_params_text(bundle: dict) -> str:
    """The declared parameters of every task (``param_spec()``, served as
    ``task_params``): name, type, default and help, so the model sets
    ``max_output_chars`` or ``num_fewshot`` with the right key and type and
    never invents one. Only a value that differs from the default belongs in
    ``sweep.tasks[].params``."""
    lines = ["### task params (sweep.tasks[].params) — write ONLY keys whose value differs from the default shown; "
             "any other key is ignored at run time and flagged as undeclared"]
    for task, spec in (bundle.get("task_params") or {}).items():
        if not spec:
            lines.append(f"- {task}: declares no parameters")
            continue
        lines.append(f"- {task}:")
        for s in spec:
            default = s["default"]
            if isinstance(default, list) and len(default) > 3:
                shown = f"a list of {len(default)} strings (the templates' block labels)"
            else:
                shown = json.dumps(default, ensure_ascii=False)
            meta = s["type"] + (" | null" if s.get("nullable") else "")
            if s.get("choices"):
                meta += "; one of " + ", ".join(json.dumps(c, ensure_ascii=False) for c in s["choices"])
            lines.append(f"  - {s['name']} ({meta}, default {shown}): {s['help']}")
    return "\n".join(lines)


def _configs_index(paths: ConsolePaths) -> str:
    lines = ["Every file under configs/experiments/, as a delta over base.yaml (what the file changes; everything else is the default):"]
    for p in sorted(paths.configs_dir.glob("*.yaml")):
        try:
            d = read_config(paths, p.name)
        except Exception as e:  # noqa: BLE001
            lines.append(f"### {p.name} — unreadable ({type(e).__name__})")
            continue
        if not d.get("valid"):
            lines.append(f"### {p.name} — does not validate")
            continue
        r = d["resolved"]
        try:
            cells = cell_names(ExperimentConfig(**r))
        except Exception:  # noqa: BLE001
            cells = []
        phases = " ".join(f"P{i + 1}" if r["training"]["phases"][k]["enabled"] else "–"
                          for i, k in enumerate(("embedding_alignment", "warmup", "sft")))
        tasks = ", ".join(t["type"] for t in (r.get("sweep") or {}).get("tasks", []))
        head = f"### {p.name} — {r.get('description') or 'no description'} | cells: {', '.join(cells) or '—'} | tasks: {tasks or '—'} | phases: {phases}"
        try:
            delta = render_yaml(paths, r, "delta")
        except ConsoleError:
            delta = ""
        lines.append(head + ("\n```yaml\n" + delta.rstrip() + "\n```" if delta else ""))
    return "\n".join(lines)


class ContextBuilder:
    """Assembles the system prompt; the static part is cached until a config file
    or ``base.yaml`` changes (mtime fingerprint)."""

    def __init__(self, paths: ConsolePaths) -> None:
        self.paths = paths
        self._fp: Optional[Tuple] = None
        self._system: Optional[str] = None

    def _fingerprint(self) -> Tuple:
        files = sorted(self.paths.configs_dir.glob("*.yaml")) if self.paths.configs_dir.exists() else []
        return tuple((p.name, p.stat().st_mtime_ns) for p in files) + (("base", self.paths.base_yaml.stat().st_mtime_ns),)

    def system_prompt(self) -> str:
        fp = self._fingerprint()
        if self._system is None or fp != self._fp:
            bundle = schema_bundle(self.paths)
            base_text = _strip_yaml_comments(self.paths.base_yaml.read_text(encoding="utf-8"))
            self._system = "\n\n".join([
                PRIMER.strip(),
                "## Field reference (path — meaning · example)\n" + field_reference(),
                "## Registries and presets\n" + _presets_text(bundle),
                "## base.yaml — the defaults every config starts from (comments removed)\n```yaml\n" + base_text + "\n```",
                "## Existing experiment configs\n" + _configs_index(self.paths),
                PROTOCOL.strip(),
            ])
            self._fp = fp
        return self._system

    def working_context(self, cfg: dict, *, from_base: bool, file: Optional[str]) -> str:
        """The dynamic tail: the working config as a delta over base.yaml (raw
        ``_deep_diff``, so a half-edited invalid form still renders) and its
        validation state."""
        base = base_resolved(self.paths)
        diff = _deep_diff(cfg, base)
        delta = {} if diff is _SAME else dict(diff)
        v = validate_config(self.paths, cfg, file=file)
        if from_base:
            head = "## Working config: a NEW config starting from base.yaml (no file yet). The user wants to create it."
        elif file:
            head = f"## Working config: configs/experiments/{file} as currently shown in the form (delta over base.yaml)"
        else:
            head = "## Working config: an unsaved config as currently shown in the form (delta over base.yaml)"
        delta_txt = yaml.safe_dump(delta, sort_keys=False, allow_unicode=True, width=110, default_flow_style=False).rstrip() if delta else "{}   # identical to base.yaml"
        if v["ok"]:
            state = (f"Validation: valid · cells: {', '.join(v['cells'])}"
                     + (" · runs as a sweep" if v.get("sweep") else " · single run (the top-level tokenizer block)")
                     + ("" if v["resolved"].get("sweep") is not None else " · no sweep block yet (sweep.tasks is required)")
                     + "".join(f"\nWARNING: {w}" for w in v.get("warnings", [])))
        else:
            state = "Validation: INVALID\n" + "\n".join(f"- {e['loc']}: {e['msg']}" for e in v["errors"])
        return f"{head}\n```yaml\n{delta_txt}\n```\n{state}"


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(limit // 2 - 20, 0)
    return text[:half] + "\n…[cut]…\n" + text[-half:]


def build_messages(builder: ContextBuilder, acfg: AssistantConfig, *, cfg: dict, history: Sequence[Dict[str, str]],
                   message: str, from_base: bool, file: Optional[str]) -> List[Dict[str, str]]:
    """System prompt (static) + working context + the last ``history_messages``
    chat messages + the new message. Only ``user`` / ``assistant`` history roles
    are kept; contents are trimmed to ``max_message_chars``."""
    msgs: List[Dict[str, str]] = [{"role": "system", "content": builder.system_prompt()}]
    msgs.append({"role": "system", "content": builder.working_context(cfg, from_base=from_base, file=file)})
    kept = [m for m in history if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    if acfg.history_messages:
        kept = kept[-acfg.history_messages:]
    else:
        kept = []
    for m in kept:
        msgs.append({"role": m["role"], "content": _trim(m["content"], acfg.max_message_chars)})
    msgs.append({"role": "user", "content": message})
    return msgs


# --------------------------------------------------------------------------
# reply parsing + edits
# --------------------------------------------------------------------------

_EDITS_RE = re.compile(r"```[ \t]*edits[^\n]*\n(.*?)\n?[ \t]*```", re.DOTALL)
_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*(\.[A-Za-z0-9_-]+)*$")


@dataclass
class ParsedReply:
    prose: str
    edits: Optional[Dict[str, Any]]
    error: Optional[str]
    raw: str


def parse_reply(text: str) -> ParsedReply:
    """The prose and the last fenced ``edits`` block (a YAML mapping). ``error``
    names a block that is present but unusable; ``edits`` is then None."""
    raw = text or ""
    blocks = list(_EDITS_RE.finditer(raw))
    if not blocks:
        return ParsedReply(raw.strip(), None, None, raw)
    m = blocks[-1]
    prose = (raw[:m.start()] + raw[m.end():]).strip()
    body = m.group(1)
    try:
        data = yaml.safe_load(body)
    except yaml.YAMLError as e:
        return ParsedReply(prose, None, f"edits block is not valid YAML: {e}", raw)
    if data is None:
        return ParsedReply(prose, {}, None, raw)
    if not isinstance(data, dict):
        return ParsedReply(prose, None, "edits block must be a mapping of dotted paths to values", raw)
    bad = [k for k in data if not isinstance(k, str)]
    if bad:
        return ParsedReply(prose, None, f"edits block has non-string keys: {bad}", raw)
    return ParsedReply(prose, data, None, raw)


def normalize_edit_path(path: str) -> str:
    p = str(path).strip().strip(".")
    if p.startswith("experiment."):
        p = p[len("experiment."):]
    if not p or not _PATH_RE.match(p):
        raise AssistantError(f"invalid edit path {path!r}")
    return p


def _get_path(obj: Any, keys: Sequence[str]) -> Tuple[bool, Any]:
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
        elif isinstance(obj, list) and k.isdigit() and int(k) < len(obj):
            obj = obj[int(k)]
        else:
            return False, None
    return True, obj


def _set_path(obj: Any, keys: Sequence[str], value: Any) -> None:
    for i, k in enumerate(keys[:-1]):
        nxt_is_index = keys[i + 1].isdigit()
        if isinstance(obj, list):
            if not k.isdigit():
                raise AssistantError(f"path segment {k!r} indexes a list; use a number")
            idx = int(k)
            if idx > len(obj):
                raise AssistantError(f"list index {idx} out of range (length {len(obj)})")
            if idx == len(obj):
                obj.append([] if nxt_is_index else {})
            if not isinstance(obj[idx], (dict, list)):
                obj[idx] = [] if nxt_is_index else {}
            obj = obj[idx]
        else:
            if not isinstance(obj, dict):
                raise AssistantError(f"cannot descend into {k!r}: parent is not a mapping")
            if not isinstance(obj.get(k), (dict, list)):
                obj[k] = [] if nxt_is_index else {}
            obj = obj[k]
    last = keys[-1]
    if isinstance(obj, list):
        if not last.isdigit():
            raise AssistantError(f"path segment {last!r} indexes a list; use a number")
        idx = int(last)
        if idx > len(obj):
            raise AssistantError(f"list index {idx} out of range (length {len(obj)})")
        if idx == len(obj):
            obj.append(value)
        else:
            obj[idx] = value
    elif isinstance(obj, dict):
        obj[last] = value
    else:
        raise AssistantError(f"cannot set {last!r}: parent is not a mapping")


def apply_edits(cfg: dict, edits: Dict[str, Any]) -> Tuple[dict, List[dict]]:
    """A deep copy of ``cfg`` with every ``dotted.path: value`` applied (replace
    semantics), and the change list ``[{path, old, new}]`` (``old`` is
    ``"<unset>"`` when the path did not exist)."""
    new = copy.deepcopy(flatten_experiment(cfg))
    changes: List[dict] = []
    for raw_path, value in edits.items():
        path = normalize_edit_path(raw_path)
        keys = path.split(".")
        existed, old = _get_path(new, keys)
        _set_path(new, keys, copy.deepcopy(value))
        changes.append({"path": path, "old": old if existed else "<unset>", "new": value})
    return new, changes


def _repair_message(errors: Sequence[dict], parse_error: Optional[str]) -> str:
    if parse_error:
        return (f"Your edits block could not be used: {parse_error}. Reply again with the same intent and a "
                f"correct ```edits block (a YAML mapping of dotted paths to values).")
    lines = "\n".join(f"- {e['loc']}: {e['msg']}" for e in errors)
    return ("Applying your edits gives a config that fails validation:\n" + lines +
            "\nReply with a corrected ```edits block (only the paths that must change on top of the working "
            "config, including your previous edits where they were right) and one line saying what you fixed.")


# --------------------------------------------------------------------------
# one chat turn
# --------------------------------------------------------------------------

def chat_turn(paths: ConsolePaths, builder: ContextBuilder, acfg: AssistantConfig, client: Any, *,
              cfg: Optional[dict], history: Sequence[Dict[str, str]], message: str, from_base: bool,
              file: Optional[str], max_repairs: int = 1) -> Iterator[dict]:
    """Yields typed events: ``token`` (a streamed delta), ``edits`` (a parsed
    block applied and validated), ``repair`` (validation failed, asking the
    model again), then ``done`` with the final reply, the applied config (when
    the last attempt produced one), its validation, the change list and usage.
    ``client`` needs ``stream(messages) -> Iterator[str]`` and ``last_usage``."""
    t0 = time.perf_counter()
    working = base_resolved(paths) if from_base or cfg is None else flatten_experiment(cfg)
    message = (message or "").strip()
    if not message:
        raise AssistantError("empty message")
    messages = build_messages(builder, acfg, cfg=working, history=history, message=message,
                              from_base=from_base, file=file)
    yield {"type": "context", "messages": len(messages), "chars": sum(len(m["content"]) for m in messages)}

    usages: List[dict] = []

    def _run(msgs: List[Dict[str, str]]) -> Iterator[Any]:
        buf: List[str] = []
        for tok in client.stream(msgs):
            buf.append(tok)
            yield {"type": "token", "text": tok}
        if getattr(client, "last_usage", None):
            usages.append(dict(client.last_usage))
        yield "".join(buf)

    text = ""
    for ev in _run(messages):
        if isinstance(ev, dict):
            yield ev
        else:
            text = ev
    parsed = parse_reply(text)
    applied: Optional[dict] = None
    rounds = 0
    while True:
        if parsed.error is None and parsed.edits is None:
            break                                  # a plain answer
        if parsed.error is None and parsed.edits is not None:
            try:
                new_cfg, changes = apply_edits(working, parsed.edits)
                v = validate_config(paths, new_cfg, file=file)
                applied = {"config": v["resolved"] if v["ok"] else new_cfg, "ok": v["ok"], "errors": v["errors"],
                           "changes": changes, "cells": v.get("cells") or [], "sweep": bool(v.get("sweep")),
                           "edits": parsed.edits}
                yield {"type": "edits", "ok": v["ok"], "errors": v["errors"], "changes": changes, "edits": parsed.edits}
                if v["ok"]:
                    break
                errors = v["errors"]
            except AssistantError as e:
                applied = None
                parsed = ParsedReply(parsed.prose, None, str(e), parsed.raw)
                errors = []
        else:
            errors = []
        if rounds >= max_repairs:
            break
        rounds += 1
        yield {"type": "repair", "errors": errors, "parse_error": parsed.error}
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": _repair_message(errors, parsed.error)}]
        text = ""
        for ev in _run(messages):
            if isinstance(ev, dict):
                yield ev
            else:
                text = ev
        parsed = parse_reply(text)
        if parsed.edits is None and parsed.error is None:
            break                                  # the model gave up on edits; keep its answer
    usage = None
    if usages:
        usage = {k: sum(u.get(k) or 0 for u in usages) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        usage["rounds"] = len(usages)
    yield {"type": "done", "reply": parsed.prose, "raw": text, "parse_error": parsed.error,
           "applied": applied, "usage": usage, "elapsed_sec": round(time.perf_counter() - t0, 2),
           "from_base": bool(from_base)}


def preview_messages(paths: ConsolePaths, builder: ContextBuilder, acfg: AssistantConfig, *, cfg: Optional[dict],
                     history: Sequence[Dict[str, str]], message: str, from_base: bool, file: Optional[str]) -> dict:
    """The messages a turn would send, with sizes — for the page's *show context* link and for tests."""
    working = base_resolved(paths) if from_base or cfg is None else flatten_experiment(cfg)
    msgs = build_messages(builder, acfg, cfg=working, history=history, message=message or "(your message)",
                          from_base=from_base, file=file)
    chars = sum(len(m["content"]) for m in msgs)
    return {"messages": msgs, "chars": chars, "approx_tokens": int(chars / 3.0), "assistant": acfg.name, "model": acfg.model}   # 3.0 chars/token measured on Gemma 4
