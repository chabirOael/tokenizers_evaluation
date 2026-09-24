"""LLM-judge scoring of the free-form generations (stage 2 of the free-form eval).

Reads every cell's ``eval_rows/freeform_cidar.parquet`` (written by the task
during the experiment), asks a judge model for a verdict per row, writes
``freeform_judge/<judge>.parquet`` next to it, merges a compact summary into
``all_metrics.json["downstream"]["freeform_cidar"]["judge"][<judge>]`` and
regenerates the sweep's comparison report. Decoupled from the experiment on
purpose: a 31B judge does not share the GPU with a training run, judges get
swapped or added, and the human check reads the same files.

Protocol: pointwise, reference-guided, 1–5 on a fixed rubric (``RUBRICS``),
plus three sub-scores and a closed set of flags, temperature 0, fixed seed.
Every variant answers the same prompts, so comparisons are paired: the
summary carries a paired-bootstrap CI on the mean difference to a baseline
cell and the win rate against it. With two or more judges per cell, their
agreement (Spearman, exact, quadratic-weighted kappa) is recorded too.

Backends: ``vllm`` (local, structured JSON through vLLM's structured outputs;
``.venv-judge``) and ``openai`` (any OpenAI-compatible chat endpoint, plain
``urllib``, ``response_format`` JSON schema when ``structured_json`` is on).
``structured_json`` is a per-judge toggle and the prose parser is the
fallback either way — constrained decoding is known to bias some models
toward the first enum value, so it must be switchable per judge.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import yaml

logger = logging.getLogger(__name__)

TASK = "freeform_cidar"
GENERATIONS_FILE = Path("eval_rows") / f"{TASK}.parquet"
JUDGE_DIR = Path("freeform_judge")
FLAGS: Tuple[str, ...] = ("repetition", "wrong_language", "empty", "off_topic", "truncated")
SUBSCORES: Tuple[str, ...] = ("correctness", "fluency", "instruction_following")

VERDICT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
        "fluency": {"type": "integer", "minimum": 1, "maximum": 5},
        "instruction_following": {"type": "integer", "minimum": 1, "maximum": 5},
        "flags": {"type": "array", "items": {"type": "string", "enum": list(FLAGS)}},
        "rationale": {"type": "string"},
    },
    "required": ["score", "correctness", "fluency", "instruction_following", "flags", "rationale"],
    "additionalProperties": False,
}

RUBRIC_V1 = """You are a strict evaluator of Arabic instruction following. You will see an instruction in Arabic, a reference answer written by a human, and a candidate answer produced by a small language model. Judge the candidate, not the reference.

Score the candidate from 1 to 5, using the reference as a guide to what a correct and complete answer contains. The reference is not the only acceptable answer: a different but correct, complete and fluent answer in Modern Standard Arabic also deserves 5. Do not reward length; a short complete answer beats a long padded one. Judge meaning, not wording.

Scale:
5 = correct, complete, fluent Modern Standard Arabic, does exactly what the instruction asks
4 = minor omissions or small language errors; still clearly useful
3 = partially correct or only partly follows the instruction; noticeable errors
2 = mostly wrong, off-topic or barely follows the instruction, but contains some relevant Arabic content
1 = empty, not Arabic, a repetition loop, gibberish, or entirely off-topic

Also give three sub-scores on the same 1-5 scale: "correctness" (agreement with what the reference establishes as true and complete), "fluency" (grammatical, natural Modern Standard Arabic), "instruction_following" (does what was asked, in the form asked).
Flags, any that apply or none: "repetition" (loops or repeats phrases), "wrong_language" (mostly not Arabic), "empty" (no answer), "off_topic" (does not address the instruction), "truncated" (cut off mid-sentence).

Reply with one JSON object and nothing else:
{"score": <1-5>, "correctness": <1-5>, "fluency": <1-5>, "instruction_following": <1-5>, "flags": [<flags>], "rationale": "<one sentence in English>"}

### Instruction
{instruction}
{context_block}### Reference answer
{reference}

### Candidate answer
{candidate}"""

RUBRICS: Dict[str, str] = {"default_v1": RUBRIC_V1}
EMPTY_CANDIDATE = "[EMPTY ANSWER]"


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class JudgeConfig:
    name: str
    backend: str                      # "vllm" | "openai"
    model: str
    rubric: str = "default_v1"
    temperature: Optional[float] = 0.0
    max_tokens: int = 256
    seed: Optional[int] = 0
    structured_json: bool = True      # per-judge toggle; the prose parser is always the fallback
    # vllm
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.92
    dtype: str = "bfloat16"
    batch_size: int = 64
    # openai-compatible
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    concurrency: int = 8
    timeout_sec: float = 120.0
    max_retries: int = 5
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in ("vllm", "openai"):
            raise ValueError(f"judge {self.name!r}: backend must be 'vllm' or 'openai', got {self.backend!r}")
        if self.rubric not in RUBRICS:
            raise ValueError(f"judge {self.name!r}: unknown rubric {self.rubric!r}; known: {sorted(RUBRICS)}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name):
            raise ValueError(f"judge name {self.name!r} must be a file-safe identifier")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "JudgeConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        d = raw.get("judge", raw)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"{path}: unknown judge keys {unknown}; known: {sorted(known)}")
        return cls(**d)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# prompt + parsing
# --------------------------------------------------------------------------

def build_messages(instruction: str, context: str, reference: str, candidate: str,
                   rubric: str = "default_v1") -> List[Dict[str, str]]:
    """One user message (Gemma's chat template has no system role)."""
    cand = candidate.strip() or EMPTY_CANDIDATE
    ctx = f"\n### Input\n{context.strip()}\n\n" if (context or "").strip() else "\n"
    text = RUBRICS[rubric]
    for key, val in (("{instruction}", instruction.strip()), ("{context_block}", ctx),
                     ("{reference}", reference.strip()), ("{candidate}", cand)):
        text = text.replace(key, val)          # not str.format: the rubric holds literal JSON braces
    return [{"role": "user", "content": text}]


@dataclass
class Verdict:
    score: Optional[int]
    correctness: Optional[int]
    fluency: Optional[int]
    instruction_following: Optional[int]
    flags: List[str]
    rationale: str
    parse_ok: bool
    raw: str

    def to_row(self) -> Dict[str, Any]:
        return {"score": self.score, "correctness": self.correctness, "fluency": self.fluency,
                "instruction_following": self.instruction_following, "flags": ",".join(self.flags),
                "rationale": self.rationale, "parse_ok": self.parse_ok, "raw": self.raw}


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*([1-5])\b', re.IGNORECASE)
_SUB_RE = {k: re.compile(rf'"?{k}"?\s*[:=]\s*([1-5])\b', re.IGNORECASE) for k in SUBSCORES}


def strip_fences(text: str) -> str:
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text.strip()


def _int_1_5(v: Any) -> Optional[int]:
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    return i if 1 <= i <= 5 else None


def parse_verdict(text: str) -> Verdict:
    """JSON first (fences stripped, then the first balanced object), the prose
    regex parser as the fallback. ``parse_ok`` is False when no 1–5 score can
    be read at all."""
    raw = text or ""
    body = strip_fences(raw)
    obj: Optional[Dict[str, Any]] = None
    for cand in (body, *_json_objects(body)):
        try:
            v = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(v, dict) and "score" in v:
            obj = v
            break
    if obj is not None:
        score = _int_1_5(obj.get("score"))
        subs = {k: _int_1_5(obj.get(k)) for k in SUBSCORES}
        flags_raw = obj.get("flags") or []
        flags = [f for f in flags_raw if isinstance(f, str) and f in FLAGS] if isinstance(flags_raw, list) else []
        rationale = str(obj.get("rationale") or "").strip()
        return Verdict(score, subs["correctness"], subs["fluency"], subs["instruction_following"],
                       flags, rationale, parse_ok=score is not None, raw=raw)
    m = _SCORE_RE.search(body)
    score = _int_1_5(m.group(1)) if m else None
    subs = {k: (_int_1_5(mm.group(1)) if (mm := rx.search(body)) else None) for k, rx in _SUB_RE.items()}
    low = body.lower()
    flags = [f for f in FLAGS if f in low]
    return Verdict(score, subs["correctness"], subs["fluency"], subs["instruction_following"],
                   flags, body[:300], parse_ok=score is not None, raw=raw)


def _json_objects(text: str) -> Iterable[str]:
    """Balanced ``{...}`` substrings, outermost first."""
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start:i + 1]


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

class JudgeBackend(Protocol):
    def complete(self, conversations: Sequence[List[Dict[str, str]]]) -> List[str]: ...


class VllmBackend:
    """Local vLLM (run from ``.venv-judge`` with ``scripts/judge/run_judge.sh``)."""

    def __init__(self, cfg: JudgeConfig) -> None:
        from vllm import LLM, SamplingParams
        self.cfg = cfg
        kw: Dict[str, Any] = dict(model=cfg.model, dtype=cfg.dtype, max_model_len=cfg.max_model_len,
                                  gpu_memory_utilization=cfg.gpu_memory_utilization, limit_mm_per_prompt={"image": 0, "audio": 0})
        try:
            self.llm = LLM(**kw)
        except TypeError:
            kw.pop("limit_mm_per_prompt")
            self.llm = LLM(**kw)
        sp: Dict[str, Any] = dict(temperature=0.0 if cfg.temperature is None else cfg.temperature,
                                  max_tokens=cfg.max_tokens, seed=cfg.seed)
        if cfg.structured_json:
            try:
                from vllm.sampling_params import StructuredOutputsParams
                sp["structured_outputs"] = StructuredOutputsParams(json=VERDICT_SCHEMA)
            except ImportError:
                from vllm.sampling_params import GuidedDecodingParams
                sp["guided_decoding"] = GuidedDecodingParams(json=VERDICT_SCHEMA)
        self.sampling = SamplingParams(**sp)
        logger.info("[judge:%s] vLLM %s, structured_json=%s", cfg.name, cfg.model, cfg.structured_json)

    def complete(self, conversations: Sequence[List[Dict[str, str]]]) -> List[str]:
        out: List[str] = []
        for s in range(0, len(conversations), self.cfg.batch_size):
            res = self.llm.chat(list(conversations[s:s + self.cfg.batch_size]), self.sampling, use_tqdm=False)
            out.extend(r.outputs[0].text for r in res)
        return out


class OpenAICompatibleBackend:
    """Any OpenAI-compatible ``/chat/completions`` endpoint over plain urllib.
    ``max_completion_tokens`` is sent (the current name); ``temperature`` and
    ``seed`` only when set; ``response_format`` JSON schema when
    ``structured_json``; ``extra_body`` is merged last for endpoint quirks."""

    def __init__(self, cfg: JudgeConfig) -> None:
        self.cfg = cfg
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise RuntimeError(f"judge {cfg.name!r}: environment variable {cfg.api_key_env} is not set")
        self._key = key
        self.url = (cfg.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"

    def _body(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        b: Dict[str, Any] = {"model": self.cfg.model, "messages": messages, "max_completion_tokens": self.cfg.max_tokens}
        if self.cfg.temperature is not None:
            b["temperature"] = self.cfg.temperature
        if self.cfg.seed is not None:
            b["seed"] = self.cfg.seed
        if self.cfg.structured_json:
            b["response_format"] = {"type": "json_schema",
                                    "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA, "strict": True}}
        b.update(self.cfg.extra_body)
        return b

    def _one(self, messages: List[Dict[str, str]]) -> str:
        data = json.dumps(self._body(messages)).encode("utf-8")
        delay = 1.0
        for attempt in range(self.cfg.max_retries + 1):
            req = urllib.request.Request(self.url, data=data, method="POST", headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self._key}"})
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_sec) as r:
                    payload = json.loads(r.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:500]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.cfg.max_retries:
                    logger.warning("[judge:%s] HTTP %s, retry in %.0fs: %s", self.cfg.name, e.code, delay, body)
                else:
                    raise RuntimeError(f"judge {self.cfg.name!r}: HTTP {e.code}: {body}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt >= self.cfg.max_retries:
                    raise
                logger.warning("[judge:%s] %s, retry in %.0fs", self.cfg.name, e, delay)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable")

    def complete(self, conversations: Sequence[List[Dict[str, str]]]) -> List[str]:
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.concurrency)) as ex:
            return list(ex.map(self._one, conversations))


def make_backend(cfg: JudgeConfig) -> JudgeBackend:
    return VllmBackend(cfg) if cfg.backend == "vllm" else OpenAICompatibleBackend(cfg)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def _mean(xs: Sequence[float]) -> Optional[float]:
    return round(statistics.fmean(xs), 4) if xs else None


def _std(xs: Sequence[float]) -> Optional[float]:
    return round(statistics.pstdev(xs), 4) if len(xs) > 1 else (0.0 if xs else None)


def paired_bootstrap(base: Dict[str, float], other: Dict[str, float], n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Mean of (other − base) over the shared ids with a percentile bootstrap
    CI, plus win / tie / loss rates of ``other`` against ``base``."""
    import numpy as np
    ids = sorted(set(base) & set(other))
    if not ids:
        return {"n": 0, "delta_mean": None, "ci_low": None, "ci_high": None, "win_rate": None, "tie_rate": None, "loss_rate": None}
    d = np.array([other[i] - base[i] for i in ids], dtype=float)
    rng = np.random.RandomState(seed)
    boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"n": len(ids), "delta_mean": round(float(d.mean()), 4),
            "ci_low": round(float(np.percentile(boots, 2.5)), 4), "ci_high": round(float(np.percentile(boots, 97.5)), 4),
            "win_rate": round(float((d > 0).mean()), 4), "tie_rate": round(float((d == 0).mean()), 4),
            "loss_rate": round(float((d < 0).mean()), 4)}


def _ranks(x: Sequence[float]) -> List[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 4) if den else None


def quadratic_weighted_kappa(x: Sequence[int], y: Sequence[int], lo: int = 1, hi: int = 5) -> Optional[float]:
    if len(x) != len(y) or not x:
        return None
    k = hi - lo + 1
    obs = [[0] * k for _ in range(k)]
    for a, b in zip(x, y):
        obs[a - lo][b - lo] += 1
    n = len(x)
    hx = [sum(row) for row in obs]
    hy = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * obs[i][j] / n
            den += w * (hx[i] * hy[j]) / (n * n)
    return round(1 - num / den, 4) if den else None


def exact_agreement(x: Sequence[int], y: Sequence[int]) -> Optional[float]:
    return round(sum(1 for a, b in zip(x, y) if a == b) / len(x), 4) if x else None


# --------------------------------------------------------------------------
# cells: generations in, verdicts out
# --------------------------------------------------------------------------

def _read_parquet(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    meta = {}
    if t.schema.metadata and b"arabic_eval" in t.schema.metadata:
        meta = json.loads(t.schema.metadata[b"arabic_eval"].decode("utf-8"))
    return t.to_pylist(), meta


def _write_parquet(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str], metadata: Dict[str, Any]) -> None:
    from arabic_eval.utils.io import write_report_table
    write_report_table(path, rows, fields, metadata=metadata)


def load_generations(cell_dir: Path) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    p = Path(cell_dir) / GENERATIONS_FILE
    return _read_parquet(p) if p.exists() else None


def judge_file(cell_dir: Path, judge_name: str) -> Path:
    return Path(cell_dir) / JUDGE_DIR / f"{judge_name}.parquet"


JUDGE_FIELDS = ["id", "stratum", "score", "correctness", "fluency", "instruction_following", "flags", "rationale", "parse_ok", "raw"]


def judge_cell(cell_dir: Path, cfg: JudgeConfig, backend: JudgeBackend, limit: Optional[int] = None,
               overwrite: bool = False) -> Optional[List[Dict[str, Any]]]:
    """Score one cell's generations; returns the verdict rows (existing file
    reused unless ``overwrite``), ``None`` when the cell has no generations."""
    cell_dir = Path(cell_dir)
    loaded = load_generations(cell_dir)
    if loaded is None:
        return None
    gens, gmeta = loaded
    if limit is not None:
        gens = gens[: int(limit)]
    out = judge_file(cell_dir, cfg.name)
    if out.exists() and not overwrite:
        rows, jmeta = _read_parquet(out)
        if jmeta.get("num_rows") == len(gens):
            logger.info("[judge:%s] %s: reusing %d verdicts", cfg.name, cell_dir.name, len(rows))
            return rows
    convs = [build_messages(g["instruction"], g.get("context") or "", g["reference"], g["generation"], cfg.rubric) for g in gens]
    t0 = time.time()
    texts = backend.complete(convs)
    dt = time.time() - t0
    rows = []
    for g, t in zip(gens, texts):
        v = parse_verdict(t)
        rows.append({"id": g["id"], "stratum": g.get("stratum") or "", **v.to_row()})
    _write_parquet(out, rows, JUDGE_FIELDS, metadata={
        "task": TASK, "kind": "freeform_judgments", "schema_version": 1, "judge": cfg.to_json(),
        "rubric_text_sha1": __import__("hashlib").sha1(RUBRICS[cfg.rubric].encode("utf-8")).hexdigest(),
        "generations_heldout_sha256": gmeta.get("heldout_sha256"), "num_rows": len(rows),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(dt, 1)})
    logger.info("[judge:%s] %s: %d verdicts in %.0fs → %s", cfg.name, cell_dir.name, len(rows), dt, out)
    return rows


def summarize_verdicts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    parsed = [r for r in rows if r.get("parse_ok") and r.get("score") is not None]
    scores = [float(r["score"]) for r in parsed]
    hist = {str(k): sum(1 for s in scores if int(s) == k) for k in range(1, 6)}
    flags_of = lambda r: [f for f in (r.get("flags") or "").split(",") if f]  # noqa: E731
    by_stratum: Dict[str, List[float]] = {}
    for r in parsed:
        by_stratum.setdefault(r.get("stratum") or "", []).append(float(r["score"]))
    return {
        "n": len(rows), "n_parsed": len(parsed),
        "parse_fail_rate": round(1 - len(parsed) / len(rows), 4) if rows else None,
        "score_mean": _mean(scores), "score_std": _std(scores),
        "score_se": round(statistics.pstdev(scores) / math.sqrt(len(scores)), 4) if len(scores) > 1 else None,
        "score_hist": hist,
        **{f"{k}_mean": _mean([float(r[k]) for r in parsed if r.get(k) is not None]) for k in SUBSCORES},
        "flag_rates": {f: round(sum(1 for r in rows if f in flags_of(r)) / len(rows), 4) if rows else None for f in FLAGS},
        "any_flag_rate": round(sum(1 for r in rows if flags_of(r)) / len(rows), 4) if rows else None,
        "score_by_stratum": {k: _mean(v) for k, v in sorted(by_stratum.items()) if k},
    }


def scores_by_id(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    return {r["id"]: float(r["score"]) for r in rows if r.get("parse_ok") and r.get("score") is not None}


# --------------------------------------------------------------------------
# sweep-level run
# --------------------------------------------------------------------------

def discover_cells(root: Path) -> List[Path]:
    root = Path(root)
    if (root / "all_metrics.json").exists():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "all_metrics.json").exists())


def choose_baseline(cells: Sequence[Path], name: Optional[str] = None) -> Optional[Path]:
    if name:
        for c in cells:
            if c.name == name:
                return c
        raise ValueError(f"baseline cell {name!r} not among {[c.name for c in cells]}")
    for c in cells:
        if c.name.startswith("native_"):
            return c
    return cells[0] if cells else None


def _load_all_metrics(cell: Path) -> Dict[str, Any]:
    with open(cell / "all_metrics.json", encoding="utf-8") as f:
        return json.load(f)


def _save_all_metrics(cell: Path, data: Dict[str, Any]) -> None:
    from arabic_eval.utils.io import save_json
    save_json(data, cell / "all_metrics.json")


def compact_summary(summary: Dict[str, Any], vs_baseline: Optional[Dict[str, Any]], cfg: JudgeConfig,
                    baseline_name: Optional[str], vs_baseline_status: str = "ok") -> Dict[str, Any]:
    """What goes into all_metrics.json — small enough for the comparison table.

    ``vs_baseline`` is ``None`` for two different reasons — this cell *is* the
    baseline, or the baseline carries no verdicts of this judge (possible since
    a run may judge a subset of the cells). ``vs_baseline_status`` says which:
    ``ok`` | ``is_baseline`` | ``baseline_not_judged`` | ``no_baseline``."""
    out = {"model": cfg.model, "backend": cfg.backend, "rubric": cfg.rubric, "structured_json": cfg.structured_json,
           "n": summary["n"], "parse_fail_rate": summary["parse_fail_rate"],
           "score_mean": summary["score_mean"], "score_std": summary["score_std"], "score_se": summary["score_se"],
           "score_hist": summary["score_hist"],
           **{f"{k}_mean": summary[f"{k}_mean"] for k in SUBSCORES},
           "any_flag_rate": summary["any_flag_rate"], "flag_rates": summary["flag_rates"],
           "score_by_stratum": summary["score_by_stratum"],
           "baseline": baseline_name, "vs_baseline": vs_baseline, "vs_baseline_status": vs_baseline_status}
    return out


def _existing_scores(cell: Path, judge_name: str) -> Optional[Dict[str, float]]:
    """The verdicts a previous run wrote for *cell*, or ``None``. Lets a run that
    judges a subset still compare against a baseline it is not re-judging — a
    file read, never a backend call."""
    f = judge_file(cell, judge_name)
    if not f.exists():
        return None
    try:
        return scores_by_id(_read_parquet(f)[0])
    except Exception as e:  # noqa: BLE001
        logger.warning("[judge:%s] %s: unreadable verdict file (%s)", judge_name, cell.name, e)
        return None


def select_cells(with_gens: Sequence[Path], names: Optional[Sequence[str]]) -> List[Path]:
    """The cells to judge: *names* in the order they were given, or every cell
    with generations when *names* is empty / None. An unknown name is an error
    naming the valid ones — never a silent skip."""
    if not names:
        return list(with_gens)
    by_name = {c.name: c for c in with_gens}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"no cell with generations named {', '.join(unknown)} "
                         f"(have: {', '.join(sorted(by_name)) or 'none'})")
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(by_name[n])
    return out


def run(root: Path | str, cfgs: Sequence[JudgeConfig], backends: Dict[str, JudgeBackend], baseline: Optional[str] = None,
        limit: Optional[int] = None, overwrite: bool = False, regenerate_report: bool = True,
        cells: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Judge the cells under ``root`` with every judge, merge the summaries into
    those cells' ``all_metrics.json`` and regenerate the sweep report.

    ``cells`` selects which cells are judged (default: every cell with
    generations). Only the selected cells are judged and rewritten; the
    comparison report is still built from **all** cells, and the paired
    comparison still runs — when the baseline is outside the selection its
    verdicts are read from the file a previous run wrote."""
    root = Path(root)
    all_cells = discover_cells(root)
    if not all_cells:
        raise FileNotFoundError(f"no cell with all_metrics.json under {root}")
    with_gens = [c for c in all_cells if (c / GENERATIONS_FILE).exists()]
    selected = select_cells(with_gens, cells)
    base = choose_baseline(with_gens, baseline)
    report: Dict[str, Any] = {"root": str(root), "cells": [c.name for c in all_cells],
                              "cells_with_generations": [c.name for c in with_gens],
                              "cells_judged": [c.name for c in selected],
                              "baseline": base.name if base else None, "judges": {}}
    if cells:
        logger.info("judging %d of %d cell(s) with generations: %s",
                    len(selected), len(with_gens), ", ".join(c.name for c in selected))
    verdicts: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}       # judge → cell → rows
    for cfg in cfgs:
        verdicts[cfg.name] = {}
        for c in selected:
            rows = judge_cell(c, cfg, backends[cfg.name], limit=limit, overwrite=overwrite)
            if rows is not None:
                verdicts[cfg.name][c.name] = rows
    # summaries + paired comparisons, merged into all_metrics.json
    for cfg in cfgs:
        per_cell: Dict[str, Any] = {}
        if base is None:
            base_scores, base_status = None, "no_baseline"
        elif base.name in verdicts[cfg.name]:                       # judged in this run
            base_scores, base_status = scores_by_id(verdicts[cfg.name][base.name]), "ok"
        else:                                                       # outside the selection: read its file
            base_scores = _existing_scores(base, cfg.name)
            base_status = "ok" if base_scores else "baseline_not_judged"
            if base_scores is None:
                logger.warning("[judge:%s] baseline %s has no verdicts — no paired comparison this run",
                               cfg.name, base.name)
        for c in selected:
            rows = verdicts[cfg.name].get(c.name)
            if rows is None:
                continue
            summ = summarize_verdicts(rows)
            is_base = base is not None and c.name == base.name
            vs = paired_bootstrap(base_scores, scores_by_id(rows)) if base_scores is not None and not is_base else None
            status = "is_baseline" if is_base else base_status
            compact = compact_summary(summ, vs, cfg, base.name if base else None, status)
            per_cell[c.name] = compact
            am = _load_all_metrics(c)
            task_block = am.setdefault("downstream", {}).setdefault(TASK, {})
            task_block.setdefault("judge", {})[cfg.name] = compact
            _save_all_metrics(c, am)
        report["judges"][cfg.name] = per_cell
    # judge-judge agreement of the cells this run touched (every judge file present, not only this run's)
    agreement: Dict[str, Any] = {}
    for c in selected:
        files = sorted((c / JUDGE_DIR).glob("*.parquet")) if (c / JUDGE_DIR).exists() else []
        by_judge = {p.stem: scores_by_id(_read_parquet(p)[0]) for p in files}
        pairs: Dict[str, Any] = {}
        names = sorted(by_judge)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = by_judge[names[i]], by_judge[names[j]]
                ids = sorted(set(a) & set(b))
                x = [int(a[k]) for k in ids]
                y = [int(b[k]) for k in ids]
                pairs[f"{names[i]}|{names[j]}"] = {"n": len(ids), "spearman": spearman(x, y),
                                                    "exact": exact_agreement(x, y), "qwk": quadratic_weighted_kappa(x, y)}
        if pairs:
            am = _load_all_metrics(c)
            am.setdefault("downstream", {}).setdefault(TASK, {})["judge_agreement"] = pairs
            _save_all_metrics(c, am)
            agreement[c.name] = pairs
    report["judge_agreement"] = agreement
    if regenerate_report and len(all_cells) > 1 and root not in all_cells:   # the report is whole-experiment
        from arabic_eval.evaluation.reporter import generate_report
        generate_report({c.name: _load_all_metrics(c) for c in all_cells}, root / "comparison_report.txt")
        report["comparison_report"] = str(root / "comparison_report.txt")
    return report


def format_table(report: Dict[str, Any]) -> str:
    from tabulate import tabulate
    lines = []
    for judge, cells in report["judges"].items():
        rows = []
        for name, s in cells.items():
            vs = s.get("vs_baseline") or {}
            se = s.get("score_se")
            rows.append([name, s["n"], f"{s['score_mean']:.3f} ± {1.96 * se:.3f}" if s["score_mean"] is not None and se is not None else "—",
                         (f"{vs['delta_mean']:+.3f} [{vs['ci_low']:+.3f}, {vs['ci_high']:+.3f}]" if vs.get("delta_mean") is not None else ("baseline" if name == report["baseline"] else "—")),
                         f"{100 * vs['win_rate']:.0f} / {100 * vs['tie_rate']:.0f} / {100 * vs['loss_rate']:.0f}" if vs.get("win_rate") is not None else "—",
                         f"{100 * (s['any_flag_rate'] or 0):.0f}%", f"{100 * (s['parse_fail_rate'] or 0):.1f}%"])
        lines.append(f"\n## judge: {judge}   (baseline: {report['baseline']})")
        lines.append(tabulate(rows, headers=["cell", "n", "score (±95%)", "Δ vs baseline [95% CI]", "win/tie/loss %", "flagged", "parse fail"], tablefmt="github"))
    for cell, pairs in (report.get("judge_agreement") or {}).items():
        for pair, a in pairs.items():
            lines.append(f"agreement {cell} {pair}: n={a['n']} spearman={a['spearman']} exact={a['exact']} qwk={a['qwk']}")
    return "\n".join(lines)
