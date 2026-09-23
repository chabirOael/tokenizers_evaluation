"""The teacher side of the distillation experiment: the fixed system prompt, the
chat turn a teacher answers, the candidate registry and JSONL helpers.

Stdlib + PyYAML only — ``scripts/distill/generate_teacher_answers.py`` imports
this module from the vLLM venv (``.venv-judge``), which has neither sacrebleu nor
the tokenizer stack of the main venv.

Rules the functions encode (docs/report.md §3.8):
  * one system prompt for every teacher (``SYSTEM_PROMPT``; its sha256 goes in
    every manifest); a chat template without a system role gets it prepended to
    the user turn (``build_conversation(system_role=False)``);
  * the user turn is the instruction, a blank line and the input when the record
    has one (``user_turn``) — never our Alpaca-AR training template, which is the
    student's prompt only;
  * the bake-off prompt set is stratified with the held-out set's reference-length
    boundaries (``stratum_of``) so ``score_by_stratum`` is comparable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

# Gloss: answer in Modern Standard Arabic; answer the request directly, no preamble or
# opening phrase, do not restate the question; length appropriate to the request; no
# markdown unless a list or a table is asked for; no language other than Arabic except
# for names or technical terms.
SYSTEM_PROMPT = (
    "أنت مساعد يجيب باللغة العربية الفصحى. أجب عن الطلب مباشرةً دون مقدمات أو عبارات افتتاحية ودون إعادة "
    "صياغة السؤال. اجعل طول الإجابة مناسبًا للطلب. لا تستخدم تنسيق ماركداون إلا إذا طُلبت قائمة أو جدول. "
    "لا تستخدم لغة غير العربية إلا للأسماء أو المصطلحات عند الضرورة."
)

# Chat-template switches every teacher gets: a "thinking" mode, where a template
# offers one, is off (the answer is the visible text only). Unknown keys are ignored
# by Jinja templates that do not read them.
CHAT_TEMPLATE_KWARGS: Dict[str, Any] = {"enable_thinking": False}

# The held-out set's reference-length tercile boundaries
# (configs/contamination/freeform_cidar_heldout_v1.manifest.json → strata.boundaries_ref_chars).
DEFAULT_STRATA_BOUNDS: Tuple[float, float] = (141.0, 300.0)
STRATA_LABELS: Tuple[str, str, str] = ("short", "medium", "long")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def user_turn(instruction: str, context: str = "") -> str:
    """The instruction, then a blank line and the input when the record has one."""
    instruction = (instruction or "").strip()
    context = (context or "").strip()
    return f"{instruction}\n\n{context}" if context else instruction


def build_conversation(instruction: str, context: str = "", system_role: bool = True,
                       system_prompt: str = SYSTEM_PROMPT) -> List[Dict[str, str]]:
    """The chat a teacher answers. ``system_role=False`` (a template without a
    system role) prepends the system text to the user turn, separated by a blank line."""
    turn = user_turn(instruction, context)
    if system_role:
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": turn}]
    return [{"role": "user", "content": f"{system_prompt}\n\n{turn}"}]


def stratum_of(ref_chars: int, bounds: Sequence[float] = DEFAULT_STRATA_BOUNDS) -> str:
    """``short`` (≤ bounds[0]) / ``medium`` (≤ bounds[1]) / ``long`` — the held-out rule."""
    for label, b in zip(STRATA_LABELS, bounds):
        if ref_chars <= b:
            return label
    return STRATA_LABELS[-1]


# --------------------------------------------------------------------------
# candidate registry (configs/distill/teacher_candidates.yaml)
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    slug: str
    repo: str                              # the repo the brief names (provenance)
    revision: str                          # the snapshot commit the run loads
    load_repo: Optional[str] = None        # where the weights are loaded from, when it differs (a cached alias)
    dtype: str = "bfloat16"
    quantization: Optional[str] = None
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    fallbacks: List[Dict[str, Any]] = field(default_factory=list)   # tried in order when loading fails
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)     # extra vLLM LLM(...) kwargs (capacity knobs)
    reference: bool = False                # a reference row, not a candidate
    params_b: Optional[float] = None       # parameter count in billions (the last tie-break: the smaller model)
    notes: str = ""

    @property
    def model(self) -> str:
        return self.load_repo or self.repo


def load_candidates(path: Path | str) -> Tuple[Dict[str, Any], List[Candidate]]:
    import yaml
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    defaults = {k: v for k, v in raw.items() if k != "candidates"}
    out = []
    for c in raw.get("candidates") or []:
        kw = {**{k: defaults[k] for k in ("max_model_len",) if k in defaults}, **c}
        out.append(Candidate(**kw))
    return defaults, out


def candidate(path: Path | str, slug: str) -> Tuple[Dict[str, Any], Candidate]:
    defaults, cands = load_candidates(path)
    for c in cands:
        if c.slug == slug:
            return defaults, c
    raise KeyError(f"no candidate {slug!r} in {path} (have {[c.slug for c in cands]})")


# --------------------------------------------------------------------------
# JSONL
# --------------------------------------------------------------------------

def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: Path | str) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n
