"""Declared parameter specs for the free ``params`` dicts of the config.

``TaskConfig.params`` (and ``TokenizerConfig.params``) are ``Dict[str, Any]``
in Pydantic on purpose — old YAMLs must stay valid — so nothing in the schema
says which keys a task accepts, what they default to or what type they are.
A :class:`ParamSpec` list, returned by ``<Task>.param_spec()``, is that
declaration: **the single source of truth** the console form (typed rows,
defaults greyed, pins), the ``?`` tooltips, the assistant's field reference,
the generated ``configs/tasks/<type>.yaml`` documentation files and the
run-start warnings all read. Validation against a spec is *advisory*
(:func:`validate_params` returns sentences, never raises): an unknown key or a
wrong type is a warning at validation time and at run start, and the run
proceeds exactly as before — the task still reads its params with ``.get``.

Torch-free: the specs live on the task classes (which need torch), this module
only defines the dataclass and the helpers over plain dicts.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ParamType = str
PARAM_TYPES = ("int", "float", "bool", "str", "list[str]", "list[int]", "path")


@dataclass(frozen=True)
class ParamSpec:
    """One declared parameter of a ``params`` dict.

    ``type`` is one of :data:`PARAM_TYPES`; ``default`` is what the code uses
    when the key is absent (the console writes a key only when its value differs
    from this, or when the row is *pinned*); ``help`` is one sentence — the
    tooltip; ``choices`` / ``min`` / ``max`` bound the value; ``nullable``
    admits ``null`` (e.g. *skip BERTScore*); ``advanced`` hides the row behind
    the card's *more…* toggle; ``group`` is a short heading inside the card.
    """
    name: str
    type: ParamType
    default: Any
    help: str
    choices: Optional[Sequence[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    nullable: bool = False
    advanced: bool = False
    group: Optional[str] = None

    def __post_init__(self) -> None:
        if self.type not in PARAM_TYPES:
            raise ValueError(f"ParamSpec {self.name!r}: type must be one of {PARAM_TYPES}, got {self.type!r}")
        if self.choices is not None:
            object.__setattr__(self, "choices", tuple(self.choices))
        if isinstance(self.default, (list, tuple)):
            object.__setattr__(self, "default", list(self.default))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["choices"] = list(self.choices) if self.choices is not None else None
        return d


def specs_by_name(spec: Iterable[ParamSpec]) -> Dict[str, ParamSpec]:
    return {s.name: s for s in spec}


def spec_defaults(spec: Iterable[ParamSpec]) -> Dict[str, Any]:
    """``{name: default}`` in declaration order — what the task uses for an empty ``params``."""
    return {s.name: (list(s.default) if isinstance(s.default, list) else s.default) for s in spec}


def resolve_params(spec: Iterable[ParamSpec], params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The spec defaults overlaid by *params* — every declared key plus any
    undeclared one the file carries (those are kept, never dropped)."""
    out = spec_defaults(spec)
    for k, v in (params or {}).items():
        out[k] = v
    return out


def _type_ok(s: ParamSpec, value: Any) -> bool:
    if value is None:
        return s.nullable
    t = s.type
    if t == "bool":
        return isinstance(value, bool)
    if t == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t in ("str", "path"):
        return isinstance(value, str)
    if t == "list[str]":
        return isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value)
    if t == "list[int]":
        return isinstance(value, (list, tuple)) and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    return True


def _fmt(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def validate_params(spec: Sequence[ParamSpec], params: Optional[Mapping[str, Any]], *,
                    owner: str = "the task") -> List[str]:
    """Advisory findings about *params* against *spec*: unknown key, wrong type,
    out of range, a value outside ``choices``. Human sentences naming the key and
    *owner* (the task type); an empty list when everything is declared and typed.
    An empty spec declares nothing, so it yields no findings (the task documents
    no parameters; nothing to check against)."""
    out: List[str] = []
    if not spec or not params:
        return out
    if not isinstance(params, Mapping):
        return [f"{owner}: params must be a mapping, got {type(params).__name__}"]
    by_name = specs_by_name(spec)
    declared = ", ".join(by_name)
    for key, value in params.items():
        s = by_name.get(str(key))
        if s is None:
            out.append(f"{owner} does not declare a parameter {key!r} — it is ignored at run time "
                       f"(declared: {declared})")
            continue
        if not _type_ok(s, value):
            want = s.type + (" or null" if s.nullable else "")
            out.append(f"{owner}.{key} should be {want}, got {_fmt(value)} ({type(value).__name__})")
            continue
        if value is None:
            continue
        if s.choices is not None and value not in s.choices:
            out.append(f"{owner}.{key} must be one of {', '.join(_fmt(c) for c in s.choices)}, got {_fmt(value)}")
            continue
        if s.type in ("int", "float"):
            if s.min is not None and value < s.min:
                out.append(f"{owner}.{key} must be ≥ {s.min:g}, got {_fmt(value)}")
            if s.max is not None and value > s.max:
                out.append(f"{owner}.{key} must be ≤ {s.max:g}, got {_fmt(value)}")
    return out


def non_default_params(spec: Sequence[ParamSpec], params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """*params* minus the keys whose value equals the spec default (undeclared keys stay)."""
    by_name = specs_by_name(spec)
    out: Dict[str, Any] = {}
    for k, v in (params or {}).items():
        s = by_name.get(k)
        if s is not None and _same(v, s.default):
            continue
        out[k] = v
    return out


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return list(a) == list(b)
    return a == b


# ---------------------------------------------------------------------------
# Generated documentation: configs/tasks/<type>.yaml
# ---------------------------------------------------------------------------

PRESET_HEADER = """\
# GENERATED by scripts/render_task_presets.py from {cls}.param_spec() — do not edit by hand,
# re-run the script after changing the spec (tests/test_task_param_specs.py pins the equality).
#
# A run NEVER reads this file. The pipeline takes a task's parameters from the experiment YAML's
# `sweep.tasks[].params` only; an absent key means the code default listed here. This file is the
# documentation of every parameter the `{type}` task declares, with its default and type; the
# experiment console renders the same spec live as a typed card (`/api/schema` → task_params).
"""


def _yaml_scalar(v: Any) -> str:
    """One YAML scalar in the repo's style: strings double-quoted (JSON escapes are valid YAML
    double-quoted escapes, so ``\\n`` in a stop marker stays visible), numbers / booleans / null
    as YAML spells them."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(v, ensure_ascii=False)


def render_task_preset(task_type: str, cls_name: str, spec: Sequence[ParamSpec]) -> str:
    """The text of ``configs/tasks/<task_type>.yaml`` for *spec*: the header, then
    ``task: {type, params: <every default>}`` with the parameter's type and help as a
    trailing comment, ``group`` headings as comment lines, ``advanced`` rows marked."""
    lines = [PRESET_HEADER.format(cls=cls_name, type=task_type).rstrip("\n"), "task:", f'  type: "{task_type}"', "  params:"]
    group: Any = object()                       # sentinel: the first group heading always prints
    for s in spec:
        if s.group != group:
            group = s.group
            if s.group:
                lines.append(f"    # — {s.group} —")
        meta = s.type + (" | null" if s.nullable else "")
        if s.choices is not None:
            meta += "; one of " + ", ".join(_fmt(c) for c in s.choices)
        if s.min is not None or s.max is not None:
            lo = f"{s.min:g}" if s.min is not None else "…"
            hi = f"{s.max:g}" if s.max is not None else "…"
            meta += f"; range {lo}–{hi}"
        if s.advanced:
            meta += "; advanced"
        lines.append(f"    # {' '.join(s.help.split())} [{meta}]")
        if isinstance(s.default, list):
            if s.default:
                lines.append(f"    {s.name}:")
                lines.extend(f"      - {_yaml_scalar(x)}" for x in s.default)
            else:
                lines.append(f"    {s.name}: []")
        else:
            lines.append(f"    {s.name}: {_yaml_scalar(s.default)}")
    return "\n".join(lines) + "\n"


def _registry_specs(registry: Any) -> Dict[str, List[ParamSpec]]:
    out: Dict[str, List[ParamSpec]] = {}
    for key in registry.list_available():
        cls = registry.get(key)
        fn = getattr(cls, "param_spec", None)
        try:
            out[key] = list(fn()) if callable(fn) else []
        except Exception:  # noqa: BLE001 — a broken spec must not take the console down
            out[key] = []
    return out


def task_param_specs() -> Dict[str, List[ParamSpec]]:
    """``{task_type: spec}`` for every registered task (an empty list for a task that
    declares none). Imports the task registry lazily; ``{}`` when torch is missing."""
    try:
        import arabic_eval.tasks  # noqa: F401  (registers the tasks)
        from arabic_eval.registry import task_registry
    except Exception:  # noqa: BLE001
        return {}
    return _registry_specs(task_registry)


def tokenizer_param_specs() -> Dict[str, List[ParamSpec]]:
    """``{tokenizer_type: spec}`` for every registered tokenizer (torch-free registry)."""
    try:
        import arabic_eval.tokenizers  # noqa: F401  (registers the tokenizers)
        from arabic_eval.registry import tokenizer_registry
    except Exception:  # noqa: BLE001
        return {}
    return _registry_specs(tokenizer_registry)
