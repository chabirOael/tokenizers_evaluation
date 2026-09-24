"""Back-end of the experiment console (``debugger/serve_experiment_console.py``).

Everything the page needs, with no HTTP in it so it is testable directly:

* **Schema + presets** — the JSON schema of :class:`ExperimentConfig`, the
  resolved ``configs/base.yaml`` defaults, the registry keys and the preset
  files under ``configs/{tokenizers,models,tasks}``.
* **Config files** — list / read / validate / render / parse / save YAML
  experiment configs under ``configs/experiments``. Validation is the real
  ``load_config`` merge with ``base.yaml`` + Pydantic; rendering produces
  either a *full* explicit YAML (every resolved value) or a *delta* over
  ``base.yaml``; both reload to the same resolved config.
* **Runs** — :class:`RunManager` launches ``scripts/run_experiment.py`` as a
  detached session leader (``setsid``) whose stdout/stderr go to a file, so
  the run outlives the browser tab, the SSH session and the server itself.
  Each run lives in ``outputs/runs/<run_id>/`` (``run.json``, the config
  snapshot the run was started from, ``console.log``, ``exit_code``), and
  the manager reconciles status from that directory alone, so a restarted
  server rediscovers every run. Cancel = SIGTERM to the process group,
  SIGKILL after a grace period.
* **Progress** — :func:`parse_progress` turns the console log into the
  current cell / stage / phase step / eval progress.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, get_args

import yaml
from pydantic import ValidationError

from arabic_eval.config import EXPERIMENT_KEYS, DatasetName, ExperimentConfig, _deep_merge, load_yaml
from arabic_eval.config_edit import (
    default_output_dir, get_created_at, now_iso, record_run_start, rewrite_experiment_keys,
    stamp_created_at, strip_runs,
)
from arabic_eval.evaluation.eval_rows import SUPERSEDED_DIR
from arabic_eval.params_spec import ParamSpec, task_param_specs, tokenizer_param_specs, validate_params
from arabic_eval.tools.config_hints import FIELD_HINTS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_EXPERIMENT_KEYS = EXPERIMENT_KEYS
_SECTION_ORDER = ("data", "tokenizer", "model", "task", "training", "evaluation", "tracking", "sweep")
log = logging.getLogger(__name__)
CANCEL_GRACE_SEC = 15.0
_MAX_LOG_CHUNK = 512 * 1024


@dataclass
class ConsolePaths:
    """Every filesystem location the console touches, relative to one repo root."""
    repo_root: Path
    configs_dir: Path = field(init=False)
    base_yaml: Path = field(init=False)
    tokenizers_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    tasks_dir: Path = field(init=False)
    runs_dir: Path = field(init=False)
    script: Path = field(init=False)
    python: Path = field(init=False)

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        self.configs_dir = self.repo_root / "configs" / "experiments"
        self.base_yaml = self.repo_root / "configs" / "base.yaml"
        self.tokenizers_dir = self.repo_root / "configs" / "tokenizers"
        self.models_dir = self.repo_root / "configs" / "models"
        self.tasks_dir = self.repo_root / "configs" / "tasks"
        self.runs_dir = self.repo_root / "outputs" / "runs"
        self.script = self.repo_root / "scripts" / "run_experiment.py"
        venv_python = self.repo_root / ".venv" / "bin" / "python"
        self.python = venv_python if venv_python.exists() else Path(sys.executable)

    def config_path(self, rel: str) -> Path:
        """Resolve a config file name / relative path, confined to ``configs/experiments``."""
        rel = str(rel or "").strip()
        if not rel:
            raise ValueError("empty config path")
        name = rel
        if name.startswith("configs/experiments/"):
            name = name[len("configs/experiments/"):]
        if not name.endswith(".yaml"):
            name += ".yaml"
        stem = name[:-5]
        if not _NAME_RE.match(stem):
            raise ValueError(f"invalid config name {stem!r} (allowed: letters, digits, _ . -)")
        path = (self.configs_dir / name).resolve()
        if path.parent != self.configs_dir.resolve():
            raise ValueError("config path escapes configs/experiments")
        return path

    def rel(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.repo_root))
        except ValueError:
            return str(path)


# ---------------------------------------------------------------------------
# Config dict helpers
# ---------------------------------------------------------------------------

def flatten_experiment(raw: dict) -> dict:
    """Same rule as ``load_config``: hoist the nested ``experiment:`` block."""
    raw = copy.deepcopy(raw or {})
    if "experiment" in raw:
        exp = raw.pop("experiment") or {}
        for k in _EXPERIMENT_KEYS:
            if k in exp:
                raw.setdefault(k, exp[k])
    return raw


def nest_experiment(flat: dict) -> dict:
    """Inverse of :func:`flatten_experiment` — the on-disk style of every
    experiment YAML in the repo (``experiment:`` first, then the sections)."""
    out: Dict[str, Any] = {}
    exp = {k: flat[k] for k in _EXPERIMENT_KEYS if k in flat}
    if exp:
        out["experiment"] = exp
    for k in _SECTION_ORDER:
        if k in flat:
            out[k] = flat[k]
    for k, v in flat.items():
        if k not in out and k not in _EXPERIMENT_KEYS and k != "experiment":
            out[k] = v
    return out


def _deep_diff(value: Any, base: Any) -> Any:
    """Keys of *value* that differ from *base*; dicts recurse, lists are atomic.
    Returns ``_SAME`` when nothing differs."""
    if isinstance(value, dict) and isinstance(base, dict):
        out = {}
        for k, v in value.items():
            if k not in base:
                out[k] = v
                continue
            d = _deep_diff(v, base[k])
            if d is not _SAME:
                out[k] = d
        return out if out else _SAME
    return _SAME if value == base else value


_SAME = object()


def _errors_of(exc: ValidationError) -> List[Dict[str, str]]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        out.append({"loc": loc, "msg": err.get("msg", ""), "type": err.get("type", "")})
    return out


def _yaml_dump(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110, default_flow_style=False)


class ConsoleError(ValueError):
    """User-facing error (bad input); the server maps it to HTTP 400."""


class RunConflict(ConsoleError):
    """A run is already active and ``force`` was not set; HTTP 409."""


# ---------------------------------------------------------------------------
# Schema, base defaults, registries, presets
# ---------------------------------------------------------------------------

def base_resolved(paths: ConsolePaths) -> dict:
    """``configs/base.yaml`` validated and dumped — the defaults every form field starts from."""
    raw = flatten_experiment(load_yaml(paths.base_yaml))
    return ExperimentConfig(**raw).model_dump(mode="json")


def _registry_keys() -> Dict[str, List[str]]:
    keys: Dict[str, List[str]] = {"tokenizers": [], "models": [], "tasks": []}
    try:
        import arabic_eval.tokenizers  # noqa: F401
        from arabic_eval.registry import tokenizer_registry
        keys["tokenizers"] = tokenizer_registry.list_available()
    except Exception:  # noqa: BLE001
        pass
    try:
        import arabic_eval.models  # noqa: F401
        import arabic_eval.tasks  # noqa: F401
        from arabic_eval.registry import model_registry, task_registry
        keys["models"] = model_registry.list_available()
        keys["tasks"] = task_registry.list_available()
    except Exception:  # noqa: BLE001
        pass
    return keys


def _load_presets(directory: Path, section: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not directory.exists():
        return out
    for p in sorted(directory.glob("*.yaml")):
        try:
            data = load_yaml(p)
        except Exception:  # noqa: BLE001
            continue
        block = data.get(section) if isinstance(data, dict) else None
        if isinstance(block, dict):
            key = block.get("type") or p.stem
            out[str(key)] = {"file": p.name, **block}
    return out


def task_specs() -> Dict[str, List[ParamSpec]]:
    """``{task_type: ParamSpec list}`` for every registered task (cached per process:
    the registry is import-time state and the specs are class constants)."""
    global _TASK_SPECS
    if _TASK_SPECS is None:
        _TASK_SPECS = task_param_specs()
    return _TASK_SPECS


_TASK_SPECS: Optional[Dict[str, List[ParamSpec]]] = None


def tokenizer_specs() -> Dict[str, List[ParamSpec]]:
    """``{tokenizer_type: ParamSpec list}`` for every registered tokenizer (cached like ``task_specs``)."""
    global _TOKENIZER_SPECS
    if _TOKENIZER_SPECS is None:
        _TOKENIZER_SPECS = tokenizer_param_specs()
    return _TOKENIZER_SPECS


_TOKENIZER_SPECS: Optional[Dict[str, List[ParamSpec]]] = None


def schema_bundle(paths: ConsolePaths) -> dict:
    keys = _registry_keys()
    presets = {
        "tokenizers": _load_presets(paths.tokenizers_dir, "tokenizer"),
        "models": _load_presets(paths.models_dir, "model"),
    }
    # Registry keys win; preset files fill in when the registries are empty
    # (e.g. torch missing) so the dropdowns are never blank. Task presets are
    # no longer loaded (``configs/tasks/*.yaml`` are generated *from* the specs
    # served as ``task_params``); their file names stand in for the task list.
    for k in ("tokenizers", "models"):
        if not keys[k]:
            keys[k] = sorted(presets[k].keys())
    if not keys["tasks"]:
        keys["tasks"] = sorted(p.stem for p in paths.tasks_dir.glob("*.yaml")) if paths.tasks_dir.exists() else []
    return {
        "schema": ExperimentConfig.model_json_schema(),
        "base": base_resolved(paths),
        "registries": {**keys, "datasets": list(get_args(DatasetName))},
        "presets": presets,
        "task_params": {t: [s.to_dict() for s in spec] for t, spec in task_specs().items()},
        "tokenizer_params": {t: [s.to_dict() for s in spec] for t, spec in tokenizer_specs().items()},
        "hints": {k: list(v) for k, v in FIELD_HINTS.items()},
        "env": {
            "hf_token_set": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
            "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
            "python": paths.rel(paths.python),
            "script": paths.rel(paths.script),
        },
    }


# ---------------------------------------------------------------------------
# Config files
# ---------------------------------------------------------------------------

def _cell_name(tok_type: str, vocab_size: Optional[int]) -> str:
    return f"{tok_type}_{vocab_size // 1000}k" if vocab_size else tok_type


def cell_names(cfg: ExperimentConfig) -> List[str]:
    """The tokenizer cells that will actually run, in run order: the
    ``sweep.tokenizers`` list (= the sub-directory names ``run_sweep`` uses)
    when the config is a sweep, else the one cell of the top-level
    ``tokenizer`` block — a single-cell run never reads ``sweep.tokenizers``."""
    if not is_sweep(cfg):
        return [_cell_name(cfg.tokenizer.type, cfg.tokenizer.vocab_size)]
    return [_cell_name(tok.type, vs) for tok in cfg.sweep.tokenizers for vs in tok.vocab_sizes]


def is_sweep(cfg: ExperimentConfig) -> bool:
    """``run_experiment.py --sweep`` only fans out with more than one cell
    declared; a single cell always runs as a plain experiment."""
    return cfg.sweep is not None and len(cfg.sweep.tokenizers) > 1


def _finding_key(owner: str, msg: str) -> str:
    head = msg.split(" ", 1)[0]
    if head.startswith(owner + "."):
        return head[len(owner) + 1:]
    if "does not declare a parameter '" in msg:
        return msg.split("does not declare a parameter '", 1)[1].split("'", 1)[0]
    return ""


def param_warnings(cfg: ExperimentConfig) -> List[Dict[str, str]]:
    """Every ``params`` dict checked against its owner's declared ``param_spec``
    (unknown key, wrong type, out of range) — ``sweep.tasks[i].params`` against
    the task, ``tokenizer.params`` / ``sweep.tokenizers[i].params`` against the
    tokenizer. Advisory, the run would proceed. Each finding carries the ``loc``
    of the params dict so the form can paint the row, the owner type (``task``
    for both kinds, historically) and the offending key."""
    out: List[Dict[str, str]] = []
    specs = task_specs()
    if specs and cfg.sweep is not None:
        for i, t in enumerate(cfg.sweep.tasks):
            loc = f"sweep.tasks.{i}.params"
            if t.type not in specs:
                out.append({"loc": f"sweep.tasks.{i}.type", "task": t.type, "key": "",
                            "msg": f"task type {t.type!r} is not in the registry ({', '.join(sorted(specs))})"})
                continue
            for msg in validate_params(specs[t.type], t.params, owner=t.type):
                out.append({"loc": loc, "task": t.type, "key": _finding_key(t.type, msg), "msg": msg})
    tspecs = tokenizer_specs()
    if tspecs:
        holders = [("tokenizer.params", cfg.tokenizer.type, cfg.tokenizer.params)]
        if cfg.sweep is not None:
            holders += [(f"sweep.tokenizers.{i}.params", t.type, t.params) for i, t in enumerate(cfg.sweep.tokenizers)]
        for loc, ttype, params in holders:
            if ttype not in tspecs:
                out.append({"loc": loc.rsplit(".", 1)[0] + ".type", "task": ttype, "key": "",
                            "msg": f"tokenizer type {ttype!r} is not in the registry ({', '.join(sorted(tspecs))})"})
                continue
            for msg in validate_params(tspecs[ttype], params, owner=ttype):
                out.append({"loc": loc, "task": ttype, "key": _finding_key(ttype, msg), "msg": msg})
    return out


def _other_output_dirs(paths: ConsolePaths, exclude_file: Optional[str]) -> Dict[str, List[str]]:
    """``{output_dir: [file, …]}`` of every other saved config (raw YAML read, no
    Pydantic — this runs on every Validate). Only the ``output_dir`` a file
    states itself; one that inherits base.yaml's is not a clash worth flagging."""
    out: Dict[str, List[str]] = {}
    if not paths.configs_dir.exists():
        return out
    skip = Path(exclude_file).name if exclude_file else None
    for p in sorted(paths.configs_dir.glob("*.yaml")):
        if p.name == skip:
            continue
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            continue
        od = (raw.get("experiment") or {}).get("output_dir") if isinstance(raw.get("experiment"), dict) else None
        od = od or raw.get("output_dir")
        if isinstance(od, str) and od:
            out.setdefault(od.rstrip("/"), []).append(p.name)
    return out


def config_warnings(cfg: ExperimentConfig, paths: Optional[ConsolePaths] = None,
                    exclude_file: Optional[str] = None) -> List[str]:
    """Things a valid config can still get wrong at launch time. The one that
    bit a real run: a single ``sweep.tokenizers`` cell that is not the top-level
    ``tokenizer`` — the run is not a sweep, so ``tokenizer.type`` is what trains
    and the sweep cell is silently ignored. Also: task params a task does not
    declare (ignored at run time — the way a preset edit never reached a run),
    and an ``output_dir`` another saved config already writes to (*paths* given;
    *exclude_file* = the file this config is, so it does not clash with itself)."""
    out: List[str] = []
    if cfg.sweep is not None and not is_sweep(cfg) and cfg.sweep.tokenizers:
        listed = [_cell_name(t.type, vs) for t in cfg.sweep.tokenizers for vs in t.vocab_sizes]
        actual = _cell_name(cfg.tokenizer.type, cfg.tokenizer.vocab_size)
        if listed != [actual]:
            out.append(
                f"single-cell run: the top-level tokenizer ({actual}) is what runs; sweep.tokenizers "
                f"({', '.join(listed)}) is only read with --sweep, which needs more than one cell. "
                f"Set tokenizer.type to {listed[0].split('_')[0] if listed else '…'} to run that cell, "
                f"or add {actual} to sweep.tokenizers for a 2-cell sweep.")
    for w in param_warnings(cfg):
        out.append(f"{w['loc']}: {w['msg']}")
    if paths is not None:
        sharing = _other_output_dirs(paths, exclude_file).get(cfg.output_dir.rstrip("/"), [])
        if sharing:
            out.append(f"output_dir {cfg.output_dir} is also the output_dir of {', '.join(sharing)} — two configs "
                       f"writing one directory share (and overwrite) results, and run_sweep skips cells whose "
                       f"all_metrics.json exists")
    return out


def results_summary(paths: ConsolePaths, cfg: ExperimentConfig) -> dict:
    """What already exists under the config's ``output_dir`` (sweep cells are
    skipped by ``run_sweep`` when their ``all_metrics.json`` exists)."""
    out_dir = paths.repo_root / cfg.output_dir
    if is_sweep(cfg):
        cells = cell_names(cfg)
        done = [c for c in cells if (out_dir / c / "all_metrics.json").exists()]
        return {
            "output_dir": cfg.output_dir, "sweep": True, "cells": cells,
            "cells_done": done, "cells_pending": [c for c in cells if c not in done],
            "exists": bool(done), "report": (out_dir / "comparison_report.txt").exists(),
        }
    # A single run writes all_metrics.json at the root of output_dir; cell
    # folders already there (a campaign dir picked by mistake) would end up
    # beside it — reported so the Start dialog can say so.
    cell_dirs = sorted(d.name for d in out_dir.iterdir()
                       if d.is_dir() and d.name != SUPERSEDED_DIR and (d / "all_metrics.json").exists()) if out_dir.is_dir() else []
    return {
        "output_dir": cfg.output_dir, "sweep": False, "cells": [],
        "cells_done": [], "cells_pending": [],
        "exists": (out_dir / "all_metrics.json").exists(), "report": False,
        "cell_dirs_with_results": cell_dirs,
    }


def _load_cfg(paths: ConsolePaths, path: Path) -> ExperimentConfig:
    raw = flatten_experiment(load_yaml(paths.base_yaml))
    _deep_merge(raw, flatten_experiment(load_yaml(path)))
    return ExperimentConfig(**raw)


def list_configs(paths: ConsolePaths) -> List[dict]:
    out = []
    for p in sorted(paths.configs_dir.glob("*.yaml")):
        st = p.stat()
        row: Dict[str, Any] = {
            "file": p.name, "path": paths.rel(p), "mtime": st.st_mtime, "size": st.st_size,
            "valid": True, "error": None,
        }
        try:
            cfg = _load_cfg(paths, p)
            row.update({
                "name": cfg.name, "description": cfg.description, "output_dir": cfg.output_dir,
                "created_at": cfg.created_at, "runs": len(cfg.runs),
                "last_run_at": cfg.runs[-1].started_at if cfg.runs else None,
                "tokenizer": cfg.tokenizer.type, "model": cfg.model.name_or_path,
                "cells": cell_names(cfg), "sweep": is_sweep(cfg),
                "tasks": [t.type for t in cfg.sweep.tasks] if cfg.sweep else [],
                "phases": {
                    k: getattr(cfg.training.phases, k).enabled
                    for k in ("embedding_alignment", "warmup", "sft")
                },
                "results": results_summary(paths, cfg),
            })
        except ValidationError as e:
            row.update({"valid": False, "error": "; ".join(f"{x['loc']}: {x['msg']}" for x in _errors_of(e)),
                        "name": p.stem})
        except Exception as e:  # noqa: BLE001
            row.update({"valid": False, "error": f"{type(e).__name__}: {e}", "name": p.stem})
        out.append(row)
    return out


def read_config(paths: ConsolePaths, rel: str) -> dict:
    path = paths.config_path(rel)
    if not path.exists():
        raise ConsoleError(f"no such config: {paths.rel(path)}")
    text = path.read_text(encoding="utf-8")
    raw = flatten_experiment(yaml.safe_load(text) or {})
    merged = flatten_experiment(load_yaml(paths.base_yaml))
    _deep_merge(merged, raw)
    result: Dict[str, Any] = {"path": paths.rel(path), "file": path.name, "yaml": text, "raw": raw}
    try:
        cfg = ExperimentConfig(**merged)
        result.update({"valid": True, "errors": [], "resolved": cfg.model_dump(mode="json"),
                       "results": results_summary(paths, cfg), "sweep": is_sweep(cfg),
                       "warnings": config_warnings(cfg, paths, exclude_file=path.name),
                       "param_warnings": param_warnings(cfg)})
    except ValidationError as e:
        result.update({"valid": False, "errors": _errors_of(e), "resolved": merged, "results": None,
                       "sweep": None, "warnings": [], "param_warnings": []})
    return result


def validate_config(paths: ConsolePaths, cfg_dict: dict, *, file: Optional[str] = None) -> dict:
    """Validate a (possibly partial) config dict the way the CLI would: merged
    over ``base.yaml``, then through Pydantic. Returns the resolved dump on
    success or ``loc``-tagged errors on failure; on success also ``warnings``
    (sentences: the launch-time traps, undeclared / mistyped task params, a
    shared output_dir) and ``param_warnings`` (the task-param findings with
    their ``loc`` for the form). *file* = the saved file this dict is, so its
    own output_dir is not reported as shared with itself."""
    merged = flatten_experiment(load_yaml(paths.base_yaml))
    _deep_merge(merged, flatten_experiment(cfg_dict or {}))
    try:
        cfg = ExperimentConfig(**merged)
    except ValidationError as e:
        return {"ok": False, "errors": _errors_of(e), "resolved": None}
    return {
        "ok": True, "errors": [], "resolved": cfg.model_dump(mode="json"),
        "sweep": is_sweep(cfg), "cells": cell_names(cfg), "results": results_summary(paths, cfg),
        "warnings": config_warnings(cfg, paths, exclude_file=file),
        "param_warnings": param_warnings(cfg),
    }


def render_yaml(paths: ConsolePaths, cfg_dict: dict, mode: str = "full") -> str:
    """Render a config dict as YAML in the repo's on-disk style.

    ``full``: every resolved value, explicitly (like ``all_tokenizers_sweep.yaml``).
    ``delta``: only what differs from ``base.yaml`` (like the native_llama files);
    the ``experiment`` block is always written. Both reload to the same config.
    """
    if mode not in ("full", "delta"):
        raise ConsoleError(f"mode must be 'full' or 'delta', got {mode!r}")
    v = validate_config(paths, cfg_dict)
    if not v["ok"]:
        raise ConsoleError("config is invalid: " + "; ".join(f"{e['loc']}: {e['msg']}" for e in v["errors"]))
    resolved = v["resolved"]
    if mode == "full":
        return _yaml_dump(nest_experiment(resolved))
    base = base_resolved(paths)
    diff = _deep_diff(resolved, base)
    flat = {} if diff is _SAME else dict(diff)
    for k in ("name", "output_dir"):
        flat.setdefault(k, resolved[k])
    if resolved.get("description"):
        flat.setdefault("description", resolved["description"])
    return _yaml_dump(nest_experiment(flat))


def parse_yaml(text: str) -> dict:
    try:
        data = yaml.safe_load(text or "")
    except yaml.YAMLError as e:
        raise ConsoleError(f"YAML parse error: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConsoleError("YAML must be a mapping at the top level")
    return flatten_experiment(data)


def save_config(paths: ConsolePaths, name: str, text: str, overwrite: bool = False,
                created_at: Optional[str] = None) -> dict:
    """Write ``configs/experiments/<name>.yaml``. A file written for the first
    time is stamped ``created_at`` (now, or *created_at*) unless the text
    already carries one; an overwrite keeps whatever the text says."""
    path = paths.config_path(name)
    if path.exists() and not overwrite:
        raise ConsoleError(f"{paths.rel(path)} exists — tick overwrite to replace it")
    v = validate_config(paths, parse_yaml(text), file=path.name)
    if not v["ok"]:
        raise ConsoleError("refusing to save an invalid config: "
                           + "; ".join(f"{e['loc']}: {e['msg']}" for e in v["errors"]))
    if not path.exists() and not get_created_at(text):
        stamped = stamp_created_at(text, created_at)
        if stamped is None:                      # a layout the rewriter cannot edit: render it, stamped
            resolved = dict(v["resolved"]); resolved["created_at"] = created_at or now_iso()
            stamped = render_yaml(paths, resolved, "delta")
        text = stamped
        v = validate_config(paths, parse_yaml(text), file=path.name)
    paths.configs_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return {"path": paths.rel(path), "file": path.name, "resolved": v["resolved"], "results": v["results"],
            "warnings": v["warnings"]}


# ---------------------------------------------------------------------------
# Clone
# ---------------------------------------------------------------------------

def clone_config(paths: ConsolePaths, source: str, new_file: str, *, name: Optional[str] = None,
                 output_dir: Optional[str] = None, description: Optional[str] = None,
                 overwrite: bool = False) -> dict:
    """Copy ``configs/experiments/<source>`` to ``<new_file>`` with a new
    experiment name / output_dir / description, a fresh ``created_at`` and an
    empty ``runs``. The copy is the source's own text with only those lines
    rewritten (comments intact — a hand-written file stays readable); when
    the file's layout defeats the rewriter, or the rewrite does not re-parse
    to exactly the intended config, a rendered delta copy is written instead
    and ``comments_kept`` is false. Defaults: ``name`` = the new file's stem,
    ``output_dir`` = ``outputs/experiments/<name>``, ``description`` untouched."""
    src_path = paths.config_path(source)
    if not src_path.exists():
        raise ConsoleError(f"no such config: {paths.rel(src_path)}")
    dst_path = paths.config_path(new_file)
    if dst_path == src_path:
        raise ConsoleError("the clone needs a different file name")
    if dst_path.exists() and not overwrite:
        raise ConsoleError(f"{paths.rel(dst_path)} exists — tick overwrite to replace it")
    src = read_config(paths, source)
    if not src["valid"]:
        raise ConsoleError("the source does not validate — fix it before cloning: "
                           + "; ".join(f"{e['loc']}: {e['msg']}" for e in src["errors"]))
    name = (name or "").strip() or dst_path.stem
    output_dir = (output_dir or "").strip() or default_output_dir(name)
    updates: Dict[str, str] = {"name": name, "output_dir": output_dir, "created_at": now_iso()}
    if description is not None:
        updates["description"] = description
    expected = dict(src["resolved"])
    expected.update(updates)
    expected["runs"] = []                        # a clone is a new experiment: its own birth date, no run history

    text = strip_runs(src["yaml"])
    text = rewrite_experiment_keys(text, updates) if text is not None else None
    comments_kept = False
    if text is not None:
        try:
            v = validate_config(paths, parse_yaml(text))
        except ConsoleError:
            v = {"ok": False}
        comments_kept = bool(v.get("ok")) and v["resolved"] == expected
    if not comments_kept:
        text = render_yaml(paths, expected, "delta")
    saved = save_config(paths, new_file, text, overwrite=overwrite)
    return {**saved, "source": paths.rel(src_path), "comments_kept": comments_kept}


# ---------------------------------------------------------------------------
# Re-eval from a checkpoint, experiment folders, compare
# ---------------------------------------------------------------------------

PHASE_ORDER = ("embedding_alignment", "warmup", "sft")
CHECKPOINT_FILES = ("model.safetensors", "model.pt", "pytorch_model.bin", "model.safetensors.index.json")
# Their adapters *replace* embed_tokens / lm_head with custom modules; ``from_pretrained``
# on a checkpoint directory rebuilds the vanilla architecture and leaves those modules
# randomly initialised (the reason scripts/dump_eval_rows.py loads the state dict itself),
# so pointing model.name_or_path at such a checkpoint does not re-evaluate the trained model.
NON_STANDARD_EMBEDDING_TOKENIZERS = frozenset({"character_bert", "farasa_character_bert", "char_jaber", "charformer"})
NATIVE_TOKENIZERS = frozenset({"native_llama", "native_qwen3"})      # train() is a no-op: load_path stays null


def experiment_dirs(paths: ConsolePaths) -> List[str]:
    """Names of the experiment folders under ``outputs/experiments`` (a folder that
    holds cells, or is itself a finished single run), plus the parent folders every
    saved config's ``output_dir`` names — for the *cell of experiment…* output_dir mode.
    ``_superseded`` is skipped."""
    base = paths.repo_root / "outputs" / "experiments"
    names = set()
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if d.is_dir() and d.name != SUPERSEDED_DIR and not d.name.startswith("."):
                names.add(d.name)
    prefix = "outputs/experiments/"
    for od in _other_output_dirs(paths, None):
        if od.startswith(prefix):
            parts = od[len(prefix):].split("/")
            if len(parts) >= 2 and parts[0]:
                names.add(parts[0])
    return sorted(names)


def _checkpoints_of(cell_dir: Path) -> List[dict]:
    """The phase checkpoints a cell directory holds, last phase first."""
    out = []
    for phase in reversed(PHASE_ORDER):
        d = cell_dir / "training" / phase
        if d.is_dir() and any((d / f).exists() for f in CHECKPOINT_FILES):
            out.append({"phase": phase, "dir": str(d), "has_config": (d / "config.json").exists()})
    return out


def reeval_options(paths: ConsolePaths, source: str) -> dict:
    """What the *Re-eval…* dialog offers for a saved config: the finished cells under
    its ``output_dir`` (the sweep cells, or the directory itself for a single run —
    discovered from disk like the Results tab, ``_superseded`` skipped) with their
    checkpoints and tokenizer, the tasks of the source config, the experiment folder."""
    src = read_config(paths, source)
    if not src["valid"]:
        raise ConsoleError("the source does not validate — fix it before building a re-eval: "
                           + "; ".join(f"{e['loc']}: {e['msg']}" for e in src["errors"]))
    cfg = ExperimentConfig(**src["resolved"])
    out_dir = paths.repo_root / cfg.output_dir
    candidates: List[Path] = []
    if out_dir.is_dir():
        candidates.append(out_dir)
        candidates += sorted(d for d in out_dir.iterdir() if d.is_dir() and d.name != SUPERSEDED_DIR and d.name != "training")
    cells = []
    for d in candidates:
        cks = _checkpoints_of(d)
        if not cks:
            continue
        cell_cfg = {}
        if (d / "config.json").exists():
            try:
                cell_cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cell_cfg = {}
        tok = (cell_cfg.get("tokenizer") or {}) if isinstance(cell_cfg, dict) else {}
        rel = paths.rel(d)
        cells.append({
            "cell": d.name, "dir": rel, "is_output_dir": d == out_dir,
            "checkpoints": [{**c, "dir": paths.rel(Path(c["dir"]))} for c in cks],
            "tokenizer": tok.get("type") or cfg.tokenizer.type,
            "tokenizer_config": tok or None,
            "results": (d / "all_metrics.json").exists(),
            "tasks_done": sorted(p.stem for p in (d / "eval_rows").glob("*.parquet")) if (d / "eval_rows").is_dir() else [],
            "non_standard_embedding": (tok.get("type") or cfg.tokenizer.type) in NON_STANDARD_EMBEDDING_TOKENIZERS,
        })
    exp_rel = cfg.output_dir.rstrip("/")
    experiment = exp_rel[len("outputs/experiments/"):].split("/")[0] if exp_rel.startswith("outputs/experiments/") else None
    return {
        "source": src["path"], "output_dir": cfg.output_dir, "experiment": experiment, "sweep": is_sweep(cfg),
        "cells": cells, "tasks": [t.type for t in cfg.sweep.tasks] if cfg.sweep else [],
        "task_params": {t.type: t.params for t in cfg.sweep.tasks} if cfg.sweep else {},
        "phases_enabled": [ph for ph in PHASE_ORDER if getattr(cfg.training.phases, ph).enabled],
    }


def reeval_config(paths: ConsolePaths, source: str, cell: str, checkpoint: str, tasks: Sequence[str], *,
                  name: Optional[str] = None, output_dir: Optional[str] = None,
                  description: Optional[str] = None) -> dict:
    """Build (never save) an eval-only config from a finished cell of *source*:
    ``model.name_or_path`` = the checkpoint directory, every phase ``enabled: false``,
    ``sweep.tasks`` = the ticked tasks with the source's params for them, the tokenizer
    block = the cell's (from its ``config.json``; ``load_path`` = its saved tokenizer for a
    from-scratch type, so nothing is retrained), ``name`` = ``<cell>_reeval``,
    ``output_dir`` = a cell folder under the same experiment directory, fresh
    provenance. Returns the flat config, its delta YAML and validation."""
    opts = reeval_options(paths, source)
    cells = {c["cell"]: c for c in opts["cells"]}
    if cell not in cells:
        raise ConsoleError(f"{cell!r} is not a finished cell of {opts['source']} "
                           f"(cells with a checkpoint: {', '.join(cells) or 'none'})")
    c = cells[cell]
    ck = next((k for k in c["checkpoints"] if k["phase"] == checkpoint or k["dir"] == checkpoint), None)
    if ck is None:
        raise ConsoleError(f"{cell} has no checkpoint {checkpoint!r} (available: "
                           f"{', '.join(k['phase'] for k in c['checkpoints'])})")
    tasks = [t for t in tasks if t]
    if not tasks:
        raise ConsoleError("tick at least one task")
    unknown = [t for t in tasks if t not in task_specs() and task_specs()]
    if unknown:
        raise ConsoleError(f"unknown task type(s): {', '.join(unknown)}")
    src = read_config(paths, source)
    resolved = copy.deepcopy(src["resolved"])
    name = (name or "").strip() or f"{cell}_reeval"
    if not _NAME_RE.match(name):
        raise ConsoleError(f"invalid experiment name {name!r}")
    # The re-eval is a cell folder under the same experiment directory: for a sweep cell or
    # a campaign cell (outputs/experiments/<exp>/<cell>) that is the cell's parent — the
    # layout the Free-form tab, the judge and compare_results.py read as one experiment;
    # a plain single run (outputs/experiments/<run>) gets a cell folder under itself.
    cell_dir = paths.repo_root / c["dir"]
    try:
        depth = len(cell_dir.relative_to(paths.repo_root / "outputs" / "experiments").parts)
    except ValueError:
        depth = 1
    exp_dir = cell_dir.parent if depth >= 2 else cell_dir
    output_dir = (output_dir or "").strip() or f"{paths.rel(exp_dir)}/{name}"
    tok = dict(c["tokenizer_config"] or resolved["tokenizer"])
    tok.setdefault("params", {})
    if tok.get("type") not in NATIVE_TOKENIZERS and tok.get("save_path") \
            and (paths.repo_root / tok["save_path"]).is_dir():
        tok["load_path"] = tok["save_path"]           # the tokenizer this checkpoint was trained with — do not retrain
    for ph in PHASE_ORDER:
        resolved["training"]["phases"][ph]["enabled"] = False
    resolved.update({
        "name": name, "output_dir": output_dir, "created_at": None, "runs": [],
        "description": description if description is not None else
        f"eval-only re-run of {opts['source'].split('/')[-1]} · cell {cell} · checkpoint training/{ck['phase']} · tasks {', '.join(tasks)}",
    })
    resolved["model"]["name_or_path"] = ck["dir"]
    resolved["tokenizer"] = {k: tok.get(k) for k in ("type", "vocab_size", "params", "save_path", "load_path")}
    src_tasks = {t["type"]: t for t in (resolved.get("sweep") or {}).get("tasks", [])}
    resolved["sweep"] = {
        "tokenizers": [{"type": tok["type"], "vocab_sizes": [tok.get("vocab_size")], "params": copy.deepcopy(tok.get("params") or {})}],
        "tasks": [copy.deepcopy(src_tasks[t]) if t in src_tasks else {"type": t, "params": {}} for t in tasks],
    }
    v = validate_config(paths, resolved)
    notes = []
    if c["non_standard_embedding"]:
        notes.append(f"{tok['type']} replaces the embedding / output head with custom modules: from_pretrained on the "
                     f"checkpoint rebuilds a vanilla architecture and leaves them randomly initialised — a re-eval through "
                     f"model.name_or_path does not reproduce this cell (see scripts/dump_eval_rows.py)")
    if not ck.get("has_config"):
        notes.append(f"{ck['dir']} has no config.json — not a save_pretrained directory; from_pretrained will fail")
    for t in tasks:
        if t not in c["tasks_done"] and c["tasks_done"]:
            notes.append(f"{cell} has no eval_rows dump for {t} (it ran: {', '.join(c['tasks_done'])})")
    return {
        "config": v["resolved"] if v["ok"] else resolved, "ok": v["ok"], "errors": v["errors"],
        "warnings": v.get("warnings", []), "param_warnings": v.get("param_warnings", []),
        "yaml": render_yaml(paths, resolved, "delta") if v["ok"] else None,
        "name": name, "output_dir": output_dir, "checkpoint": ck["dir"], "cell": cell, "tasks": tasks,
        "source": opts["source"], "notes": notes,
    }


def _flat_diff(a: Any, b: Any, prefix: str, out: List[dict]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(a) + [k for k in b if k not in a]:
            path = f"{prefix}.{k}" if prefix else str(k)
            if k not in b:
                out.append({"path": path, "a": a[k], "b": _ABSENT})
            elif k not in a:
                out.append({"path": path, "a": _ABSENT, "b": b[k]})
            else:
                _flat_diff(a[k], b[k], path, out)
        return
    # lists of mappings of equal length (sweep.tasks, sweep.tokenizers, the mix sources)
    # are compared element by element so a differing task param gets its own row;
    # every other list is atomic, like _deep_diff
    if isinstance(a, list) and isinstance(b, list) and a and len(a) == len(b) \
            and all(isinstance(x, dict) for x in a) and all(isinstance(x, dict) for x in b):
        for i, (x, y) in enumerate(zip(a, b)):
            _flat_diff(x, y, f"{prefix}.{i}", out)
        return
    if a != b:
        out.append({"path": prefix, "a": a, "b": b})


_ABSENT = "<absent>"


def diff_configs(paths: ConsolePaths, a_dict: dict, b_rel: str) -> List[dict]:
    """Every config path whose *resolved* value differs between the working
    config *a_dict* (a form dict; merged over base.yaml like ``validate_config``)
    and the saved file *b_rel*: ``[{path, a, b}]`` in the schema's order, lists
    atomic (the same rule as ``_deep_diff``), ``"<absent>"`` on one side when a
    key exists only on the other. Provenance (``created_at`` / ``runs``) is left
    out — two files always differ there."""
    va = validate_config(paths, a_dict)
    if va["ok"]:
        a = va["resolved"]
    else:
        a = flatten_experiment(load_yaml(paths.base_yaml))
        _deep_merge(a, flatten_experiment(a_dict or {}))
    rb = read_config(paths, b_rel)
    b = rb["resolved"]
    out: List[dict] = []
    _flat_diff(a, b, "", out)
    return [r for r in out if r["path"] not in ("created_at", "runs") and not r["path"].startswith("runs.")]


# ---------------------------------------------------------------------------
# Progress parser
# ---------------------------------------------------------------------------

_STAGE_LABELS = {
    "1": "loading corpus", "2": "tokenizer", "3": "intrinsic eval",
    "4": "loading model", "5": "training", "6-7": "downstream eval",
}
_RE_CELL = re.compile(r"SWEEP cell: (\S+)")
_RE_EXP_START = re.compile(r"^Experiment: (\S+)$")
_RE_SKIP = re.compile(r"SWEEP: skipping (\S+) \(results exist\)")
_RE_STAGE = re.compile(r"Step (\d+(?:-\d+)?)/7: (.+?)\.{0,3}$")
_RE_STEP = re.compile(r"\[(\w+)\] step (\d+)/(\d+) loss=([\d.]+|nan|inf) lr=(\S+)")
_RE_EVAL_LOSS = re.compile(r"\[(\w+)\] step (\d+) eval_loss=([\d.]+|nan)")
_RE_PHASE_DONE = re.compile(r"\[(\w+)\] complete: steps=(\d+)")
_RE_PHASE_SKIP = re.compile(r"\[(\w+)\] skipped \(enabled=false\)")
_RE_EARLY = re.compile(r"\[(\w+)\] early-stop at step (\d+)")
_RE_TASK_START = re.compile(r"(\w+): evaluating (\d+) examples")
_RE_TASK_DONE = re.compile(r"^\s*\[(\w+)\] .*\(eval=([\d.]+)s\)")
_RE_TQDM = re.compile(r"LightEval MCQ.*?(\d+)/(\d+)")
_RE_EXP_DONE = re.compile(r"Experiment '(.+?)' done")
_RE_CELL_FAILED = re.compile(r"Cell (\S+) failed: (.*)")
_RE_TS = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \w+ [\w.]+: ")
_RE_TS_CAP = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] \w+ [\w.]+: ")
_RE_TOK_LOAD = re.compile(r"^\s*loading from (\S+)")
_RE_TOK_TRAIN = re.compile(r"^\s*training on (\d+) texts")
_RE_PHASE_ANY = re.compile(r"^\[(embedding_alignment|warmup|sft)\] ")


def _strip_ts(line: str) -> str:
    return _RE_TS.sub("", line, count=1)


_RE_JUDGE_LOAD = re.compile(r"\[judge:([\w.-]+)\] vLLM (\S+?),")
_RE_JUDGE_CELL = re.compile(r"\[judge:([\w.-]+)\] (\S+): (reusing )?(\d+) verdicts(?: in ([\d.]+)s)?")
_RE_JUDGE_REPORT = re.compile(r"^report → (\S+)")


def parse_judge_progress(text: str) -> Optional[dict]:
    """The judge stage's state from its console log (``scripts/judge/judge_freeform.py``):
    per judge the model being loaded and the cells scored so far (``n`` verdicts,
    ``reused`` when an earlier file was kept), plus the report path once written.
    ``None`` when the log carries no judge line at all."""
    judges: Dict[str, dict] = {}
    report = None
    for raw in re.split(r"\r\n|\n|\r", text):
        line = _RE_TS.sub("", raw.rstrip())
        m = _RE_JUDGE_LOAD.search(line)
        if m:
            judges.setdefault(m.group(1), {"model": None, "cells": {}})["model"] = m.group(2)
            continue
        m = _RE_JUDGE_CELL.search(line)
        if m:
            j = judges.setdefault(m.group(1), {"model": None, "cells": {}})
            j["cells"][m.group(2)] = {"n": int(m.group(4)), "reused": bool(m.group(3)),
                                      "seconds": float(m.group(5)) if m.group(5) else None}
            continue
        m = _RE_JUDGE_REPORT.match(line)
        if m:
            report = m.group(1)
    if not judges and report is None:
        return None
    return {"judges": judges, "report": report, "done": report is not None}


def parse_progress(text: str) -> dict:
    """Reduce a console log to the run's current state (see module doc)."""
    state: Dict[str, Any] = {
        "cell": None, "experiment": None, "cells_done": [], "cells_skipped": [], "cells_failed": {},
        "stage": None, "stage_label": None,
        "phase": None, "phases_done": [],
        "eval": None, "tasks_done": [],
        "done": False, "error": None, "last_line": None,
        # timings for the Runs-tab step panel: every stage / phase / task with the
        # timestamp of its first and last log line (log timestamps, not wall clock)
        "stages": [], "phase_times": {}, "task_times": {}, "tokenizer": None,
        "last_ts": None, "done_at": None,
    }
    tb_lines: List[str] = []
    in_tb = False

    def _reset_experiment() -> None:
        state.update({"stage": None, "stage_label": None, "phase": None, "phases_done": [], "eval": None,
                      "tasks_done": [], "done": False, "stages": [], "phase_times": {}, "task_times": {},
                      "tokenizer": None, "done_at": None})

    def _close_stage(ts: Optional[str]) -> None:
        if state["stages"] and state["stages"][-1]["end"] is None:
            state["stages"][-1]["end"] = ts

    for raw in re.split(r"\r\n|\n|\r", text):
        line = raw.rstrip()
        if not line:
            continue
        state["last_line"] = line
        mts = _RE_TS_CAP.match(line)
        ts = mts.group(1) if mts else None
        if ts:
            state["last_ts"] = ts
        if line.startswith("Traceback (most recent call last)"):
            in_tb, tb_lines = True, []
            continue
        if in_tb:
            if line.startswith((" ", "\t")):
                continue
            in_tb = False
            state["error"] = line
            continue
        msg = _strip_ts(line)
        m = _RE_CELL.search(msg)
        if m:
            _reset_experiment()
            state["cell"] = m.group(1)
            continue
        m = _RE_EXP_START.match(msg)
        if m:
            # A new run_experiment() begins (a cell, or a restart appended to
            # the same log): per-experiment counters start over.
            _reset_experiment()
            state["experiment"] = m.group(1)
            continue
        m = _RE_SKIP.search(msg)
        if m:
            state["cells_skipped"].append(m.group(1))
            continue
        m = _RE_CELL_FAILED.search(msg)
        if m:
            state["cells_failed"][m.group(1)] = m.group(2)
            continue
        m = _RE_STAGE.search(msg)
        if m:
            _close_stage(ts)
            state["stage"] = m.group(1)
            state["stage_label"] = _STAGE_LABELS.get(m.group(1), m.group(2))
            state["stages"].append({"id": m.group(1), "label": state["stage_label"], "start": ts, "end": None})
            continue
        m = _RE_TOK_LOAD.match(msg)
        if m and state["stage"] == "2":
            state["tokenizer"] = {"mode": "load", "detail": m.group(1)}
            continue
        m = _RE_TOK_TRAIN.match(msg)
        if m and state["stage"] == "2":
            state["tokenizer"] = {"mode": "train", "detail": f"{int(m.group(1)):,} texts"}
            continue
        m = _RE_PHASE_ANY.match(msg)
        if m:
            pt = state["phase_times"].setdefault(m.group(1), {"start": ts, "end": None, "status": "running", "steps": None})
            if pt["start"] is None:
                pt["start"] = ts
        m = _RE_STEP.search(msg)
        if m:
            state["phase_times"].setdefault(m.group(1), {"start": ts, "end": None, "status": "running", "steps": None})["steps"] = int(m.group(3))
            state["phase"] = {"name": m.group(1), "step": int(m.group(2)), "steps": int(m.group(3)),
                              "loss": float(m.group(4)), "lr": m.group(5),
                              "eval_loss": (state["phase"] or {}).get("eval_loss")
                              if (state["phase"] or {}).get("name") == m.group(1) else None}
            continue
        m = _RE_EVAL_LOSS.search(msg)
        if m:
            ph = state["phase"] if (state["phase"] or {}).get("name") == m.group(1) else {"name": m.group(1), "step": int(m.group(2)), "steps": None, "loss": None, "lr": None}
            ph["eval_loss"] = float(m.group(3))
            ph["eval_step"] = int(m.group(2))
            state["phase"] = ph
            continue
        m = _RE_EARLY.search(msg)
        if m and state["phase"] and state["phase"].get("name") == m.group(1):
            state["phase"]["early_stopped"] = True
            continue
        m = _RE_PHASE_DONE.search(msg)
        if m:
            state["phases_done"].append({"name": m.group(1), "steps": int(m.group(2)), "skipped": False})
            state["phase"] = None
            pt = state["phase_times"].setdefault(m.group(1), {"start": ts, "end": None, "status": "running", "steps": None})
            pt.update({"end": ts, "status": "done"})
            continue
        m = _RE_PHASE_SKIP.search(msg)
        if m:
            state["phases_done"].append({"name": m.group(1), "steps": 0, "skipped": True})
            state["phase_times"][m.group(1)] = {"start": ts, "end": ts, "status": "skipped", "steps": 0}
            continue
        m = _RE_TASK_START.search(msg)
        if m:
            state["eval"] = {"task": m.group(1), "done": 0, "total": int(m.group(2))}
            state["task_times"][m.group(1)] = {"start": ts, "end": None, "seconds": None}
            continue
        m = _RE_TQDM.search(line)
        if m:
            ev = state["eval"] or {"task": None, "total": int(m.group(2))}
            ev["done"], ev["total"] = int(m.group(1)), int(m.group(2))
            state["eval"] = ev
            continue
        m = _RE_TASK_DONE.search(msg)
        if m:
            state["tasks_done"].append({"task": m.group(1), "seconds": float(m.group(2))})
            state["eval"] = None
            tt = state["task_times"].setdefault(m.group(1), {"start": None, "end": None, "seconds": None})
            tt.update({"end": ts, "seconds": float(m.group(2))})
            continue
        m = _RE_EXP_DONE.search(msg)
        if m:
            state["done"] = True
            state["done_at"] = ts
            _close_stage(ts)
            if state["cell"]:
                state["cells_done"].append(state["cell"])
            continue
    judge = parse_judge_progress(text)
    if judge is not None:
        state["judge"] = judge
        if judge["done"]:
            state["done"] = True
    return state

def plan_of(cfg: ExperimentConfig) -> dict:
    """What a run *will* execute, for the Runs-tab step panel: the phases with
    their enabled flag and step budget (post-validation, so derived ``steps``
    are filled in), whether the intrinsic and downstream stages run, the tasks."""
    return {
        "phases": {k: {"enabled": ph.enabled, "steps": ph.steps}
                   for k, ph in ((n, getattr(cfg.training.phases, n)) for n in ("embedding_alignment", "warmup", "sft"))},
        "intrinsic": bool(cfg.evaluation.intrinsic_metrics),
        "downstream": bool(cfg.evaluation.downstream_metrics),
        "tasks": [t.type for t in cfg.sweep.tasks] if cfg.sweep else [],
    }


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

STATUS_ACTIVE = ("running", "cancelling")
RUN_DIR_PLACEHOLDER = "{RUN_DIR}"      # in an argv before launch: the run directory once allocated


def _proc_start_ticks(pid: int) -> Optional[int]:
    """Field 22 of ``/proc/<pid>/stat`` — the process start time in clock
    ticks. Together with the pid it identifies a process uniquely, so a
    recycled pid is never mistaken for a run that is still alive."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    rest = stat[stat.rindex(")") + 2:].split()
    try:
        return int(rest[19])
    except (IndexError, ValueError):
        return None


def _proc_state(pid: int) -> Optional[str]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat[stat.rindex(")") + 2:].split()[0]


def process_alive(pid: Optional[int], start_ticks: Optional[int]) -> bool:
    if not pid:
        return False
    state = _proc_state(pid)
    if state is None or state in ("Z", "X"):
        return False
    if start_ticks is not None:
        return _proc_start_ticks(pid) == start_ticks
    return True


def group_alive(pgid: Optional[int]) -> bool:
    """True while *any* process of the run's process group exists (the shell
    may be gone while the python child still runs)."""
    if not pgid:
        return False
    pgid = int(pgid)
    # /proc scan rather than killpg(pgid, 0): the latter also counts zombies
    # (a finished shell nobody has reaped yet), which would keep a run
    # "running" forever after a server restart.
    try:
        entries = os.listdir("/proc")
    except OSError:
        return False
    for name in entries:
        if not name.isdigit():
            continue
        try:
            stat = Path(f"/proc/{name}/stat").read_text()
        except OSError:
            continue
        rest = stat[stat.rindex(")") + 2:].split()
        if len(rest) > 2 and rest[0] not in ("Z", "X") and rest[2] == str(pgid):
            return True
    return False


def gpu_snapshot() -> Optional[List[dict]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append({"index": int(parts[0]), "name": parts[1], "memory_used_mib": int(parts[2]),
                         "memory_total_mib": int(parts[3]), "utilization_pct": int(parts[4])})
        except ValueError:
            continue
    return gpus


# ---------------------------------------------------------------------------
# Judge stage (free-form eval): configs, readiness, command
# ---------------------------------------------------------------------------

def judge_headers_ok(repo_root: Path) -> bool:
    """The locally extracted Python headers vLLM's Triton JIT needs (see scripts/judge/setup_judge_env.sh)."""
    return any((repo_root / ".local-pkgs" / "extracted" / "usr" / "include").glob("python3*/Python.h"))


def judge_ready(paths: ConsolePaths, cfg: Any) -> Tuple[bool, str]:
    """Whether a judge can be launched from this machine right now, and why not."""
    if cfg.backend == "vllm":
        venv = paths.repo_root / ".venv-judge" / "bin" / "python"
        if not venv.exists():
            return False, "no .venv-judge — run scripts/judge/setup_judge_env.sh"
        if not judge_headers_ok(paths.repo_root):
            return False, "Python headers missing under .local-pkgs/extracted — run scripts/judge/setup_judge_env.sh"
        return True, ""
    if not os.environ.get(cfg.api_key_env):
        return False, f"{cfg.api_key_env} is not set in the console server's environment (export it before starting the server)"
    return True, ""


def judge_config_path(paths: ConsolePaths, ref: str) -> Path:
    """``<name>`` or ``configs/judges/<name>.yaml`` → the file, confined to configs/judges."""
    d = (paths.repo_root / "configs" / "judges").resolve()
    ref = str(ref or "").strip()
    if not ref:
        raise ConsoleError("judge required")
    cand = (paths.repo_root / ref).resolve() if ref.endswith(".yaml") else (d / f"{ref}.yaml").resolve()
    if d not in cand.parents:
        raise ConsoleError("judge configs live under configs/judges/")
    if not cand.is_file():
        raise ConsoleError(f"no such judge config: {ref}")
    return cand


def judge_configs(paths: ConsolePaths) -> List[dict]:
    """Every ``configs/judges/*.yaml`` with its parsed essentials and readiness."""
    from arabic_eval.judge.freeform_judge import JudgeConfig
    d = paths.repo_root / "configs" / "judges"
    out: List[dict] = []
    for p in sorted(d.glob("*.yaml")) if d.is_dir() else []:
        entry: Dict[str, Any] = {"file": paths.rel(p), "id": p.stem}
        try:
            cfg = JudgeConfig.from_yaml(p)
            ready, why = judge_ready(paths, cfg)
            entry.update({"name": cfg.name, "backend": cfg.backend, "model": cfg.model, "rubric": cfg.rubric,
                          "structured_json": cfg.structured_json, "gpu": cfg.backend == "vllm",
                          "api_key_env": cfg.api_key_env if cfg.backend == "openai" else None,
                          "ready": ready, "why": why})
        except Exception as e:  # noqa: BLE001 — a broken YAML is listed, not hidden
            entry.update({"name": p.stem, "error": f"{type(e).__name__}: {e}", "ready": False, "why": "invalid config"})
        out.append(entry)
    return out


def judge_cells(repo_root: Path, experiment: str) -> Tuple[Path, List[str]]:
    """The experiment dir (confined to outputs/experiments) and its cells that have free-form generations."""
    base = (repo_root / "outputs" / "experiments").resolve()
    exp = (repo_root / str(experiment or "")).resolve()
    if base not in exp.parents or not exp.is_dir():
        raise ConsoleError(f"no such experiment under outputs/experiments: {experiment}")
    gen = Path("eval_rows") / "freeform_cidar.parquet"
    if (exp / gen).exists():
        return exp, [exp.name]
    cells = sorted(p.name for p in exp.iterdir() if p.is_dir() and (p / gen).exists())
    if not cells:
        raise ConsoleError(f"{experiment} has no cell with eval_rows/freeform_cidar.parquet — run the free-form task first")
    return exp, cells


class RunManager:
    """Detached experiment runs under ``outputs/runs/<run_id>/`` — experiments
    (``start``) and free-form judge stages (``start_judge``), one lifecycle."""

    def __init__(self, paths: ConsolePaths, grace_sec: float = CANCEL_GRACE_SEC) -> None:
        self.paths = paths
        self.grace_sec = grace_sec
        self._lock = threading.Lock()
        self._procs: Dict[str, subprocess.Popen] = {}

    # ---- record io ---------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        if not _NAME_RE.match(run_id or ""):
            raise ConsoleError(f"invalid run id {run_id!r}")
        return self.paths.runs_dir / run_id

    def _record_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run.json"

    def _read(self, run_id: str) -> dict:
        p = self._record_path(run_id)
        if not p.exists():
            raise ConsoleError(f"no such run: {run_id}")
        return json.loads(p.read_text(encoding="utf-8"))

    def _write(self, rec: dict) -> None:
        p = self._record_path(rec["run_id"])
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)

    # ---- command -----------------------------------------------------------
    def build_command(self, snapshot: Path, sweep: bool, device: Optional[str], seed: Optional[int],
                      base_config: Optional[Path] = None) -> List[str]:
        cmd = [str(self.paths.python), str(self.paths.script), "--config", str(snapshot)]
        if base_config is not None:
            cmd += ["--base-config", str(base_config)]
        if sweep:
            cmd.append("--sweep")
        if device:
            cmd += ["--device", str(device)]
        if seed is not None:
            cmd += ["--seed", str(int(seed))]
        return cmd

    def build_judge_command(self, experiment_rel: str, judge_yamls: Sequence[str], gpu: bool,
                            baseline: Optional[str], limit: Optional[int], overwrite: bool,
                            cells: Optional[Sequence[str]] = None) -> List[str]:
        """``scripts/judge/run_judge.sh`` (the vLLM venv with its environment) when any judge
        needs the GPU, else ``scripts/judge/judge_freeform.py`` in the main venv."""
        if gpu:
            cmd = [str(self.paths.repo_root / "scripts" / "judge" / "run_judge.sh")]
        else:
            cmd = [str(self.paths.python), str(self.paths.repo_root / "scripts" / "judge" / "judge_freeform.py")]
        cmd += ["--experiment", experiment_rel]
        for y in judge_yamls:
            cmd += ["--judge", str(y)]
        if cells:
            cmd += ["--cells", *[str(c) for c in cells]]
        if baseline:
            cmd += ["--baseline", str(baseline)]
        if limit:
            cmd += ["--limit", str(int(limit))]
        if overwrite:
            cmd.append("--overwrite")
        return cmd

    # ---- lifecycle -----------------------------------------------------------
    def active(self) -> List[dict]:
        return [r for r in self.list() if r["status"] in STATUS_ACTIVE]

    def start(self, config: str, *, yaml_text: Optional[str] = None, sweep: Optional[bool] = None,
              device: Optional[str] = None, seed: Optional[int] = None, force: bool = False,
              command: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None) -> dict:
        """Launch a run from a saved config file (``config`` = file name) or
        from ``yaml_text`` (``config`` is then only a label). The config is
        snapshotted into the run directory and the run is started *from the
        snapshot*, so later edits to the file never reach a running job.

        ``command`` overrides the launched argv (tests use a stub)."""
        if yaml_text is None:
            src = self.paths.config_path(config)
            if not src.exists():
                raise ConsoleError(f"no such config: {self.paths.rel(src)}")
            yaml_text = src.read_text(encoding="utf-8")
            label = src.stem
            config_ref = self.paths.rel(src)
        else:
            label = re.sub(r"[^A-Za-z0-9_.-]", "_", str(config or "unsaved")) or "unsaved"
            config_ref = None
        v = validate_config(self.paths, parse_yaml(yaml_text), file=config_ref)
        if not v["ok"]:
            raise ConsoleError("config is invalid: " + "; ".join(f"{e['loc']}: {e['msg']}" for e in v["errors"]))
        cfg = ExperimentConfig(**v["resolved"])
        if sweep is None:
            sweep = is_sweep(cfg)
        # The snapshot lives in the run dir, which is allocated inside _launch;
        # RUN_DIR_PLACEHOLDER is substituted there.
        argv = command or self.build_command(Path(RUN_DIR_PLACEHOLDER) / "config.yaml", sweep, device, seed)
        record = {
            "kind": "experiment", "config": config_ref,
            "experiment_name": cfg.name, "output_dir": cfg.output_dir,
            "sweep": bool(sweep), "cells": cell_names(cfg) if sweep else [],
            "tasks": [t.type for t in cfg.sweep.tasks] if cfg.sweep else [],
            "plan": plan_of(cfg),
            "tokenizer": cfg.tokenizer.type, "model": cfg.model.name_or_path,
            "experiment_log": f"outputs/logs/{cfg.name}/experiment.log",
        }
        snapshots = {"config.yaml": yaml_text if yaml_text.endswith("\n") else yaml_text + "\n"}
        rec = self._launch(label, argv, record, snapshots, snapshot_main="config.yaml", force=force, env=env)
        if config_ref is not None:               # a run from a saved file goes into that file's run log
            try:
                stamp = record_run_start(src, source="console", run_id=rec["run_id"], when=rec["started_at"])
            except OSError as e:
                log.warning("could not record the run start in %s: %s", config_ref, e)
                stamp = None
            rec["run_stamp"] = stamp
            self._write(rec)
        return rec

    def start_judge(self, experiment: str, judges: Sequence[str], *, baseline: Optional[str] = None,
                    limit: Optional[int] = None, overwrite: bool = False, force: bool = False,
                    cells: Optional[Sequence[str]] = None,
                    command: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None) -> dict:
        """Launch the free-form judge stage on a finished experiment (or sweep) as
        a detached run: the judge YAMLs are snapshotted into the run dir and the
        script is pointed at the snapshots. A GPU judge (vLLM) conflicts with an
        active run like an experiment does; API-only judges run alongside.

        ``cells`` judges only those cells (default: every cell with generations);
        the baseline may sit outside the selection — its verdicts are then read
        from the file a previous run wrote.

        ``command`` overrides the launched argv (tests use a stub)."""
        from arabic_eval.judge.freeform_judge import JudgeConfig
        cells_arg = cells
        exp_dir, cells = judge_cells(self.paths.repo_root, experiment)
        exp_rel = self.paths.rel(exp_dir)
        if not judges:
            raise ConsoleError("pick at least one judge")
        cfgs = []
        for ref in judges:
            path = judge_config_path(self.paths, ref)
            cfg = JudgeConfig.from_yaml(path)
            ready, why = judge_ready(self.paths, cfg)
            if not ready:
                raise ConsoleError(f"judge {cfg.name}: {why}")
            cfgs.append((path, cfg))
        names = [c.name for _, c in cfgs]
        if len(set(names)) != len(names):
            raise ConsoleError(f"duplicate judge names: {names}")
        if baseline and baseline not in cells:
            raise ConsoleError(f"baseline {baseline!r} is not a cell with generations ({', '.join(cells)})")
        selected = list(dict.fromkeys(str(c) for c in (cells_arg or [])))   # order kept, duplicates dropped
        unknown = [c for c in selected if c not in cells]
        if unknown:
            raise ConsoleError(f"not cells with generations: {', '.join(unknown)} (have: {', '.join(cells)})")
        judged = selected or cells
        gpu = any(c.backend == "vllm" for _, c in cfgs)
        snapshots = {f"judges/{c.name}.yaml": path.read_text(encoding="utf-8") for path, c in cfgs}
        request = {"experiment": exp_rel, "judges": names, "cells": judged, "baseline": baseline, "limit": limit,
                   "overwrite": bool(overwrite), "gpu": gpu, "backends": {c.name: c.backend for _, c in cfgs}}
        snapshots["judge.json"] = json.dumps(request, ensure_ascii=False, indent=1) + "\n"
        argv = command or self.build_judge_command(
            exp_rel, [f"{RUN_DIR_PLACEHOLDER}/judges/{c.name}.yaml" for _, c in cfgs], gpu, baseline, limit, overwrite,
            selected)
        record = {
            "kind": "judge", "config": None,
            "experiment_name": exp_dir.name, "output_dir": exp_rel,
            "sweep": False, "cells": judged, "cells_available": cells, "tasks": ["freeform_cidar"],
            "tokenizer": "judge", "model": ", ".join(c.model for _, c in cfgs),
            "judges": names, "judge_backends": request["backends"], "judge_models": {c.name: c.model for _, c in cfgs},
            "baseline": baseline, "limit": limit, "overwrite": bool(overwrite), "gpu": gpu,
            "experiment_log": None,
        }
        return self._launch(f"judge_{exp_dir.name}", argv, record, snapshots, snapshot_main="judge.json",
                            force=force or not gpu, env=env)

    def _launch(self, label: str, argv: List[str], record: dict, snapshots: Dict[str, str], *,
                snapshot_main: str, force: bool, env: Optional[Dict[str, str]] = None) -> dict:
        """Allocate the run dir, write the snapshots, start the detached session
        and write ``run.json``. ``RUN_DIR_PLACEHOLDER`` in ``argv`` becomes the
        run dir (snapshots are addressed through it)."""
        with self._lock:
            active = self.active()
            if active and not force:
                raise RunConflict(
                    f"run {active[0]['run_id']} is still {active[0]['status']}; tick 'run concurrently' to start anyway"
                )
            ts = datetime.now().astimezone()
            run_id = f"{ts.strftime('%Y%m%d-%H%M%S')}_{label}"
            n = 1
            while self.run_dir(run_id).exists():
                n += 1
                run_id = f"{ts.strftime('%Y%m%d-%H%M%S')}_{label}_{n}"
            rd = self.run_dir(run_id)
            rd.mkdir(parents=True)
            for name, text in snapshots.items():
                p = rd / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
            argv = [a.replace(RUN_DIR_PLACEHOLDER, str(rd)) for a in argv]
            console_log = rd / "console.log"
            exit_file = rd / "exit_code"
            # bash owns the session; it records the exit code itself so the
            # status survives a server restart (the child is reparented to
            # init, nobody else can wait() on it). The TERM trap keeps bash
            # alive through a cancel (SIGTERM goes to the whole group) until
            # the python child has exited, so the code still gets written;
            # a *handled* trap (not '') is reset to default on exec, so the
            # child itself still dies on SIGTERM.
            shell = (f"trap 'true' TERM; "
                     f"{' '.join(shlex.quote(a) for a in argv)} > {shlex.quote(str(console_log))} 2>&1; "
                     f"echo $? > {shlex.quote(str(exit_file))}")
            run_env = dict(os.environ)
            run_env.setdefault("PYTHONUNBUFFERED", "1")
            if env:
                run_env.update({str(k): str(v) for k, v in env.items()})
            proc = subprocess.Popen(
                ["bash", "-c", shell], cwd=str(self.paths.repo_root), env=run_env,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._procs[run_id] = proc
            rec = {
                "run_id": run_id, "label": label, **record,
                "argv": argv, "pid": proc.pid, "pgid": proc.pid,
                "start_ticks": _proc_start_ticks(proc.pid),
                "started_at": ts.isoformat(timespec="seconds"), "finished_at": None,
                "status": "running", "exit_code": None, "cancel_requested_at": None,
                "console_log": self.paths.rel(console_log), "snapshot": self.paths.rel(rd / snapshot_main),
            }
            self._write(rec)
        return rec

    def _reconcile(self, rec: dict) -> dict:
        if rec["status"] not in STATUS_ACTIVE:
            return rec
        proc = self._procs.get(rec["run_id"])
        if proc is not None:
            proc.poll()  # reap so the shell never lingers as a zombie
        exit_file = self.run_dir(rec["run_id"]) / "exit_code"
        code: Optional[int] = None
        if exit_file.exists():
            try:
                code = int(exit_file.read_text().strip() or "1")
            except ValueError:
                code = 1
        alive = process_alive(rec.get("pid"), rec.get("start_ticks"))
        if code is None and (alive or group_alive(rec.get("pgid"))):
            return rec
        rec["finished_at"] = datetime.now().isoformat(timespec="seconds")
        if code is None and rec.get("cancel_requested_at"):
            rec["status"], rec["exit_code"] = "cancelled", None   # group killed before bash could record it
        elif code is None:
            rec["status"] = "lost"      # process gone, no exit code written (killed -9 / reboot)
        elif rec.get("cancel_requested_at") and code != 0:
            rec["status"], rec["exit_code"] = "cancelled", code
        elif code == 0:
            rec["status"], rec["exit_code"] = "finished", 0
        else:
            rec["status"], rec["exit_code"] = "failed", code
        self._write(rec)
        self._procs.pop(rec["run_id"], None)
        return rec

    def list(self) -> List[dict]:
        if not self.paths.runs_dir.exists():
            return []
        out = []
        for d in sorted(self.paths.runs_dir.iterdir(), reverse=True):
            p = d / "run.json"
            if not p.is_file():
                continue
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append(self._reconcile(rec))
        return out

    def get(self, run_id: str) -> dict:
        return self._reconcile(self._read(run_id))

    def detail(self, run_id: str, tail_bytes: int = 64 * 1024) -> dict:
        rec = self.get(run_id)
        log = self.paths.repo_root / rec["console_log"]
        text = ""
        size = 0
        if log.exists():
            size = log.stat().st_size
            with open(log, "rb") as fh:
                if size > tail_bytes:
                    fh.seek(size - tail_bytes)
                text = fh.read().decode("utf-8", errors="replace")
        progress = parse_progress(_read_all(log)) if log.exists() else parse_progress("")
        rec = dict(rec)
        if rec.get("kind", "experiment") == "experiment" and not rec.get("plan"):
            rec["plan"] = self._plan_from_snapshot(rec)      # records written before the plan existed
        rec["progress"] = progress
        rec["server_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")   # same clock and format as the log timestamps
        rec["log_tail"] = text
        rec["log_size"] = size
        rec["results"] = self.results(run_id)
        return rec

    def _plan_from_snapshot(self, rec: dict) -> Optional[dict]:
        snap = self.paths.repo_root / str(rec.get("snapshot") or "")
        if not snap.is_file() or snap.suffix != ".yaml":
            return None
        try:
            v = validate_config(self.paths, parse_yaml(snap.read_text(encoding="utf-8")))
            return plan_of(ExperimentConfig(**v["resolved"])) if v["ok"] else None
        except Exception:  # noqa: BLE001 — a plan is a convenience, never a failure of the detail view
            return None

    def log(self, run_id: str, offset: int = 0, max_bytes: int = _MAX_LOG_CHUNK) -> dict:
        rec = self.get(run_id)
        log = self.paths.repo_root / rec["console_log"]
        if not log.exists():
            return {"offset": 0, "data": "", "size": 0, "status": rec["status"]}
        size = log.stat().st_size
        offset = max(0, min(int(offset or 0), size))
        with open(log, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read(max_bytes)
        return {"offset": offset + len(chunk), "data": chunk.decode("utf-8", errors="replace"),
                "size": size, "status": rec["status"]}

    def cancel(self, run_id: str) -> dict:
        rec = self.get(run_id)
        if rec["status"] not in STATUS_ACTIVE:
            raise ConsoleError(f"run {run_id} is {rec['status']}, nothing to cancel")
        pgid = int(rec.get("pgid") or rec["pid"])
        if rec["status"] == "running":
            rec["cancel_requested_at"] = datetime.now().isoformat(timespec="seconds")
            rec["status"] = "cancelling"
            self._write(rec)
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                return self._reconcile(rec)
            threading.Thread(target=self._kill_after_grace, args=(run_id, pgid), daemon=True).start()
        return rec

    def _kill_after_grace(self, run_id: str, pgid: int) -> None:
        deadline = time.monotonic() + self.grace_sec
        while time.monotonic() < deadline:
            time.sleep(0.25)
            if self.get(run_id)["status"] not in STATUS_ACTIVE and not group_alive(pgid):
                return
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # bash dies with the group, so no exit_code is written: mark it ourselves.
        rec = self._read(run_id)
        if rec["status"] in STATUS_ACTIVE:
            rec.update({"status": "cancelled", "exit_code": -9,
                        "finished_at": datetime.now().isoformat(timespec="seconds")})
            self._write(rec)
            self._procs.pop(run_id, None)

    # ---- results -----------------------------------------------------------
    def results(self, run_id: str) -> Optional[dict]:
        rec = self._read(run_id)
        out_dir = self.paths.repo_root / rec["output_dir"]
        if rec.get("kind") == "judge":
            report = out_dir / "freeform_judge_report.json"
            if not report.exists():
                return None
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            summary = {j: {cell: {"score_mean": sm.get("score_mean"), "n": sm.get("n"),
                                  "delta": (sm.get("vs_baseline") or {}).get("delta_mean"),
                                  "parse_fail_rate": sm.get("parse_fail_rate")}
                           for cell, sm in cells.items()}
                       for j, cells in (data.get("judges") or {}).items()}
            return {"judge": True, "report": self.paths.rel(report), "baseline": data.get("baseline"),
                    "summary": summary, "comparison_report": data.get("comparison_report")}
        if rec["sweep"]:
            cells = {}
            for c in rec.get("cells") or []:
                m = _metrics_summary(out_dir / c / "all_metrics.json")
                if m is not None:
                    cells[c] = m
            if not cells:
                return None
            return {"sweep": True, "cells": cells,
                    "report": self.paths.rel(out_dir / "comparison_report.txt")
                    if (out_dir / "comparison_report.txt").exists() else None}
        m = _metrics_summary(out_dir / "all_metrics.json")
        return None if m is None else {"sweep": False, "cells": {rec["experiment_name"]: m}}


def _read_all(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _metrics_summary(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if "error" in data and "downstream" not in data:
        return {"error": data["error"]}
    intrinsic = data.get("intrinsic") or {}
    downstream = {}
    for task, m in (data.get("downstream") or {}).items():
        if not isinstance(m, dict):
            continue
        downstream[task] = {
            "accuracy": m.get("accuracy"),
            "accuracy_pmi": m.get("accuracy_pmi"),
            "accuracy_char_norm": m.get("accuracy_char_norm"),
            "num_samples": m.get("num_samples"),
            "inference_time_sec": m.get("inference_time_sec"),
        }
    mei = {}
    for task, m in (data.get("mei") or {}).items():
        if isinstance(m, dict):
            mei[task] = m.get("mei")
    training = {}
    for phase, m in (data.get("training") or {}).items():
        if isinstance(m, dict):
            training[phase] = {k: m.get(k) for k in ("status", "steps_completed", "final_train_loss",
                                                      "best_eval_loss", "early_stopped", "wall_time_sec")}
    return {
        "intrinsic": {k: intrinsic.get(k) for k in ("fertility", "compression_ratio", "unk_rate",
                                                     "vocab_coverage", "root_conservation_rate")},
        "downstream": downstream, "mei": mei, "training": training,
        "path": str(path),
    }


def config_results(paths: ConsolePaths, rel: str) -> dict:
    """Per-cell metrics summary of a config's ``output_dir`` (what a finished
    or partly finished run left behind), for the config list."""
    path = paths.config_path(rel)
    if not path.exists():
        raise ConsoleError(f"no such config: {paths.rel(path)}")
    cfg = _load_cfg(paths, path)
    summary = results_summary(paths, cfg)
    out_dir = paths.repo_root / cfg.output_dir
    cells: Dict[str, Any] = {}
    if summary["sweep"]:
        for c in summary["cells_done"]:
            m = _metrics_summary(out_dir / c / "all_metrics.json")
            if m is not None:
                cells[c] = m
    else:
        m = _metrics_summary(out_dir / "all_metrics.json")
        if m is not None:
            cells[cfg.name] = m
    report = out_dir / "comparison_report.txt"
    return {**summary, "path": paths.rel(path), "cells_metrics": cells,
            "report_path": paths.rel(report) if report.exists() else None,
            "experiment_log": f"outputs/logs/{cfg.name}/experiment.log"
            if (paths.repo_root / "outputs/logs" / cfg.name / "experiment.log").exists() else None}
