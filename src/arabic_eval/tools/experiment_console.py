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
from typing import Any, Dict, List, Optional, get_args

import yaml
from pydantic import ValidationError

from arabic_eval.config import DatasetName, ExperimentConfig, _deep_merge, load_yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_EXPERIMENT_KEYS = ("name", "description", "output_dir", "seed", "deterministic")
_SECTION_ORDER = ("data", "tokenizer", "model", "task", "training", "evaluation", "tracking", "sweep")
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


def schema_bundle(paths: ConsolePaths) -> dict:
    keys = _registry_keys()
    presets = {
        "tokenizers": _load_presets(paths.tokenizers_dir, "tokenizer"),
        "models": _load_presets(paths.models_dir, "model"),
        "tasks": _load_presets(paths.tasks_dir, "task"),
    }
    # Registry keys win; preset files fill in when the registries are empty
    # (e.g. torch missing) so the dropdowns are never blank.
    for k in ("tokenizers", "models", "tasks"):
        if not keys[k]:
            keys[k] = sorted(presets[k].keys())
    return {
        "schema": ExperimentConfig.model_json_schema(),
        "base": base_resolved(paths),
        "registries": {**keys, "datasets": list(get_args(DatasetName))},
        "presets": presets,
        "env": {
            "hf_token_set": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
            "python": paths.rel(paths.python),
            "script": paths.rel(paths.script),
        },
    }


# ---------------------------------------------------------------------------
# Config files
# ---------------------------------------------------------------------------

def cell_names(cfg: ExperimentConfig) -> List[str]:
    """The sub-directory names ``run_sweep`` uses, in run order."""
    if cfg.sweep is None:
        return []
    out = []
    for tok in cfg.sweep.tokenizers:
        for vs in tok.vocab_sizes:
            out.append(f"{tok.type}_{vs // 1000}k" if vs else tok.type)
    return out


def is_sweep(cfg: ExperimentConfig) -> bool:
    """``run_experiment.py --sweep`` only fans out with more than one cell
    declared; a single cell always runs as a plain experiment."""
    return cfg.sweep is not None and len(cfg.sweep.tokenizers) > 1


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
    return {
        "output_dir": cfg.output_dir, "sweep": False, "cells": [],
        "cells_done": [], "cells_pending": [],
        "exists": (out_dir / "all_metrics.json").exists(), "report": False,
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
                       "results": results_summary(paths, cfg), "sweep": is_sweep(cfg)})
    except ValidationError as e:
        result.update({"valid": False, "errors": _errors_of(e), "resolved": merged, "results": None,
                       "sweep": None})
    return result


def validate_config(paths: ConsolePaths, cfg_dict: dict) -> dict:
    """Validate a (possibly partial) config dict the way the CLI would: merged
    over ``base.yaml``, then through Pydantic. Returns the resolved dump on
    success or ``loc``-tagged errors on failure."""
    merged = flatten_experiment(load_yaml(paths.base_yaml))
    _deep_merge(merged, flatten_experiment(cfg_dict or {}))
    try:
        cfg = ExperimentConfig(**merged)
    except ValidationError as e:
        return {"ok": False, "errors": _errors_of(e), "resolved": None}
    return {
        "ok": True, "errors": [], "resolved": cfg.model_dump(mode="json"),
        "sweep": is_sweep(cfg), "cells": cell_names(cfg), "results": results_summary(paths, cfg),
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


def save_config(paths: ConsolePaths, name: str, text: str, overwrite: bool = False) -> dict:
    path = paths.config_path(name)
    if path.exists() and not overwrite:
        raise ConsoleError(f"{paths.rel(path)} exists — tick overwrite to replace it")
    v = validate_config(paths, parse_yaml(text))
    if not v["ok"]:
        raise ConsoleError("refusing to save an invalid config: "
                           + "; ".join(f"{e['loc']}: {e['msg']}" for e in v["errors"]))
    paths.configs_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return {"path": paths.rel(path), "file": path.name, "resolved": v["resolved"]}


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


def _strip_ts(line: str) -> str:
    return _RE_TS.sub("", line, count=1)


def parse_progress(text: str) -> dict:
    """Reduce a console log to the run's current state (see module doc)."""
    state: Dict[str, Any] = {
        "cell": None, "experiment": None, "cells_done": [], "cells_skipped": [], "cells_failed": {},
        "stage": None, "stage_label": None,
        "phase": None, "phases_done": [],
        "eval": None, "tasks_done": [],
        "done": False, "error": None, "last_line": None,
    }
    tb_lines: List[str] = []
    in_tb = False
    for raw in re.split(r"\r\n|\n|\r", text):
        line = raw.rstrip()
        if not line:
            continue
        state["last_line"] = line
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
            state.update({"cell": m.group(1), "stage": None, "stage_label": None, "phase": None,
                          "phases_done": [], "eval": None, "tasks_done": [], "done": False})
            continue
        m = _RE_EXP_START.match(msg)
        if m:
            # A new run_experiment() begins (a cell, or a restart appended to
            # the same log): per-experiment counters start over.
            state.update({"experiment": m.group(1), "stage": None, "stage_label": None, "phase": None,
                          "phases_done": [], "eval": None, "tasks_done": [], "done": False})
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
            state["stage"] = m.group(1)
            state["stage_label"] = _STAGE_LABELS.get(m.group(1), m.group(2))
            continue
        m = _RE_STEP.search(msg)
        if m:
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
            continue
        m = _RE_PHASE_SKIP.search(msg)
        if m:
            state["phases_done"].append({"name": m.group(1), "steps": 0, "skipped": True})
            continue
        m = _RE_TASK_START.search(msg)
        if m:
            state["eval"] = {"task": m.group(1), "done": 0, "total": int(m.group(2))}
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
            continue
        m = _RE_EXP_DONE.search(msg)
        if m:
            state["done"] = True
            if state["cell"]:
                state["cells_done"].append(state["cell"])
            continue
    return state


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

STATUS_ACTIVE = ("running", "cancelling")


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


class RunManager:
    """Detached experiment runs under ``outputs/runs/<run_id>/``."""

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
        v = validate_config(self.paths, parse_yaml(yaml_text))
        if not v["ok"]:
            raise ConsoleError("config is invalid: " + "; ".join(f"{e['loc']}: {e['msg']}" for e in v["errors"]))
        cfg = ExperimentConfig(**v["resolved"])
        if sweep is None:
            sweep = is_sweep(cfg)

        with self._lock:
            active = self.active()
            if active and not force:
                raise RunConflict(
                    f"run {active[0]['run_id']} is still {active[0]['status']}; tick 'run concurrently' to start anyway"
                )
            ts = datetime.now()
            run_id = f"{ts.strftime('%Y%m%d-%H%M%S')}_{label}"
            n = 1
            while self.run_dir(run_id).exists():
                n += 1
                run_id = f"{ts.strftime('%Y%m%d-%H%M%S')}_{label}_{n}"
            rd = self.run_dir(run_id)
            rd.mkdir(parents=True)
            snapshot = rd / "config.yaml"
            snapshot.write_text(yaml_text if yaml_text.endswith("\n") else yaml_text + "\n", encoding="utf-8")
            console_log = rd / "console.log"
            exit_file = rd / "exit_code"
            argv = command or self.build_command(snapshot, sweep, device, seed)
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
                "run_id": run_id, "config": config_ref, "label": label,
                "experiment_name": cfg.name, "output_dir": cfg.output_dir,
                "sweep": bool(sweep), "cells": cell_names(cfg) if sweep else [],
                "tasks": [t.type for t in cfg.sweep.tasks] if cfg.sweep else [],
                "tokenizer": cfg.tokenizer.type, "model": cfg.model.name_or_path,
                "argv": argv, "pid": proc.pid, "pgid": proc.pid,
                "start_ticks": _proc_start_ticks(proc.pid),
                "started_at": ts.isoformat(timespec="seconds"), "finished_at": None,
                "status": "running", "exit_code": None, "cancel_requested_at": None,
                "console_log": self.paths.rel(console_log), "snapshot": self.paths.rel(snapshot),
                "experiment_log": f"outputs/logs/{cfg.name}/experiment.log",
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
        rec["progress"] = progress
        rec["log_tail"] = text
        rec["log_size"] = size
        rec["results"] = self.results(run_id)
        return rec

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
