"""Where the experiment data lives, and the report scripts the helpers reuse."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional

REPO_ROOT = Path(os.environ.get("ARABIC_EVAL_REPO") or Path(__file__).resolve().parents[3])

_EXPERIMENTS: Optional[Path] = None


def repo_root() -> Path:
    return REPO_ROOT


def experiments_root() -> Path:
    """``outputs/experiments`` of the repo, unless overridden (tests, another checkout)."""
    if _EXPERIMENTS is not None:
        return _EXPERIMENTS
    env = os.environ.get("ARABIC_EVAL_EXPERIMENTS")
    return Path(env) if env else REPO_ROOT / "outputs" / "experiments"


def set_experiments_root(path: Optional[os.PathLike]) -> None:
    """Point every helper at another experiments folder (``None`` restores the default)."""
    global _EXPERIMENTS
    _EXPERIMENTS = None if path is None else Path(path)
    from arabic_eval.analysis.inventory import clear_cache
    clear_cache()


_SCRIPTS: Dict[str, ModuleType] = {}


def script_module(rel: str) -> ModuleType:
    """Import a repo script (``scripts/mcq_compare.py``) as a module, once.

    The comparisons of the report are computed by these scripts; the helpers
    call the same functions so a number read here equals the one they print.
    """
    if rel in _SCRIPTS:
        return _SCRIPTS[rel]
    path = REPO_ROOT / rel
    name = "_ae_script_" + rel.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - a missing script is a broken checkout
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _SCRIPTS[rel] = mod
    return mod
