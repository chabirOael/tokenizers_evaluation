#!/usr/bin/env python
"""Regenerate ``configs/tasks/<type>.yaml`` from every task's ``param_spec()``.

The files are *documentation*: every parameter a task declares, its default,
type and one-line help. A run never reads them — the pipeline takes a task's
parameters from the experiment YAML's ``sweep.tasks[].params`` only — and the
experiment console renders the same specs live (``/api/schema`` →
``task_params``), so the files exist for readers (CLAUDE.md links them) and
must equal what the spec says; ``tests/test_task_param_specs.py`` pins that.

    .venv/bin/python scripts/render_task_presets.py            # write the files
    .venv/bin/python scripts/render_task_presets.py --check    # exit 1 when a file is stale
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from arabic_eval.params_spec import render_task_preset, task_param_specs  # noqa: E402


def expected_presets() -> dict:
    """``{task_type: text}`` for every registered task that declares parameters."""
    from arabic_eval.registry import task_registry
    out = {}
    for task, spec in task_param_specs().items():
        if spec:
            out[task] = render_task_preset(task, task_registry.get(task).__name__, spec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare only; exit 1 when any file differs")
    ap.add_argument("--out-dir", default=str(REPO / "configs" / "tasks"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    stale = []
    for task, text in expected_presets().items():
        path = out_dir / f"{task}.yaml"
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            print(f"up to date  {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")
            continue
        stale.append(path)
        if args.check:
            print(f"STALE       {path}")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"written     {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")
    if args.check and stale:
        print(f"{len(stale)} preset file(s) differ from the specs — run scripts/render_task_presets.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
