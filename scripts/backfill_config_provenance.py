#!/usr/bin/env python
"""Backfill the provenance fields of the experiment configs that predate them.

``experiment.created_at`` and ``experiment.runs`` are stamped by the tooling
from 2026-09-20 on (console save / clone, console and CLI starts). For the
files written before that, this script reconstructs what the repo still
knows, editing each file's own text (comments intact):

* ``created_at`` ← the author date of the commit that added the file
  (``git log --diff-filter=A --follow``); an untracked file gets its mtime
  and is flagged.
* ``runs`` ← every console run record under ``outputs/runs/*/run.json``
  whose ``config`` names the file (``source: console``), oldest first.
  CLI starts of the past left no record and cannot be recovered.

Idempotent: a file that already has ``created_at`` keeps it, run ids already
listed are skipped. ``--dry-run`` prints the plan without writing.

    .venv/bin/python scripts/backfill_config_provenance.py --dry-run
    .venv/bin/python scripts/backfill_config_provenance.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arabic_eval.config_edit import append_run, get_created_at, stamp_created_at  # noqa: E402


def git_added_at(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "log", "--diff-filter=A", "--follow", "--format=%aI", "--", str(path.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out[-1] if out else None


def console_runs(rel: str) -> list[dict]:
    runs = []
    for rec_path in sorted((REPO_ROOT / "outputs" / "runs").glob("*/run.json")):
        try:
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if rec.get("kind", "experiment") == "experiment" and rec.get("config") == rel:
            # records before 2026-09-20 carry a naive local time; give it its offset
            when = datetime.fromisoformat(rec["started_at"]).astimezone().isoformat(timespec="seconds")
            runs.append({"started_at": when, "run_id": rec["run_id"], "source": "console"})
    return sorted(runs, key=lambda r: r["started_at"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg_dir = REPO_ROOT / "configs" / "experiments"
    changed = 0
    for path in sorted(cfg_dir.glob("*.yaml")):
        rel = str(path.relative_to(REPO_ROOT))
        original = text = path.read_text(encoding="utf-8")
        notes = []
        if get_created_at(text):
            notes.append(f"created_at kept ({get_created_at(text)})")
        else:
            when = git_added_at(path)
            flag = ""
            if when is None:
                when = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
                flag = " (untracked — file mtime)"
            new = stamp_created_at(text, when)
            if new is None:
                notes.append("created_at: layout not editable — skipped")
            else:
                text = new
                notes.append(f"created_at ← {when}{flag}")
        have = {r.get("run_id") for r in _runs_of(text)}
        added = 0
        for entry in console_runs(rel):
            if entry["run_id"] in have:
                continue
            new = append_run(text, entry)
            if new is None:
                notes.append("runs: layout not editable — skipped")
                break
            text, added = new, added + 1
        if added:
            notes.append(f"runs +{added}")
        status = "unchanged"
        if text != original:
            changed += 1
            status = "would write" if args.dry_run else "wrote"
            if not args.dry_run:
                tmp = path.with_suffix(".yaml.tmp")
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, path)
        print(f"{status:11s} {path.name:48s} {'; '.join(notes)}")
    print(f"{changed} file(s) {'to change' if args.dry_run else 'changed'}")
    return 0


def _runs_of(text: str) -> list:
    import yaml
    data = yaml.safe_load(text) or {}
    exp = data.get("experiment") if isinstance(data.get("experiment"), dict) else data
    return exp.get("runs") or []


if __name__ == "__main__":
    sys.exit(main())
