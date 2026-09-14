#!/usr/bin/env python
"""Build, calibrate, or inspect the Phase 1/2 pretraining-mix pool (Stage A).

    # Build (or reuse) the cached pool declared in an experiment YAML
    .venv/bin/python scripts/build_pretraining_mix.py --config configs/experiments/<exp>.yaml

    # Calibration: stream N raw docs per source, apply every filter without
    # stopping at the word target, print drop-reason tables + dialect samples.
    # Writes under <cache_dir>/_calibration/ and never touches the real pool.
    .venv/bin/python scripts/build_pretraining_mix.py --config ... --calibrate 2000

    # Print the manifest of an already-built pool
    .venv/bin/python scripts/build_pretraining_mix.py --config ... --report

Gated sources (ArabicWeb24) need ``HF_TOKEN`` exported in this shell.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.config import load_config  # noqa: E402
from arabic_eval.data.pretraining_mix.pool import (  # noqa: E402
    build_pool, format_manifest, load_pool_manifest, pool_dir, pool_fingerprint,
)
from arabic_eval.utils.logging import setup_logger  # noqa: E402


def _print_dialect_samples(directory: Path, per_source: int) -> None:
    import csv
    path = directory / "dropped_dialect.csv"
    if not path.exists():
        return
    by_source: dict = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            by_source.setdefault(row["source"], []).append(row)
    for source, rows in by_source.items():
        print(f"\n--- dialect-gate drops from {source}: {len(rows)} total, first {per_source} ---")
        for row in rows[:per_source]:
            print(f"  [{row['per_1k_words']}/1k, {row['n_markers']} markers: {row['top_markers']}]")
            print(f"    {row['snippet'][:160]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build / calibrate / inspect the pretraining-mix pool")
    parser.add_argument("--config", required=True, help="Experiment YAML with training.pretraining_mix")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--force", action="store_true", help="Rebuild even if a complete pool is cached")
    parser.add_argument("--calibrate", type=int, default=None, metavar="N",
                        help="Calibration mode: stream N raw docs per source, no early stop, no cache")
    parser.add_argument("--report", action="store_true", help="Print the manifest of the cached pool and exit")
    parser.add_argument("--samples", type=int, default=15, help="Dialect-drop snippets to print per source")
    args = parser.parse_args()

    base_path = args.base_config
    if base_path is None:
        default_base = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
        if default_base.exists():
            base_path = str(default_base)
    config = load_config(args.config, base_path=base_path)
    mix = config.training.pretraining_mix
    if mix is None:
        print("training.pretraining_mix is not configured in this YAML", file=sys.stderr)
        return 2
    setup_logger("arabic_eval")

    if args.report:
        directory = pool_dir(mix)
        manifest = load_pool_manifest(directory)
        if manifest is None:
            print(f"no pool at {directory} (fingerprint {pool_fingerprint(mix)})")
            return 1
        print(format_manifest(manifest))
        return 0

    if args.calibrate:
        out_dir = Path(mix.cache_dir) / "_calibration" / f"{pool_fingerprint(mix)}_n{args.calibrate}"
        directory = build_pool(
            mix, out_dir=out_dir, force=True,
            doc_limit_per_source=args.calibrate, stop_at_target=False,
        )
    else:
        directory = build_pool(mix, force=args.force)

    manifest = load_pool_manifest(directory)
    print(format_manifest(manifest))
    _print_dialect_samples(directory, args.samples)
    print(f"\nartifacts: {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
