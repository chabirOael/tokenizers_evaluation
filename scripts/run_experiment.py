#!/usr/bin/env python3
"""CLI: Run a full experiment from a YAML config file.

Usage:
    python scripts/run_experiment.py --config configs/experiments/native_llama_3phase_with_sft.yaml
    python scripts/run_experiment.py --config configs/experiments/<sweep>.yaml --sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.config import load_config
from arabic_eval.pipeline.experiment import run_experiment, run_sweep
from arabic_eval.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Arabic tokenizer evaluation experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--base-config", type=str, default=None, help="Path to base YAML config")
    parser.add_argument("--sweep", action="store_true", help="Run as sweep over combinations")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., cuda:0)")
    args = parser.parse_args()

    # Load config
    base_path = args.base_config
    if base_path is None:
        default_base = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
        if default_base.exists():
            base_path = str(default_base)

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["model"] = {"device": args.device}

    config = load_config(args.config, base_path=base_path, overrides=overrides or None)

    # Setup logging — full log + an error-only log for quick post-run triage.
    # Both go under outputs/logs/<experiment_name>/ so all logs live in one place.
    log_dir = Path("outputs/logs") / config.name
    log_file = log_dir / "experiment.log"
    error_log_file = log_dir / "errors.log"
    setup_logger("arabic_eval", log_file=log_file, error_log_file=error_log_file)

    # Run. The 3-phase pipeline trains once per (tokenizer, vocab_size) and
    # evaluates on every task in ``sweep.tasks``. ``run_experiment`` covers
    # the single-cell case; ``run_sweep`` covers multiple tokenizer cells.
    if args.sweep and config.sweep is not None and len(config.sweep.tokenizers) > 1:
        results = run_sweep(config)
    else:
        results = run_experiment(config)

    print(f"\nExperiment complete. Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
