#!/usr/bin/env python3
"""CLI: Run a full experiment from a YAML config file.

Usage:
    python scripts/run_experiment.py --config configs/experiments/bpe_32k_generation.yaml
    python scripts/run_experiment.py --config configs/experiments/full_sweep.yaml --sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.config import load_config
from arabic_eval.pipeline.experiment import run_single_experiment, run_sweep
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

    # Setup logging
    log_file = Path(config.output_dir) / "experiment.log"
    setup_logger("arabic_eval", log_file=log_file)

    # Run
    if args.sweep or config.sweep is not None:
        results = run_sweep(config)
    else:
        results = run_single_experiment(config)

    print(f"\nExperiment complete. Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
