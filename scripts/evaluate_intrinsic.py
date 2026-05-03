#!/usr/bin/env python3
"""CLI: Compute intrinsic tokenizer metrics on evaluation data.

Usage:
    python scripts/evaluate_intrinsic.py --tokenizer-path outputs/tokenizers/bpe_32k --type bpe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.data.loader import load_arabic_dataset, extract_texts
from arabic_eval.evaluation.intrinsic_metrics import compute_intrinsic_metrics
from arabic_eval.utils.logging import setup_logger
from arabic_eval.utils.io import save_json

import arabic_eval.tokenizers  # noqa: F401
from arabic_eval.registry import tokenizer_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tokenizer intrinsic metrics")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                        help="Path to saved tokenizer directory")
    parser.add_argument("--type", type=str, required=True,
                        help="Tokenizer type (registry key)")
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Number of eval texts to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    setup_logger("arabic_eval")

    # Load tokenizer
    tokenizer_cls = tokenizer_registry.get(args.type)
    tokenizer = tokenizer_cls()
    tokenizer.load(args.tokenizer_path)
    print(f"Loaded {args.type} tokenizer (vocab_size={tokenizer.vocab_size})")

    # Load eval texts
    dataset = load_arabic_dataset(max_eval_samples=args.num_samples)
    eval_texts = extract_texts(dataset.get("eval", dataset["train"]))[:args.num_samples]
    print(f"Evaluating on {len(eval_texts)} texts...")

    # Compute metrics
    metrics = compute_intrinsic_metrics(tokenizer, eval_texts)

    print("\nIntrinsic Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save
    if args.output:
        save_json(metrics, args.output)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
