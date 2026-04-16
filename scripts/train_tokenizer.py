#!/usr/bin/env python3
"""CLI: Train a single tokenizer on the Arabic dataset.

Usage:
    python scripts/train_tokenizer.py --type bpe --vocab-size 32000
    python scripts/train_tokenizer.py --type morpho_bpe --vocab-size 16000
    python scripts/train_tokenizer.py --type character_bert
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.data.loader import load_arabic_dataset, extract_texts
from arabic_eval.utils.logging import setup_logger
from arabic_eval.utils.reproducibility import set_seed

# Trigger registry
import arabic_eval.tokenizers  # noqa: F401
from arabic_eval.registry import tokenizer_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Arabic tokenizer")
    parser.add_argument("--type", type=str, required=True,
                        help=f"Tokenizer type: {tokenizer_registry.list_available()}")
    parser.add_argument("--vocab-size", type=int, default=32_000,
                        help="Target vocabulary size (ignored for character-level)")
    parser.add_argument("--dataset", type=str, default="Jr23xd23/ArabicText-Large",
                        help="HuggingFace dataset name")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max training samples")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: outputs/tokenizers/<type>_<vocab>)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logger("arabic_eval")
    set_seed(args.seed)

    # Default output path
    if args.output is None:
        vs = f"_{args.vocab_size // 1000}k" if args.vocab_size else ""
        args.output = f"outputs/tokenizers/{args.type}{vs}"

    # Load data
    print(f"Loading dataset: {args.dataset}")
    dataset = load_arabic_dataset(
        dataset_name=args.dataset,
        max_train_samples=args.max_samples,
        seed=args.seed,
    )
    texts = extract_texts(dataset["train"])
    print(f"Loaded {len(texts)} training texts")

    # Train tokenizer
    print(f"Training {args.type} tokenizer (vocab_size={args.vocab_size})...")
    tokenizer_cls = tokenizer_registry.get(args.type)
    tokenizer = tokenizer_cls()
    tokenizer.train(texts, vocab_size=args.vocab_size)

    # Save
    tokenizer.save(args.output)
    print(f"Tokenizer saved to: {args.output}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Quick test
    test_text = "مرحبا بكم في منصة تقييم المجزئات العربية"
    enc = tokenizer.encode(test_text)
    print(f"\nTest encode: '{test_text}'")
    print(f"  Tokens ({len(enc.input_ids)}): {enc.tokens[:20]}")
    print(f"  IDs: {enc.input_ids[:20]}")
    dec = tokenizer.decode(enc.input_ids)
    print(f"  Decoded: '{dec}'")


if __name__ == "__main__":
    main()
