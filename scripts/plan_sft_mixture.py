#!/usr/bin/env python
"""Dry-run a phase's ``mixture`` without training.

Loads the corpora, composes the exact training set the phase would train
on with the tokenizer the experiment uses, and prints the manifest: per
corpus (pool size, quota, drawn / dropped / kept, loss tokens) and per
category (example share against loss-token share), plus the largest
``total_examples`` these shares admit. Run it before spending GPU hours on
a new ratio.

    .venv/bin/python scripts/plan_sft_mixture.py --config configs/experiments/native_llama_3phase_sft_mixture.yaml
    .venv/bin/python scripts/plan_sft_mixture.py --config <yaml> --tokenizer-path outputs/tokenizers/bpe_32k --type bpe
    .venv/bin/python scripts/plan_sft_mixture.py --config <yaml> --plan-only        # pool sizes + quotas, no tokenization

The tokenizer comes from ``tokenizer.load_path`` (or ``--tokenizer-path``);
native wrappers (``native_llama`` / ``native_qwen3``) need no path. A
from-scratch tokenizer that has not been trained yet can only be planned
(``--plan-only``): quotas depend on pool sizes alone, drop counts and
loss-token shares depend on the tokenizer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.config import CORPUS_CATEGORY, load_config
from arabic_eval.data.sft_mixture import compose_mixture, load_mixture_pools, manifest_summary, plan_mixture
from arabic_eval.registry import tokenizer_registry
from arabic_eval.utils.logging import setup_logger
import arabic_eval.tokenizers  # noqa: F401  (registers the tokenizers)


def _table(rows, headers):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run a phase's mixture (no training)")
    parser.add_argument("--config", required=True, help="Experiment YAML whose phase has a mixture")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--phase", default="sft", choices=["embedding_alignment", "warmup", "sft"])
    parser.add_argument("--tokenizer-path", default=None, help="Saved tokenizer directory (overrides tokenizer.load_path)")
    parser.add_argument("--type", default=None, help="Tokenizer registry key (overrides tokenizer.type)")
    parser.add_argument("--plan-only", action="store_true", help="Pool sizes and quotas only; skip tokenization")
    parser.add_argument("--out", default=None, help="Write the full manifest (with record ids) to this JSON file")
    args = parser.parse_args()

    base_path = args.base_config
    if base_path is None:
        default_base = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
        if default_base.exists():
            base_path = str(default_base)
    config = load_config(args.config, base_path=base_path)
    setup_logger("arabic_eval")

    phase = getattr(config.training.phases, args.phase)
    if phase.mixture is None:
        print(f"training.phases.{args.phase} has no mixture in this YAML", file=sys.stderr)
        return 2
    mixture = phase.mixture

    from arabic_eval.data.contamination import load_exclusions
    exclusions = load_exclusions(config.training.contamination_exclusions)
    pools, before = load_mixture_pools(phase.datasets, config.training.corpus_params, phase.clean_latin_rows,
                                       exclusions)
    if exclusions is not None and exclusions.dropped:
        print(f"contamination exclusions ({exclusions.path}): "
              + ", ".join(f"{k} −{v}" for k, v in exclusions.dropped.items()))
    capacities = {n: len(p) for n, p in pools.items()}
    plan = plan_mixture(mixture, phase.datasets, capacities, phase.batch_size)

    print(f"\n{args.phase}: total_examples {mixture.total_examples} → steps {phase.steps} "
          f"(batch_size {phase.batch_size}); max total_examples at these shares: "
          f"{plan['max_total_examples_at_these_shares']}\n")
    rows = [
        (n, CORPUS_CATEGORY[n], before[n], capacities[n], plan["weights"][n], plan["allocation"][n],
         plan["residual"].get(n, 0))
        for n in phase.datasets
    ]
    print(_table(rows, ["corpus", "category", "available", "after_latin", "weight", "planned", "beyond_pool"]))
    if args.plan_only:
        return 0

    tok_type = args.type or config.tokenizer.type
    tok_path = args.tokenizer_path or config.tokenizer.load_path
    tokenizer = tokenizer_registry.get(tok_type)(**config.tokenizer.params)
    if tok_path:
        tokenizer.load(tok_path)
    elif not tok_type.startswith("native_"):
        print(f"\n{tok_type} needs a saved tokenizer (--tokenizer-path) to measure drops and loss tokens; "
              f"use --plan-only for the quotas alone", file=sys.stderr)
        return 2

    encodings, manifest = compose_mixture(
        mixture, phase.datasets, pools, tokenizer, phase.max_length, phase.loss_target,
        batch_size=phase.batch_size, pool_sizes_before_filter=before, clean_latin_rows=phase.clean_latin_rows,
    )
    print(f"\ntokenizer {tok_type}: {len(encodings)} examples composed, {manifest['loss_tokens']} loss tokens\n")
    rows = [
        (n, d["planned"], d["drawn"], d["dropped_truncation"], d["dropped_cut_answer"], d["kept"], d["repeated"],
         d["loss_tokens"], round(d["loss_tokens"] / max(d["kept"], 1), 1))
        for n, d in manifest["datasets"].items()
    ]
    print(_table(rows, ["corpus", "planned", "drawn", "drop_trunc", "drop_cut", "kept", "repeated", "loss_tokens", "per_example"]))
    print()
    rows = [
        (c, f"{v['share']:.0%}", v["quota"], v["kept"], f"{v['example_share']:.1%}", v["loss_tokens"], f"{v['loss_token_share']:.1%}")
        for c, v in manifest["categories"].items()
    ]
    print(_table(rows, ["category", "share", "quota", "kept", "example_share", "loss_tokens", "loss_token_share"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"phase": args.phase, **manifest}, f, ensure_ascii=False, indent=2)
        print(f"\nmanifest → {args.out}")
    else:
        print("\n" + json.dumps(manifest_summary(manifest)["categories"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
