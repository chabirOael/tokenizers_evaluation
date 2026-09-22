#!/usr/bin/env python
"""Embedding-initialisation probe: step-0 loss and a 50-update Phase-1 probe per method.

"Beyond Initialization Loss" (arXiv 2608.03494) measured that the loss *at*
initialisation predicts a method's CPT outcome poorly while a short CPT probe
ranks methods reliably. So for every ``model.embedding_init`` method this
script loads the base model, adapts it to the tokenizer with that method,
measures the raw-text loss of the untrained rows (``diag_heldout_loss.lm_loss``
on the 120 held-out pool documents), runs **50 Phase-1-style updates** through
the real phase runner (embeddings only, LR 5e-4 constant, batch 8 × 512, the
first blocks of the tokenizer's packed pretraining mix — never the held-out
documents) and measures the same loss again. The table it prints — and writes
as JSON — is what gates the method carried into a full run: it must beat
``legacy`` at step 0 *and* at step 50, and ``surface_avg`` must beat ``mean``
at step 50 to be preferred over it.

    .venv/bin/python scripts/probe_embedding_init.py \\
        --tokenizer-path outputs/tokenizers/araroopat_maxpat40k_v3 --tokenizer-type araroopat \\
        --config configs/experiments/qwen_araroopat_3phase_v2.yaml \\
        --out outputs/experiments/qwen_native_vs_araroopat/embedding_init_probe.json

``--config`` supplies the model (``model.*``) and the pretraining-mix block
(``training.pretraining_mix``: pool, block size, seed); the packed corpus for
the tokenizer is built when missing (~11 min for AraRooPat on the 20 M pool).
A method is written as ``method[:weighting][+norm]`` —
``surface_avg:char_len+base_mean``. GPU: one model at a time, ~2–3 min each.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import arabic_eval.models  # noqa: E402,F401 — registries
import arabic_eval.tokenizers  # noqa: E402,F401
from arabic_eval.config import PhaseConfig, load_config  # noqa: E402
from arabic_eval.registry import model_registry, tokenizer_registry  # noqa: E402

import diag_heldout_loss as D  # noqa: E402

log = logging.getLogger("probe_embedding_init")

DEFAULT_METHODS = ["legacy", "random", "mean", "surface_avg:uniform", "surface_avg:char_len",
                   "surface_avg:char_len+base_mean"]


def parse_method(spec: str) -> Dict[str, Any]:
    """``surface_avg:char_len+base_mean`` → {method, weighting, norm}."""
    norm = "none"
    if "+" in spec:
        spec, norm = spec.split("+", 1)
    weighting = "uniform"
    if ":" in spec:
        spec, weighting = spec.split(":", 1)
    return {"method": spec, "weighting": weighting, "norm": norm}


def probe_one(spec: str, *, cfg, tokenizer, tokenizer_type: str, blocks, chars_per_token: float,
              packed, steps: int, lr: float, batch_size: int, device: str, seed: int) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from arabic_eval.data.collation import get_collator
    from arabic_eval.training.phases import run_phase

    init = {**parse_method(spec), "base_tokenizer": cfg.model.embedding_init.base_tokenizer, "seed": seed}
    t0 = time.perf_counter()
    model_cls = model_registry.get(cfg.model.type)
    adapter = model_cls(model_name_or_path=cfg.model.name_or_path, device=device, dtype=cfg.model.dtype,
                        embedding_init=init, **cfg.model.params)
    adapter.adapt_to_tokenizer(tokenizer)
    report = adapter.embedding_init_report.to_json() if adapter.embedding_init_report else None
    t_init = time.perf_counter() - t0

    adapter.model.eval()
    step0 = D.lm_loss(adapter.model, blocks, adapter.device)

    block_size = packed.block_size
    dataset, info = packed.take(f"probe:{spec}", steps * batch_size)
    packed.cursor = 0                                   # every method probes on the same first blocks
    collator = get_collator(tokenizer.embedding_type, pad_token_id=getattr(tokenizer, "pad_token_id", 0),
                            max_length=block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    phase_cfg = PhaseConfig(
        datasets=["pretraining_mix"], trainable_parameters=["embed_tokens", "lm_head"], steps=steps,
        mix_tokens=steps * batch_size * block_size, learning_rate=lr, batch_size=batch_size,
        gradient_accumulation_steps=1, max_length=block_size, loss_target="full_sequence",
        lr_scheduler="constant", warmup_steps=0, save_checkpoint=False,
    )
    t1 = time.perf_counter()
    res = run_phase(phase_name=f"probe:{spec}", adapter=adapter, phase_cfg=phase_cfg, train_loader=loader,
                    output_dir=Path("/tmp") / "probe_embedding_init", logging_steps=10)
    t_train = time.perf_counter() - t1
    adapter.model.eval()
    step_n = D.lm_loss(adapter.model, blocks, adapter.device)
    train_losses = [round(l, 4) for _, l in res.train_losses]

    out = {
        "spec": spec, **parse_method(spec),
        "step0_loss_per_token": step0["loss_per_token"],
        "step0_loss_per_char": round(step0["loss_per_token"] / chars_per_token, 4),
        f"step{steps}_loss_per_token": step_n["loss_per_token"],
        f"step{steps}_loss_per_char": round(step_n["loss_per_token"] / chars_per_token, 4),
        "train_loss_first": train_losses[0] if train_losses else None,
        "train_loss_last10_mean": round(sum(train_losses[-10:]) / max(len(train_losses[-10:]), 1), 4) if train_losses else None,
        "train_losses": train_losses,
        "blocks": info,
        "init_report": report,
        "wall_sec": {"load_and_init": round(t_init, 1), "train": round(t_train, 1),
                     "total": round(time.perf_counter() - t0, 1)},
    }
    del adapter
    torch.cuda.empty_cache()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--tokenizer-type", default="araroopat")
    ap.add_argument("--config", required=True, help="experiment YAML: model.* and training.pretraining_mix")
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-docs", type=int, default=D.N_DOCS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import pyarrow.parquet as pq

    from arabic_eval.data.pretraining_mix.packing import build_packed_corpus
    from arabic_eval.data.pretraining_mix.pool import build_pool

    cfg = load_config(args.config, base_path=str(REPO_ROOT / "configs/base.yaml"))
    mix_cfg = cfg.training.pretraining_mix
    tok_cls = tokenizer_registry.get(args.tokenizer_type)
    tokenizer = tok_cls()
    tokenizer.load(args.tokenizer_path)
    specials = tokenizer.special_tokens or {}
    eos, bos = specials.get("eos_token"), specials.get("bos_token")
    log.info("tokenizer %s: vocab %d", args.tokenizer_path, tokenizer.vocab_size)

    pool_dir = build_pool(mix_cfg)                      # cached: same fingerprint → no rebuild
    docs = pq.read_table(pool_dir / "fineweb2_arb.parquet", columns=["text"]).column("text").to_pylist()[-args.n_docs:]
    doc_ids = [D.strip_specials(tokenizer.encode(d).input_ids, bos, eos) for d in docs]
    chars = sum(len(d) for d in docs)
    toks = sum(len(x) for x in doc_ids)
    cpt = chars / max(toks, 1)
    blocks = D.blocks_from(doc_ids, int(eos), mix_cfg.block_size)
    log.info("held-out: %d docs, %d tokens, %.3f chars/token, %d blocks", len(docs), toks, cpt, len(blocks))

    packed = build_packed_corpus(mix_cfg, tokenizer, args.tokenizer_type, pool_dir=pool_dir)
    log.info("packed corpus: %d blocks × %d", packed.n_blocks, packed.block_size)

    results: List[Dict[str, Any]] = []
    for spec in args.methods:
        log.info("=== %s ===", spec)
        results.append(probe_one(spec, cfg=cfg, tokenizer=tokenizer, tokenizer_type=args.tokenizer_type,
                                 blocks=blocks, chars_per_token=cpt, packed=packed, steps=args.steps, lr=args.lr,
                                 batch_size=args.batch_size, device=args.device, seed=args.seed))
        r = results[-1]
        log.info("%s: step0 %.4f/tok (%.4f/char) → step%d %.4f/tok (%.4f/char), %.0fs", spec,
                 r["step0_loss_per_token"], r["step0_loss_per_char"], args.steps,
                 r[f"step{args.steps}_loss_per_token"], r[f"step{args.steps}_loss_per_char"], r["wall_sec"]["total"])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"tokenizer_path": args.tokenizer_path, "tokenizer_type": args.tokenizer_type,
                       "vocab_size": tokenizer.vocab_size, "model": cfg.model.name_or_path, "config": args.config,
                       "steps": args.steps, "lr": args.lr, "batch_size": args.batch_size,
                       "block_size": mix_cfg.block_size, "heldout_docs": len(docs), "heldout_tokens": toks,
                       "chars_per_token": round(cpt, 4), "pool_dir": str(pool_dir), "results": results},
                      f, ensure_ascii=False, indent=2)

    k = f"step{args.steps}"
    print(f"\n{'method':34} {'step-0 /tok':>12} {'step-0 /char':>13} {k + ' /tok':>13} {k + ' /char':>14} {'wall s':>8}")
    for r in results:
        print(f"{r['spec']:34} {r['step0_loss_per_token']:12.4f} {r['step0_loss_per_char']:13.4f} "
              f"{r[k + '_loss_per_token']:13.4f} {r[k + '_loss_per_char']:14.4f} {r['wall_sec']['total']:8.0f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
