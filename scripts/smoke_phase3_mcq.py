"""Smoke test: 100-step Phase 3 dry-run with the new arabic_squad_mcq corpus.

Verifies the integration end-to-end: corpus loading, tokenization (LCP
masking on letter answers), forward pass, backward pass, optimizer step.
Does not check loss quality — that's the full sweep's job.

Runtime ~2-3 min on H100.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from arabic_eval.config import PhaseConfig, EarlyStoppingConfig
from arabic_eval.data.finetune_corpora import build_qa_dataloader, load_corpora
from arabic_eval.models.llama_adapter import LlamaAdapter
from arabic_eval.tokenizers.native_llama import NativeLlamaTokenizer
from arabic_eval.training.phases import run_phase


def main():
    print("Loading native_llama tokenizer...")
    tok = NativeLlamaTokenizer()
    tok.train([], vocab_size=128256)

    print("Loading Llama-3.2-1B...")
    adapter = LlamaAdapter(
        model_name_or_path="meta-llama/Llama-3.2-1B",
        device="auto",
        dtype="bfloat16",
    )
    adapter.adapt_to_tokenizer(tok)

    print("Loading the new Phase-3 corpora (tydiqa + arcd + synthetic MCQ)...")
    train_records = load_corpora(
        ["tydiqa_arabic", "arcd", "arabic_squad_mcq"], "train"
    )
    print(f"Total training records: {len(train_records)}")
    by_src = {}
    for r in train_records:
        by_src[r.source] = by_src.get(r.source, 0) + 1
    print(f"By source: {by_src}")

    # Build a tiny train loader
    train_loader = build_qa_dataloader(
        train_records[:200],  # small subset for the smoke test
        tok,
        batch_size=4,
        max_length=512,
        loss_target="answer_only",
        shuffle=True,
    )

    phase_cfg = PhaseConfig(
        enabled=True,
        datasets=["tydiqa_arabic", "arcd", "arabic_squad_mcq"],
        trainable_parameters=["*"],
        steps=20,
        learning_rate=2.0e-4,
        batch_size=4,
        gradient_accumulation_steps=4,
        optimizer="adamw",
        weight_decay=0.01,
        max_length=512,
        loss_target="answer_only",
        lr_scheduler="cosine",
        warmup_steps=0,
        max_grad_norm=1.0,
        save_checkpoint=False,
        early_stopping=None,
    )

    print("Running 20-step Phase 3 dry-run...")
    out_dir = Path("/tmp/smoke_phase3_mcq")
    result = run_phase(
        phase_name="sft",
        adapter=adapter,
        phase_cfg=phase_cfg,
        train_loader=train_loader,
        eval_loader=None,
        output_dir=out_dir,
        bf16=True,
        fp16=False,
        logging_steps=5,
    )

    print()
    print(f"Steps completed: {result.steps_completed}")
    print(f"Final train loss: {result.final_train_loss:.4f}")
    print(f"Wall time: {result.wall_time_sec:.1f}s")
    print()
    print("DONE — Phase 5 smoke test PASSED.")


if __name__ == "__main__":
    main()
