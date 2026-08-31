"""Compare embedding distributions: pretrained Llama vs after Phase 1.

Phase 1 trains embed_tokens (+ tied lm_head) at LR=1e-3 for 1000 steps. For
native_llama (vocab=128256, no resize), the starting embeddings are byte-
identical to pretrained Llama. After Phase 1, how much have they drifted?

Aggressive drift on a high-LR phase could distort the pretrained Arabic
representations, hurting downstream eval.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM


def main():
    print("Loading pretrained Llama-3.2-1B embeddings...")
    pretrained = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B", torch_dtype=torch.float32
    )
    pre_emb = pretrained.get_input_embeddings().weight.detach().cpu()
    pre_lm = pretrained.lm_head.weight.detach().cpu()
    tied_pre = (pre_emb.data_ptr() == pre_lm.data_ptr())

    print(f"  embed_tokens shape: {pre_emb.shape}")
    print(f"  lm_head shape: {pre_lm.shape}")
    print(f"  tied (same pointer): {tied_pre}")
    print(f"  embed L2 mean: {pre_emb.norm(dim=1).mean().item():.4f}")
    print(f"  embed L2 std:  {pre_emb.norm(dim=1).std().item():.4f}")
    del pretrained

    print("\nLoading Phase 1 checkpoint (after embedding alignment)...")
    p1 = AutoModelForCausalLM.from_pretrained(
        "outputs/experiments/native_llama_3phase_with_sft/training/embedding_alignment",
        torch_dtype=torch.float32,
    )
    p1_emb = p1.get_input_embeddings().weight.detach().cpu()
    p1_lm = p1.lm_head.weight.detach().cpu()
    tied_p1 = (p1_emb.data_ptr() == p1_lm.data_ptr())
    print(f"  tied after Phase 1: {tied_p1}")

    diff = (p1_emb - pre_emb).abs()
    print(f"\nDrift after Phase 1 (1000 steps, LR=1e-3, full-sequence loss):")
    print(f"  per-row L2 of (p1 - pre): mean={diff.norm(dim=1).mean().item():.4f}  "
          f"max={diff.norm(dim=1).max().item():.4f}")
    print(f"  pretrained per-row L2:    mean={pre_emb.norm(dim=1).mean().item():.4f}")
    rel_drift = (diff.norm(dim=1) / pre_emb.norm(dim=1).clamp(min=1e-8)).mean().item()
    print(f"  Relative drift (||delta||/||pre||):  mean={rel_drift:.4f}  "
          f"({rel_drift * 100:.1f}% of original norm)")

    print("\nLoading after Phase 2 (warmup) checkpoint...")
    p2 = AutoModelForCausalLM.from_pretrained(
        "outputs/experiments/native_llama_3phase_with_sft/training/warmup",
        torch_dtype=torch.float32,
    )
    p2_emb = p2.get_input_embeddings().weight.detach().cpu()
    diff2 = (p2_emb - pre_emb).abs()
    rel2 = (diff2.norm(dim=1) / pre_emb.norm(dim=1).clamp(min=1e-8)).mean().item()
    print(f"  After Phase 2 cumulative drift: {rel2 * 100:.1f}% of original embed norm")

    print("\nLoading after Phase 3 (sft) checkpoint...")
    p3 = AutoModelForCausalLM.from_pretrained(
        "outputs/experiments/native_llama_3phase_with_sft/training/sft",
        torch_dtype=torch.float32,
    )
    p3_emb = p3.get_input_embeddings().weight.detach().cpu()
    diff3 = (p3_emb - pre_emb).abs()
    rel3 = (diff3.norm(dim=1) / pre_emb.norm(dim=1).clamp(min=1e-8)).mean().item()
    print(f"  After Phase 3 cumulative drift: {rel3 * 100:.1f}% of original embed norm")

    print("\n--- Comparing transformer body weights (Phase 1 should NOT modify these) ---")
    p1_body_q = p1.model.layers[0].self_attn.q_proj.weight.detach().cpu()
    pre = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.float32)
    pre_body_q = pre.model.layers[0].self_attn.q_proj.weight.detach().cpu()
    body_diff = (p1_body_q - pre_body_q).abs().max().item()
    print(f"  Layer 0 q_proj max-abs diff: {body_diff:.6f}  (should be 0.0 — body was frozen)")


if __name__ == "__main__":
    main()
