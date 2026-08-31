"""Diagnose how much of Phase 3's training data is being truncated.

Hypothesis: TyDiQA-Arabic contexts can be very long. With max_length=512,
many training examples may have their answer span truncated out, leaving
the model with little gradient signal.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from transformers import AutoTokenizer

from arabic_eval.data.finetune_corpora import (
    _format_qa_full, _format_qa_prompt, load_corpora,
)
from arabic_eval.data.answer_only_masking import compute_answer_only_labels


def main():
    print("Loading native Llama tokenizer...")
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # Phase 3 training corpora
    print("\nLoading TyDiQA-Arabic + ARCD train splits...")
    records = load_corpora(["tydiqa_arabic", "arcd"], "train")
    print(f"Total Phase 3 training records: {len(records)}")

    by_source = {}
    for r in records:
        by_source.setdefault(r.source, []).append(r)
    for src, rs in by_source.items():
        print(f"  {src}: {len(rs)}")

    print("\n" + "=" * 70)
    print("Phase 3 truncation analysis (max_length=512)")
    print("=" * 70)

    full_token_lens = []
    prompt_token_lens = []
    answer_token_lens = []
    n_dropped_by_lcp = 0
    n_truncated_full = 0
    n_truncated_prompt = 0
    n_no_answer_signal = 0  # full truncated to <= prompt length
    n_partial_answer = 0     # answer span partially truncated

    for r in records:
        prompt_text = _format_qa_prompt(r)
        full_text = _format_qa_full(r)

        # Encode without truncation to see real lengths
        prompt_full_enc = tok.encode(prompt_text, add_special_tokens=True)
        full_full_enc = tok.encode(full_text, add_special_tokens=True)
        prompt_token_lens.append(len(prompt_full_enc))
        full_token_lens.append(len(full_full_enc))
        answer_token_lens.append(len(full_full_enc) - len(prompt_full_enc))
        if len(full_full_enc) > 512:
            n_truncated_full += 1
        if len(prompt_full_enc) > 512:
            n_truncated_prompt += 1

        # Encode WITH truncation as the pipeline does
        prompt_enc_t = tok.encode(prompt_text, max_length=512, truncation=True, add_special_tokens=True)
        full_enc_t = tok.encode(full_text, max_length=512, truncation=True, add_special_tokens=True)

        labels = compute_answer_only_labels(prompt_enc_t, full_enc_t)
        if labels is None:
            n_dropped_by_lcp += 1
            continue
        # answer span = positions where label != -100
        ans_signal_count = sum(1 for l in labels if l != -100)
        if ans_signal_count == 0:
            n_no_answer_signal += 1
        # If full_full_enc has more answer tokens than full_enc_t has after the prompt,
        # answer was partially truncated.
        actual_answer_tokens = len(full_enc_t) - len(prompt_enc_t)
        original_answer_tokens = len(full_full_enc) - len(prompt_full_enc)
        if (actual_answer_tokens < original_answer_tokens and
            actual_answer_tokens > 0 and len(full_enc_t) == 512):
            n_partial_answer += 1

    full_arr = np.array(full_token_lens)
    prompt_arr = np.array(prompt_token_lens)
    ans_arr = np.array(answer_token_lens)

    n = len(records)
    print(f"\nFull text token lengths (no trunc):")
    print(f"  mean={full_arr.mean():.0f}  p50={np.percentile(full_arr, 50):.0f}  "
          f"p75={np.percentile(full_arr, 75):.0f}  p90={np.percentile(full_arr, 90):.0f}  "
          f"p99={np.percentile(full_arr, 99):.0f}  max={full_arr.max()}")
    print(f"\nPrompt text token lengths (no trunc):")
    print(f"  mean={prompt_arr.mean():.0f}  p50={np.percentile(prompt_arr, 50):.0f}  "
          f"p75={np.percentile(prompt_arr, 75):.0f}  p90={np.percentile(prompt_arr, 90):.0f}  "
          f"p99={np.percentile(prompt_arr, 99):.0f}  max={prompt_arr.max()}")
    print(f"\nAnswer span token lengths (no trunc):")
    print(f"  mean={ans_arr.mean():.1f}  p50={np.percentile(ans_arr, 50):.0f}  "
          f"p99={np.percentile(ans_arr, 99):.0f}  max={ans_arr.max()}")

    print(f"\nTruncation impact (max_length=512):")
    print(f"  Records with full_text > 512: {n_truncated_full}/{n} ({100*n_truncated_full/n:.1f}%)")
    print(f"  Records with prompt > 512:    {n_truncated_prompt}/{n} ({100*n_truncated_prompt/n:.1f}%)")
    print(f"  Records dropped by LCP (None): {n_dropped_by_lcp}/{n} ({100*n_dropped_by_lcp/n:.1f}%)")
    print(f"  Records with 0 answer-signal tokens: {n_no_answer_signal}/{n} ({100*n_no_answer_signal/n:.1f}%)")
    print(f"  Records with PARTIALLY truncated answer: {n_partial_answer}/{n} ({100*n_partial_answer/n:.1f}%)")

    if n_truncated_full / n > 0.5:
        print("\n!!! WARNING: more than half of Phase 3 training data exceeds max_length=512.")
        print("    Most training examples are losing context or answer span.")
    elif n_truncated_full / n > 0.1:
        print("\n  Note: substantial fraction of Phase 3 data is being truncated.")


if __name__ == "__main__":
    main()
