"""End-to-end diagnostics for the 3-phase pipeline + LightEval eval.

Probes:
  1) For each task, sample N examples and compute prompt token lengths under
     the native_llama tokenizer. How often does the prompt exceed max_length=512?
  2) Show the actual PROMPT STRING that hits the model (formatted, not parsed).
  3) Compare it to the Phase 3 SFT format the model was trained on.
  4) Check whether the answer prefix `### الإجابة:` survives truncation.

Goal: confirm or refute the format-mismatch hypothesis.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

from arabic_eval.data.finetune_corpora import _format_qa_full, _format_qa_prompt, QARecord
from arabic_eval.tasks.lighteval.acva import ACVATask
from arabic_eval.tasks.lighteval.alghafa import AlghafaTask
from arabic_eval.tasks.lighteval.arabic_exam import ArabicExamTask


def main():
    print("Loading native Llama tokenizer...")
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # Load each task & sample examples
    print("\n" + "=" * 70)
    print("PROMPT FORMAT INSPECTION")
    print("=" * 70)

    for task_name, task_cls in [
        ("acva", ACVATask),
        ("alghafa", AlghafaTask),
        ("arabic_exam", ArabicExamTask),
    ]:
        print(f"\n--- {task_name} ---")
        try:
            task = task_cls({})
            examples = task.get_eval_examples()[:1000]
            print(f"  Loaded {len(examples)} examples (first 1000)")

            ctx_lens = []
            cont_letters = Counter()
            example_shown = False
            truncated_count = 0

            for ex in examples:
                ctx = task._format_eval_context(ex)
                conts = task._build_continuations(ex)
                ctx_tokens = tok.encode(ctx, add_special_tokens=True)
                ctx_lens.append(len(ctx_tokens))
                if len(ctx_tokens) > 512:
                    truncated_count += 1
                cont_letters[conts[0].lstrip()[:5]] += 1

                if not example_shown:
                    example_shown = True
                    print(f"\n  Sample prompt (first example):")
                    print("  " + "-" * 60)
                    print(ctx[:1500])
                    if len(ctx) > 1500:
                        print("  ... [truncated for display, full length: %d chars]" % len(ctx))
                    print("  " + "-" * 60)
                    print(f"  Continuations: {conts}")
                    print(f"  Gold answer idx: {ex['answer']}")

            ctx_lens = np.array(ctx_lens)
            print(f"\n  Prompt token length stats:")
            print(f"    mean={ctx_lens.mean():.1f}  p50={np.percentile(ctx_lens, 50):.0f}  "
                  f"p90={np.percentile(ctx_lens, 90):.0f}  p99={np.percentile(ctx_lens, 99):.0f}  "
                  f"max={ctx_lens.max()}")
            print(f"  >512 tokens: {truncated_count}/{len(examples)} ({100*truncated_count/len(examples):.1f}%)")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Phase 3 SFT format comparison
    print("\n" + "=" * 70)
    print("PHASE 3 SFT FORMAT (what the model was trained on)")
    print("=" * 70)
    rec = QARecord(
        id="example",
        question="ما هي عاصمة المغرب؟",
        context="المغرب دولة عربية تقع في شمال أفريقيا. عاصمتها هي الرباط.",
        answer="الرباط",
        source="arabic_squad",
    )
    print("  Phase 3 trained prompt format:")
    print("  " + "-" * 60)
    print(_format_qa_full(rec))
    print("  " + "-" * 60)


if __name__ == "__main__":
    main()
