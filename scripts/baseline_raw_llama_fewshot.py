"""Phase 3 smoke baseline: raw Llama-3.2-1B with 3-shot prompts on 200 examples.

Compares against the zero-shot baseline (Phase 2) to verify few-shot adds lift
on at least one task. Runtime ~5 min total.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from arabic_eval.tokenizers.native_llama import NativeLlamaTokenizer
from arabic_eval.models.llama_adapter import LlamaAdapter
from arabic_eval.tasks.lighteval.acva import ACVATask
from arabic_eval.tasks.lighteval.alghafa import AlghafaTask
from arabic_eval.tasks.lighteval.arabic_exam import ArabicExamTask


N_SAMPLES = 200
NUM_FEWSHOT = 3


def main():
    print("Loading native_llama tokenizer...")
    tok = NativeLlamaTokenizer()
    tok.train([], vocab_size=128256)

    print("Loading raw Llama-3.2-1B (no training)...")
    adapter = LlamaAdapter(
        model_name_or_path="meta-llama/Llama-3.2-1B",
        device="auto",
        dtype="bfloat16",
    )
    adapter.adapt_to_tokenizer(tok)
    adapter.model.eval()

    summary = {}
    for task_name, task_cls in [
        ("acva", ACVATask),
        ("alghafa", AlghafaTask),
        ("arabic_exam", ArabicExamTask),
    ]:
        print(f"\n{'=' * 60}")
        print(f"Evaluating raw Llama on {task_name} "
              f"(first {N_SAMPLES} examples, {NUM_FEWSHOT}-shot)")
        print(f"{'=' * 60}")
        task = task_cls({"num_fewshot": NUM_FEWSHOT})
        # Force the eval cache to the first N_SAMPLES so few-shot pool is
        # restricted to that subset (deterministic + bounded compute).
        all_examples = task.get_eval_examples()
        task._cached_examples = all_examples[: N_SAMPLES]
        task._fewshot_pool_by_config = None
        t0 = time.perf_counter()
        metrics = task.evaluate(
            adapter, tok,
            score_normalization="char+pmi",
        )
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s")
        print(f"  accuracy_char_norm = {metrics.get('accuracy_char_norm'):.4f}")
        print(f"  accuracy_pmi       = {metrics.get('accuracy_pmi'):.4f}")
        print(f"  num_samples        = {metrics.get('num_samples')}")
        summary[task_name] = {
            "char_norm": metrics.get("accuracy_char_norm"),
            "pmi": metrics.get("accuracy_pmi"),
            "num_samples": metrics.get("num_samples"),
        }

    print("\n\n" + "=" * 60)
    print(f"RAW LLAMA-3.2-1B SUMMARY (no training, {NUM_FEWSHOT}-shot, "
          f"first {N_SAMPLES} examples each)")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
