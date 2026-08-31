"""Sanity baseline: raw Llama-3.2-1B (no training) on a sample of each eval task.

If Phase 1's high LR (1e-3) is damaging pretrained native_llama embeddings,
the raw model should outperform native_llama_3phase_*. If it underperforms,
the pipeline is adding value (or at least not destroying things).

We sample 500 examples per task to keep runtime short (~5 min/task).
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import time

# Force tokenizers to be quiet
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from arabic_eval.tokenizers.native_llama import NativeLlamaTokenizer
from arabic_eval.models.llama_adapter import LlamaAdapter
from arabic_eval.tasks.lighteval.acva import ACVATask
from arabic_eval.tasks.lighteval.alghafa import AlghafaTask
from arabic_eval.tasks.lighteval.arabic_exam import ArabicExamTask


N_SAMPLES = 500


def main():
    print("Loading native_llama tokenizer (raw Llama-3.2-1B tokenizer)...")
    tok = NativeLlamaTokenizer()
    tok.train([], vocab_size=128256)
    print("  vocab=", tok.vocab_size)

    print("\nLoading raw Llama-3.2-1B (no training)...")
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
        print(f"Evaluating raw Llama on {task_name} (first {N_SAMPLES} examples)")
        print(f"{'=' * 60}")
        task = task_cls({})
        # Truncate the eval set to N_SAMPLES.
        all_examples = task.get_eval_examples()
        # Save and clobber via override
        task._cached_examples = all_examples[:N_SAMPLES]
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
    print("RAW LLAMA-3.2-1B SUMMARY (no training, first 500 examples each)")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
