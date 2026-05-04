"""Step 5 smoke test for the clean_latin_rows flag: per-task offline drop
rates on real HF data, no model loading.

Iterates the 4 LightEval tasks twice each (flag off / on), reports drop rate
and verifies that no Latin-script letters remain in any kept row's inspected
text fields. Runs in the main env, ~30 s on warm cache.
"""
from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from arabic_eval.tokenizers.utils.arabic_text import contains_latin_letters
from arabic_eval.tasks.lighteval.acva import ACVATask
from arabic_eval.tasks.lighteval.alghafa import AlghafaTask
from arabic_eval.tasks.lighteval.arabic_exam import ArabicExamTask
from arabic_eval.tasks.lighteval.culture_arabic_mmlu import CultureArabicMMLUTask


def run() -> int:
    print()
    print("=" * 70)
    print("Step 5: per-task offline filter sanity check on real HF data")
    print("=" * 70)
    results = []
    for cls in (ACVATask, AlghafaTask, CultureArabicMMLUTask, ArabicExamTask):
        print(f"\n--- {cls.__name__} ---", flush=True)
        on = cls({"clean_latin_rows": True})
        off = cls({"clean_latin_rows": False})
        sft_off, ev_off = off._get_splits()
        sft_on, ev_on = on._get_splits()
        n_off = len(sft_off) + len(ev_off)
        n_on = len(sft_on) + len(ev_on)
        pct = 100.0 * (n_off - n_on) / max(n_off, 1)
        leak_count = 0
        for ex in sft_on + ev_on:
            for f in on._text_fields(ex):
                if contains_latin_letters(f):
                    leak_count += 1
                    break
        results.append(
            {
                "task": cls.__name__,
                "rows_off": n_off,
                "rows_on": n_on,
                "dropped_pct": pct,
                "leaks": leak_count,
            }
        )
        print(
            f"    rows: {n_off} -> {n_on}  ({pct:.2f}% dropped)  leaks={leak_count}",
            flush=True,
        )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(
            f"  {r['task']:25s}  off={r['rows_off']:>6d}  on={r['rows_on']:>6d}  "
            f"dropped={r['dropped_pct']:5.2f}%  leaks={r['leaks']}"
        )
    total_leaks = sum(r["leaks"] for r in results)
    print()
    if total_leaks == 0:
        print("Step 5 OK — zero leaks across all 4 tasks")
        return 0
    print(f"Step 5 FAIL — {total_leaks} leaks detected")
    return 1


if __name__ == "__main__":
    raise SystemExit(run())
