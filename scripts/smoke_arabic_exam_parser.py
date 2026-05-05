"""End-to-end smoke check for the arabic_exam parser fixes (2026-05-04).

Run from the project root:

    .venv/bin/python scripts/smoke_arabic_exam_parser.py

Hits the real ``MBZUAI/ArabicMMLU`` dataset (cached under
``outputs/data_cache``) and asserts the four bug-fix invariants:

  1. ``All`` aggregate config is excluded → no row appears under that group.
  2. Total examples = sum of the 40 subject configs minus 120 ``is_few_shot``
     rows ⇒ 14455 rows on the current dataset snapshot.
  3. ~706 rows carry a non-empty ``Context`` and the eval prompt prepends
     a ``### السياق:`` block for those, byte-identical to the inherited
     base helper for the rest.
  4. 340 rows have 5 choices and 139 of those are ``Answer Key=E`` (recovered
     from the pre-fix silent-drop branch). Note: the raw dataset has 344 /
     141 of these respectively; subtracting the few_shot rows that hit those
     buckets yields the post-filter counts.

Mirrors the pattern in ``scripts/smoke_morph_metrics.py`` — runnable
standalone, prints a green/red summary, exits non-zero on any failure.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Project src on sys.path (matches the other smoke scripts).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arabic_eval.tasks.lighteval.arabic_exam import ArabicExamTask  # noqa: E402
from arabic_eval.tasks.lighteval.utils import format_mcq_context  # noqa: E402


EXPECTED_TOTAL = 14455           # 14575 raw - 120 is_few_shot
MIN_CONTEXT_ROWS = 700           # 706 in current snapshot (709 raw - 3 few_shot)
EXPECTED_5_CHOICE = 340          # 344 raw - 4 few_shot
EXPECTED_E_ANSWER = 139          # 141 raw - 2 few_shot


def _check(label: str, ok: bool, detail: str = "") -> bool:
    flag = "[ OK ]" if ok else "[FAIL]"
    msg = f"{flag} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return ok


def main() -> int:
    task = ArabicExamTask({"cache_dir": "outputs/data_cache", "seed": 42})
    exs = task._load_all_examples()

    cfg_hist = Counter(e.get("_source_config", "_default") for e in exs)
    five_choice = [e for e in exs if len(e["choices"]) == 5]
    e_answer = [e for e in exs if e["answer"] == 4]
    with_ctx = [e for e in exs if e.get("context")]
    no_ctx = [e for e in exs if not e.get("context")]

    all_ok = True

    # --- Step 1 invariant: All config excluded ----------------------------
    all_ok &= _check(
        "EXCLUDED_CONFIGS removed 'All' config",
        "All" not in cfg_hist,
        f"num_configs={len(cfg_hist)}",
    )

    # --- Step 4 invariant: total = 14575 - 120 ----------------------------
    all_ok &= _check(
        f"total examples = {EXPECTED_TOTAL}",
        len(exs) == EXPECTED_TOTAL,
        f"actual={len(exs)}",
    )

    # --- Step 2 invariants: Context captured & rendered -------------------
    all_ok &= _check(
        f"≥ {MIN_CONTEXT_ROWS} rows carry non-empty context",
        len(with_ctx) >= MIN_CONTEXT_ROWS,
        f"actual={len(with_ctx)}",
    )
    if with_ctx:
        sample = with_ctx[0]
        rendered = task._format_eval_context(sample)
        all_ok &= _check(
            "context-bearing prompt prepends '### السياق:'",
            rendered.startswith("### السياق:\n"),
        )

    # Byte-identical fallback for context-free rows
    fallback_mismatches = 0
    for e in no_ctx[:500]:
        if task._format_eval_context(e) != format_mcq_context(e["question"], e["choices"]):
            fallback_mismatches += 1
    all_ok &= _check(
        "context-free prompt is byte-identical to inherited helper (sample 500)",
        fallback_mismatches == 0,
        f"mismatches={fallback_mismatches}",
    )

    # --- Step 3 invariants: 5-choice rows + Answer=E recovered ------------
    all_ok &= _check(
        f"5-choice rows == {EXPECTED_5_CHOICE}",
        len(five_choice) == EXPECTED_5_CHOICE,
        f"actual={len(five_choice)}",
    )
    all_ok &= _check(
        f"answer=E rows == {EXPECTED_E_ANSWER} (recovered from silent drop)",
        len(e_answer) == EXPECTED_E_ANSWER,
        f"actual={len(e_answer)}",
    )
    if five_choice:
        rendered = task._format_eval_context(five_choice[0])
        five_letters_ok = all(letter in rendered for letter in ("أ", "ب", "ج", "د", "هـ"))
        all_ok &= _check("5-choice prompt renders all 5 Arabic letters", five_letters_ok)
        conts = task._build_continuations(five_choice[0])
        all_ok &= _check(
            "5-choice continuations have 5 entries",
            len(conts) == 5,
            f"continuations={conts}",
        )

    print()
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
