"""Smoke test for the per-phase clean_latin_rows flag: offline filter check
on synthetic QARecords. No network, no model loading. Companion to
``smoke_clean_latin_rows.py`` (which covers the eval-side LightEval flag) —
this one covers the *training-side* filter on the 3-phase pipeline.

Exits 0 on success; raises (non-zero exit) on any unexpected outcome.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("smoke_clean_latin_records")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.data.finetune_corpora import QARecord, filter_latin_records
from arabic_eval.tokenizers.utils.arabic_text import contains_latin_letters


def _qa(id_, question="ما هي عاصمة مصر؟", context="", answer="القاهرة",
        prompt_template="qa", choices=None):
    return QARecord(
        id=id_, question=question, context=context, answer=answer,
        source="smoke", prompt_template=prompt_template, choices=choices,
    )


def main() -> int:
    recs = [
        _qa("arabic_only"),
        _qa("latin_in_question", question="What is X?"),
        _qa("latin_in_context", context="Egypt is a country"),
        _qa("latin_in_answer", answer="Cairo"),
        _qa("mcq_clean", prompt_template="mcq_letter",
            choices=["القاهرة", "بغداد", "دمشق", "الرياض"]),
        _qa("mcq_latin_choice", prompt_template="mcq_letter",
            choices=["القاهرة", "Cairo", "دمشق", "الرياض"]),
        _qa("arabic_with_digits", question="ما حدث عام 2024؟"),
        _qa("diacritized", question="الْكِتَابُ جَمِيلٌ"),
        _qa("latin_acronym", question="ما هو IBM؟"),
    ]
    expected_kept = {"arabic_only", "mcq_clean", "arabic_with_digits", "diacritized"}
    expected_dropped = {
        "latin_in_question", "latin_in_context", "latin_in_answer",
        "mcq_latin_choice", "latin_acronym",
    }

    kept = filter_latin_records(recs)
    kept_ids = {r.id for r in kept}

    log.info("input=%d kept=%d dropped=%d", len(recs), len(kept), len(recs) - len(kept))
    log.info("kept ids: %s", sorted(kept_ids))

    if kept_ids != expected_kept:
        log.error("FAIL: kept set mismatch")
        log.error("  expected: %s", sorted(expected_kept))
        log.error("  got:      %s", sorted(kept_ids))
        return 1

    # No kept record contains Latin letters in any inspected field.
    for r in kept:
        fields = [r.question, r.context, r.answer]
        if r.choices:
            fields.extend(r.choices)
        for f in fields:
            if contains_latin_letters(f):
                log.error("FAIL: kept record %s has Latin in field %r", r.id, f)
                return 1

    # Sanity: dropped IDs are exactly the complement.
    dropped_ids = {r.id for r in recs} - kept_ids
    if dropped_ids != expected_dropped:
        log.error("FAIL: dropped set mismatch")
        log.error("  expected: %s", sorted(expected_dropped))
        log.error("  got:      %s", sorted(dropped_ids))
        return 1

    # Empty input edge case.
    if filter_latin_records([]) != []:
        log.error("FAIL: empty input should return empty list")
        return 1

    log.info("OK: filter_latin_records behaves as expected on synthetic records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
