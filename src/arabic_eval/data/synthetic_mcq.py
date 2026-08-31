"""Synthetic MCQ corpus generated from Arabic-SQuAD.

Why this exists. Phase 3 SFT trained on extractive QA only (TyDiQA-Arabic +
ARCD), so the model never saw the *MCQ format* at train time. The eval
benchmarks (ACVA, AlGhafa, ArabicMMLU) are MCQ. Without MCQ training data
the model has to learn the eval format from few-shot demos alone, which the
2026-05-06 diagnostic showed is insufficient — the model's class-collapse
on letter / word continuations is the dominant failure mode.

This module synthesizes a 4-way MCQ corpus from the existing Arabic-SQuAD
extractive-QA data, **disjoint from every eval benchmark by construction**
(Arabic-SQuAD is the Phase 1 + 2 corpus, not used for evaluation).

Construction recipe (per Arabic-SQuAD row):
  1. Take the gold extractive answer span.
  2. Sample ``num_choices - 1`` distractor spans from the **same context**,
     of the same word-count as the gold answer, that don't equal the gold.
  3. Randomly assign the gold a position (A/B/C/D) so the model can't
     shortcut on letter prior.
  4. Emit a QARecord with ``prompt_template='mcq_letter'`` and ``answer``
     set to the gold's Arabic letter.

The synthetic distractors are not factually grounded NPs; they're simply
same-length spans from the passage. This is sufficient for **format-teaching**
— the goal is to make the model learn that ``الإجابة:`` is followed by a
single Arabic letter (``أ`` / ``ب`` / ``ج`` / ``د``). Evaluation correctness
of synthetic distractors does not matter for that loss target.

Determinism: seeded by ``seed`` (default 42); regenerating with the same
seed produces byte-identical output. Rows that can't supply enough
distractors (very short contexts) are skipped.
"""
from __future__ import annotations

import logging
import random
from typing import List, Optional

from .finetune_corpora import QARecord, _load_arabic_squad

logger = logging.getLogger(__name__)


# Imported lazily to avoid pulling all of `tasks` at module-import time.
def _arabic_choice_letters() -> List[str]:
    from arabic_eval.tasks.lighteval.utils import ARABIC_CHOICE_LETTERS
    return ARABIC_CHOICE_LETTERS


def _sample_distractors(
    context: str,
    gold_answer: str,
    n: int,
    rng: random.Random,
    max_attempts: int = 60,
) -> List[str]:
    """Sample ``n`` distinct word-aligned spans from ``context`` of the same
    word-count as ``gold_answer``, none equal to ``gold_answer``.

    Returns an empty list if fewer than ``n`` distinct candidates exist
    after ``max_attempts`` rejection samples — the caller should skip the row.
    """
    ctx_words = context.split()
    gold_words = gold_answer.split()
    span_len = max(1, len(gold_words))
    # Need at least span_len + n words to even attempt n distinct distractors.
    if len(ctx_words) < span_len + n:
        return []
    n_starts = len(ctx_words) - span_len + 1
    if n_starts < 1:
        return []

    found: List[str] = []
    seen = {gold_answer}
    attempts = 0
    while len(found) < n and attempts < max_attempts:
        i = rng.randrange(0, n_starts)
        span = " ".join(ctx_words[i : i + span_len]).strip()
        if span and span not in seen:
            seen.add(span)
            found.append(span)
        attempts += 1
    if len(found) < n:
        return []
    return found


def build_synthetic_mcq_corpus(
    seed: int = 42,
    num_choices: int = 4,
    max_records: Optional[int] = None,
) -> List[QARecord]:
    """Build the ``arabic_squad_mcq`` synthetic MCQ corpus.

    Loads Arabic-SQuAD train and converts each (question, context, answer)
    triple into a 4-way MCQ record. Rows that can't supply ``num_choices - 1``
    distinct distractors are skipped (rare; happens only on very short
    contexts). The output is returned in original Arabic-SQuAD order so
    downstream sampling stays deterministic.

    ``max_records`` (optional) caps the corpus size — useful for smoke tests.
    """
    extractive = _load_arabic_squad("train")
    rng = random.Random(seed)
    letters = _arabic_choice_letters()
    if num_choices < 2 or num_choices > len(letters):
        raise ValueError(
            f"num_choices={num_choices} must be in [2, {len(letters)}]"
        )

    out: List[QARecord] = []
    n_skipped = 0
    for r in extractive:
        if max_records is not None and len(out) >= max_records:
            break
        distractors = _sample_distractors(
            r.context, r.answer, n=num_choices - 1, rng=rng
        )
        if len(distractors) < num_choices - 1:
            n_skipped += 1
            continue
        gold_pos = rng.randrange(0, num_choices)
        choices = list(distractors)
        choices.insert(gold_pos, r.answer)
        gold_letter = letters[gold_pos]
        out.append(
            QARecord(
                id=f"sq_mcq_{r.id}",
                question=r.question,
                context=r.context,
                answer=gold_letter,
                source="arabic_squad_mcq",
                prompt_template="mcq_letter",
                choices=choices,
            )
        )

    logger.info(
        "arabic_squad_mcq: built %d synthetic MCQ records "
        "(skipped %d short-context rows; %d total Arabic-SQuAD rows)",
        len(out), n_skipped, len(extractive),
    )
    return out


def load_arabic_squad_mcq(split: str) -> List[QARecord]:
    """Loader entry-point matching the ``_LOADERS`` signature in
    ``finetune_corpora.py``. Currently only ``split == "train"`` is supported
    (Arabic-SQuAD has no validation split)."""
    if split != "train":
        raise ValueError(
            f"arabic_squad_mcq has no '{split}' split (only 'train' is available)"
        )
    return build_synthetic_mcq_corpus()
