"""Tests for the synthetic MCQ corpus generator.

Pin down:
  * Distractor sampling: same word-count as gold, drawn from the same
    context, no duplicates with gold.
  * Gold-letter assignment: every Arabic letter (أ/ب/ج/د) is reachable;
    no letter-position shortcut.
  * Output format matches the LightEval-official letter MCQ prompt
    (verifies the model sees the same prefix tokens as eval).
  * Determinism under seed.
  * QARecord dispatches to the MCQ formatter when ``prompt_template ==
    "mcq_letter"``.
"""
from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from arabic_eval.data.finetune_corpora import (
    QARecord,
    _format_qa_full,
    _format_qa_prompt,
)
from arabic_eval.data.synthetic_mcq import (
    _sample_distractors,
    build_synthetic_mcq_corpus,
)


# ---------------------------------------------------------------------------
# Distractor sampling
# ---------------------------------------------------------------------------

def test_distractors_are_word_aligned_and_same_word_count():
    rng = random.Random(0)
    ctx = "كلمة واحدة اثنتان ثلاث أربع خمس ست سبع"
    gold = "ثلاث"  # 1 word
    out = _sample_distractors(ctx, gold, n=3, rng=rng)
    assert len(out) == 3
    for span in out:
        assert len(span.split()) == 1
        assert span != gold
        # All chars must come from the context vocabulary.
        assert span in ctx.split()


def test_distractors_match_multi_word_gold_length():
    rng = random.Random(1)
    ctx = " ".join([f"w{i}" for i in range(20)])
    gold = "w5 w6 w7"  # 3 words
    out = _sample_distractors(ctx, gold, n=3, rng=rng)
    assert len(out) == 3
    for span in out:
        assert len(span.split()) == 3
        assert span != gold


def test_distractors_are_distinct():
    rng = random.Random(2)
    ctx = "a b c d e f g h i j"
    gold = "x"  # not in context — sampling will produce only context tokens
    out = _sample_distractors(ctx, gold, n=3, rng=rng)
    assert len(out) == 3
    assert len(set(out)) == 3   # distinct


def test_distractors_empty_when_context_too_short():
    rng = random.Random(3)
    out = _sample_distractors("a b", "x", n=3, rng=rng)
    assert out == []


# ---------------------------------------------------------------------------
# build_synthetic_mcq_corpus — needs Arabic-SQuAD; we patch the loader
# ---------------------------------------------------------------------------

def _fake_arabic_squad_records(n: int = 50):
    """Synthetic Arabic-SQuAD-shaped records — enough variety in context
    word count to satisfy the distractor sampler."""
    base_words = ["البحر", "الأرض", "الجبل", "النهر", "الشمس", "القمر",
                  "السماء", "النجوم", "الغابة", "الصحراء", "الحديقة",
                  "المدينة", "القرية", "الشارع", "البيت", "الباب", "النافذة"]
    out = []
    for i in range(n):
        # 30-word context built from base_words
        ctx_words = [base_words[(i + j) % len(base_words)] for j in range(30)]
        gold_pos = i % 25
        gold = ctx_words[gold_pos]
        out.append(QARecord(
            id=str(i),
            question=f"سؤال {i}؟",
            context=" ".join(ctx_words),
            answer=gold,
            source="arabic_squad",
        ))
    return out


@patch("arabic_eval.data.synthetic_mcq._load_arabic_squad")
def test_corpus_marks_records_as_mcq_letter_template(mock_load):
    mock_load.return_value = _fake_arabic_squad_records(20)
    out = build_synthetic_mcq_corpus(seed=42, num_choices=4)
    assert len(out) > 0
    for rec in out:
        assert rec.prompt_template == "mcq_letter"
        assert rec.source == "arabic_squad_mcq"
        assert rec.choices is not None and len(rec.choices) == 4
        # Answer is one of the four Arabic letters.
        assert rec.answer in {"أ", "ب", "ج", "د"}
        # The gold answer (one of the choices) sits at the position
        # indicated by rec.answer.
        from arabic_eval.tasks.lighteval.utils import ARABIC_CHOICE_LETTERS
        gold_idx = ARABIC_CHOICE_LETTERS.index(rec.answer)
        assert gold_idx < len(rec.choices)


@patch("arabic_eval.data.synthetic_mcq._load_arabic_squad")
def test_corpus_gold_letters_distribution_is_not_collapsed(mock_load):
    """Random-position assignment must spread the gold across multiple letters
    — otherwise the model could shortcut on letter prior."""
    mock_load.return_value = _fake_arabic_squad_records(200)
    out = build_synthetic_mcq_corpus(seed=42, num_choices=4)
    letters_used = {r.answer for r in out}
    # All 4 letters should appear at least once across 200 records.
    assert letters_used == {"أ", "ب", "ج", "د"}


@patch("arabic_eval.data.synthetic_mcq._load_arabic_squad")
def test_corpus_is_deterministic_under_seed(mock_load):
    mock_load.return_value = _fake_arabic_squad_records(50)
    a = build_synthetic_mcq_corpus(seed=42)
    mock_load.return_value = _fake_arabic_squad_records(50)
    b = build_synthetic_mcq_corpus(seed=42)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.choices == y.choices
        assert x.answer == y.answer


@patch("arabic_eval.data.synthetic_mcq._load_arabic_squad")
def test_max_records_caps_corpus_size(mock_load):
    mock_load.return_value = _fake_arabic_squad_records(50)
    out = build_synthetic_mcq_corpus(seed=42, max_records=10)
    assert len(out) == 10


# ---------------------------------------------------------------------------
# Prompt format dispatch
# ---------------------------------------------------------------------------

def test_qa_template_uses_extractive_format():
    """Default prompt_template='qa' produces the existing
    `السياق:`/`السؤال:`/`الإجابة:` extractive prompt."""
    rec = QARecord(
        id="x", question="ما هي العاصمة؟", context="عاصمة المغرب الرباط.",
        answer="الرباط", source="arabic_squad",
    )
    assert rec.prompt_template == "qa"
    p = _format_qa_prompt(rec)
    assert "السياق:" in p
    assert "السؤال:" in p
    assert p.endswith("الإجابة:")
    assert "أ." not in p


def test_mcq_letter_template_produces_lighteval_format():
    rec = QARecord(
        id="x", question="ما هي العاصمة؟", context="",
        answer="ب", source="arabic_squad_mcq",
        prompt_template="mcq_letter",
        choices=["مدينة1", "مدينة2", "مدينة3", "مدينة4"],
    )
    p = _format_qa_prompt(rec)
    # LightEval-official: instruction + question + letter listing.
    assert "الأسئلة التالية" in p
    for letter, choice in zip("أبجد", rec.choices):
        assert f"{letter}. {choice}" in p
    assert "###" not in p
    assert p.endswith("الإجابة:")


def test_mcq_letter_template_full_includes_letter_answer():
    rec = QARecord(
        id="x", question="س", context="",
        answer="ج", source="arabic_squad_mcq",
        prompt_template="mcq_letter",
        choices=["a", "b", "c", "d"],
    )
    full = _format_qa_full(rec)
    assert full.endswith("الإجابة: ج")


def test_mcq_letter_template_with_context_prepends_context_line():
    rec = QARecord(
        id="x", question="س", context="نص السياق هنا.",
        answer="أ", source="arabic_squad_mcq",
        prompt_template="mcq_letter",
        choices=["a", "b", "c", "d"],
    )
    p = _format_qa_prompt(rec)
    assert p.startswith("السياق: نص السياق هنا.\n")


def test_mcq_letter_template_requires_choices():
    rec = QARecord(
        id="x", question="س", context="",
        answer="أ", source="arabic_squad_mcq",
        prompt_template="mcq_letter",
        # choices=None — error
    )
    with pytest.raises(ValueError, match="prompt_template='mcq_letter' requires record.choices"):
        _format_qa_prompt(rec)
