"""Tests for ``arabic_eval.data.finetune_corpora`` and the LCP masking helper.

The corpus loaders themselves are network-bound (HF Hub) and are exercised
via ``pytest.mark.integration``-style live tests in
``scripts/smoke_finetune_corpora.py``. The unit tests here focus on:

  - LCP masking math (no network)
  - Prompt format strings (no network)
  - Tokenize-and-build-dataloader integration with a stub tokenizer
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.data.answer_only_masking import compute_answer_only_labels
from arabic_eval.data.finetune_corpora import (
    QARecord,
    _format_qa_full,
    _format_qa_prompt,
    build_qa_dataloader,
    filter_latin_records,
    tokenize_records,
)
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput


# --------------------------------------------------------------------------
# LCP masking math
# --------------------------------------------------------------------------

def test_lcp_masks_prompt_keeps_answer():
    prompt = [1, 2, 3, 4, 5]
    full = [1, 2, 3, 4, 5, 100, 101, 102]
    labels = compute_answer_only_labels(prompt, full)
    assert labels == [-100, -100, -100, -100, -100, 100, 101, 102]


def test_lcp_handles_eos_appended_to_prompt():
    """Prompt-only encoding has EOS at index P-1; full encoding has answer there."""
    prompt = [1, 2, 3, 999]      # 999 = EOS
    full = [1, 2, 3, 100, 101]   # 100 = first answer token (where prompt had EOS)
    labels = compute_answer_only_labels(prompt, full)
    # LCP is 3 (positions 0..2 match); position 3 onwards is the answer.
    assert labels == [-100, -100, -100, 100, 101]


def test_lcp_returns_none_when_truncation_eats_answer():
    """If LCP >= len(full), the full text was truncated to the prompt — no answer left."""
    prompt = [1, 2, 3, 4, 5]
    full = [1, 2, 3]  # truncated below the prompt length
    assert compute_answer_only_labels(prompt, full) is None


def test_lcp_full_equals_prompt_returns_none():
    prompt = [1, 2, 3]
    full = [1, 2, 3]
    assert compute_answer_only_labels(prompt, full) is None


# --------------------------------------------------------------------------
# Prompt format
# --------------------------------------------------------------------------

def _rec(answer: str = "بيير كوري") -> QARecord:
    return QARecord(
        id="test-1",
        question="من هو مكتشف المرو؟",
        context="المرو معدن. اكتشفه بيير كوري.",
        answer=answer,
        source="arabic_squad",
    )


def test_prompt_template_format_with_marker():
    """Post-2026-05-06: prompts use bare ``السياق:`` / ``السؤال:`` / ``الإجابة:``
    (no ``###`` markers) so train and eval share the same prefix tokens."""
    rec = QARecord(
        id="x", question="ما هو ASCII؟", context="نظام ترميز.",
        answer="معيار", source="arabic_squad",  # 'معيار' not in context
    )
    p = _format_qa_prompt(rec)
    # Bare LightEval-aligned labels.
    assert "السياق:" in p
    assert "السؤال:" in p
    assert "الإجابة:" in p
    # Legacy ``###`` markers must NOT appear.
    assert "###" not in p
    # Answer label must be the last line (no trailing newline, no answer text)
    assert p.endswith("الإجابة:")
    # No answer leaks into the prompt-only form
    assert rec.answer not in p


def test_full_includes_answer_with_single_space():
    f = _format_qa_full(_rec("بيير كوري"))
    assert f.endswith("الإجابة: بيير كوري")
    # Full text starts with the same prefix as the prompt
    p = _format_qa_prompt(_rec("بيير كوري"))
    assert f.startswith(p)


# --------------------------------------------------------------------------
# tokenize_records / build_qa_dataloader integration with a stub tokenizer
# --------------------------------------------------------------------------

class _StubTokenizer(BaseTokenizer):
    """A toy whitespace tokenizer with a deterministic vocab.

    Mimics a real tokenizer enough that ``tokenize_records`` can run:
    appends EOS=999 to standalone encodings (matches the real auto-EOS
    behavior), supports ``max_length`` and ``truncation``.
    """
    EOS = 999
    PAD = 0

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {"<pad>": 0}

    def _ids(self, text: str) -> list[int]:
        ids = []
        for tok in text.split():
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)
            ids.append(self._vocab[tok])
        ids.append(self.EOS)  # auto-EOS
        return ids

    def train(self, texts, vocab_size, **kwargs):  # pragma: no cover
        pass

    def encode(self, text, max_length=None, padding=False, truncation=False):
        ids = self._ids(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids), tokens=text.split())

    def decode(self, ids):  # pragma: no cover
        return " ".join(t for t, i in self._vocab.items() if i in ids)

    def save(self, path):  # pragma: no cover
        pass

    def load(self, path):  # pragma: no cover
        pass

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def embedding_type(self) -> str:
        return EmbeddingType.STANDARD

    @property
    def special_tokens(self) -> dict:
        return {"<pad>": 0, "</s>": 999}

    @property
    def pad_token_id(self) -> int:
        return self.PAD


def test_tokenize_records_answer_only_masks_prompt_span():
    tok = _StubTokenizer()
    rec = _rec("بيير كوري")
    encs = tokenize_records([rec], tok, max_length=128, loss_target="answer_only")
    assert len(encs) == 1
    e = encs[0]
    assert "input_ids" in e and "labels" in e
    assert len(e["labels"]) == len(e["input_ids"])
    # All masked positions are -100; trailing positions are real answer tokens.
    assert e["labels"][0] == -100
    real_label_positions = [i for i, l in enumerate(e["labels"]) if l != -100]
    assert real_label_positions, "at least one answer token must be unmasked"
    # The real labels equal the corresponding input_ids (no shift here; collator handles).
    for i in real_label_positions:
        assert e["labels"][i] == e["input_ids"][i]


def test_tokenize_records_full_sequence_no_labels():
    tok = _StubTokenizer()
    rec = _rec("بيير كوري")
    encs = tokenize_records([rec], tok, max_length=128, loss_target="full_sequence")
    assert len(encs) == 1
    e = encs[0]
    assert "input_ids" in e
    assert "labels" not in e  # collator default fallback handles it


def test_tokenize_records_drops_truncated_examples():
    """If max_length is shorter than the prompt, the answer is gone — drop the row."""
    tok = _StubTokenizer()
    rec = _rec("بيير كوري")
    # Full text has many tokens; max_length=2 keeps only the first two.
    encs = tokenize_records([rec], tok, max_length=2, loss_target="answer_only")
    assert encs == []


def test_tokenize_records_invalid_loss_target():
    tok = _StubTokenizer()
    with pytest.raises(ValueError, match="unknown loss_target"):
        tokenize_records([_rec()], tok, max_length=64, loss_target="frobnicate")


def test_build_qa_dataloader_yields_padded_batch():
    tok = _StubTokenizer()
    recs = [
        QARecord(id=f"r{i}", question=f"س{i}", context=f"ج{i}",
                 answer=f"إ{i} كلمة كلمة", source="arabic_squad")
        for i in range(4)
    ]
    loader = build_qa_dataloader(
        recs, tok, batch_size=2, max_length=64,
        loss_target="answer_only", shuffle=False,
    )
    batches = list(loader)
    assert len(batches) == 2
    for batch in batches:
        # Standard collator outputs
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        # Same shape across the three
        assert batch["input_ids"].shape == batch["attention_mask"].shape
        assert batch["input_ids"].shape == batch["labels"].shape
        # Some labels are -100 (prompt mask + pad mask), some are real
        assert (batch["labels"] == -100).any()
        assert (batch["labels"] != -100).any()


# --------------------------------------------------------------------------
# filter_latin_records — per-phase clean_latin_rows predicate
# --------------------------------------------------------------------------

def _qa(id_: str, question: str = "ما هي عاصمة مصر؟", context: str = "",
        answer: str = "القاهرة", source: str = "test",
        prompt_template: str = "qa", choices=None) -> QARecord:
    return QARecord(
        id=id_, question=question, context=context, answer=answer,
        source=source, prompt_template=prompt_template, choices=choices,
    )


class TestFilterLatinRecords:
    def test_pure_arabic_passes_through(self):
        recs = [_qa("a"), _qa("b"), _qa("c")]
        out = filter_latin_records(recs)
        assert len(out) == 3
        assert [r.id for r in out] == ["a", "b", "c"]

    def test_latin_in_question_dropped(self):
        recs = [_qa("a"), _qa("b", question="What is X?")]
        out = filter_latin_records(recs)
        assert [r.id for r in out] == ["a"]

    def test_latin_in_context_dropped(self):
        recs = [_qa("a", context="مصر بلد عربي"),
                _qa("b", context="Egypt is a country")]
        out = filter_latin_records(recs)
        assert [r.id for r in out] == ["a"]

    def test_latin_in_answer_dropped(self):
        recs = [_qa("a"), _qa("b", answer="Cairo")]
        out = filter_latin_records(recs)
        assert [r.id for r in out] == ["a"]

    def test_latin_in_mcq_choice_dropped(self):
        recs = [
            _qa("a", prompt_template="mcq_letter",
                choices=["القاهرة", "بغداد", "دمشق", "الرياض"]),
            _qa("b", prompt_template="mcq_letter",
                choices=["القاهرة", "Cairo", "دمشق", "الرياض"]),
        ]
        out = filter_latin_records(recs)
        assert [r.id for r in out] == ["a"]

    def test_arabic_with_digits_passes(self):
        # ASCII digits + Arabic-Indic digits are not Latin letters.
        recs = [_qa("a", question="ما حدث عام 2024؟"),
                _qa("b", question="ما حدث عام ١٤٤٥؟")]
        out = filter_latin_records(recs)
        assert [r.id for r in out] == ["a", "b"]

    def test_empty_input_returns_empty(self):
        assert filter_latin_records([]) == []

    def test_mixed_batch_preserves_order_and_originals(self):
        recs = [
            _qa("a"),
            _qa("b", question="What?"),
            _qa("c"),
            _qa("d", answer="Hello"),
            _qa("e"),
        ]
        out = filter_latin_records(recs)
        assert [r.id for r in out] == ["a", "c", "e"]
        # Originals untouched
        assert recs[1].question == "What?"
        assert len(recs) == 5

    def test_diacritized_arabic_passes(self):
        # Diacritics are not Latin letters.
        recs = [_qa("a", question="الْكِتَابُ جَمِيلٌ")]
        assert filter_latin_records(recs) == recs

    def test_arabic_with_latin_acronym_dropped(self):
        recs = [_qa("a", question="ما هو IBM؟")]
        assert filter_latin_records(recs) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
