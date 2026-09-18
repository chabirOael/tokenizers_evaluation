"""Free-form generation task (``tasks/freeform``): prompt equality with the
training template, the character budget → token cap, EOS stripping, marker
truncation, the degeneration detector, the typed status for non-generative
tokenizers, and an end-to-end ``evaluate`` on a tiny random Llama with a
word-level stub tokenizer (real HF greedy generation on CPU, row dump, metrics)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from arabic_eval.data.finetune_corpora import QARecord, _format_qa_prompt  # noqa: E402
from arabic_eval.data.freeform_heldout import HeldoutRow, write_heldout_jsonl  # noqa: E402
from arabic_eval.models.llama_adapter import LlamaAdapter  # noqa: E402
from arabic_eval.tasks.freeform import metrics as M  # noqa: E402
from arabic_eval.tasks.freeform.cidar import METRIC_NAMES, ROW_FIELDS, FreeformCidarTask  # noqa: E402
from arabic_eval.tasks.freeform.generation import (  # noqa: E402
    DecodingConfig, derive_token_cap, generate_freeform, generation_supported, measure_chars_per_token,
    strip_trailing_eos, truncate_at_marker,
)
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput  # noqa: E402

VOCAB = 96
PAD, BOS, EOS, UNK = 0, 1, 2, 3
TEXTS = ["الكتاب على الطاولة", "ذهب الولد إلى المدرسة", "يدرس الطلاب اللغة العربية", "كتبت الطالبة رسالة طويلة",
         "في المكتبة كتب كثيرة", "قرأت مقالة عن التاريخ", "السؤال: اكتب جملة الإجابة: نعم"]


class _WordTokenizer(BaseTokenizer):
    """Word-level tokenizer that appends BOS/EOS to every encoding — the
    from-scratch convention the generator has to cope with."""

    def __init__(self, embedding_type: str = EmbeddingType.STANDARD) -> None:
        self._et = embedding_type
        self._w2i: Dict[str, int] = {}
        self._i2w: Dict[int, str] = {}

    def train(self, texts, vocab_size, **kw):
        for t in texts:
            for w in t.split():
                if w not in self._w2i and len(self._w2i) < VOCAB - 4:
                    self._w2i[w] = len(self._w2i) + 4
                    self._i2w[self._w2i[w]] = w

    def encode(self, text, max_length=None, padding=False, truncation=False):
        ids = [BOS] + [self._w2i.get(w, UNK) for w in text.split()] + [EOS]
        if truncation and max_length:
            ids = ids[:max_length]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids), tokens=text.split())

    def decode(self, ids):
        return " ".join(self._i2w.get(i, "" if i in (PAD, BOS, EOS) else "<unk>") for i in ids).strip()

    def save(self, path): ...
    def load(self, path): ...

    @property
    def vocab_size(self): return VOCAB
    @property
    def embedding_type(self): return self._et
    @property
    def special_tokens(self): return {"pad_token": PAD, "bos_token": BOS, "eos_token": EOS, "unk_token": UNK}


@pytest.fixture(scope="module")
def tiny_llama(tmp_path_factory) -> Path:
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=True, bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD)
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_llama")
    LlamaForCausalLM(cfg).save_pretrained(path)
    return path


@pytest.fixture
def adapter(tiny_llama) -> LlamaAdapter:
    return LlamaAdapter(str(tiny_llama), device="cpu", dtype="float32")


@pytest.fixture
def tokenizer() -> _WordTokenizer:
    t = _WordTokenizer()
    t.train(TEXTS, VOCAB)
    return t


@pytest.fixture
def heldout(tmp_path) -> Path:
    rows = [HeldoutRow(id=f"cidar-{i}", prompt=f"اكتب جملة عن {TEXTS[i].split()[0]}", context="",
                       reference=TEXTS[i], stratum="short", ref_chars=len(TEXTS[i])) for i in range(6)]
    p = tmp_path / "heldout.jsonl"
    write_heldout_jsonl(rows, p)
    return p


# --------------------------------------------------------------------------

class TestPromptAndBudget:
    def test_prompt_is_the_training_prompt(self, heldout):
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None, "num_fewshot": 3})
        row = task.load_examples()[0]
        rec = QARecord(id=row["id"], question=row["prompt"], context="", answer=row["reference"],
                       source="cidar", prompt_template="instruction")
        p = task.build_prompt(row)
        assert p == _format_qa_prompt(rec) and p.endswith("\nالإجابة:") and p.startswith("السؤال: ")
        assert "السياق:" not in p

    def test_context_row_gets_the_context_line(self):
        p = FreeformCidarTask.build_prompt({"id": "x", "prompt": "لخص", "context": "نص", "reference": ""})
        assert p == "السياق: نص\nالسؤال: لخص\nالإجابة:"

    def test_token_cap_from_character_budget(self):
        cfg = DecodingConfig(max_output_chars=1200, token_cap_margin=1.15)
        assert derive_token_cap(cfg, 4.0) == 353            # ceil(1200/4*1.15)+8
        assert derive_token_cap(cfg, 1.0) == 1388           # char-level: ~4x the BPE cap
        assert derive_token_cap(cfg, 100.0) == cfg.token_cap_floor
        assert derive_token_cap(cfg, 0.01) == cfg.token_cap_ceiling

    def test_chars_per_token_measured_on_the_tokenizer(self, tokenizer):
        cpt = measure_chars_per_token(tokenizer, ["الكتاب على الطاولة"])   # 18 chars / (3 words + BOS + EOS)
        assert cpt == pytest.approx(18 / 5)

    def test_strip_trailing_eos_and_markers(self):
        assert strip_trailing_eos([1, 5, 6, 2], 2) == [1, 5, 6]
        assert strip_trailing_eos([1, 5, 6], 2) == [1, 5, 6]
        assert strip_trailing_eos([1, 5, 6, 2], None) == [1, 5, 6, 2]
        assert truncate_at_marker("جواب\nالسؤال: آخر", ("\nالسؤال:",)) == ("جواب", True)
        assert truncate_at_marker("جواب طويل", ("\nالسؤال:",)) == ("جواب طويل", False)
        assert truncate_at_marker("أ\n###ب\nالسؤال:ج", ("\nالسؤال:", "\n###")) == ("أ", True)

    def test_generation_support_by_embedding_family(self):
        for et in (EmbeddingType.STANDARD, EmbeddingType.CHAR_JABER):
            assert generation_supported(_WordTokenizer(et))[0]
        for et in (EmbeddingType.CHARACTER_CNN, EmbeddingType.CHARFORMER):
            ok, why = generation_supported(_WordTokenizer(et))
            assert not ok and why


class TestMetrics:
    def test_degeneration_detector(self):
        assert M.is_degenerate("الطبيب الطبيب الطبيب الطبيب الطبيب الطبيب الطبيب")
        assert M.is_degenerate("هذا نص عادي " * 6)                              # repeated 4-grams
        assert M.is_degenerate("ن" * 25)
        assert M.is_degenerate("الطبيب" * 7)                                        # space-less loop, 42 chars
        assert not M.is_degenerate("البومة طائر ليلي بعينين كبيرتين ووجه مستدير أما الصقر فطائر نهاري حاد البصر")
        assert not M.is_degenerate("")

    def test_arabic_ratio_and_chrf(self):
        assert M.arabic_letter_ratio("نص عربي") == 1.0
        assert M.arabic_letter_ratio("نص Latin") == pytest.approx(2 / 7)
        assert M.arabic_letter_ratio("123 ...") is None
        assert M.chrf_sentence("الحياة رحلة قصيرة", "الحياة رحلة قصيرة") == pytest.approx(100.0)
        assert M.chrf_sentence("", "مرجع") == 0.0
        assert 0 < M.chrf_sentence("الحياة رحلة", "الحياة رحلة قصيرة جدا") < 100

    def test_summary_shape(self):
        rows = [{"gen_chars": 10, "gen_tokens": 4, "ref_chars": 12, "chrf": 50.0, "generation": "أ ب", "reference": "أ ب ج",
                 "bertscore_f1": 0.8, "bertscore_p": 0.8, "bertscore_r": 0.8, "empty": False, "degenerate": False,
                 "latin": False, "arabic_letter_ratio": 1.0, "hit_cap": False, "stop_reason": "eos", "char_truncated": False,
                 "reference_roundtrip_chrf": 100.0}]
        s = M.summarize(rows, gen_wall_sec=2.0)
        assert s["num_samples"] == 1 and s["chrf"] == 50.0 and s["eos_rate"] == 1.0 and s["gen_chars_per_sec"] == 5.0


class TestEvaluate:
    def test_unsupported_family_returns_typed_status_without_touching_the_model(self, heldout):
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None})
        for et in (EmbeddingType.CHARACTER_CNN, EmbeddingType.CHARFORMER):
            out = task.evaluate(model=None, tokenizer=_WordTokenizer(et))
            assert out["status"] == "generation_unsupported" and out["num_samples"] == 0
            assert out["chrf"] is None and out["embedding_type"] == et and out["reason"]

    def test_prompt_fed_to_the_model_has_no_trailing_eos(self, adapter, tokenizer, heldout, monkeypatch):
        seen: List[torch.Tensor] = []

        def fake_generate(input_ids, **kw):
            seen.append(input_ids.clone())
            cont = torch.full((input_ids.shape[0], 3), tokenizer._w2i["الكتاب"], dtype=torch.long)
            return torch.cat([input_ids, cont, torch.full((input_ids.shape[0], 1), EOS)], dim=1)

        monkeypatch.setattr(adapter, "generate", fake_generate)
        adapter.adapt_to_tokenizer(tokenizer)
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None, "batch_size": 4})
        out = task.evaluate(adapter, tokenizer)
        assert seen and all((ids[:, -1] != EOS).all() for ids in seen)          # no prompt ends with EOS
        assert all((ids[:, -1] != PAD).all() for ids in seen)                   # left padding
        assert out["status"] == "ok" and out["num_samples"] == 6
        assert out["eos_rate"] == 1.0 and out["hit_cap_rate"] == 0.0 and out["mean_gen_tokens"] == 3.0

    def test_marker_truncation_and_stop_reason(self, adapter, tokenizer, heldout, monkeypatch):
        w = tokenizer._w2i

        def fake_generate(input_ids, **kw):
            # "نعم\nالسؤال: ..." cannot be produced by a word tokenizer (no newline) — emit the marker words
            cont = torch.tensor([[w["نعم"], w["السؤال:"], w["اكتب"]]] * input_ids.shape[0])
            return torch.cat([input_ids, cont], dim=1)

        monkeypatch.setattr(adapter, "generate", fake_generate)
        adapter.adapt_to_tokenizer(tokenizer)
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None,
                                  "stop_markers": [" السؤال:"], "batch_size": 8})
        out = task.evaluate(adapter, tokenizer)
        assert out["marker_stop_rate"] == 1.0 and out["mean_gen_chars"] == len("نعم")

    def test_end_to_end_real_greedy_generation_with_row_dump(self, adapter, tokenizer, heldout, tmp_path):
        adapter.adapt_to_tokenizer(tokenizer)
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None, "batch_size": 4,
                                  "max_output_chars": 60, "token_cap_floor": 6, "token_cap_ceiling": 12})
        out = task.evaluate(adapter, tokenizer, max_samples=5, row_dump_dir=tmp_path / "eval_rows")
        assert out["status"] == "ok" and out["num_samples"] == 5
        assert set(METRIC_NAMES) <= set(out) and out["bertscore_f1"] is None
        assert out["token_cap"] == 12 and out["generation_wall_sec"] > 0
        assert 0.0 <= out["degenerate_rate"] <= 1.0 and out["reference_roundtrip_chrf"] == pytest.approx(100.0)
        import pyarrow.parquet as pq
        t = pq.read_table(tmp_path / "eval_rows" / "freeform_cidar.parquet")
        assert t.num_rows == 5 and list(t.column_names) == ROW_FIELDS
        meta = json.loads(t.schema.metadata[b"arabic_eval"].decode("utf-8"))
        assert meta["kind"] == "freeform_generations" and meta["decoding"]["sampling"] == "greedy"
        assert meta["token_cap"] == 12 and meta["embedding_type"] == "standard" and meta["bertscore"] is None
        rows = t.to_pylist()
        assert all(r["prompt_text"].endswith("الإجابة:") for r in rows)
        assert all(r["gen_tokens"] <= 12 and len(r["generation"]) <= 60 for r in rows)
        assert all(r["stop_reason"] in ("eos", "marker", "cap") for r in rows)

    def test_determinism(self, adapter, tokenizer, heldout):
        adapter.adapt_to_tokenizer(tokenizer)
        task = FreeformCidarTask({"heldout_path": str(heldout), "bertscore_model": None, "token_cap_floor": 6, "token_cap_ceiling": 8})
        a = task.evaluate(adapter, tokenizer, max_samples=4)
        b = task.evaluate(adapter, tokenizer, max_samples=4)
        for k in ("chrf", "mean_gen_tokens", "degenerate_rate", "eos_rate"):
            assert a[k] == b[k]
