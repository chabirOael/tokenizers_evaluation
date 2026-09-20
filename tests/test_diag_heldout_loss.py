"""``scripts/diag_heldout_loss.py``: block building, the answer-only encodings
under the current and the v1 template, the loss / EOS statistics on a tiny
random Llama (CPU), the standard-embedding guard and the pool lookup."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from arabic_eval.config import TokenizerConfig  # noqa: E402
from arabic_eval.data.finetune_corpora import QARecord, _format_qa_prompt  # noqa: E402
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput  # noqa: E402

spec = importlib.util.spec_from_file_location("diag_heldout_loss", REPO / "scripts" / "diag_heldout_loss.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)  # type: ignore[union-attr]

VOCAB = 64
PAD, BOS, EOS, UNK = 0, 1, 2, 3


class _WordTok(BaseTokenizer):
    def __init__(self, add_specials: bool = True, embedding_type: str = EmbeddingType.STANDARD):
        self._add = add_specials
        self._et = embedding_type

    def train(self, texts, vocab_size, **kw): ...

    def encode(self, text, max_length=None, padding=False, truncation=False):
        words = text.split()
        ids = [4 + (hash(w) % (VOCAB - 4)) for w in words]
        if self._add:
            ids = [BOS] + ids + [EOS]
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids), tokens=words)

    def decode(self, ids): return ""
    def save(self, path): ...
    def load(self, path): ...
    @property
    def vocab_size(self): return VOCAB
    @property
    def embedding_type(self): return self._et
    @property
    def special_tokens(self): return {"pad_token": PAD, "bos_token": BOS, "eos_token": EOS, "unk_token": UNK}


def _recs(n: int = 5):
    return [QARecord(id=f"c{i}", question=f"اكتب جملة {i}", context="", answer=" ".join(f"كلمة{i}" for _ in range(3 + i)),
                     source="cidar", prompt_template="instruction") for i in range(n)]


def test_strip_specials_and_blocks():
    assert D.strip_specials([1, 5, 6, 2], 1, 2) == [5, 6]
    assert D.strip_specials([5, 6], 1, 2) == [5, 6]
    assert D.strip_specials([5, 6, 2], None, 2) == [5, 6]
    blocks = D.blocks_from([[5, 6, 7], [8, 9]], eos=2, block=4)
    assert blocks == [[5, 6, 7, 2]]                       # [8, 9, 2] tail dropped
    assert D.blocks_from([[5] * 10], eos=2, block=4) == [[5, 5, 5, 5], [5, 5, 5, 5]]
    assert D.blocks_from([], eos=2, block=4) == []


def test_answer_encodings_current_and_v1_template():
    recs = _recs(3)
    encs, dropped = D.answer_encodings(recs, _WordTok(), max_length=256)
    assert dropped == 0 and len(encs) == 3
    for rec, (ids, labels) in zip(recs, encs):
        n_prompt = 1 + len(_format_qa_prompt(rec).split())          # BOS + prompt words
        n_answer = len(rec.answer.split())
        assert len(ids) == n_prompt + n_answer + 1 and ids[-1] == EOS
        assert labels[:n_prompt] == [-100] * n_prompt and labels[n_prompt:] == ids[n_prompt:]
        assert labels[-1] == EOS                                    # the EOS is a loss target
    # v1: flat template, far fewer prompt tokens, EOS still at the end
    encs_v1, _ = D.answer_encodings(recs, _WordTok(), max_length=256, v1=True)
    for rec, (ids, labels) in zip(recs, encs_v1):
        n_prompt = 1 + len(D.v1_format_prompt(rec).split())
        assert D.v1_format_prompt(rec) == f"السؤال: {rec.question}\nالإجابة:"
        assert labels[:n_prompt] == [-100] * n_prompt and ids[-1] == EOS and labels[-1] == EOS
        assert len(ids) < len(encs[0][0]) + 20
    # a tokenizer that emits no EOS gets one appended (the training rule)
    encs_noeos, _ = D.answer_encodings(recs, _WordTok(add_specials=False), max_length=256)
    assert all(ids[-1] == EOS and labels[-1] == EOS for ids, labels in encs_noeos)
    # truncation that eats the answer drops the reference
    encs_short, dropped = D.answer_encodings(recs, _WordTok(), max_length=8)
    assert dropped == 3 and encs_short == []


@pytest.fixture(scope="module")
def tiny_model():
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True, bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def test_lm_loss_and_answer_stats_on_tiny_model(tiny_model):
    tok = _WordTok()
    docs = [" ".join(f"كلمة{i % 7}" for i in range(40)) for _ in range(4)]
    doc_ids = [D.strip_specials(tok.encode(d).input_ids, BOS, EOS) for d in docs]
    blocks = D.blocks_from(doc_ids, EOS, block=32)
    mix = D.lm_loss(tiny_model, blocks, device="cpu", batch_size=2)
    assert mix["blocks"] == len(blocks) == 5 and mix["tokens_scored"] == 5 * 31
    assert math.isfinite(mix["loss_per_token"]) and 0 < mix["loss_per_token"] < 2 * math.log(VOCAB)
    encs, _ = D.answer_encodings(_recs(5), tok, max_length=256)
    ans = D.answer_stats(tiny_model, encs, PAD, device="cpu", batch_size=2)
    assert ans["n"] == 5 and ans["answer_tokens"] == sum(sum(1 for l in lab if l != -100) for _, lab in encs)
    assert math.isfinite(ans["nll_per_answer_token"]) and ans["nll_per_answer_token"] > 0
    assert 0.0 <= ans["p_eos_at_end_mean"] <= 1.0 and 0.0 <= ans["p_eos_at_end_median"] <= 1.0
    assert 0.0 <= ans["eos_rank1_share"] <= 1.0
    # batching must not change the numbers (padding is masked)
    ans1 = D.answer_stats(tiny_model, encs, PAD, device="cpu", batch_size=1)
    assert ans1["nll_per_answer_token"] == pytest.approx(ans["nll_per_answer_token"], abs=2e-3)
    assert ans1["p_eos_at_end_mean"] == pytest.approx(ans["p_eos_at_end_mean"], abs=2e-3)


def test_eos_summary():
    s = D.summarize_eos([0.2, 0.6, 0.9], [3, 1, 1])
    assert s == {"p_eos_at_end_mean": pytest.approx(0.5667, abs=1e-4), "p_eos_at_end_median": 0.6,
                 "eos_rank1_share": pytest.approx(0.6667, abs=1e-4), "n": 3}
    assert D.summarize_eos([], []) == {"p_eos_at_end_mean": None, "p_eos_at_end_median": None,
                                       "eos_rank1_share": None, "n": 0}


def test_build_tokenizer_refuses_non_standard_embeddings(monkeypatch):
    class _CharCNNTok(_WordTok):
        def __init__(self, **kw):
            super().__init__(embedding_type=EmbeddingType.CHARACTER_CNN)

    monkeypatch.setattr(D.tokenizer_registry, "get", lambda name: _CharCNNTok)
    with pytest.raises(SystemExit, match="character_cnn"):
        D.build_tokenizer(TokenizerConfig(type="character_bert", vocab_size=None, params={}, save_path=""))
    monkeypatch.setattr(D.tokenizer_registry, "get", lambda name: _WordTok)
    tok = D.build_tokenizer(TokenizerConfig(type="bpe", vocab_size=None, params={}, save_path=""))
    assert tok.embedding_type == EmbeddingType.STANDARD
    with pytest.raises(SystemExit, match="is gone"):
        D.build_tokenizer(TokenizerConfig(type="bpe", vocab_size=None, params={}, save_path="no/such/dir"))


def test_pool_dir_and_checkpoint_lookup(tmp_path):
    cell = tmp_path / "cell"
    (cell / "data" / "pretraining_mix").mkdir(parents=True)
    assert D.pool_dir_for(cell, None) == REPO / D.DEFAULT_POOL                 # no manifest → default
    (cell / "data" / "pretraining_mix" / "packed_manifest.json").write_text(
        json.dumps({"pool_dir": "outputs/data_cache/pretraining_mix/abc"}), encoding="utf-8")
    assert D.pool_dir_for(cell, None) == REPO / "outputs/data_cache/pretraining_mix/abc"
    assert D.pool_dir_for(cell, "some/pool") == REPO / "some/pool"
    from arabic_eval.config import ModelConfig
    mc = ModelConfig(type="qwen3", name_or_path="Qwen/Qwen3-4B-Base")
    assert D.checkpoint_for(cell, True, None, mc) == "Qwen/Qwen3-4B-Base"
    assert D.checkpoint_for(cell, False, "x/y", mc) == "x/y"
    with pytest.raises(SystemExit, match="no SFT checkpoint"):
        D.checkpoint_for(cell, False, None, mc)
    (cell / "training" / "sft").mkdir(parents=True)
    (cell / "training" / "sft" / "model.safetensors").write_bytes(b"")
    assert D.checkpoint_for(cell, False, None, mc) == str(cell / "training" / "sft")
