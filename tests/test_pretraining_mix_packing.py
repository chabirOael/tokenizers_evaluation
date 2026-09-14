"""Stage B of the pretraining mix: token-budget fill, packing, cursor,
cache, and the pipeline wiring (offline, tiny random model)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.config import (  # noqa: E402
    EarlyStoppingConfig, PhaseConfig, PhasesConfig, TrainingConfig,
)
from arabic_eval.data.pretraining_mix.packing import (  # noqa: E402
    EncodedDoc, PackedBlockDataset, PackedCorpus, build_packed_corpus, fill_token_budget,
    pack_blocks, resolve_token_budget, source_doc_order,
)
from arabic_eval.data.pretraining_mix.pool import build_pool, load_pool_manifest  # noqa: E402
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput  # noqa: E402
from tests.test_pretraining_mix_pool import ListSource, SourceDoc, mix_cfg, msa_doc  # noqa: E402


# --------------------------------------------------------------------------
# Toy tokenizers with very different fertility
# --------------------------------------------------------------------------

class _WordTok(BaseTokenizer):
    """1 token per word; adds nothing. EOS must be appended by the packer."""
    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, vocab_size: int = 64):
        self._v = vocab_size

    def train(self, texts, vocab_size, **kw): ...
    def encode(self, text, max_length=None, padding=False, truncation=False):
        ids = [4 + (hash(w) % (self._v - 4)) for w in text.split()]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids), tokens=text.split())
    def decode(self, ids): return ""
    def save(self, path): ...
    def load(self, path): ...
    @property
    def vocab_size(self): return self._v
    @property
    def embedding_type(self): return EmbeddingType.STANDARD
    @property
    def special_tokens(self): return {"pad_token": 0, "bos_token": 1, "eos_token": 2, "unk_token": 3}


class _CharTok(_WordTok):
    """1 token per character; emits BOS … EOS itself (like the real char tokenizers)."""
    def encode(self, text, max_length=None, padding=False, truncation=False):
        ids = [1] + [4 + (ord(c) % (self._v - 4)) for c in text] + [2]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids), tokens=list(text))


class _CharCNNTok(_WordTok):
    """Word ids + a [words, 4] char_ids matrix with BOS/EOS rows (CharBERT shape)."""
    def encode(self, text, max_length=None, padding=False, truncation=False):
        words = text.split()
        ids = [1] + [4 + (hash(w) % (self._v - 4)) for w in words] + [2]
        rows = [[1, 0, 0, 0]] + [[(ord(c) % 40) + 2 for c in w[:4]] + [0] * (4 - len(w[:4])) for w in words] + [[2, 0, 0, 0]]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids), tokens=words, char_ids=rows)
    @property
    def embedding_type(self): return EmbeddingType.CHARACTER_CNN
    def get_embedding_config(self):
        return {"char_vocab_size": 48, "char_embed_dim": 8, "max_char_len": 4,
                "cnn_filters": [[1, 8], [2, 8]], "num_highway_layers": 1, "output_vocab_size": self._v}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _pool(tmp_path, sources=(("web", 0.7), ("wiki", 0.3)), total_words=6000, docs_per_source=80):
    cfg = mix_cfg(tmp_path, list(sources), total_words=total_words, block_size=16)
    override = {
        name: ListSource(name, [SourceDoc(id=f"{name}{i}", text=msa_doc(4, salt=1000 * k + i), kind="web")
                                for i in range(docs_per_source)])
        for k, (name, _) in enumerate(sources)
    }
    out = build_pool(cfg, sources_override=override)
    assert load_pool_manifest(out)["all_targets_reached"]
    return cfg, out


def _training_cfg(mix, p1_steps=3, p2_steps=4, bs=2, p2_on_mix=True, sft_enabled=False) -> TrainingConfig:
    def phase(**kw):
        base = dict(datasets=["pretraining_mix"], trainable_parameters=["*"], steps=p1_steps, learning_rate=1e-3,
                    batch_size=bs, loss_target="full_sequence", max_length=mix.block_size, save_checkpoint=False)
        base.update(kw)
        return PhaseConfig(**base)
    return TrainingConfig(
        phases=PhasesConfig(
            embedding_alignment=phase(trainable_parameters=["embed_tokens", "lm_head"]),
            warmup=phase(steps=p2_steps) if p2_on_mix else phase(datasets=["arabic_squad"], loss_target="answer_only", steps=p2_steps),
            sft=phase(datasets=["arcd"], loss_target="answer_only", enabled=sft_enabled,
                      early_stopping=EarlyStoppingConfig(enabled=False)),
        ),
        pretraining_mix=mix,
    )


# --------------------------------------------------------------------------
# Fill: shares hold in tokens, not documents
# --------------------------------------------------------------------------

def test_fill_holds_token_shares_under_different_fertilities(tmp_path):
    cfg, pool = _pool(tmp_path, total_words=20000, docs_per_source=250)
    # Budgets ≈ 100 docs per tokenizer so one-document granularity is ~1 %.
    for tok, budget in ((_WordTok(), 8000), (_CharTok(), 50000)):
        docs, stats = fill_token_budget(pool, cfg, tok, budget, log_every=0)
        by = {s.name: s for s in stats}
        for s in stats:
            assert s.tokens >= s.target_tokens
            assert s.tokens - s.target_tokens < 700  # overshoot ≤ one (char-tokenized) doc
        total = sum(s.tokens for s in stats)
        assert abs(by["web"].tokens / total - 0.7) < 0.03
        assert docs[0].ids[-1] == 2  # EOS separator guaranteed
    word_stats = fill_token_budget(pool, cfg, _WordTok(), 8000, log_every=0)[1]
    char_stats = fill_token_budget(pool, cfg, _CharTok(), 8000, log_every=0)[1]
    # Same token budget → the char tokenizer takes far fewer documents.
    assert char_stats[0].docs_taken < word_stats[0].docs_taken
    assert char_stats[0].fertility > 4 * word_stats[0].fertility


def test_fill_is_a_nested_prefix_of_one_fixed_order(tmp_path):
    cfg, pool = _pool(tmp_path)
    small, _ = fill_token_budget(pool, cfg, _WordTok(), 1500, log_every=0)
    large, _ = fill_token_budget(pool, cfg, _WordTok(), 3000, log_every=0)
    small_web = [d.ids.tolist() for d in small if d.source == "web"]
    large_web = [d.ids.tolist() for d in large if d.source == "web"]
    assert large_web[: len(small_web)] == small_web
    assert source_doc_order(10, 42, "web").tolist() == source_doc_order(10, 42, "web").tolist()
    assert source_doc_order(10, 42, "web").tolist() != source_doc_order(10, 42, "wiki").tolist()


def test_fill_raises_when_pool_is_too_small(tmp_path):
    cfg, pool = _pool(tmp_path, total_words=1000, docs_per_source=20)
    with pytest.raises(ValueError, match="pool is exhausted"):
        fill_token_budget(pool, cfg, _CharTok(), 10_000_000, log_every=0)


# --------------------------------------------------------------------------
# Pack
# --------------------------------------------------------------------------

def test_pack_blocks_shapes_tail_and_char_ids():
    docs = [EncodedDoc("a", np.arange(10) + 4), EncodedDoc("b", np.arange(7) + 4), EncodedDoc("a", np.arange(20) + 4)]
    ids, chars, tail = pack_blocks(docs, block_size=8, seed=0)
    assert ids.shape == (4, 8) and ids.dtype == np.int32 and tail == 37 - 32 and chars is None
    with pytest.raises(ValueError, match="< one block"):
        pack_blocks([EncodedDoc("a", np.arange(3))], block_size=8, seed=0)
    cdocs = [EncodedDoc("a", np.arange(9), np.ones((9, 4), dtype=np.int64)),
             EncodedDoc("a", np.arange(9), 2 * np.ones((9, 4), dtype=np.int64))]
    ids, chars, tail = pack_blocks(cdocs, block_size=6, seed=0)
    assert chars.shape == (3, 6, 4) and chars.dtype == np.int16 and tail == 0


def test_pack_blocks_is_seeded_and_uses_every_doc():
    docs = [EncodedDoc("a", np.full(5, i)) for i in range(20)]
    a, _, _ = pack_blocks(docs, 10, seed=1)
    b, _, _ = pack_blocks(docs, 10, seed=1)
    c, _, _ = pack_blocks(docs, 10, seed=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert sorted(np.unique(a).tolist()) == list(range(20))


# --------------------------------------------------------------------------
# Cursor
# --------------------------------------------------------------------------

def _corpus(n_blocks=10, block=4, sequential=True):
    ids = np.arange(n_blocks * block, dtype=np.int32).reshape(n_blocks, block)
    return PackedCorpus(ids, None, {"n_blocks": n_blocks}, consume_sequentially=sequential)


def test_take_sequential_continues_and_wraps(caplog):
    c = _corpus()
    d1, i1 = c.take("embedding_alignment", 6)
    d2, i2 = c.take("warmup", 3)
    assert (i1["block_start"], i1["block_end"]) == (0, 6) and (i2["block_start"], i2["block_end"]) == (6, 9)
    assert d1[0]["input_ids"].tolist() == [0, 1, 2, 3] and d2[0]["input_ids"].tolist() == list(range(24, 28))
    assert not i2["wrapped"]
    with caplog.at_level("WARNING"):
        d3, i3 = c.take("sft", 5)
    assert i3["wrapped"] and i3["block_start"] == 9 and "wrapping around" in caplog.text
    assert d3[1]["input_ids"].tolist() == [0, 1, 2, 3]  # wrapped to block 0
    assert c.consumption.keys() == {"embedding_alignment", "warmup", "sft"}


def test_take_non_sequential_restarts_at_zero():
    c = _corpus(sequential=False)
    _, i1 = c.take("embedding_alignment", 4)
    _, i2 = c.take("warmup", 4)
    assert i1["block_start"] == 0 and i2["block_start"] == 0


def test_packed_block_dataset_emits_char_ids():
    ids = np.zeros((2, 3), dtype=np.int32)
    chars = np.ones((2, 3, 5), dtype=np.int16)
    ex = PackedBlockDataset(ids, chars, 0, 2)[1]
    assert ex["char_ids"].shape == (3, 5) and ex["char_ids"].dtype == np.int64


# --------------------------------------------------------------------------
# Budget + build/cache
# --------------------------------------------------------------------------

def test_resolve_token_budget(tmp_path):
    cfg, _ = _pool(tmp_path)
    tc = _training_cfg(cfg, p1_steps=3, p2_steps=4, bs=2)
    assert resolve_token_budget(tc) == (3 * 2 + 4 * 2) * 16
    tc2 = _training_cfg(cfg, p2_on_mix=False)
    assert resolve_token_budget(tc2) == 3 * 2 * 16
    cfg.token_budget = 999
    assert resolve_token_budget(_training_cfg(cfg)) == 999


def test_build_packed_corpus_caches_and_records_shares(tmp_path, caplog):
    cfg, pool = _pool(tmp_path)
    tc = _training_cfg(cfg)
    out = tmp_path / "cell" / "data"
    corpus = build_packed_corpus(tc, _WordTok(), "word", out, pool_dir=pool)
    m = json.load(open(out / "packed_manifest.json", encoding="utf-8"))
    assert m["n_blocks"] == corpus.n_blocks >= (3 * 2 + 4 * 2)
    assert m["token_budget"] == 224 and m["block_size"] == 16
    assert abs(m["sources"][0]["achieved_share"] - 0.7) < 0.15
    assert m["tokenizer"]["type"] == "word"
    with caplog.at_level("INFO"):
        again = build_packed_corpus(tc, _WordTok(), "word", out, pool_dir=pool)
    assert "reusing packed corpus" in caplog.text and again.n_blocks == corpus.n_blocks
    # a different tokenizer identity repacks
    other = build_packed_corpus(tc, _CharTok(), "char", out, pool_dir=pool)
    assert other.manifest["fingerprint"] != corpus.manifest["fingerprint"]


def test_build_packed_corpus_char_cnn(tmp_path):
    cfg, pool = _pool(tmp_path)
    corpus = build_packed_corpus(_training_cfg(cfg), _CharCNNTok(), "charcnn", tmp_path / "d", pool_dir=pool)
    assert corpus.char_ids is not None and corpus.char_ids.shape[:2] == corpus.input_ids.shape
    assert (tmp_path / "d" / "packed_char_ids.npy").exists()


# --------------------------------------------------------------------------
# Pipeline wiring on a tiny random model
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory) -> Path:
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
                      tie_word_embeddings=True, bos_token_id=None, eos_token_id=2, pad_token_id=0)
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_model")
    Qwen3ForCausalLM(cfg).save_pretrained(path)
    return path


def test_run_all_phases_consumes_packed_mix_sequentially(tmp_path, tiny_model_path):
    from arabic_eval.models.qwen3_adapter import Qwen3Adapter
    from arabic_eval.pipeline.experiment import _run_all_phases

    cfg, pool = _pool(tmp_path)
    cfg.cache_dir = str(pool.parent)  # build_pool() inside the pipeline must find the cached pool
    tc = _training_cfg(cfg, p1_steps=3, p2_steps=4, bs=2)
    adapter = Qwen3Adapter(str(tiny_model_path), device="cpu", dtype="float32")
    tok = _WordTok()
    adapter.adapt_to_tokenizer(tok)
    history = _run_all_phases(adapter, tok, tc, tmp_path / "cell" / "training",
                              tokenizer_type="word", data_dir=tmp_path / "cell" / "data")
    p1, p2 = history["embedding_alignment"], history["warmup"]
    assert p1["status"] == "ok" and p1["steps_completed"] == 3
    assert p1["data"] == {**p1["data"], "dataset": "pretraining_mix", "block_start": 0, "block_end": 6, "n_tokens": 96}
    assert p2["data"]["block_start"] == 6 and p2["data"]["block_end"] == 14 and not p2["data"]["wrapped"]
    assert history["sft"] == {"status": "skipped"}
    assert history["pretraining_mix"]["consumption"].keys() == {"embedding_alignment", "warmup"}
    assert (tmp_path / "cell" / "data" / "packed_manifest.json").exists()
    assert np.isfinite(p1["final_train_loss"]) and np.isfinite(p2["final_train_loss"])


def test_run_all_phases_char_cnn_branch_on_packed_mix(tmp_path, tiny_model_path):
    from arabic_eval.models.qwen3_adapter import Qwen3Adapter
    from arabic_eval.pipeline.experiment import _run_all_phases

    cfg, pool = _pool(tmp_path)
    cfg.cache_dir = str(pool.parent)
    tc = _training_cfg(cfg, p1_steps=2, p2_steps=2, bs=2)
    adapter = Qwen3Adapter(str(tiny_model_path), device="cpu", dtype="float32")
    tok = _CharCNNTok()
    adapter.adapt_to_tokenizer(tok)
    history = _run_all_phases(adapter, tok, tc, tmp_path / "cell" / "training",
                              tokenizer_type="charcnn", data_dir=tmp_path / "cell" / "data")
    assert history["warmup"]["status"] == "ok" and np.isfinite(history["warmup"]["final_train_loss"])
