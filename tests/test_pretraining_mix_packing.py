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
    EncodedDoc, PackedBlockDataset, PackedCorpus, build_packed_corpus, exact_share_budget,
    pack_blocks, select_docs, source_doc_order, tokenize_pool,
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


def _training_cfg(mix, p1_tokens, p2_tokens=None, bs=2, sft_enabled=False) -> TrainingConfig:
    """Mix phases sized in tokens; steps derived (block_size 16)."""
    def phase(**kw):
        base = dict(datasets=["pretraining_mix"], trainable_parameters=["*"], learning_rate=1e-3,
                    batch_size=bs, loss_target="full_sequence", max_length=mix.block_size, save_checkpoint=False)
        base.update(kw)
        return PhaseConfig(**base)
    warmup = (phase(mix_tokens=p2_tokens) if p2_tokens
              else phase(datasets=["arabic_squad"], loss_target="answer_only", steps=4))
    return TrainingConfig(
        phases=PhasesConfig(
            embedding_alignment=phase(trainable_parameters=["embed_tokens", "lm_head"], mix_tokens=p1_tokens),
            warmup=warmup,
            sft=phase(datasets=["arcd"], loss_target="answer_only", enabled=sft_enabled, steps=4,
                      early_stopping=EarlyStoppingConfig(enabled=False)),
        ),
        pretraining_mix=mix,
    )


# --------------------------------------------------------------------------
# Tokenize the whole pool + exact-share budget
# --------------------------------------------------------------------------

def test_tokenize_pool_covers_every_doc_and_shares_hold_under_different_fertilities(tmp_path):
    cfg, pool = _pool(tmp_path, total_words=20000, docs_per_source=250)
    shares = {s.name: s.share for s in cfg.sources}
    for tok in (_WordTok(), _CharTok()):
        docs, stats = tokenize_pool(pool, cfg, tok, log_every=0)
        for name, st in stats.items():
            assert st.docs_available == len(docs[name])
            assert st.available_tokens == sum(int(d.ids.shape[0]) for d in docs[name])
            assert docs[name][0].ids[-1] == 2  # EOS separator guaranteed
        budget = exact_share_budget({n: s.available_tokens for n, s in stats.items()}, shares)
        # the binding source supplies exactly its share; nobody is asked for more than it has
        assert all(round(shares[n] * budget) <= stats[n].available_tokens for n in shares)
        assert any(abs(shares[n] * budget - stats[n].available_tokens) < 1 for n in shares)
        selected = select_docs(docs, stats, budget)
        total = sum(st.tokens for st in stats.values())
        for n, st in stats.items():
            assert st.tokens >= st.target_tokens and st.tokens <= st.available_tokens
            assert abs(st.tokens / total - shares[n]) < 0.03
        assert len(selected) == sum(st.docs_taken for st in stats.values())
    # fertility drives docs, not tokens
    _, w = tokenize_pool(pool, cfg, _WordTok(), log_every=0)
    _, c = tokenize_pool(pool, cfg, _CharTok(), log_every=0)
    assert c["web"].fertility > 4 * w["web"].fertility and c["web"].docs_available == w["web"].docs_available


def test_source_doc_order_is_fixed_per_source():
    assert source_doc_order(10, 42, "web").tolist() == source_doc_order(10, 42, "web").tolist()
    assert source_doc_order(10, 42, "web").tolist() != source_doc_order(10, 42, "wiki").tolist()


def test_exact_share_budget_math():
    assert exact_share_budget({"a": 700, "b": 300}, {"a": 0.7, "b": 0.3}) == 1000
    assert exact_share_budget({"a": 700, "b": 600}, {"a": 0.7, "b": 0.3}) == 1000   # b's surplus unused
    assert exact_share_budget({"a": 350, "b": 300}, {"a": 0.7, "b": 0.3}) == 500    # a binds


# --------------------------------------------------------------------------
# Pack
# --------------------------------------------------------------------------

def test_pack_blocks_shapes_tail_and_char_ids():
    docs = [EncodedDoc("a", np.arange(10) + 4), EncodedDoc("b", np.arange(7) + 4), EncodedDoc("a", np.arange(20) + 4)]
    ids, chars, tail = pack_blocks(docs, block_size=8, seed=0)
    assert ids.shape == (4, 8) and ids.dtype == np.int32 and tail == 37 - 32 and chars is None
    with pytest.raises(ValueError, match="< one block"):
        pack_blocks([EncodedDoc("a", np.arange(3))], block_size=8, seed=0)
    cdocs = [EncodedDoc("a", np.arange(9), np.ones((9, 4), dtype=np.int16)),
             EncodedDoc("a", np.arange(9), 2 * np.ones((9, 4), dtype=np.int16))]
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
# Strict, disjoint slices
# --------------------------------------------------------------------------

def _corpus(n_blocks=10, block=4):
    ids = np.arange(n_blocks * block, dtype=np.int32).reshape(n_blocks, block)
    return PackedCorpus(ids, None, {"n_blocks": n_blocks, "fertility_min": 1.0})


def test_take_is_consecutive_disjoint_and_never_wraps():
    c = _corpus()
    d1, i1 = c.take("embedding_alignment", 6)
    d2, i2 = c.take("warmup", 3)
    assert (i1["block_start"], i1["block_end"]) == (0, 6) and (i2["block_start"], i2["block_end"]) == (6, 9)
    assert d1[0]["input_ids"].tolist() == [0, 1, 2, 3] and d2[0]["input_ids"].tolist() == list(range(24, 28))
    assert i1["corpus_share"] == 0.6
    with pytest.raises(ValueError, match="raise pretraining_mix.pool.total_words"):
        c.take("sft", 2)  # only 1 block left
    assert c.consumption.keys() == {"embedding_alignment", "warmup"}
    with pytest.raises(ValueError, match="exceeds corpus"):
        PackedBlockDataset(c.input_ids, None, 8, 12)


def test_packed_block_dataset_emits_char_ids():
    ids = np.zeros((2, 3), dtype=np.int32)
    chars = np.ones((2, 3, 5), dtype=np.int16)
    ex = PackedBlockDataset(ids, chars, 0, 2)[1]
    assert ex["char_ids"].shape == (3, 5) and ex["char_ids"].dtype == np.int64


# --------------------------------------------------------------------------
# Build / shared cache
# --------------------------------------------------------------------------

def test_build_packed_corpus_packs_whole_pool_and_is_shared_across_cells(tmp_path, caplog):
    cfg, pool = _pool(tmp_path)
    cell_a, cell_b = tmp_path / "exp1" / "cell" / "data", tmp_path / "exp2" / "cell" / "data"
    corpus = build_packed_corpus(cfg, _WordTok(), "word", cell_data_dir=cell_a, pool_dir=pool)
    m = corpus.manifest
    assert m["n_tokens_packed"] <= m["exact_share_budget"] + 200  # ≤ one doc per source overshoot
    assert sum(s["tokens"] for s in m["sources"]) >= m["exact_share_budget"]
    assert abs(m["sources"][0]["achieved_share"] - 0.7) < 0.03
    assert m["tokenizer"]["type"] == "word" and len(m["tokenizer"]["content_hash"]) == 16
    packed = Path(m["pool_dir"]) / "packed" / m["fingerprint"]
    assert packed.exists() and (packed / "packed_input_ids.npy").exists()
    assert json.load(open(cell_a / "packed_manifest.json", encoding="utf-8"))["packed_dir"] == str(packed)
    with caplog.at_level("INFO"):
        again = build_packed_corpus(cfg, _WordTok(), "word", cell_data_dir=cell_b, pool_dir=pool)
    assert "reusing packed corpus" in caplog.text and again.n_blocks == corpus.n_blocks
    assert (cell_b / "packed_manifest.json").exists()


def test_packed_cache_keys_on_tokenizer_content(tmp_path):
    cfg, pool = _pool(tmp_path)
    a = build_packed_corpus(cfg, _WordTok(), "word", pool_dir=pool)

    class _WordTokV2(_WordTok):
        def encode(self, text, max_length=None, padding=False, truncation=False):
            ids = [5 + (hash(w) % (self._v - 5)) for w in text.split()]   # different learned "vocab"
            return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids), tokens=text.split())
    b = build_packed_corpus(cfg, _WordTokV2(), "word", pool_dir=pool)   # same type / vocab_size / specials
    assert a.manifest["fingerprint"] != b.manifest["fingerprint"]
    c = build_packed_corpus(cfg, _CharTok(), "char", pool_dir=pool)
    assert c.manifest["fingerprint"] not in (a.manifest["fingerprint"], b.manifest["fingerprint"])


def test_build_packed_corpus_char_cnn(tmp_path):
    cfg, pool = _pool(tmp_path)
    corpus = build_packed_corpus(cfg, _CharCNNTok(), "charcnn", cell_data_dir=tmp_path / "d", pool_dir=pool)
    assert corpus.char_ids is not None and corpus.char_ids.shape[:2] == corpus.input_ids.shape
    assert (Path(corpus.manifest["pool_dir"]) / "packed" / corpus.manifest["fingerprint"] / "packed_char_ids.npy").exists()


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


def test_run_all_phases_draws_disjoint_slices(tmp_path, tiny_model_path):
    from arabic_eval.models.qwen3_adapter import Qwen3Adapter
    from arabic_eval.pipeline.experiment import _run_all_phases

    cfg, pool = _pool(tmp_path)
    cfg.cache_dir = str(pool.parent)  # build_pool() inside the pipeline must find the cached pool
    tc = _training_cfg(cfg, p1_tokens=3 * 2 * 16, p2_tokens=4 * 2 * 16, bs=2)   # 6 + 8 blocks
    assert (tc.phases.embedding_alignment.steps, tc.phases.warmup.steps) == (3, 4)
    adapter = Qwen3Adapter(str(tiny_model_path), device="cpu", dtype="float32")
    tok = _WordTok()
    adapter.adapt_to_tokenizer(tok)
    history = _run_all_phases(adapter, tok, tc, tmp_path / "cell" / "training",
                              tokenizer_type="word", data_dir=tmp_path / "cell" / "data")
    p1, p2 = history["embedding_alignment"], history["warmup"]
    assert p1["status"] == "ok" and p1["steps_completed"] == 3
    assert (p1["data"]["block_start"], p1["data"]["block_end"], p1["data"]["n_tokens"]) == (0, 6, 96)
    assert (p2["data"]["block_start"], p2["data"]["block_end"]) == (6, 14)
    assert history["sft"] == {"status": "skipped"}
    assert history["pretraining_mix"]["consumption"].keys() == {"embedding_alignment", "warmup"}
    assert (tmp_path / "cell" / "data" / "packed_manifest.json").exists()
    assert np.isfinite(p1["final_train_loss"]) and np.isfinite(p2["final_train_loss"])


def test_run_all_phases_raises_when_pool_too_small(tmp_path, tiny_model_path):
    from arabic_eval.models.qwen3_adapter import Qwen3Adapter
    from arabic_eval.pipeline.experiment import _run_all_phases

    cfg, pool = _pool(tmp_path, total_words=1000, docs_per_source=20)
    cfg.cache_dir = str(pool.parent)
    tc = _training_cfg(cfg, p1_tokens=10 * 2 * 16, p2_tokens=1000 * 2 * 16, bs=2)
    adapter = Qwen3Adapter(str(tiny_model_path), device="cpu", dtype="float32")
    tok = _WordTok()
    adapter.adapt_to_tokenizer(tok)
    with pytest.raises(ValueError, match="raise pretraining_mix.pool.total_words"):
        _run_all_phases(adapter, tok, tc, tmp_path / "cell" / "training",
                        tokenizer_type="word", data_dir=tmp_path / "cell" / "data")


def test_run_all_phases_char_cnn_branch_on_packed_mix(tmp_path, tiny_model_path):
    from arabic_eval.models.qwen3_adapter import Qwen3Adapter
    from arabic_eval.pipeline.experiment import _run_all_phases

    cfg, pool = _pool(tmp_path)
    cfg.cache_dir = str(pool.parent)
    tc = _training_cfg(cfg, p1_tokens=2 * 2 * 16, p2_tokens=2 * 2 * 16, bs=2)
    adapter = Qwen3Adapter(str(tiny_model_path), device="cpu", dtype="float32")
    tok = _CharCNNTok()
    adapter.adapt_to_tokenizer(tok)
    history = _run_all_phases(adapter, tok, tc, tmp_path / "cell" / "training",
                              tokenizer_type="charcnn", data_dir=tmp_path / "cell" / "data")
    assert history["warmup"]["status"] == "ok" and np.isfinite(history["warmup"]["final_train_loss"])
