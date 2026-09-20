"""Tests for Qwen3 support — ``Qwen3Adapter`` + ``NativeQwen3Tokenizer``.

Everything runs offline against a tiny random-init ``Qwen3ForCausalLM``
built from ``Qwen3Config`` and saved to ``tmp_path`` — the *real* adapter
then loads it through ``from_pretrained`` exactly as it would load
``Qwen/Qwen3-4B-Base``. The point is to pin the claim that Qwen3 exposes
the attribute surface ``LlamaAdapter`` relies on, for every embedding
branch, without downloading 8 GB of weights.

The last test is network-gated: it pulls only the Qwen3-4B-Base tokenizer
files + ``config.json`` and asserts the real ID map the wrapper hard-codes.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoConfig, Qwen3Config, Qwen3ForCausalLM

import arabic_eval.models  # noqa: F401 — registers adapters
import arabic_eval.tokenizers  # noqa: F401 — registers tokenizers
from arabic_eval.config import PhaseConfig, load_config
from arabic_eval.evaluation.intrinsic_metrics import compute_intrinsic_metrics
from arabic_eval.evaluation.unk_reports import INTRINSIC_UNK_FIELDS, scan_text
from arabic_eval.models.llama_adapter import LlamaAdapter
from arabic_eval.models.qwen3_adapter import Qwen3Adapter
from arabic_eval.registry import model_registry, tokenizer_registry
from arabic_eval.tasks.lighteval.base import _compute_loglikelihood
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput
from arabic_eval.tokenizers.char_jaber import CharJaberTokenizer
from arabic_eval.tokenizers.character_bert import CharacterBERTTokenizer
from arabic_eval.tokenizers.charformer import CharformerTokenizer
from arabic_eval.tokenizers.native_qwen3 import ENDOFTEXT_TOKEN_ID, NativeQwen3Tokenizer
from arabic_eval.training.freezing import apply_trainable_filter
from arabic_eval.training.phases import run_phase

REPO = Path(__file__).resolve().parents[1]

TINY_VOCAB = 128
TINY_HIDDEN = 32

ARABIC_TEXTS = [
    "الكتاب على الطاولة",
    "ذهب الولد إلى المدرسة",
    "يدرس الطلاب اللغة العربية",
    "كتبت الطالبة رسالة طويلة",
    "في المكتبة كتب كثيرة",
    "قرأت مقالة عن التاريخ",
] * 5


# --------------------------------------------------------------------------
# Fixtures: tiny Qwen3 checkpoint on disk + stub tokenizers
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_qwen3_path(tmp_path_factory) -> Path:
    """Random-init Qwen3 saved with ``save_pretrained`` (tied embeddings)."""
    cfg = Qwen3Config(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        bos_token_id=None,
        eos_token_id=TINY_VOCAB - 1,
        pad_token_id=TINY_VOCAB - 1,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    path = tmp_path_factory.mktemp("tiny_qwen3")
    model.save_pretrained(path)
    return path


def _load(tiny_qwen3_path: Path) -> Qwen3Adapter:
    return Qwen3Adapter(str(tiny_qwen3_path), device="cpu", dtype="float32")


class _StubStandardTokenizer(BaseTokenizer):
    """BOS-less word tokenizer over a fixed vocab — mirrors Qwen's no-BOS shape.

    ``vocab_size`` is configurable so the same stub covers the native case
    (== model rows) and the from-scratch case (< model rows).
    """

    def __init__(self, vocab_size: int = TINY_VOCAB) -> None:
        self._vocab_size = vocab_size
        self._w2i: Dict[str, int] = {}

    def train(self, texts: List[str], vocab_size: int, **kw) -> None:
        for t in texts:
            for w in t.split():
                if w not in self._w2i and len(self._w2i) < self._vocab_size - 2:
                    self._w2i[w] = len(self._w2i) + 1  # 0 = pad, last = eos

    def encode(self, text, max_length=None, padding=False, truncation=False) -> TokenizerOutput:
        ids = [self._w2i.get(w, 1) for w in text.split()]
        if truncation and max_length:
            ids = ids[:max_length]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids),
                               tokens=text.split())

    def decode(self, ids):  # pragma: no cover
        inv = {v: k for k, v in self._w2i.items()}
        return " ".join(inv.get(i, "?") for i in ids)

    def save(self, path):  # pragma: no cover
        pass

    def load(self, path):  # pragma: no cover
        pass

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def embedding_type(self) -> str:
        return EmbeddingType.STANDARD

    @property
    def special_tokens(self) -> Dict[str, int]:
        # No unk_token key — same contract as NativeQwen3Tokenizer.
        return {"pad_token": 0, "bos_token": self._vocab_size - 1,
                "eos_token": self._vocab_size - 1}


def _std_tok(vocab_size: int = TINY_VOCAB) -> _StubStandardTokenizer:
    t = _StubStandardTokenizer(vocab_size)
    t.train(ARABIC_TEXTS, vocab_size)
    return t


def _std_batch(tok: BaseTokenizer, texts: List[str]) -> Dict[str, torch.Tensor]:
    encs = [tok.encode(t) for t in texts]
    L = max(len(e.input_ids) for e in encs)
    pad = tok.special_tokens["pad_token"]
    ids = torch.tensor([e.input_ids + [pad] * (L - len(e.input_ids)) for e in encs])
    mask = torch.tensor([e.attention_mask + [0] * (L - len(e.attention_mask)) for e in encs])
    labels = ids.clone()
    labels[mask == 0] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# --------------------------------------------------------------------------
# 1. Registry
# --------------------------------------------------------------------------

def test_registry_keys_resolve():
    assert model_registry.get("qwen3") is Qwen3Adapter
    assert model_registry.get("llama") is LlamaAdapter
    assert issubclass(Qwen3Adapter, LlamaAdapter)
    assert tokenizer_registry.get("native_qwen3") is NativeQwen3Tokenizer
    # The Llama keys still resolve to the Llama classes (no accidental rebinding).
    assert tokenizer_registry.get("native_llama") is not NativeQwen3Tokenizer


# --------------------------------------------------------------------------
# 2. Attribute surface: every embedding branch adapts + forwards
# --------------------------------------------------------------------------

def test_tiny_qwen3_has_llama_attribute_surface(tiny_qwen3_path):
    adapter = _load(tiny_qwen3_path)
    m = adapter.model
    assert isinstance(m, Qwen3ForCausalLM)
    assert hasattr(m.model, "embed_tokens")
    assert hasattr(m.model, "layers") and len(m.model.layers) == 2
    assert hasattr(m, "lm_head")
    assert m.config.hidden_size == TINY_HIDDEN
    # Tied embeddings, like Qwen3-4B-Base and Llama-3.2-1B.
    assert m.config.tie_word_embeddings
    names = [n for n, _ in m.named_parameters()]
    assert "model.embed_tokens.weight" in names
    assert not any("lm_head" in n for n in names), "lm_head must be tied (absent)"


def test_standard_branch_forward(tiny_qwen3_path):
    adapter = _load(tiny_qwen3_path)
    tok = _std_tok()
    adapter.adapt_to_tokenizer(tok)
    out = adapter.forward(_std_batch(tok, ARABIC_TEXTS[:4]))
    assert torch.isfinite(out["loss"])
    assert out["logits"].shape[-1] == TINY_VOCAB
    # adapt_to_tokenizer writes the special ids into the HF config.
    assert adapter.model.config.pad_token_id == 0
    assert adapter.model.config.eos_token_id == TINY_VOCAB - 1


def test_character_cnn_branch_forward(tiny_qwen3_path):
    adapter = _load(tiny_qwen3_path)
    tok = CharacterBERTTokenizer(max_char_len=12)
    tok.train(ARABIC_TEXTS, vocab_size=200)
    adapter.adapt_to_tokenizer(tok)
    encs = [tok.encode(t) for t in ARABIC_TEXTS[:3]]
    S = max(len(e.char_ids) for e in encs)
    C = len(encs[0].char_ids[0])
    char_ids = torch.zeros(len(encs), S, C, dtype=torch.long)
    mask = torch.zeros(len(encs), S, dtype=torch.long)
    labels = torch.full((len(encs), S), -100, dtype=torch.long)
    for i, e in enumerate(encs):
        n = len(e.char_ids)
        char_ids[i, :n] = torch.tensor(e.char_ids)
        mask[i, :n] = 1
        labels[i, :n] = torch.tensor(e.input_ids)
    out = adapter.forward({"char_ids": char_ids, "attention_mask": mask, "labels": labels})
    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == (len(encs), S, tok.vocab_size)


def test_char_jaber_branch_forward(tiny_qwen3_path):
    adapter = _load(tiny_qwen3_path)
    tok = CharJaberTokenizer()
    tok.train(ARABIC_TEXTS, vocab_size=0)
    adapter.adapt_to_tokenizer(tok)
    out = adapter.forward(_std_batch(tok, ARABIC_TEXTS[:3]))
    assert torch.isfinite(out["loss"])
    assert out["logits"].shape[-1] == tok.vocab_size


def test_charformer_branch_forward(tiny_qwen3_path):
    adapter = _load(tiny_qwen3_path)
    tok = CharformerTokenizer()
    tok.train(ARABIC_TEXTS, vocab_size=0)
    adapter.adapt_to_tokenizer(tok)
    batch = _std_batch(tok, ARABIC_TEXTS[:3])
    d_s = tok.get_embedding_config()["downsample_rate"]
    assert d_s == 2
    # Regression: an odd byte length used to produce L+1 logits vs L labels
    # (killed the charformer cell of all_tokenizers_sweep on Llama). Exercise
    # both parities explicitly.
    for L in (batch["input_ids"].shape[1] // 2 * 2, batch["input_ids"].shape[1] // 2 * 2 + 1):
        b = {k: v[:, :L] for k, v in batch.items()}
        out = adapter.forward(b)
        assert torch.isfinite(out["loss"]), f"L={L}"
        # Output head upsamples back to (padded) byte length so labels align.
        assert out["logits"].shape[1] == -(-L // d_s) * d_s, f"L={L}"
        assert out["logits"].shape[-1] == tok.vocab_size


def test_generate_works_on_standard_branch(tiny_qwen3_path):
    adapter = _load(tiny_qwen3_path)
    tok = _std_tok()
    adapter.adapt_to_tokenizer(tok)
    ids = torch.tensor([tok.encode(ARABIC_TEXTS[0]).input_ids])
    out = adapter.generate(ids, max_new_tokens=3, do_sample=False)
    assert out.shape[1] == ids.shape[1] + 3


# --------------------------------------------------------------------------
# 3 + 4. resize semantics on Qwen3
# --------------------------------------------------------------------------

class TestSdpaBackends:
    """``configure_sdpa_backends`` (called at adapter load) turns the cuDNN SDPA
    backend off: it host-compiles a kernel per (batch, kv_length) shape, which
    billed ~300 s to the first free-form batch of every Qwen3-4B cell. The flag
    is process-global, so each test restores it."""

    @pytest.fixture(autouse=True)
    def _restore(self, monkeypatch):
        import torch
        if not hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            pytest.skip("torch without the cuDNN SDPA knob")
        before = torch.backends.cuda.cudnn_sdp_enabled()
        yield
        torch.backends.cuda.enable_cudnn_sdp(before)

    def test_disabled_when_cuda_is_available(self, monkeypatch):
        import torch
        from arabic_eval.models import llama_adapter as la
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.delenv(la.CUDNN_SDP_ENV, raising=False)
        torch.backends.cuda.enable_cudnn_sdp(True)
        assert la.configure_sdpa_backends() is False
        assert torch.backends.cuda.cudnn_sdp_enabled() is False
        assert la.configure_sdpa_backends() is False           # idempotent
        # the other exact-attention backends are untouched
        assert torch.backends.cuda.flash_sdp_enabled() and torch.backends.cuda.math_sdp_enabled()

    def test_env_opt_out_keeps_it(self, monkeypatch):
        import torch
        from arabic_eval.models import llama_adapter as la
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setenv(la.CUDNN_SDP_ENV, "1")
        torch.backends.cuda.enable_cudnn_sdp(True)
        assert la.configure_sdpa_backends() is None
        assert torch.backends.cuda.cudnn_sdp_enabled() is True

    def test_noop_without_cuda(self, monkeypatch):
        import torch
        from arabic_eval.models import llama_adapter as la
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.delenv(la.CUDNN_SDP_ENV, raising=False)
        torch.backends.cuda.enable_cudnn_sdp(True)
        assert la.configure_sdpa_backends() is None
        assert torch.backends.cuda.cudnn_sdp_enabled() is True

    def test_adapter_load_calls_it(self, tiny_qwen3_path, monkeypatch):
        from arabic_eval.models import llama_adapter as la
        calls = []
        monkeypatch.setattr(la, "configure_sdpa_backends", lambda: calls.append(1))
        LlamaAdapter(str(tiny_qwen3_path), device="cpu", dtype="float32")
        assert calls == [1]


def test_native_vocab_is_noop_resize(tiny_qwen3_path):
    adapter = _load(tiny_qwen3_path)
    before = adapter.model.model.embed_tokens.weight.detach().clone()
    adapter.adapt_to_tokenizer(_std_tok(TINY_VOCAB))
    after = adapter.model.model.embed_tokens.weight.detach()
    assert after.shape == before.shape
    assert torch.equal(before, after), "no-op resize must leave rows byte-identical"


def test_from_scratch_vocab_keeps_first_rows_and_tie(tiny_qwen3_path):
    adapter = _load(tiny_qwen3_path)
    before = adapter.model.model.embed_tokens.weight.detach().clone()
    small = 48
    adapter.adapt_to_tokenizer(_std_tok(small))
    m = adapter.model
    emb = m.model.embed_tokens.weight
    assert emb.shape == (small, TINY_HIDDEN)
    assert torch.equal(emb.detach(), before[:small]), "first N pretrained rows preserved"
    assert m.lm_head.weight.shape == (small, TINY_HIDDEN)
    assert m.lm_head.weight.data_ptr() == emb.data_ptr(), "tie must survive resize"
    assert m.config.vocab_size == small
    out = adapter.forward(_std_batch(_std_tok(small), ARABIC_TEXTS[:2]))
    assert torch.isfinite(out["loss"])


# --------------------------------------------------------------------------
# 5. Phase-1 freezing on tied Qwen3
# --------------------------------------------------------------------------

def test_phase1_filter_on_tied_qwen3_warns_not_raises(tiny_qwen3_path, caplog):
    adapter = _load(tiny_qwen3_path)
    adapter.adapt_to_tokenizer(_std_tok())
    with caplog.at_level(logging.WARNING, logger="arabic_eval.training.freezing"):
        trainable = apply_trainable_filter(adapter.model, ["embed_tokens", "lm_head"])
    assert trainable == ["model.embed_tokens.weight"]
    assert any("lm_head" in r.getMessage() for r in caplog.records), \
        "expected the tied-weight warning for lm_head"
    for n, p in adapter.model.named_parameters():
        assert p.requires_grad == ("embed_tokens" in n), n


# --------------------------------------------------------------------------
# 6. Mini 3-phase run through the real phase runner
# --------------------------------------------------------------------------

class _ListDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _loader(tok: BaseTokenizer, texts: List[str], bs: int = 4) -> DataLoader:
    items = []
    for t in texts:
        e = tok.encode(t)
        items.append({"input_ids": e.input_ids, "attention_mask": e.attention_mask})

    def collate(rows):
        L = max(len(r["input_ids"]) for r in rows)
        pad = tok.special_tokens["pad_token"]
        ids = torch.tensor([r["input_ids"] + [pad] * (L - len(r["input_ids"])) for r in rows])
        mask = torch.tensor([r["attention_mask"] + [0] * (L - len(r["attention_mask"])) for r in rows])
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return DataLoader(_ListDataset(items), batch_size=bs, shuffle=False, collate_fn=collate)


def _phase_cfg(**kw) -> PhaseConfig:
    base = dict(
        datasets=["arabic_squad"], trainable_parameters=["*"], steps=3,
        learning_rate=1e-3, batch_size=4, gradient_accumulation_steps=1,
        weight_decay=0.0, max_length=64, loss_target="full_sequence",
        lr_scheduler="constant", warmup_steps=0, max_grad_norm=1.0,
        save_checkpoint=False,
    )
    base.update(kw)
    return PhaseConfig(**base)


def _snapshot(model) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _changed(before, model) -> List[str]:
    return [n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())]


def test_three_phases_on_tiny_qwen3(tiny_qwen3_path, tmp_path):
    adapter = _load(tiny_qwen3_path)
    tok = _std_tok(48)  # from-scratch shape: Phase 1 is meaningful
    adapter.adapt_to_tokenizer(tok)
    loader = _loader(tok, ARABIC_TEXTS)

    # Phase 1 — only embeddings move (lm_head is tied to them).
    snap = _snapshot(adapter.model)
    r1 = run_phase(
        phase_name="embedding_alignment", adapter=adapter,
        phase_cfg=_phase_cfg(trainable_parameters=["embed_tokens", "lm_head"], learning_rate=1e-2),
        train_loader=loader, output_dir=tmp_path / "p1", bf16=False, fp16=False, logging_steps=10,
    )
    assert r1.steps_completed == 3
    assert _changed(snap, adapter.model) == ["model.embed_tokens.weight"]

    # Phase 2 — everything moves.
    snap = _snapshot(adapter.model)
    run_phase(
        phase_name="warmup", adapter=adapter,
        phase_cfg=_phase_cfg(trainable_parameters=["*"], learning_rate=1e-2),
        train_loader=loader, output_dir=tmp_path / "p2", bf16=False, fp16=False, logging_steps=10,
    )
    changed = set(_changed(snap, adapter.model))
    all_names = {n for n, _ in adapter.model.named_parameters()}
    assert changed == all_names, f"unchanged after full unfreeze: {all_names - changed}"

    # Phase 3 — with checkpoint; then reload and compare.
    r3 = run_phase(
        phase_name="sft", adapter=adapter,
        phase_cfg=_phase_cfg(trainable_parameters=["*"], save_checkpoint=True),
        train_loader=loader, output_dir=tmp_path / "p3", bf16=False, fp16=False, logging_steps=10,
    )
    assert r3.checkpoint_path is not None
    trained = _snapshot(adapter.model)
    reloaded = Qwen3Adapter(str(r3.checkpoint_path), device="cpu", dtype="float32")
    assert reloaded.model.config.vocab_size == 48
    for n, p in reloaded.model.named_parameters():
        assert torch.equal(p.detach(), trained[n]), f"checkpoint round-trip mismatch: {n}"


# --------------------------------------------------------------------------
# 7. BOS-less LightEval scoring
# --------------------------------------------------------------------------

def test_loglikelihood_without_bos(tiny_qwen3_path):
    adapter = _load(tiny_qwen3_path)
    tok = _std_tok()
    adapter.adapt_to_tokenizer(tok)
    adapter.model.eval()
    ctx = "الكتاب على"
    ll_a = _compute_loglikelihood(adapter, tok, ctx, " الطاولة")
    ll_b = _compute_loglikelihood(adapter, tok, ctx, " المدرسة")
    assert ll_a < 0 and ll_b < 0
    assert ll_a != ll_b
    # Two-token continuation sums two log-probs (each ≤ 0), so it is ≤ the
    # one-token continuation that shares its first token.
    ll_ab = _compute_loglikelihood(adapter, tok, ctx, " الطاولة المدرسة")
    assert ll_ab <= ll_a


# --------------------------------------------------------------------------
# 8. No-UNK path
# --------------------------------------------------------------------------

def test_no_unk_key_yields_zero_unk_rate_and_empty_report(tmp_path):
    import pyarrow.parquet as pq

    tok = _std_tok()
    assert "unk_token" not in tok.special_tokens
    report_path = tmp_path / "intrinsic_unks.parquet"
    m = compute_intrinsic_metrics(tok, ARABIC_TEXTS, morphological_metrics=False,
                                  unk_report_path=report_path)
    assert m["unk_rate"] == 0.0
    assert m["vocab_coverage"] == 1.0
    # Qwen has byte fallback and therefore no UNK id at all: the report is a
    # readable, empty file carrying the full schema, so every (tokenizer, task)
    # pair in a sweep still produces a comparable artifact.
    table = pq.read_table(report_path)
    assert list(table.schema.names) == list(INTRINSIC_UNK_FIELDS)
    assert table.num_rows == 0
    assert scan_text(tok, ARABIC_TEXTS[0]) == []


# --------------------------------------------------------------------------
# 9. Experiment YAMLs are consumable
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,p1,p3", [
    ("qwen3_4b_3phase_with_sft", False, True),
    ("qwen3_4b_3phase_no_sft", False, False),
    ("_smoke_qwen3_4b", True, True),
])
def test_experiment_yaml_loads(name, p1, p3):
    cfg = load_config(str(REPO / "configs" / "experiments" / f"{name}.yaml"),
                      base_path=str(REPO / "configs" / "base.yaml"))
    assert cfg.model.type == "qwen3"
    assert cfg.model.name_or_path == "Qwen/Qwen3-4B-Base"
    assert cfg.tokenizer.type == "native_qwen3"
    assert cfg.tokenizer.params["model_name_or_path"] == "Qwen/Qwen3-4B-Base"
    assert cfg.training.phases.embedding_alignment.enabled is p1
    assert cfg.training.phases.sft.enabled is p3
    assert cfg.evaluation.score_normalization == "char+pmi"
    assert cfg.sweep.tokenizers[0].type == "native_qwen3"
    # The registry can build both halves of the cell.
    assert model_registry.get(cfg.model.type) is Qwen3Adapter
    assert tokenizer_registry.get(cfg.tokenizer.type) is NativeQwen3Tokenizer


def test_model_yaml_loads():
    import yaml
    d = yaml.safe_load((REPO / "configs" / "models" / "qwen3_4b.yaml").read_text())
    assert d["model"]["type"] == "qwen3"
    assert d["model"]["name_or_path"] == "Qwen/Qwen3-4B-Base"


# --------------------------------------------------------------------------
# 10. Real Qwen3-4B-Base tokenizer (network-gated)
# --------------------------------------------------------------------------

def _hub_reachable() -> bool:
    try:
        AutoConfig.from_pretrained("Qwen/Qwen3-4B-Base")
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _hub_reachable(), reason="HF Hub not reachable")
def test_real_qwen3_tokenizer_facts(tmp_path):
    from arabic_eval.tokenizers.utils.arabic_text import clean_token_string

    tok = NativeQwen3Tokenizer()
    hf = tok._hf_tokenizer
    assert len(hf) == 151669
    assert tok.vocab_size == 151936, "must report the padded embedding-row count"
    assert tok.vocab_size == AutoConfig.from_pretrained("Qwen/Qwen3-4B-Base").vocab_size
    st = tok.special_tokens
    assert st == {"pad_token": ENDOFTEXT_TOKEN_ID, "bos_token": ENDOFTEXT_TOKEN_ID,
                  "eos_token": ENDOFTEXT_TOKEN_ID}
    assert "unk_token" not in st
    assert tok.embedding_type == EmbeddingType.STANDARD

    text = "الكتاب على الطاولة"
    enc = tok.encode(text)
    assert enc.input_ids[0] != ENDOFTEXT_TOKEN_ID, "Qwen adds no BOS"
    assert enc.input_ids[-1] != ENDOFTEXT_TOKEN_ID, "Qwen adds no EOS"
    assert len(enc.input_ids) == len(enc.attention_mask) == len(enc.tokens)
    assert tok.decode(enc.input_ids) == text
    # ByteLevel token strings must clean back to Arabic for the morph metrics.
    cleaned = "".join(clean_token_string(t) for t in enc.tokens)
    assert cleaned == text.replace(" ", "")

    # Intrinsic metrics on the real tokenizer: no UNKs, finite numbers.
    m = compute_intrinsic_metrics(tok, ARABIC_TEXTS, morphological_metrics=False)
    assert m["unk_rate"] == 0.0 and m["fertility"] > 0

    # save → load round-trip keeps the same vocab contract.
    tok.save(tmp_path / "tok")
    tok2 = NativeQwen3Tokenizer()
    tok2.load(tmp_path / "tok")
    assert tok2.vocab_size == 151936
    assert tok2.encode(text).input_ids == enc.input_ids
