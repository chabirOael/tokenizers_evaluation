"""Tests for Gemma-3 support — ``Gemma3Adapter`` + ``NativeGemmaTokenizer`` (brief v6 §5.2, docs/report.md §3.17).

Offline, against tiny random-init checkpoints saved to ``tmp_path``: a *multimodal* ``Gemma3ForConditionalGeneration``
(the shape of ``google/gemma-3-4b-pt``: SigLIP tower + projector + text model) and the text-only
``Gemma3ForCausalLM`` our phase checkpoints save. What is pinned:

* the multimodal checkpoint loads the WHOLE text model (logits equal to the multimodal model's on text, tied
  embeddings, only vision / projector tensors left out) — and the plain ``Gemma3ForCausalLM.from_pretrained`` trap
  this avoids (every text weight random, only a warning) is still a trap, so the key mapping stays necessary;
* a checkpoint missing a text weight is refused, never loaded partly random;
* the Llama attribute surface; all four embedding branches forward; generation;
* the scaled embedding (``Gemma3TextScaledWordEmbedding``, × √hidden) survives the resize and an embedding init,
  with the tie; the native resize is a no-op;
* a mini three-phase run through the real phase runner and a checkpoint round trip;
* ``NativeGemmaTokenizer`` reads the special ids from the tokenizer and the padded row count from the TEXT config.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import (Gemma3Config, Gemma3ForCausalLM, Gemma3ForConditionalGeneration, Gemma3TextConfig,  # noqa: E402
                          SiglipVisionConfig)

import arabic_eval.models  # noqa: E402,F401
import arabic_eval.tokenizers  # noqa: E402,F401
import test_qwen3_support as Q  # noqa: E402 — the stub tokenizers, batch / loader helpers and phase config
from arabic_eval.models.gemma3_adapter import Gemma3Adapter, Gemma3LoadError, load_gemma3_text_model  # noqa: E402
from arabic_eval.models.llama_adapter import LlamaAdapter  # noqa: E402
from arabic_eval.registry import model_registry, tokenizer_registry  # noqa: E402
from arabic_eval.tokenizers.char_jaber import CharJaberTokenizer  # noqa: E402
from arabic_eval.tokenizers.character_bert import CharacterBERTTokenizer  # noqa: E402
from arabic_eval.tokenizers.charformer import CharformerTokenizer  # noqa: E402
from arabic_eval.tokenizers.native_gemma import NativeGemmaTokenizer  # noqa: E402
from arabic_eval.training.phases import run_phase  # noqa: E402

TINY_VOCAB = Q.TINY_VOCAB            # 128 — the stub tokenizers' native size
HIDDEN = 32
TEXT_KW = dict(vocab_size=TINY_VOCAB, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=2,
               num_attention_heads=2, num_key_value_heads=1, head_dim=16, sliding_window=8,
               max_position_embeddings=256, tie_word_embeddings=True, pad_token_id=0, eos_token_id=TINY_VOCAB - 1,
               bos_token_id=TINY_VOCAB - 1)


def _mm_config() -> Gemma3Config:
    vc = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                            image_size=28, patch_size=14)
    return Gemma3Config(text_config=Gemma3TextConfig(**TEXT_KW).to_dict(), vision_config=vc.to_dict(),
                        mm_tokens_per_image=4, image_token_index=TINY_VOCAB - 4, boi_token_index=TINY_VOCAB - 3,
                        eoi_token_index=TINY_VOCAB - 2)


@pytest.fixture(scope="module")
def mm_path(tmp_path_factory) -> Path:
    torch.manual_seed(0)
    p = tmp_path_factory.mktemp("tiny_gemma3_mm")
    Gemma3ForConditionalGeneration(_mm_config()).save_pretrained(p)
    return p


@pytest.fixture(scope="module")
def text_path(tmp_path_factory) -> Path:
    torch.manual_seed(1)
    p = tmp_path_factory.mktemp("tiny_gemma3_text")
    Gemma3ForCausalLM(Gemma3TextConfig(**TEXT_KW)).save_pretrained(p)
    return p


def _load(path: Path, **kw) -> Gemma3Adapter:
    return Gemma3Adapter(str(path), device="cpu", dtype="float32", **kw)


X = torch.tensor([[TINY_VOCAB - 1, 10, 20, 30, 40, 50]])


# ---- registry / loading --------------------------------------------------------------------------------------

def test_registry_keys_resolve():
    assert model_registry.get("gemma3") is Gemma3Adapter and issubclass(Gemma3Adapter, LlamaAdapter)
    assert tokenizer_registry.get("native_gemma") is NativeGemmaTokenizer
    assert model_registry.get("qwen3") is not Gemma3Adapter


def test_multimodal_checkpoint_loads_the_whole_text_model(mm_path):
    ref = Gemma3ForConditionalGeneration.from_pretrained(mm_path).eval()
    a = _load(mm_path)
    m = a.model.eval()
    assert isinstance(m, Gemma3ForCausalLM)
    with torch.no_grad():
        assert torch.equal(m(input_ids=X).logits, ref(input_ids=X).logits)
    assert m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
    assert m.config.hidden_size == HIDDEN and len(m.model.layers) == 2
    names = [n for n, _ in m.named_parameters()]
    assert "model.embed_tokens.weight" in names and not any("vision" in n or "projector" in n for n in names)


def test_plain_from_pretrained_is_the_trap_the_mapping_avoids(mm_path):
    """Pins *why* the adapter maps keys: plain ``Gemma3ForCausalLM.from_pretrained`` on the multimodal
    checkpoint leaves the text weights random (transformers 5.7). If a future version fixes it, this test fails
    and the mapping can be revisited — it never makes a run silently random."""
    ref = Gemma3ForConditionalGeneration.from_pretrained(mm_path).eval()
    plain, info = Gemma3ForCausalLM.from_pretrained(mm_path, output_loading_info=True)
    assert "model.embed_tokens.weight" in info["missing_keys"]
    with torch.no_grad():
        assert not torch.allclose(plain.eval()(input_ids=X).logits, ref(input_ids=X).logits)


def test_text_only_checkpoint_loads(text_path):
    ref = Gemma3ForCausalLM.from_pretrained(text_path).eval()
    a = _load(text_path)
    with torch.no_grad():
        assert torch.equal(a.model.eval()(input_ids=X).logits, ref(input_ids=X).logits)


def test_heldout_diagnostic_loads_gemma_through_the_adapter_loader(mm_path, text_path):
    """scripts/diag_heldout_loss.py --base on a Gemma-3 cell reads the multimodal release checkpoint: it must get the
    text model with every weight (the adapter's loader), not the auto class's multimodal model; a phase checkpoint
    loads the same way. The loss call the diagnostic makes (labels = inputs) runs on the result."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "diag_heldout_loss", Path(__file__).resolve().parents[1] / "scripts" / "diag_heldout_loss.py")
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)
    ref = Gemma3ForConditionalGeneration.from_pretrained(mm_path).eval()
    m = diag.load_causal_lm(str(mm_path), "gemma3", "float32").eval()
    assert isinstance(m, Gemma3ForCausalLM)
    with torch.no_grad():
        assert torch.equal(m(input_ids=X).logits, ref(input_ids=X).logits)
        assert torch.isfinite(m(input_ids=X, labels=X).loss)
    t = diag.load_causal_lm(str(text_path), "gemma3", "float32").eval()
    with torch.no_grad():
        assert torch.equal(t(input_ids=X).logits, Gemma3ForCausalLM.from_pretrained(text_path).eval()(input_ids=X).logits)


def test_partial_checkpoint_is_refused(mm_path, tmp_path):
    from safetensors.torch import load_file, save_file
    import shutil
    bad = tmp_path / "bad"
    shutil.copytree(mm_path, bad)
    st = load_file(str(bad / "model.safetensors"))
    drop = next(k for k in st if "layers.1.mlp.down_proj" in k)
    del st[drop]
    save_file(st, str(bad / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(Gemma3LoadError, match="did not load whole"):
        load_gemma3_text_model(str(bad), torch_dtype=torch.float32)


# ---- branches -------------------------------------------------------------------------------------------------

def test_standard_branch_forward_and_generate(mm_path):
    a = _load(mm_path)
    tok = Q._std_tok()
    a.adapt_to_tokenizer(tok)
    out = a.forward(Q._std_batch(tok, Q.ARABIC_TEXTS[:4]))
    assert torch.isfinite(out["loss"]) and out["logits"].shape[-1] == TINY_VOCAB
    ids = torch.tensor([tok.encode(Q.ARABIC_TEXTS[0]).input_ids])
    assert a.generate(ids, max_new_tokens=3, do_sample=False).shape[1] == ids.shape[1] + 3


def test_character_cnn_branch_forward(mm_path):
    a = _load(mm_path)
    tok = CharacterBERTTokenizer(max_char_len=12)
    tok.train(Q.ARABIC_TEXTS, vocab_size=200)
    a.adapt_to_tokenizer(tok)
    encs = [tok.encode(t) for t in Q.ARABIC_TEXTS[:3]]
    S, C = max(len(e.char_ids) for e in encs), len(encs[0].char_ids[0])
    char_ids = torch.zeros(len(encs), S, C, dtype=torch.long)
    mask = torch.zeros(len(encs), S, dtype=torch.long)
    labels = torch.full((len(encs), S), -100, dtype=torch.long)
    for i, e in enumerate(encs):
        n = len(e.char_ids)
        char_ids[i, :n], mask[i, :n], labels[i, :n] = torch.tensor(e.char_ids), 1, torch.tensor(e.input_ids)
    out = a.forward({"char_ids": char_ids, "attention_mask": mask, "labels": labels})
    assert torch.isfinite(out["loss"]) and out["logits"].shape == (len(encs), S, tok.vocab_size)


def test_char_jaber_and_charformer_branches_forward(mm_path):
    a = _load(mm_path)
    tok = CharJaberTokenizer()
    tok.train(Q.ARABIC_TEXTS, vocab_size=0)
    a.adapt_to_tokenizer(tok)
    out = a.forward(Q._std_batch(tok, Q.ARABIC_TEXTS[:3]))
    assert torch.isfinite(out["loss"]) and out["logits"].shape[-1] == tok.vocab_size
    a = _load(mm_path)
    tok = CharformerTokenizer()
    tok.train(Q.ARABIC_TEXTS, vocab_size=0)
    a.adapt_to_tokenizer(tok)
    batch = Q._std_batch(tok, Q.ARABIC_TEXTS[:3])
    L = batch["input_ids"].shape[1] // 2 * 2 + 1                           # odd length: the padding path
    out = a.forward({k: v[:, :L] for k, v in batch.items()})
    assert torch.isfinite(out["loss"]) and out["logits"].shape[1] == L + 1


# ---- embeddings ----------------------------------------------------------------------------------------------

def test_native_resize_is_a_noop(mm_path):
    a = _load(mm_path)
    before = a.model.model.embed_tokens.weight.detach().clone()
    a.adapt_to_tokenizer(Q._std_tok(TINY_VOCAB))
    assert torch.equal(a.model.model.embed_tokens.weight.detach(), before)


def test_scaled_embedding_survives_resize_and_init(mm_path):
    a = _load(mm_path)
    emb0 = a.model.model.embed_tokens
    scale = float(emb0.embed_scale)
    assert type(emb0).__name__ == "Gemma3TextScaledWordEmbedding" and scale == pytest.approx(HIDDEN ** 0.5)
    before = emb0.weight.detach().clone()
    small = 48
    a.adapt_to_tokenizer(Q._std_tok(small))
    emb = a.model.model.embed_tokens
    assert type(emb).__name__ == "Gemma3TextScaledWordEmbedding" and float(emb.embed_scale) == pytest.approx(scale)
    assert torch.equal(emb.weight.detach(), before[:small])
    assert a.model.lm_head.weight.data_ptr() == emb.weight.data_ptr()
    # the forward still scales: the embedding output is the row × √hidden
    with torch.no_grad():
        assert torch.allclose(emb(torch.tensor([[3]]))[0, 0], emb.weight[3] * scale)
    m = _load(mm_path, embedding_init={"method": "mean"})
    m.adapt_to_tokenizer(Q._std_tok(small))
    w = m.model.model.embed_tokens
    assert type(w).__name__ == "Gemma3TextScaledWordEmbedding"
    assert torch.allclose(w.weight.detach(), before.mean(0).expand(small, -1), atol=1e-6)
    assert m.model.lm_head.weight.data_ptr() == w.weight.data_ptr()


# ---- training ------------------------------------------------------------------------------------------------

def test_three_phases_and_checkpoint_round_trip(mm_path, tmp_path):
    a = _load(mm_path)
    tok = Q._std_tok(48)
    a.adapt_to_tokenizer(tok)
    loader = Q._loader(tok, Q.ARABIC_TEXTS)
    snap = Q._snapshot(a.model)
    run_phase(phase_name="embedding_alignment", adapter=a,
              phase_cfg=Q._phase_cfg(trainable_parameters=["embed_tokens", "lm_head"], learning_rate=1e-2),
              train_loader=loader, output_dir=tmp_path / "p1", bf16=False, fp16=False, logging_steps=10)
    assert Q._changed(snap, a.model) == ["model.embed_tokens.weight"]
    snap = Q._snapshot(a.model)
    run_phase(phase_name="warmup", adapter=a, phase_cfg=Q._phase_cfg(learning_rate=1e-2),
              train_loader=loader, output_dir=tmp_path / "p2", bf16=False, fp16=False, logging_steps=10)
    assert set(Q._changed(snap, a.model)) == {n for n, _ in a.model.named_parameters()}
    r3 = run_phase(phase_name="sft", adapter=a, phase_cfg=Q._phase_cfg(save_checkpoint=True),
                   train_loader=loader, output_dir=tmp_path / "p3", bf16=False, fp16=False, logging_steps=10)
    trained = Q._snapshot(a.model)
    cfg = json.loads((Path(r3.checkpoint_path) / "config.json").read_text())
    assert cfg["model_type"] == "gemma3_text" and cfg["vocab_size"] == 48
    # plain keys on disk, so dump_eval_rows.load_state_into and a plain from_pretrained read it
    from safetensors import safe_open
    keys = list(safe_open(str(Path(r3.checkpoint_path) / "model.safetensors"), "pt").keys())
    assert keys and all(k.startswith("model.") for k in keys), keys[:3]
    plain = Gemma3ForCausalLM.from_pretrained(r3.checkpoint_path)
    assert torch.equal(plain.model.embed_tokens.weight.detach(), trained["model.embed_tokens.weight"])
    back = _load(Path(r3.checkpoint_path))
    for n, p in back.model.named_parameters():
        assert torch.equal(p.detach(), trained[n]), n


# ---- the native wrapper --------------------------------------------------------------------------------------

def _tiny_gemma_tokenizer_dir(tmp_path: Path) -> Path:
    """A local tokenizer with Gemma's special ids (<pad> 0, <eos> 1, <bos> 2, <unk> 3) that prepends <bos>,
    next to a multimodal config whose TEXT vocab is padded beyond the tokenizer."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
    for w in "الكتاب على الطاولة ذهب الولد إلى المدرسة".split():
        vocab[w] = len(vocab)
    t = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    t.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    fast = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>",
                                   unk_token="<unk>")
    d = tmp_path / "tiny_gemma_tok"
    fast.save_pretrained(d)
    cfg = _mm_config()
    cfg.text_config.vocab_size = len(vocab) + 5           # padded rows, as 262 208 vs 262 144
    cfg.save_pretrained(d)
    return d


def test_native_gemma_wrapper_facts(tmp_path):
    d = _tiny_gemma_tokenizer_dir(tmp_path)
    tok = NativeGemmaTokenizer(model_name_or_path=str(d))
    assert tok.special_tokens == {"pad_token": 0, "bos_token": 2, "eos_token": 1, "unk_token": 3}
    assert tok.vocab_size == len(tok._hf_tokenizer) + 5               # the TEXT config's padded rows
    ids = tok.encode("الكتاب على الطاولة").input_ids
    assert ids[0] == 2 and ids[-1] != 1                                 # <bos> first, no <eos> appended
    assert tok.decode(ids) == "الكتاب على الطاولة"
