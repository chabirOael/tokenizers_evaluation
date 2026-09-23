"""``scripts/diag_uncertainty.py``: the per-token statistics (a uniform head has
entropy ln V), the loop-onset index on a synthetic loop row, the position-matched
control, the decile curve, and the reference pass against the existing
``diag_heldout_loss.answer_stats`` on a tiny random Llama (CPU)."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, List

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from arabic_eval.data.finetune_corpora import QARecord  # noqa: E402
from arabic_eval.tasks.freeform.generation import text_stop  # noqa: E402
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput  # noqa: E402

spec = importlib.util.spec_from_file_location("diag_uncertainty", REPO / "scripts" / "diag_uncertainty.py")
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)  # type: ignore[union-attr]
DH = U.DH

VOCAB = 96
PAD, BOS, EOS, UNK = 0, 1, 2, 3


class _VocabTok(BaseTokenizer):
    """Whitespace word tokenizer with a growing vocabulary; decode joins words by one space."""

    def __init__(self, add_specials: bool = True):
        self._add = add_specials
        self._w2i: Dict[str, int] = {}
        self._i2w: Dict[int, str] = {}

    def _id(self, w: str) -> int:
        if w not in self._w2i:
            i = 4 + len(self._w2i)
            assert i < VOCAB, "test vocabulary overflow"
            self._w2i[w], self._i2w[i] = i, w
        return self._w2i[w]

    def train(self, texts, vocab_size, **kw): ...

    def encode(self, text, max_length=None, padding=False, truncation=False):
        words = text.split()
        ids = [self._id(w) for w in words]
        if self._add:
            ids = [BOS] + ids + [EOS]
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids), tokens=words)

    def decode(self, ids):
        return " ".join(self._i2w[i] for i in ids if i in self._i2w)

    def save(self, path): ...
    def load(self, path): ...
    @property
    def vocab_size(self): return VOCAB
    @property
    def embedding_type(self): return EmbeddingType.STANDARD
    @property
    def special_tokens(self): return {"pad_token": PAD, "bos_token": BOS, "eos_token": EOS, "unk_token": UNK}


@pytest.fixture(scope="module")
def tiny_model():
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True, bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def test_uniform_head_has_entropy_ln_v():
    V = 1000
    logits = torch.zeros(5, V, dtype=torch.bfloat16)
    targets = torch.tensor([0, 1, 2, 3, 999])
    H, p1, margin, nll = U.token_stats(logits, targets, chunk=2)
    assert torch.allclose(H, torch.full((5,), math.log(V)), atol=1e-5)
    assert torch.allclose(p1, torch.full((5,), 1.0 / V), atol=1e-7)
    assert torch.allclose(margin, torch.zeros(5), atol=1e-7)
    assert torch.allclose(nll, torch.full((5,), math.log(V)), atol=1e-5)
    # a peaked row: p1 near 1, margin near 1, entropy near 0, NLL of the peak near 0
    peaked = torch.zeros(1, V)
    peaked[0, 7] = 50.0
    H, p1, margin, nll = U.token_stats(peaked, torch.tensor([7]))
    assert H.item() < 1e-6 and p1.item() > 0.999 and margin.item() > 0.999 and nll.item() < 1e-6


def test_onset_index_on_a_synthetic_loop_row():
    tok = _VocabTok(add_specials=False)
    prefix = "هذه مقدمة قصيرة عن الموضوع ثم"
    unit = "العالقين في المكان"
    raw = "\n" + prefix + " " + " ".join([unit] * 5)
    stop = text_stop(raw, [], loop_stop=True)
    assert stop.reason == "loop" and stop.loop.period == 3
    generation = stop.text.strip()
    assert generation == prefix + " " + unit          # cut right after the first copy
    oc = U.onset_char_offset(raw, generation, stop.loop.period)
    assert raw[oc:].startswith(unit) and oc == 1 + len(prefix) + 1
    ids = tok.encode(raw).input_ids
    k = U.onset_token_index(ids, oc - 1, tok.decode)  # decode drops the leading newline: shift by it
    assert k == len(prefix.split())                   # the first token of the first copy
    assert tok.decode(ids[k:k + 3]) == unit
    # a generation that is not a prefix of the raw text, or shorter than the period, has no onset
    assert U.onset_char_offset(raw, "شيء آخر", 3) is None
    assert U.onset_char_offset(raw, "ثم", 3) is None
    # the offset past the whole text has no onset token
    assert U.onset_token_index(ids, 10_000, tok.decode) is None


def _stats(values: List[float]) -> Dict[str, List[float]]:
    return {"H": list(values), "p1": [1 - v / 10 for v in values], "margin": [0.5 - v / 20 for v in values],
            "nll": list(values)}


def test_position_matched_control_picks_rows_long_enough():
    eos_rows = [_stats([9.0] * 4), _stats([1.0] * 5), _stats([2.0] * 3 + [4.0] * 7)]
    control, n = U.position_matched_control(onset=5, window=3, eos_rows=eos_rows)
    # rows with ≥ 5 answer tokens: the second (window [2, 5) = 1, 1, 1) and the third (2, 4, 4)
    assert n == 2
    assert control["H"] == pytest.approx((1.0 + (2.0 + 4.0 + 4.0) / 3) / 2)
    assert control["p1"] == pytest.approx(1 - control["H"] / 10)
    # a window that starts before position 0 is clipped, not shifted
    control, n = U.position_matched_control(onset=2, window=32, eos_rows=eos_rows)
    assert n == 3 and control["H"] == pytest.approx((9.0 + 1.0 + 2.0) / 3)
    assert U.position_matched_control(onset=11, window=3, eos_rows=eos_rows) == (None, 0)
    assert U.window_means(_stats([1.0, 2.0]), 1, 1) is None


def test_decile_curve_per_row_then_over_rows():
    rows = [_stats([float(j) for j in range(20)]), _stats([5.0] * 10), _stats([100.0] * 5)]   # the last is too short
    curve = U.decile_curve(rows)
    assert len(curve) == 10
    assert curve[0] == pytest.approx(((0 + 1) / 2 + 5.0) / 2)
    assert curve[9] == pytest.approx(((18 + 19) / 2 + 5.0) / 2)


def test_reference_pass_nll_equals_the_existing_helper(tiny_model):
    tok = _VocabTok()
    recs = [QARecord(id=f"c{i}", question=f"اكتب جملة رقم {i}", context="",
                     answer=" ".join(f"كلمة{j}" for j in range(3 + i)), source="cidar", prompt_template="instruction")
            for i in range(6)]
    encs, dropped = DH.answer_encodings(recs, tok, max_length=256)
    assert dropped == 0
    ref = DH.answer_stats(tiny_model, encs, PAD, device="cpu", batch_size=8)
    stats = U.score_sequences(tiny_model, encs, PAD, device="cpu", batch_size=8)
    total = sum(v for s in stats for v in s["nll"])
    assert sum(len(s["nll"]) for s in stats) == ref["answer_tokens"]
    assert abs(total - ref["answer_nll_total"]) / ref["answer_nll_total"] < 1e-6
    # the per-row answer spans cover the LCP-masked labels, EOS included
    for (ids, labels), s in zip(encs, stats):
        assert len(s["H"]) == sum(1 for l in labels if l != -100)
    # batching / order do not change the per-row numbers
    stats2 = U.score_sequences(tiny_model, encs, PAD, device="cpu", batch_size=2, order=list(reversed(range(len(encs)))))
    for a, b in zip(stats, stats2):
        assert a["nll"] == pytest.approx(b["nll"], abs=1e-5) and a["H"] == pytest.approx(b["H"], abs=1e-5)
    summary = U.summarize_positions(stats, chars=100)
    assert summary["tokens"] == ref["answer_tokens"]
    assert summary["entropy_per_char"] == pytest.approx(summary["entropy_total"] / 100, abs=1e-6)
    assert 0 < summary["entropy_per_token"] <= math.log(VOCAB) + 1e-6


def test_generation_sequence_mirrors_the_generation_prompt():
    tok = _VocabTok()
    (ids, labels), g = U.generation_sequence("اكتب شيئا ### الإجابة:", "جواب قصير هنا", tok)
    # prompt: BOS + 4 words, trailing EOS stripped; generation: its own BOS / EOS stripped
    assert ids[0] == BOS and EOS not in ids and len(g) == 3
    assert labels == [-100] * 5 + g and ids[5:] == g
    (ids, labels), g = U.generation_sequence("اكتب", "", tok)
    assert g == [] and all(l == -100 for l in labels)
