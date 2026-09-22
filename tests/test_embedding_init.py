"""``model.embedding_init`` — informed initialisation of a swapped vocabulary's rows (2026-09-22).

On the tiny random Qwen3 of ``test_qwen3_support`` (tied embeddings) with a tiny real
HF WordLevel tokenizer as the *base* tokenizer and a stub tokenizer whose
``token_surfaces`` is hand-written: the ``surface_avg`` rows equal the expected
averages to 1e-6, a token without surfaces gets the base matrix's global mean,
``legacy`` is byte-identical to the plain resize, the tie survives every method,
and ``AraRooPatTokenizer.token_surfaces`` covers every ROOT / PAT id that has a
reconstruction entry plus every closed-class token.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.config import EmbeddingInitConfig, ModelConfig, load_config  # noqa: E402
from arabic_eval.models.embeddings.init_from_surfaces import (  # noqa: E402
    build_init_matrix,
    capture_base_embeddings,
    surface_vectors,
)
from arabic_eval.models.qwen3_adapter import Qwen3Adapter  # noqa: E402
from arabic_eval.tokenizers.araroopat import (  # noqa: E402
    PFX_CLITICE,
    PFX_CLITICP,
    PFX_PAT,
    PFX_PREP,
    PFX_ROOT,
    SFX,
    SPECIAL_TOKENS_ORDERED,
    TOK_LIT_BEGIN,
    TOK_PROP_END,
    AraRooPatTokenizer,
)
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput  # noqa: E402

from tests.test_qwen3_support import TINY_HIDDEN, TINY_VOCAB, tiny_qwen3_path  # noqa: E402,F401

REPO = Path(__file__).resolve().parents[1]

# The tiny base HF tokenizer: WordLevel over these ids (all < TINY_VOCAB).
BASE_VOCAB = {"[UNK]": 0, "ال": 1, "كتاب": 2, "على": 3, "الطاولة": 4, "من": 5, "ب": 6, "ي": 7, "درس": 8}


@pytest.fixture(scope="module")
def base_hf_dir(tmp_path_factory) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(WordLevel(BASE_VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    path = tmp_path_factory.mktemp("tiny_base_tok")
    hf.save_pretrained(path)
    return path


class _SurfaceTokenizer(BaseTokenizer):
    """Standard-embedding stub whose ``token_surfaces`` is hand-written."""

    def __init__(self, vocab_size: int, surfaces: Dict[int, List[Tuple[str, float]]]) -> None:
        self._n = vocab_size
        self._surfaces = surfaces
        self._reverse_vocab = {i: f"[T_{i}]" for i in range(vocab_size)}

    def train(self, texts, vocab_size, **kw):  # pragma: no cover
        pass

    def encode(self, text, max_length=None, padding=False, truncation=False) -> TokenizerOutput:
        ids = [min(len(w), self._n - 1) for w in text.split()]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids), tokens=text.split())

    def decode(self, ids):  # pragma: no cover
        return " ".join(str(i) for i in ids)

    def save(self, path):  # pragma: no cover
        pass

    def load(self, path):  # pragma: no cover
        pass

    @property
    def vocab_size(self) -> int:
        return self._n

    @property
    def embedding_type(self) -> str:
        return EmbeddingType.STANDARD

    @property
    def special_tokens(self) -> Dict[str, int]:
        return {"pad_token": 0, "bos_token": 1, "eos_token": 2, "unk_token": 3}

    def token_surfaces(self):
        return {i: self._surfaces.get(i, []) for i in range(self._n)}


SMALL = 12
SURFACES = {
    5: [("كتاب", 1.0)],                    # one piece → that row
    6: [("ال كتاب", 1.0)],                 # two pieces → their mean (char_len: 2:4)
    7: [("كتاب", 1.0), ("على", 3.0)],      # two surfaces, weighted 1:3
    8: [],                                 # no surface → global mean
    9: [("zzz unknown", 1.0)],             # tokenizes to [UNK] only — a special id → no piece → global mean
    10: [("", 2.0)],                       # empty surface → no surface → global mean
}


def _adapter(tiny_qwen3_path: Path, **init) -> Qwen3Adapter:
    return Qwen3Adapter(str(tiny_qwen3_path), device="cpu", dtype="float32", embedding_init=init or None)


# ---------------------------------------------------------------------------
# build_init_matrix — pure
# ---------------------------------------------------------------------------

class TestBuildInitMatrix:
    def _base(self, tiny_qwen3_path):
        return capture_base_embeddings(_adapter(tiny_qwen3_path).model)

    def test_surface_avg_uniform_matches_hand_computed_rows(self, tiny_qwen3_path, base_hf_dir):
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(base_hf_dir)
        base = self._base(tiny_qwen3_path)
        tok = _SurfaceTokenizer(SMALL, SURFACES)
        m, stats = build_init_matrix("surface_avg", base, SMALL, tok.token_surfaces(), hf,
                                     token_strings=tok._reverse_vocab)
        gm = base.mean(0)
        assert m.shape == (SMALL, TINY_HIDDEN)
        assert torch.allclose(m[5], base[2], atol=1e-6)
        assert torch.allclose(m[6], (base[1] + base[2]) / 2, atol=1e-6)
        assert torch.allclose(m[7], (base[2] + 3 * base[3]) / 4, atol=1e-6)
        assert torch.allclose(m[8], gm, atol=1e-6) and torch.allclose(m[10], gm, atol=1e-6)
        assert torch.allclose(m[9], gm, atol=1e-6)                # special ids are never averaged in
        for i in (0, 1, 2, 3, 4, 11):                              # ids with no surfaces at all
            assert torch.allclose(m[i], gm, atol=1e-6)
        assert stats["rows_from_surfaces"] == 3 and stats["rows_fallback_mean"] == SMALL - 3
        assert stats["surfaces_unique"] == 4 and stats["families"] == {"T": {"tokens": SMALL, "with_surface": 3}}   # كتاب, ال كتاب, على, zzz unknown

    def test_char_len_weighting(self, tiny_qwen3_path, base_hf_dir):
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(base_hf_dir)
        base = self._base(tiny_qwen3_path)
        m, _ = build_init_matrix("surface_avg", base, SMALL, {6: SURFACES[6]}, hf, weighting="char_len")
        assert torch.allclose(m[6], (2 * base[1] + 4 * base[2]) / 6, atol=1e-6)
        vecs, ok = surface_vectors(["ال كتاب", ""], base, hf, "char_len")
        assert ok.tolist() == [True, False] and torch.allclose(vecs[0], m[6], atol=1e-6)

    def test_base_mean_norm_rescales_every_row(self, tiny_qwen3_path, base_hf_dir):
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(base_hf_dir)
        base = self._base(tiny_qwen3_path)
        m, _ = build_init_matrix("surface_avg", base, SMALL, SURFACES, hf, norm="base_mean")
        target = base.norm(dim=1).mean()
        assert torch.allclose(m.norm(dim=1), target.expand(SMALL), atol=1e-4)

    def test_mean_random_and_legacy(self, tiny_qwen3_path):
        base = self._base(tiny_qwen3_path)
        m, s = build_init_matrix("mean", base, SMALL)
        assert torch.allclose(m, base.mean(0).expand(SMALL, -1)) and s["rows_fallback_mean"] == SMALL
        r1, s1 = build_init_matrix("random", base, SMALL, seed=7)
        r2, _ = build_init_matrix("random", base, SMALL, seed=7)
        r3, _ = build_init_matrix("random", base, SMALL, seed=8)
        assert torch.equal(r1, r2) and not torch.equal(r1, r3) and s1["rows_random"] == SMALL
        assert 0.005 < float(r1.std()) < 0.05
        lg, _ = build_init_matrix("legacy", base, SMALL)
        assert torch.equal(lg, base[:SMALL])

    def test_bad_arguments(self, tiny_qwen3_path):
        base = self._base(tiny_qwen3_path)
        with pytest.raises(ValueError):
            build_init_matrix("magic", base, SMALL)
        with pytest.raises(ValueError):
            build_init_matrix("surface_avg", base, SMALL)          # no surfaces / tokenizer
        with pytest.raises(ValueError):
            build_init_matrix("mean", base, SMALL, weighting="cubic")


# ---------------------------------------------------------------------------
# Through the adapter
# ---------------------------------------------------------------------------

class TestAdapter:
    def test_legacy_is_byte_identical_to_the_plain_resize(self, tiny_qwen3_path):
        plain = _adapter(tiny_qwen3_path)
        legacy = _adapter(tiny_qwen3_path, method="legacy")
        tok = _SurfaceTokenizer(SMALL, SURFACES)
        plain.adapt_to_tokenizer(tok)
        legacy.adapt_to_tokenizer(tok)
        assert torch.equal(plain.model.model.embed_tokens.weight, legacy.model.model.embed_tokens.weight)
        assert legacy.embedding_init_report is None and plain.embedding_init_report is None

    def test_surface_avg_writes_rows_keeps_tie_and_reports(self, tiny_qwen3_path, base_hf_dir):
        adapter = _adapter(tiny_qwen3_path, method="surface_avg", base_tokenizer=str(base_hf_dir))
        base = capture_base_embeddings(adapter.model)
        adapter.adapt_to_tokenizer(_SurfaceTokenizer(SMALL, SURFACES))
        m = adapter.model
        emb = m.model.embed_tokens.weight
        assert emb.shape == (SMALL, TINY_HIDDEN)
        assert m.lm_head.weight.data_ptr() == emb.data_ptr(), "tie must survive the init"
        assert torch.allclose(emb[5].detach(), base[2], atol=1e-6)
        assert torch.allclose(emb[7].detach(), (base[2] + 3 * base[3]) / 4, atol=1e-6)
        assert torch.allclose(emb[8].detach(), base.mean(0), atol=1e-6)
        rep = adapter.embedding_init_report
        assert rep is not None and rep.method == "surface_avg" and rep.tied_embeddings is True
        assert (rep.rows_total, rep.rows_from_surfaces, rep.rows_fallback_mean) == (SMALL, 3, SMALL - 3)
        assert rep.base_rows == TINY_VOCAB and rep.base_tokenizer == str(base_hf_dir)
        j = rep.to_json()
        assert j["families"] == {"T": {"tokens": SMALL, "with_surface": 3}} and j["wall_sec"] >= 0
        out = adapter.forward({"input_ids": torch.tensor([[5, 6, 7]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
                               "labels": torch.tensor([[5, 6, 7]])})
        assert torch.isfinite(out["loss"])

    def test_mean_and_random_through_the_adapter(self, tiny_qwen3_path):
        a = _adapter(tiny_qwen3_path, method="mean")
        base = capture_base_embeddings(a.model)
        a.adapt_to_tokenizer(_SurfaceTokenizer(SMALL, SURFACES))
        assert torch.allclose(a.model.model.embed_tokens.weight.detach(), base.mean(0).expand(SMALL, -1), atol=1e-6)
        r = _adapter(tiny_qwen3_path, method="random", seed=3)
        r.adapt_to_tokenizer(_SurfaceTokenizer(SMALL, SURFACES))
        assert r.embedding_init_report.rows_random == SMALL
        assert r.model.lm_head.weight.data_ptr() == r.model.model.embed_tokens.weight.data_ptr()

    def test_native_vocab_is_never_touched(self, tiny_qwen3_path):
        a = _adapter(tiny_qwen3_path, method="mean")
        before = a.model.model.embed_tokens.weight.detach().clone()
        a.adapt_to_tokenizer(_SurfaceTokenizer(TINY_VOCAB, {}))
        assert torch.equal(a.model.model.embed_tokens.weight.detach(), before)
        assert a.embedding_init_report is None

    def test_config_object_and_dict_are_accepted(self, tiny_qwen3_path):
        cfg = EmbeddingInitConfig(method="mean")
        a = _adapter(tiny_qwen3_path, **cfg.model_dump())
        assert a._embedding_init["method"] == "mean"
        b = Qwen3Adapter(str(tiny_qwen3_path), device="cpu", dtype="float32", embedding_init=cfg)
        assert b._embedding_init == a._embedding_init
        with pytest.raises(TypeError):
            Qwen3Adapter(str(tiny_qwen3_path), device="cpu", dtype="float32", embedding_init="mean")


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_is_legacy_and_round_trips(self):
        m = ModelConfig()
        assert m.embedding_init.method == "legacy" and m.model_dump()["embedding_init"]["method"] == "legacy"
        assert EmbeddingInitConfig(method="surface_avg", weighting="char_len", norm="base_mean").norm == "base_mean"
        with pytest.raises(Exception):
            EmbeddingInitConfig(method="magic")

    def test_every_experiment_yaml_still_loads_with_legacy_default(self):
        for p in sorted((REPO / "configs/experiments").glob("*.yaml")):
            cfg = load_config(str(p), base_path=str(REPO / "configs/base.yaml"))
            assert cfg.model.embedding_init.method in ("legacy", "random", "mean", "surface_avg"), p.name


# ---------------------------------------------------------------------------
# AraRooPat surfaces
# ---------------------------------------------------------------------------

class _NoGen:
    def generate(self, root, pattern):  # pragma: no cover
        return None


def _small_araroopat() -> AraRooPatTokenizer:
    tok = AraRooPatTokenizer()
    tok._backend = _NoGen()
    tok._build_vocab(Counter({"درس": 5, "كتب": 3, "زرف": 2}), Counter({"مَ1ْ2َ3َ": 5, "1ِ2ا3": 3, "1َ2َ3َ": 2}),
                     Counter({"و": 5, "ال": 5}), Counter({"ة": 9, "ه": 5}))
    v = tok._vocab
    tok._reconstruction = {
        (v[f"{PFX_ROOT}درس{SFX}"], v[f"{PFX_PAT}مَ1ْ2َ3َ{SFX}"]): "مدرس",
        (v[f"{PFX_ROOT}كتب{SFX}"], v[f"{PFX_PAT}1ِ2ا3{SFX}"]): "كتاب",
        (v[f"{PFX_ROOT}كتب{SFX}"], v[f"{PFX_PAT}1َ2َ3َ{SFX}"]): "كتب",
        # زرف has no reconstruction entry → no surface → fallback
    }
    tok._metadata = {"roots": {"درس": {"freq": 5}, "كتب": {"freq": 3}, "زرف": {"freq": 2}},
                     "patterns": {"مَ1ْ2َ3َ": {"freq": 5}, "1ِ2ا3": {"freq": 3}, "1َ2َ3َ": {"freq": 2}}}
    return tok


class TestAraRooPatSurfaces:
    def test_every_pair_bearing_root_and_pattern_and_every_closed_class_token_is_covered(self):
        tok = _small_araroopat()
        v, ts = tok._vocab, tok.token_surfaces()
        assert set(ts) == set(v.values())
        with_pair = {rid for rid, _ in tok._reconstruction} | {pid for _, pid in tok._reconstruction}
        for tid in with_pair:
            assert ts[tid], tok._reverse_vocab[tid]
        assert ts[v[f"{PFX_ROOT}زرف{SFX}"]] == []                        # no entry → fallback
        for tok_str, tid in v.items():
            if tok_str.startswith((PFX_CLITICP, PFX_CLITICE, PFX_PREP, "[FUNC_", "[CHAR_", "[DIGIT_", "[PUNCT_")):
                assert ts[tid], tok_str
            if tok_str in SPECIAL_TOKENS_ORDERED or tok_str in (TOK_LIT_BEGIN, TOK_PROP_END):
                assert ts[tid] == [], tok_str

    def test_weights_and_space_variants(self):
        tok = _small_araroopat()
        v, ts = tok._vocab, tok.token_surfaces()
        root = dict(ts[v[f"{PFX_ROOT}كتب{SFX}"]])
        assert root == {"كتاب": 3.0, " كتاب": 3.0, "كتب": 2.0, " كتب": 2.0}     # weighted by the pattern's freq
        pat = dict(ts[v[f"{PFX_PAT}1ِ2ا3{SFX}"]])
        assert pat == {"كتاب": 3.0, " كتاب": 3.0}                            # weighted by the root's freq
        assert ts[v[f"{PFX_CLITICP}ال{SFX}"]] == [("ال", 1.0), (" ال", 1.0)]
        assert ts[v[f"{PFX_CLITICE}ه{SFX}"]] == [("ه", 1.0)]
        assert ts[v[f"{PFX_PREP}من{SFX}"]] == [("من", 1.0), (" من", 1.0)]
        assert ts[v["[PUNCT_.]"]] == [(".", 1.0)] and ts[v["[DIGIT_3]"]] == [("3", 1.0)]

    def test_base_default_uses_decode(self):
        from arabic_eval.tokenizers.bpe import BPETokenizer
        bpe = BPETokenizer()
        bpe.train(["الكتاب على الطاولة", "ذهب الولد إلى المدرسة"] * 10, vocab_size=300)
        ts = bpe.token_surfaces()
        assert set(ts) == set(range(bpe.vocab_size))
        for sid in bpe.special_tokens.values():
            assert ts[sid] == []
        covered = [tid for tid, items in ts.items() if items]
        assert len(covered) >= 10                                     # the merged pieces; lone bytes are excluded
        assert all("\ufffd" not in s_ for items in ts.values() for s_, _ in items)   # lone bytes have no surface
        tid = bpe.encode("الكتاب").input_ids[1]
        assert ts[tid][0][0].strip() and ts[tid][0][1] == 1.0 and "Ø" not in ts[tid][0][0]
