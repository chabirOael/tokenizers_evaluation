"""Tests for the ratio-controlled Phase 3 mixture (``PhaseConfig.mixture``,
``data/sft_mixture.py``) and the free-form corpora loaders.

Offline: every corpus is an in-memory list of ``QARecord`` and the
tokenizer is one token per word with BOS/EOS, so truncation behaviour is
exact and predictable.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.config import (
    CORPUS_CATEGORY,
    DatasetName,
    ExperimentConfig,
    MixtureConfig,
    PhaseConfig,
    TrainingConfig,
    load_config,
)
from arabic_eval.data import finetune_corpora as fc
from arabic_eval.data.finetune_corpora import (
    PINNED_REVISIONS,
    QARecord,
    _LOADERS,
    _format_qa_full,
    _format_qa_prompt,
    load_corpus,
    tokenize_record,
)
from arabic_eval.data.sft_mixture import (
    MixtureShortfallError,
    build_mixture_dataloader,
    category_quotas,
    compose_mixture,
    largest_remainder,
    load_mixture_pools,
    manifest_summary,
    max_total_at_shares,
    plan_mixture,
    water_fill,
)
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput
from typing import get_args


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

class _WordTok(BaseTokenizer):
    """One token per whitespace word, BOS … EOS, honours max_length truncation."""
    def __init__(self, vocab_size: int = 64):
        self._v = vocab_size

    def train(self, texts, vocab_size, **kw): ...

    def encode(self, text, max_length=None, padding=False, truncation=False):
        words = text.split()
        ids = [1] + [4 + (hash(w) % (self._v - 4)) for w in words] + [2]
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids), tokens=words)

    def decode(self, ids): return ""
    def save(self, path): ...
    def load(self, path): ...
    @property
    def vocab_size(self): return self._v
    @property
    def embedding_type(self): return EmbeddingType.STANDARD
    @property
    def special_tokens(self): return {"pad_token": 0, "bos_token": 1, "eos_token": 2, "unk_token": 3}


def _qa(source: str, n: int, *, ctx_words: int = 3, answer_words: int = 1) -> List[QARecord]:
    return [
        QARecord(id=f"{source}-{i:04d}", question=f"سؤال {i}", context=" ".join(f"سياق{i}" for _ in range(ctx_words)),
                 answer=" ".join(f"جواب{i}" for _ in range(answer_words)), source=source)
        for i in range(n)
    ]


def _mcq(n: int) -> List[QARecord]:
    return [
        QARecord(id=f"mcq-{i:04d}", question=f"سؤال {i}", context="", answer="أ", source="arabic_squad_mcq",
                 prompt_template="mcq_letter", choices=["خيار1", "خيار2", "خيار3", "خيار4"])
        for i in range(n)
    ]


def _free(source: str, n: int, *, answer_words: int = 8, ctx: bool = False) -> List[QARecord]:
    return [
        QARecord(id=f"{source}-{i:04d}", question=f"اكتب {i}", context=("مدخل نص" if ctx else ""),
                 answer=" ".join(f"كلمة{i}" for _ in range(answer_words)), source=source, prompt_template="instruction")
        for i in range(n)
    ]


def _mix(**kw) -> MixtureConfig:
    base = dict(total_examples=100, shares={"extractive": 0.4, "mcq": 0.3, "free_form": 0.3})
    base.update(kw)
    return MixtureConfig(**base)


POOLS = {
    "tydiqa_arabic": _qa("tydiqa_arabic", 200),
    "arcd": _qa("arcd", 10),
    "arabic_squad_mcq": _mcq(300),
    "cidar": _free("cidar", 50),
    "bactrian_x_ar": _free("bactrian_x_ar", 200, ctx=True),
}
DATASETS = list(POOLS)


# --------------------------------------------------------------------------
# Registry consistency
# --------------------------------------------------------------------------

def test_dataset_name_category_and_loaders_are_in_sync():
    names = set(get_args(DatasetName)) - {"pretraining_mix"}
    assert names == set(CORPUS_CATEGORY) == set(_LOADERS)
    assert set(CORPUS_CATEGORY.values()) == {"extractive", "mcq", "free_form"}
    for name in ("cidar", "bactrian_x_ar", "aya_ar"):
        assert CORPUS_CATEGORY[name] == "free_form"
        assert len(PINNED_REVISIONS[name]) == 40


# --------------------------------------------------------------------------
# Instruction template
# --------------------------------------------------------------------------

def test_instruction_template_without_context():
    rec = _free("cidar", 1)[0]
    assert _format_qa_prompt(rec) == (
        f"{fc.INSTRUCTION_HEADER}\n\n{fc.INSTRUCTION_LABEL}\nاكتب 0\n\n{fc.ANSWER_LABEL}\n"
    )
    assert _format_qa_full(rec) == _format_qa_prompt(rec) + rec.answer     # no separator: the cue line ends in \n
    assert rec.category == "free_form"
    assert "السياق" not in _format_qa_prompt(rec) and fc.INPUT_LABEL not in _format_qa_prompt(rec)


def test_instruction_template_with_context_adds_the_input_section():
    rec = _free("bactrian_x_ar", 1, ctx=True)[0]
    assert _format_qa_prompt(rec) == (
        f"{fc.INSTRUCTION_HEADER_WITH_INPUT}\n\n{fc.INSTRUCTION_LABEL}\nاكتب 0\n\n"
        f"{fc.INPUT_LABEL}\nمدخل نص\n\n{fc.ANSWER_LABEL}\n"
    )
    assert fc.INSTRUCTION_HEADER not in _format_qa_prompt(rec)             # the with-input header replaces it


def test_unknown_template_raises():
    rec = QARecord(id="x", question="q", context="", answer="a", source="cidar", prompt_template="bogus")
    with pytest.raises(ValueError, match="unknown prompt_template"):
        _format_qa_prompt(rec)


def test_tokenize_record_reports_loss_tokens_and_truncation():
    tok = _WordTok()
    rec = _free("cidar", 1, answer_words=5)[0]
    n_prompt = 1 + len(_format_qa_prompt(rec).split())                  # BOS + the prompt words (no EOS)
    entry, loss_tokens, truncated = tokenize_record(rec, tok, max_length=64, loss_target="answer_only")
    assert entry is not None and loss_tokens == 6 and not truncated   # 5 answer words + EOS
    entry, loss_tokens, truncated = tokenize_record(rec, tok, max_length=n_prompt + 1, loss_target="answer_only")
    assert entry is not None and truncated and loss_tokens == 1        # 1 answer token survives
    entry, _, _ = tokenize_record(rec, tok, max_length=n_prompt, loss_target="answer_only")
    assert entry is None                                                # truncation ate the answer
    entry, loss_tokens, _ = tokenize_record(rec, tok, max_length=64, loss_target="full_sequence")
    assert "labels" not in entry and loss_tokens == len(entry["input_ids"])


class _WordTokNoEos(_WordTok):
    """The native Llama / Qwen3 wrappers' shape: no EOS (nor BOS) on encode."""
    def encode(self, text, max_length=None, padding=False, truncation=False):
        words = text.split()
        ids = [4 + (hash(w) % (self._v - 4)) for w in words]
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids), tokens=words)


def test_tokenize_record_appends_eos_when_the_tokenizer_does_not():
    """A model trained on answers without a trailing EOS never learns to stop:
    the native_qwen3 run of 2026-09-18 generated to the cap on 249 / 250 prompts."""
    tok = _WordTokNoEos()
    rec = _free("cidar", 1, answer_words=5)[0]
    n_words = len(fc._format_qa_full(rec).split())
    entry, loss_tokens, truncated = tokenize_record(rec, tok, max_length=64, loss_target="answer_only")
    assert entry["input_ids"][-1] == 2 and len(entry["input_ids"]) == n_words + 1 and not truncated
    assert loss_tokens == 6 and entry["labels"][-1] == 2                 # 5 answer words + the EOS, all loss targets
    entry, _, _ = tokenize_record(rec, tok, max_length=64, loss_target="full_sequence")
    assert entry["input_ids"][-1] == 2
    entry, _, truncated = tokenize_record(rec, tok, max_length=n_words - 1, loss_target="answer_only")
    assert truncated and entry["input_ids"][-1] != 2                     # a cut record gets no EOS: there is no answer end to mark
    entry, _, _ = tokenize_record(rec, _WordTok(), max_length=64, loss_target="answer_only")
    assert entry["input_ids"][-2:] != [2, 2]                             # a tokenizer that emits EOS is left alone


# --------------------------------------------------------------------------
# Quota arithmetic
# --------------------------------------------------------------------------

def test_largest_remainder_sums_exactly_and_is_deterministic():
    assert largest_remainder({"a": 1, "b": 1, "c": 1}, 100) == {"a": 34, "b": 33, "c": 33}
    assert largest_remainder({"a": 0.4, "b": 0.3, "c": 0.3}, 30000) == {"a": 12000, "b": 9000, "c": 9000}
    out = largest_remainder({"a": 0.333, "b": 0.333, "c": 0.334}, 7)
    assert sum(out.values()) == 7
    assert largest_remainder({"a": 1}, 0) == {"a": 0}


def test_category_quotas():
    assert category_quotas({"extractive": 0.4, "mcq": 0.3, "free_form": 0.3}, 100) == {"extractive": 40, "mcq": 30, "free_form": 30}


def test_water_fill_caps_small_corpus_and_hands_remainder_over():
    alloc, short = water_fill(12000, {"tydiqa_arabic": 14805, "arcd": 693}, {"tydiqa_arabic": 1, "arcd": 1})
    assert alloc == {"tydiqa_arabic": 11307, "arcd": 693} and short == 0


def test_water_fill_equal_and_proportional_and_weights():
    caps = {"a": 100, "b": 300}
    assert water_fill(40, caps, {"a": 1, "b": 1})[0] == {"a": 20, "b": 20}
    assert water_fill(40, caps, {"a": 100, "b": 300})[0] == {"a": 10, "b": 30}
    assert water_fill(40, caps, {"a": 3, "b": 1})[0] == {"a": 30, "b": 10}


def test_water_fill_reports_shortfall():
    alloc, short = water_fill(500, {"a": 100, "b": 300}, {"a": 1, "b": 1})
    assert alloc == {"a": 100, "b": 300} and short == 100


def test_max_total_at_shares():
    shares = {"extractive": 0.4, "mcq": 0.3, "free_form": 0.3}
    caps = {"extractive": 15498, "mcq": 48344, "free_form": 96000}
    assert max_total_at_shares(shares, caps) == 38745
    assert max_total_at_shares(shares, caps, batch_size=4) == 38744


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------

def test_plan_equal_water_fills_within_category():
    caps = {n: len(p) for n, p in POOLS.items()}
    plan = plan_mixture(_mix(), DATASETS, caps)
    assert plan["category_quotas"] == {"extractive": 40, "mcq": 30, "free_form": 30}
    assert plan["allocation"] == {"tydiqa_arabic": 30, "arcd": 10, "arabic_squad_mcq": 30, "cidar": 15, "bactrian_x_ar": 15}
    assert plan["residual"] == {}


def test_plan_proportional_and_explicit_weights():
    caps = {n: len(p) for n, p in POOLS.items()}
    plan = plan_mixture(_mix(within_category="proportional"), DATASETS, caps)
    assert plan["allocation"]["cidar"] == 6 and plan["allocation"]["bactrian_x_ar"] == 24
    plan = plan_mixture(_mix(weights={"cidar": 2, "bactrian_x_ar": 1}), DATASETS, caps)
    assert plan["allocation"]["cidar"] == 20 and plan["allocation"]["bactrian_x_ar"] == 10
    assert plan["allocation"]["tydiqa_arabic"] == 30   # unweighted corpora keep the equal rule


def test_plan_shortfall_error_names_numbers_and_ceiling():
    caps = {n: len(p) for n, p in POOLS.items()}
    with pytest.raises(MixtureShortfallError) as ei:
        plan_mixture(_mix(total_examples=600), DATASETS, caps, batch_size=4)
    msg = str(ei.value)
    assert "'extractive' needs 240 examples" in msg and "hold 210" in msg
    assert "at most 524" in msg          # min(210/0.4, 300/0.3, 250/0.3) = 525 → multiple of 4
    assert "upsample" in msg


def test_plan_upsample_spreads_residual_by_weights():
    caps = {n: len(p) for n, p in POOLS.items()}
    plan = plan_mixture(_mix(total_examples=600, upsample=True), DATASETS, caps)
    assert plan["allocation"]["tydiqa_arabic"] + plan["allocation"]["arcd"] == 240
    assert plan["residual"] == {"tydiqa_arabic": 15, "arcd": 15}
    assert plan["allocation"]["arcd"] == 25


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------

def test_compose_exact_total_and_per_category_counts():
    tok = _WordTok()
    enc, manifest = compose_mixture(_mix(), DATASETS, POOLS, tok, max_length=64, loss_target="answer_only")
    assert len(enc) == 100
    cats = manifest["categories"]
    assert {c: cats[c]["kept"] for c in cats} == {"extractive": 40, "mcq": 30, "free_form": 30}
    assert abs(sum(c["loss_token_share"] for c in cats.values()) - 1.0) < 1e-3
    # Free-form answers are 8 words + EOS; MCQ letters 1 + EOS; extractive 1 + EOS.
    assert cats["free_form"]["loss_tokens"] == 30 * 9 and cats["mcq"]["loss_tokens"] == 60
    assert cats["free_form"]["loss_token_share"] > cats["free_form"]["example_share"]
    ds = manifest["datasets"]
    assert ds["arcd"]["kept"] == 10 and ds["arcd"]["available"] == 10 and ds["cidar"]["revision"] == PINNED_REVISIONS["cidar"]
    assert all(d["drawn"] == d["kept"] == d["planned"] and d["repeated"] == 0 for d in ds.values())
    assert len(ds["tydiqa_arabic"]["ids"]) == 30 and len(set(ds["tydiqa_arabic"]["ids"])) == 30
    assert manifest["max_total_examples_at_these_shares"] == 525
    assert all("labels" in e for e in enc)


def test_compose_is_deterministic_under_seed_and_changes_with_it():
    tok = _WordTok()
    _, m1 = compose_mixture(_mix(), DATASETS, POOLS, tok, max_length=64, loss_target="answer_only")
    _, m2 = compose_mixture(_mix(), DATASETS, POOLS, tok, max_length=64, loss_target="answer_only")
    assert m1["datasets"]["tydiqa_arabic"]["ids"] == m2["datasets"]["tydiqa_arabic"]["ids"]
    _, m3 = compose_mixture(_mix(seed=7), DATASETS, POOLS, tok, max_length=64, loss_target="answer_only")
    assert m1["datasets"]["tydiqa_arabic"]["ids"] != m3["datasets"]["tydiqa_arabic"]["ids"]
    # Independent of the loader's row order.
    shuffled = {n: list(reversed(p)) for n, p in POOLS.items()}
    _, m4 = compose_mixture(_mix(), DATASETS, shuffled, tok, max_length=64, loss_target="answer_only")
    assert m1["datasets"]["cidar"]["ids"] == m4["datasets"]["cidar"]["ids"]


def test_compose_tops_up_after_truncation_drops():
    """Records whose answer is cut away by max_length do not count; the
    draw keeps walking so the quota is met exactly."""
    tok = _WordTok()
    pools = dict(POOLS)
    # Half the extractive pool has a 60-word context → prompt alone exceeds max_length 40
    # (the MCQ prompt is 23 tokens, so 40 keeps every other pool intact).
    pools["tydiqa_arabic"] = _qa("tydiqa_arabic", 100, ctx_words=60) + _qa("tydiqa_arabic", 100, ctx_words=3)
    pools["tydiqa_arabic"] = [QARecord(**{**r.__dict__, "id": f"t-{i:04d}"}) for i, r in enumerate(pools["tydiqa_arabic"])]
    enc, manifest = compose_mixture(_mix(), DATASETS, pools, tok, max_length=40, loss_target="answer_only")
    d = manifest["datasets"]["tydiqa_arabic"]
    assert d["kept"] == 30 and d["dropped_truncation"] > 0 and d["drawn"] == d["kept"] + d["dropped_truncation"]
    assert len(enc) == 100


def test_compose_drop_truncated_answers():
    tok = _WordTok()
    pools = dict(POOLS)
    # cidar answers of 50 words: prompt (4 tokens incl. BOS) + 50 + EOS = 55 > 40 → cut.
    pools["cidar"] = _free("cidar", 20, answer_words=50) + _free("cidar", 40, answer_words=2)
    pools["cidar"] = [QARecord(**{**r.__dict__, "id": f"c-{i:04d}"}) for i, r in enumerate(pools["cidar"])]
    pools["bactrian_x_ar"] = _free("bactrian_x_ar", 200, answer_words=2)
    _, m_keep = compose_mixture(_mix(), DATASETS, pools, tok, max_length=40, loss_target="answer_only")
    assert m_keep["datasets"]["cidar"]["dropped_cut_answer"] == 0
    _, m_drop = compose_mixture(_mix(drop_truncated_answers=True), DATASETS, pools, tok, max_length=40, loss_target="answer_only")
    d = m_drop["datasets"]["cidar"]
    assert d["kept"] == 15 and d["dropped_cut_answer"] > 0
    assert all(int(i.split("-")[1]) >= 20 for i in d["ids"])   # only the short-answer records survive


def test_compose_shortfall_after_tokenization_is_loud():
    tok = _WordTok()
    pools = dict(POOLS)
    pools["arcd"] = _qa("arcd", 10, ctx_words=60)   # every ARCD record is untokenizable at 40 → TyDiQA covers
    enc, manifest = compose_mixture(_mix(), DATASETS, pools, tok, max_length=40, loss_target="answer_only")
    assert manifest["datasets"]["arcd"]["kept"] == 0 and manifest["datasets"]["tydiqa_arabic"]["kept"] == 40
    assert len(enc) == 100
    # Every extractive corpus loses records: the category cannot reach 40.
    pools["arcd"] = _qa("arcd", 5, ctx_words=60) + _qa("arcd", 5, ctx_words=3)
    pools["arcd"] = [QARecord(**{**r.__dict__, "id": f"a-{i:04d}"}) for i, r in enumerate(pools["arcd"])]
    pools["tydiqa_arabic"] = _qa("tydiqa_arabic", 30, ctx_words=60) + _qa("tydiqa_arabic", 30, ctx_words=3)
    pools["tydiqa_arabic"] = [QARecord(**{**r.__dict__, "id": f"t-{i:04d}"}) for i, r in enumerate(pools["tydiqa_arabic"])]
    with pytest.raises(MixtureShortfallError, match="category 'extractive' kept 35 of its 40"):
        compose_mixture(_mix(), DATASETS, pools, tok, max_length=40, loss_target="answer_only")


def test_compose_spills_a_dry_corpus_deficit_over_to_its_category():
    """ARCD is allocated its whole pool (10), loses 4 to truncation, and
    TyDiQA absorbs the 4 — the category quota still lands exactly."""
    tok = _WordTok()
    pools = dict(POOLS)
    pools["arcd"] = _qa("arcd", 4, ctx_words=60) + _qa("arcd", 6, ctx_words=3)
    pools["arcd"] = [QARecord(**{**r.__dict__, "id": f"a-{i:04d}"}) for i, r in enumerate(pools["arcd"])]
    enc, manifest = compose_mixture(_mix(), DATASETS, pools, tok, max_length=40, loss_target="answer_only")
    d = manifest["datasets"]
    assert d["arcd"]["planned"] == 10 and d["arcd"]["kept"] == 6 and d["arcd"]["dropped_truncation"] == 4
    assert d["tydiqa_arabic"]["planned"] == 30 and d["tydiqa_arabic"]["kept"] == 34
    assert manifest["categories"]["extractive"]["kept"] == 40 and len(enc) == 100


def test_compose_upsample_repeats_a_further_permutation():
    tok = _WordTok()
    enc, manifest = compose_mixture(_mix(total_examples=600, upsample=True), DATASETS, POOLS, tok,
                                    max_length=64, loss_target="answer_only")
    assert len(enc) == 600
    d = manifest["datasets"]["arcd"]
    assert d["kept"] == 25 and d["passes"] == 3 and d["repeated"] == 15 and d["planned"] == 25
    assert len(set(d["ids"])) == 10


def test_manifest_summary_drops_ids():
    tok = _WordTok()
    _, manifest = compose_mixture(_mix(), DATASETS, POOLS, tok, max_length=64, loss_target="answer_only")
    summary = manifest_summary(manifest)
    assert "ids" not in summary["datasets"]["cidar"] and "ids" in manifest["datasets"]["cidar"]
    assert summary["categories"] == manifest["categories"]


def test_build_mixture_dataloader_uses_loader_params_and_latin_filter(monkeypatch):
    seen = {}

    def fake_load(name, split, **params):
        seen[name] = params
        pool = list(POOLS[name])
        if name == "cidar":
            pool[0] = QARecord(**{**pool[0].__dict__, "answer": "Latin word"})
        return pool
    monkeypatch.setattr("arabic_eval.data.sft_mixture.load_corpus", fake_load)
    loader, manifest = build_mixture_dataloader(
        _mix(), DATASETS, _WordTok(), batch_size=4, max_length=64, loss_target="answer_only",
        corpus_params={"cidar": {"include_datasets": ["x"]}}, clean_latin_rows=True,
    )
    assert seen["cidar"] == {"include_datasets": ["x"]} and seen["arcd"] == {}
    assert manifest["datasets"]["cidar"]["available"] == 50 and manifest["datasets"]["cidar"]["after_latin_filter"] == 49
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 4 and "labels" in batch
    assert sum(len(b["input_ids"]) for b in loader) == 100


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------

def _phase(**kw) -> PhaseConfig:
    base = dict(
        datasets=["tydiqa_arabic", "arcd", "arabic_squad_mcq", "cidar"],
        trainable_parameters=["*"], learning_rate=1e-4, batch_size=4,
        mixture={"total_examples": 400, "shares": {"extractive": 0.4, "mcq": 0.3, "free_form": 0.3}},
    )
    base.update(kw)
    return PhaseConfig(**base)


def test_mixture_derives_steps_and_validates_consistency():
    assert _phase().steps == 100
    assert _phase(steps=100).steps == 100
    with pytest.raises(ValueError, match="implies steps = 100"):
        _phase(steps=99)
    with pytest.raises(ValueError, match="multiple of batch_size"):
        _phase(mixture={"total_examples": 402, "shares": {"extractive": 0.4, "mcq": 0.3, "free_form": 0.3}})


def test_mixture_refuses_category_mismatch():
    with pytest.raises(ValueError, match="is 'free_form' but mixture.shares has no 'free_form'"):
        _phase(mixture={"total_examples": 400, "shares": {"extractive": 0.5, "mcq": 0.5}})
    with pytest.raises(ValueError, match="names 'free_form' but no listed dataset"):
        _phase(datasets=["tydiqa_arabic", "arabic_squad_mcq"])
    with pytest.raises(ValueError, match="names datasets the phase does not list"):
        _phase(mixture={"total_examples": 400, "shares": {"extractive": 0.4, "mcq": 0.3, "free_form": 0.3},
                        "weights": {"bactrian_x_ar": 1}})
    with pytest.raises(ValueError, match="duplicates"):
        _phase(datasets=["tydiqa_arabic", "tydiqa_arabic", "arabic_squad_mcq", "cidar"])


def test_mixture_shares_and_fields_validated():
    with pytest.raises(ValueError, match="sum to 1"):
        MixtureConfig(total_examples=10, shares={"extractive": 0.5, "mcq": 0.4})
    with pytest.raises(ValueError, match="positive"):
        MixtureConfig(total_examples=0, shares={"extractive": 1.0})
    with pytest.raises(ValueError, match="positive"):
        MixtureConfig(total_examples=10, shares={"extractive": 1.0}, weights={"arcd": 0})
    with pytest.raises(ValueError):
        MixtureConfig(total_examples=10, shares={"other": 1.0})


def test_mixture_refused_on_pretraining_mix_phase():
    with pytest.raises(ValueError, match="only valid on a QA phase"):
        PhaseConfig(datasets=["pretraining_mix"], trainable_parameters=["*"], learning_rate=1e-4, batch_size=4,
                    loss_target="full_sequence", mix_tokens=4096,
                    mixture={"total_examples": 400, "shares": {"extractive": 1.0}})


def test_phase_without_mixture_still_requires_steps():
    with pytest.raises(ValueError, match="steps is required"):
        PhaseConfig(datasets=["arcd"], trainable_parameters=["*"], learning_rate=1e-4, batch_size=4)


def test_corpus_params_refuses_pretraining_mix():
    cfg = load_config("configs/base.yaml")
    with pytest.raises(ValueError, match="pretraining_mix"):
        TrainingConfig(phases=cfg.training.phases, corpus_params={"pretraining_mix": {}})
    ok = TrainingConfig(phases=cfg.training.phases, corpus_params={"aya_ar": {"include_datasets": ["Aya-Dataset"]}})
    assert ok.corpus_params["aya_ar"] == {"include_datasets": ["Aya-Dataset"]}


def test_reference_mixture_config_loads():
    cfg = load_config("configs/experiments/native_llama_3phase_sft_mixture.yaml", base_path="configs/base.yaml")
    sft = cfg.training.phases.sft
    assert cfg.training.corpus_params["aya_ar"]["include_datasets"] == ["Aya-Dataset", "Dolly-v2 (T)"]
    assert sft.mixture is not None and sft.mixture.total_examples == 30000
    assert sft.steps == 30000 // sft.batch_size
    assert set(sft.datasets) == {"tydiqa_arabic", "arcd", "arabic_squad_mcq", "cidar", "bactrian_x_ar", "aya_ar"}


# --------------------------------------------------------------------------
# Free-form loaders on fixture rows
# --------------------------------------------------------------------------

def test_cidar_loader_normalizes_and_dedups(monkeypatch):
    rows = [
        {"index": 1, "instruction": " اكتب قصيدة ", "output": "بيت\r\nبيت آخر\r\n"},
        {"index": 2, "instruction": "لخص", "output": "ملخص"},
        {"index": 2, "instruction": "لخص", "output": "ملخص"},          # exact duplicate → dropped
        {"index": 1, "instruction": "شيء آخر", "output": "جواب"},      # repeated index, new content → suffixed id
        {"index": 3, "instruction": "", "output": "جواب"},             # empty instruction → dropped
    ]

    class _DS(list):
        pass
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: _DS(rows))
    recs = load_corpus("cidar", "train")
    assert [r.id for r in recs] == ["cidar-1", "cidar-2", "cidar-1-3"]
    assert recs[0].question == "اكتب قصيدة" and recs[0].answer == "بيت\nبيت آخر" and recs[0].context == ""
    assert all(r.prompt_template == "instruction" and r.source == "cidar" for r in recs)
    with pytest.raises(ValueError, match="no 'validation' split"):
        load_corpus("cidar", "validation")


def test_bactrian_loader_reads_gzip_json_and_handles_null_input(tmp_path, monkeypatch):
    rows = [
        {"instruction": "قم بإنشاء قائمة", "input": "", "id": "alpaca-1", "output": "1- أ\n2- ب"},
        {"instruction": "اذكر العدد", "input": "شركة كوستكو", "id": "dolly-2", "output": "584"},
        {"instruction": "س", "input": None, "id": "alpaca-3", "output": "ج"},
        {"instruction": "س", "input": None, "id": "alpaca-4", "output": ""},   # empty output → dropped
    ]
    path = tmp_path / "ar.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    seen = {}

    def fake_download(repo, filename, repo_type=None, revision=None):
        seen.update(repo=repo, filename=filename, revision=revision)
        return str(path)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    recs = load_corpus("bactrian_x_ar", "train")
    assert seen == {"repo": "MBZUAI/Bactrian-X", "filename": "data/ar.json.gz", "revision": PINNED_REVISIONS["bactrian_x_ar"]}
    assert [r.id for r in recs] == ["bactrian-alpaca-1", "bactrian-dolly-2", "bactrian-alpaca-3"]
    assert recs[1].context == "شركة كوستكو" and recs[2].context == ""
    assert f"{fc.INPUT_LABEL}\nشركة كوستكو\n" in _format_qa_prompt(recs[1])


def test_aya_loader_splits_context_label_and_checks_allowlist(monkeypatch):
    import pyarrow as pa
    table = pa.table({
        "id": [1, 2, 3],
        "inputs": ["متى بدأت الشركة؟\nContext:الشركة هي شركة طيران.", "كم عدد الخلفاء؟", "سؤال بلا جواب"],
        "targets": ["في عام 2000.", "أربعة.", ""],
        "dataset_name": ["Dolly-v2 (T)", "Aya-Dataset", "Aya-Dataset"],
        "script": ["Arab", "Arab", "Arab"],
    })
    seen = {}

    def fake_subset(cache_path, include):
        seen["include"] = list(include)
        seen["cache"] = cache_path
        return table
    monkeypatch.setattr(fc, "_aya_cached_subset", fake_subset)
    recs = load_corpus("aya_ar", "train")
    assert seen["include"] == list(fc.AYA_DEFAULT_INCLUDE) and seen["cache"].suffix == ".parquet"
    assert [r.id for r in recs] == ["aya-1", "aya-2"]
    assert recs[0].question == "متى بدأت الشركة؟" and recs[0].context == "الشركة هي شركة طيران."
    assert "Context" not in _format_qa_full(recs[0])
    recs = load_corpus("aya_ar", "train", include_datasets=["Aya-Dataset"])
    assert seen["include"] == ["Aya-Dataset"]
    with pytest.raises(ValueError, match="matched no Arab-script row"):
        load_corpus("aya_ar", "train", include_datasets=["Aya-Dataset", "Nope"])
    with pytest.raises(ValueError, match="at least one"):
        load_corpus("aya_ar", "train", include_datasets=[])


def test_aya_cache_fingerprint_depends_on_allowlist(monkeypatch):
    import pyarrow as pa
    table = pa.table({"id": [1, 2], "inputs": ["س", "س"], "targets": ["ج", "ج"],
                      "dataset_name": ["Aya-Dataset", "Dolly-v2 (T)"], "script": ["Arab", "Arab"]})
    paths = []
    def fake_subset(cache_path, include):
        paths.append(cache_path)
        return table
    monkeypatch.setattr(fc, "_aya_cached_subset", fake_subset)
    load_corpus("aya_ar", "train", include_datasets=["Aya-Dataset"])
    load_corpus("aya_ar", "train", include_datasets=["Aya-Dataset"])
    load_corpus("aya_ar", "train", include_datasets=["Aya-Dataset", "Dolly-v2 (T)"])
    assert paths[0] == paths[1] != paths[2]


# --------------------------------------------------------------------------
# Early-stop eval mixture (added 2026-09-22)
# --------------------------------------------------------------------------

class TestEvalMixture:
    """``early_stopping.eval_mixture`` composes the stop signal from the dev
    pools of the phase's own corpora, at the training mixture's ratio."""

    def test_attach_category_tags_every_encoding(self):
        tok = _WordTok()
        encs, _man = compose_mixture(_mix(total_examples=100), DATASETS, POOLS, tok, 128, "answer_only",
                                     batch_size=4, attach_category=True)
        assert len(encs) == 100
        cats = [e["_category"] for e in encs]
        assert set(cats) == {"extractive", "mcq", "free_form"}
        assert cats.count("extractive") == 40 and cats.count("mcq") == 30 and cats.count("free_form") == 30

    def test_without_attach_category_entries_are_untouched(self):
        tok = _WordTok()
        encs, _ = compose_mixture(_mix(total_examples=100), DATASETS, POOLS, tok, 128, "answer_only", batch_size=4)
        assert all("_category" not in e for e in encs)

    def test_composing_from_dev_pools_honours_the_shares(self):
        """The eval mixture is the same function over smaller (dev) pools."""
        dev_pools = {k: v[: max(4, len(v) // 20)] for k, v in POOLS.items()}
        tok = _WordTok()
        encs, man = compose_mixture(_mix(total_examples=20), list(dev_pools), dev_pools, tok, 128,
                                    "answer_only", batch_size=4, attach_category=True)
        assert len(encs) == 20
        assert man["categories"]["extractive"]["kept"] == 8
        assert man["categories"]["mcq"]["kept"] == 6
        assert man["categories"]["free_form"]["kept"] == 6

    def test_a_dev_pool_too_small_is_an_error_not_a_repeat(self):
        tiny = {k: v[:2] for k, v in POOLS.items()}
        tok = _WordTok()
        with pytest.raises(MixtureShortfallError):
            compose_mixture(_mix(total_examples=100, upsample=False), list(tiny), tiny, tok, 128,
                            "answer_only", batch_size=4, attach_category=True)

    def test_load_mixture_pools_reads_the_requested_split(self, monkeypatch):
        seen = []

        def fake_load_corpus(name, split, **kw):
            seen.append((name, split))
            return POOLS[name][:5]
        monkeypatch.setattr("arabic_eval.data.sft_mixture.load_corpus", fake_load_corpus)
        pools, before = load_mixture_pools(["cidar", "arcd"], None, False, None, split="dev")
        assert seen == [("cidar", "dev"), ("arcd", "dev")]
        assert set(pools) == {"cidar", "arcd"} and before["cidar"] == 5

    def test_eval_mixture_is_seed_deterministic(self):
        tok = _WordTok()
        a, _ = compose_mixture(_mix(total_examples=40, seed=7), DATASETS, POOLS, tok, 128, "answer_only",
                               batch_size=4, attach_category=True)
        b, _ = compose_mixture(_mix(total_examples=40, seed=7), DATASETS, POOLS, tok, 128, "answer_only",
                               batch_size=4, attach_category=True)
        c, _ = compose_mixture(_mix(total_examples=40, seed=9), DATASETS, POOLS, tok, 128, "answer_only",
                               batch_size=4, attach_category=True)
        assert [e["input_ids"] for e in a] == [e["input_ids"] for e in b]
        assert [e["input_ids"] for e in a] != [e["input_ids"] for e in c]
