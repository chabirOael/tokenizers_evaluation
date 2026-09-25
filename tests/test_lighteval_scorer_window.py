"""The LightEval MCQ scorer's continuation window (fixed 2026-09-24).

``_compute_loglikelihood`` used to score ``full[len(context_ids):]``, which
assumed the context encoding is a token prefix of the context + continuation
encoding. Every from-scratch tokenizer appends ``</s>`` to a standalone
encoding, so that window skipped the first continuation token and scored the
trailing ``</s>`` (a single-piece letter under BPE: ``</s>`` alone). The
tests before this file checked accuracies and dumps, never *which* tokens
were summed — this file pins that, against a tiny random Llama:

  * the scored ids are the continuation's own encoding minus specials, for a
    tokenizer that appends BOS / EOS and for one that appends nothing, for a
    one-token and a multi-token continuation;
  * for the tokenizer that appends nothing the value is byte-identical to the
    old window (the fix is a no-op for the native wrappers);
  * the sentinel when truncation leaves no room, under both truncation styles
    (AraRooPat cuts after the EOS; HF ``tokenizers`` keeps the EOS);
  * the CharacterBERT ``char_ids`` branch;
  * ``cont_tokens`` in the row dump (schema 2), and old dumps stay readable;
  * ``token_logprobs`` (schema 3, ``cont_token_ll``): one addend per scored
    token, summing to the value, empty for the sentinel.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

import arabic_eval.models  # noqa: E402,F401 — registers adapters
from arabic_eval.data.answer_only_masking import (  # noqa: E402
    common_prefix_len,
    compute_answer_only_labels,
    continuation_start,
    strip_trailing_eos,
)
from arabic_eval.evaluation.eval_rows import (  # noqa: E402
    SCHEMA_VERSION,
    SENTINEL_LL,
    EvalRowFile,
    build_row_record,
    eval_row_schema,
)
from arabic_eval.models.llama_adapter import LlamaAdapter  # noqa: E402
from arabic_eval.tasks.lighteval.base import (  # noqa: E402
    LightEvalBenchmarkTask,
    ScoredLogLikelihood,
    _compute_loglikelihood,
)
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput  # noqa: E402
from arabic_eval.tokenizers.character_bert import CharacterBERTTokenizer  # noqa: E402
from arabic_eval.tools.eval_rows_browser import _flatten_for_csv  # noqa: E402

VOCAB = 128
PAD, BOS, EOS, UNK = 0, 1, 2, 3

TEXTS = [
    "الكتاب على الطاولة",
    "ذهب الولد إلى المدرسة",
    "يدرس الطلاب اللغة العربية",
    "كتبت الطالبة رسالة طويلة",
    "في المكتبة كتب كثيرة",
    "قرأت مقالة عن التاريخ",
    "السؤال الإجابة: أ ب ج د",
]

CTX = "ذهب الولد إلى"
ONE = " الطاولة"            # one word = one token
MULTI = " الطاولة المدرسة"  # two tokens


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_llama_path(tmp_path_factory) -> Path:
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, tie_word_embeddings=True,
        bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD,
    )
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_llama")
    LlamaForCausalLM(cfg).save_pretrained(path)
    return path


class _WordTok(BaseTokenizer):
    """Whitespace word tokenizer over a fixed vocab.

    ``specials=True`` mirrors every from-scratch tokenizer (``<s>`` words
    ``</s>`` on every standalone encoding); ``specials=False`` mirrors native
    Qwen (nothing added). ``keep_eos_on_truncation`` picks the truncation
    style: False = AraRooPat / CharacterBERT (cut the list, the EOS goes),
    True = HF ``tokenizers`` with a post-processor (the EOS survives).
    """

    def __init__(self, specials: bool, keep_eos_on_truncation: bool = False) -> None:
        self.specials = specials
        self.keep_eos = keep_eos_on_truncation
        self._w2i: Dict[str, int] = {}
        for t in TEXTS:
            for w in t.split():
                if w not in self._w2i:
                    self._w2i[w] = len(self._w2i) + 4

    def train(self, texts, vocab_size, **kw):  # pragma: no cover
        pass

    def word_ids(self, text: str) -> List[int]:
        return [self._w2i.get(w, UNK) for w in text.split()]

    def encode(self, text, max_length=None, padding=False, truncation=False) -> TokenizerOutput:
        body = self.word_ids(text)
        if self.specials:
            if truncation and max_length and self.keep_eos:
                body = body[: max(max_length - 2, 0)]
            ids = [BOS] + body + [EOS]
        else:
            ids = body
        if truncation and max_length:
            ids = ids[:max_length]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids))

    def decode(self, ids):  # pragma: no cover
        return ""

    def save(self, path):  # pragma: no cover
        pass

    def load(self, path):  # pragma: no cover
        pass

    @property
    def vocab_size(self) -> int:
        return VOCAB

    @property
    def embedding_type(self) -> str:
        return EmbeddingType.STANDARD

    @property
    def special_tokens(self) -> Dict[str, int]:
        return {"pad_token": PAD, "bos_token": BOS, "eos_token": EOS, "unk_token": UNK}


def _adapter(path: Path, tok: BaseTokenizer) -> LlamaAdapter:
    adapter = LlamaAdapter(str(path), device="cpu", dtype="float32")
    adapter.adapt_to_tokenizer(tok)
    adapter.model.eval()
    return adapter


@torch.no_grad()
def _log_probs(adapter: LlamaAdapter, ids: List[int]) -> torch.Tensor:
    t = torch.tensor([ids])
    out = adapter.forward({"input_ids": t, "attention_mask": torch.ones_like(t), "labels": t.clone()})
    return F.log_softmax(out["logits"][0], dim=-1)


def _sum_window(lp: torch.Tensor, ids: List[int], start: int, end: int) -> float:
    """Σ log P(ids[start:end]) — the scorer's own summation order."""
    total = 0.0
    for pos in range(start - 1, end - 1):
        total += lp[pos, ids[pos + 1]].item()
    return total


def _old_window_ll(adapter, tok, context, continuation, max_length=512) -> float:
    """The pre-fix scorer, verbatim: ``full[len(context_ids):]`` on the raw encodings."""
    ctx = tok.encode(context, max_length=max_length, truncation=True).input_ids
    full = tok.encode(context + continuation, max_length=max_length, truncation=True).input_ids
    if len(full) <= len(ctx):
        return SENTINEL_LL
    return _sum_window(_log_probs(adapter, full), full, len(ctx), len(full))


# --------------------------------------------------------------------------
# 1. The shared helpers
# --------------------------------------------------------------------------

class TestContinuationStart:
    def test_eos_appending_encodings_meet_at_the_continuation(self):
        full, k = continuation_start([1, 5, 6, 2], [1, 5, 6, 7, 8, 2], eos_id=2)
        assert full == [1, 5, 6, 7, 8] and k == 3 and full[k:] == [7, 8]

    def test_nothing_appended_is_the_plain_prefix(self):
        full, k = continuation_start([5, 6], [5, 6, 7], eos_id=2)
        assert full == [5, 6, 7] and k == 2

    def test_truncated_full_without_eos(self):
        # AraRooPat-style truncation cut the EOS off the full encoding only.
        full, k = continuation_start([1, 5, 6, 2], [1, 5, 6, 7], eos_id=2)
        assert full[k:] == [7]

    def test_no_eos_id_strips_nothing(self):
        full, k = continuation_start([1, 5, 2], [1, 5, 7, 2], eos_id=None)
        assert full == [1, 5, 7, 2] and k == 2

    def test_helpers(self):
        assert common_prefix_len([1, 2, 3], [1, 2, 4]) == 2
        assert strip_trailing_eos([1, 5, 2], 2) == [1, 5]
        assert strip_trailing_eos([1, 5], 2) == [1, 5]
        # The training labels still use the same prefix rule (EOS stays a target there).
        assert compute_answer_only_labels([1, 5, 2], [1, 5, 7, 2]) == [-100, -100, 7, 2]


# --------------------------------------------------------------------------
# 2. Which tokens are scored
# --------------------------------------------------------------------------

@pytest.mark.parametrize("specials", [True, False], ids=["appends_eos", "appends_nothing"])
@pytest.mark.parametrize("continuation", [ONE, MULTI], ids=["one_token", "multi_token"])
def test_scored_ids_are_the_continuations_own(tiny_llama_path, specials, continuation):
    tok = _WordTok(specials=specials)
    adapter = _adapter(tiny_llama_path, tok)
    cont_ids = [i for i in tok.encode(continuation).input_ids if i not in (BOS, EOS, PAD)]
    assert cont_ids == tok.word_ids(continuation) and UNK not in cont_ids

    ll = _compute_loglikelihood(adapter, tok, CTX, continuation)
    assert isinstance(ll, ScoredLogLikelihood) and isinstance(ll, float)
    assert ll.n_tokens == len(cont_ids)
    assert not ll.truncated

    # Independently: the full encoding without its trailing EOS ends in exactly
    # the continuation's ids; sum their log-probs at their own positions.
    full = strip_trailing_eos(tok.encode(CTX + continuation).input_ids, EOS)
    assert full[-len(cont_ids):] == cont_ids
    expected = _sum_window(_log_probs(adapter, full), full, len(full) - len(cont_ids), len(full))
    assert float(ll) == expected

    if specials:
        # The defect this file exists for: the old window skipped the first
        # continuation token and summed log P(</s>) instead.
        raw_full = tok.encode(CTX + continuation).input_ids
        raw_ctx = tok.encode(CTX).input_ids
        assert raw_full[len(raw_ctx):] == cont_ids[1:] + [EOS]
        assert _old_window_ll(adapter, tok, CTX, continuation) != float(ll)


@pytest.mark.parametrize("context,continuation", [
    (CTX, ONE), (CTX, MULTI), ("الكتاب على", " الطاولة"),
    ("السؤال الإجابة:", " أ"), ("السؤال الإجابة:", " ب ج"),
])
def test_no_eos_tokenizer_value_is_byte_identical_to_the_old_window(tiny_llama_path, context, continuation):
    tok = _WordTok(specials=False)
    adapter = _adapter(tiny_llama_path, tok)
    new = _compute_loglikelihood(adapter, tok, context, continuation)
    old = _old_window_ll(adapter, tok, context, continuation)
    assert float(new) == old  # exact, not approx: same ids, same forward, same order


# --------------------------------------------------------------------------
# 3. Truncation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("keep_eos", [False, True], ids=["cut_style", "hf_keep_eos"])
def test_sentinel_when_the_context_fills_the_window(tiny_llama_path, keep_eos):
    tok = _WordTok(specials=True, keep_eos_on_truncation=keep_eos)
    adapter = _adapter(tiny_llama_path, tok)
    ctx = "يدرس الطلاب اللغة العربية كتبت الطالبة"   # 6 words + BOS/EOS
    ll = _compute_loglikelihood(adapter, tok, ctx, ONE, max_length=5)
    assert float(ll) == SENTINEL_LL
    assert ll.n_tokens == 0 and ll.truncated


def test_sentinel_without_specials(tiny_llama_path):
    tok = _WordTok(specials=False)
    adapter = _adapter(tiny_llama_path, tok)
    ll = _compute_loglikelihood(adapter, tok, "يدرس الطلاب اللغة العربية", ONE, max_length=4)
    assert float(ll) == SENTINEL_LL and ll.n_tokens == 0 and ll.truncated


def test_cap_cutting_the_continuation_scores_what_is_left(tiny_llama_path):
    tok = _WordTok(specials=True)                     # cut style
    adapter = _adapter(tiny_llama_path, tok)
    # BOS + 3 context words + 2 continuation words = 6 before the EOS; cap 5.
    ll = _compute_loglikelihood(adapter, tok, CTX, MULTI, max_length=5)
    assert ll.n_tokens == 1 and ll.truncated
    full = tok.encode(CTX + MULTI, max_length=5, truncation=True).input_ids
    assert full[-1] == tok.word_ids(MULTI)[0]
    assert float(ll) == _sum_window(_log_probs(adapter, full), full, 4, 5)


def test_empty_shared_prefix_is_the_sentinel(tiny_llama_path):
    tok = _WordTok(specials=False)
    adapter = _adapter(tiny_llama_path, tok)
    ll = _compute_loglikelihood(adapter, tok, "", ONE)
    assert float(ll) == SENTINEL_LL and ll.n_tokens == 0


# --------------------------------------------------------------------------
# 3b. Per-token log-probabilities (schema 3)
# --------------------------------------------------------------------------

THREE = " الطاولة المدرسة الكتاب"   # three tokens, like [LIT_BEGIN] [CHAR_x] [LIT_END]


@pytest.mark.parametrize("specials,continuation", [
    (False, ONE), (False, MULTI), (True, ONE), (True, THREE),
], ids=["no_eos_one", "no_eos_two", "eos_one", "eos_three"])
def test_token_logprobs_are_the_addends(tiny_llama_path, specials, continuation):
    tok = _WordTok(specials=specials)
    adapter = _adapter(tiny_llama_path, tok)
    ll = _compute_loglikelihood(adapter, tok, CTX, continuation)
    assert isinstance(ll.token_logprobs, tuple)
    assert len(ll.token_logprobs) == ll.n_tokens == len(tok.word_ids(continuation))
    assert abs(sum(ll.token_logprobs) - float(ll)) < 1e-6
    # In the summation order, each addend is log P(token | its left context).
    full = strip_trailing_eos(tok.encode(CTX + continuation).input_ids, EOS)
    lp = _log_probs(adapter, full)
    start = len(full) - ll.n_tokens
    assert list(ll.token_logprobs) == [lp[p - 1, full[p]].item() for p in range(start, len(full))]


def test_token_logprobs_empty_for_the_sentinel(tiny_llama_path):
    tok = _WordTok(specials=True)
    adapter = _adapter(tiny_llama_path, tok)
    ll = _compute_loglikelihood(adapter, tok, "يدرس الطلاب اللغة العربية كتبت الطالبة", ONE, max_length=5)
    assert float(ll) == SENTINEL_LL
    assert ll.n_tokens == 0 and ll.token_logprobs == ()


def test_scored_float_default_has_no_addends():
    v = ScoredLogLikelihood(-1.5, 2, False)
    assert v.token_logprobs == () and float(v) == -1.5


# --------------------------------------------------------------------------
# 4. CharacterBERT (char_ids, word-level logits)
# --------------------------------------------------------------------------

def test_char_ids_branch_scores_the_continuation_words(tiny_llama_path):
    tok = CharacterBERTTokenizer(max_char_len=12)
    tok.train(TEXTS * 3, vocab_size=200)
    adapter = LlamaAdapter(str(tiny_llama_path), device="cpu", dtype="float32")
    adapter.adapt_to_tokenizer(tok)
    adapter.model.eval()
    eos = tok.special_tokens["eos_token"]

    for continuation, n_words in ((ONE, 1), (MULTI, 2)):
        ll = _compute_loglikelihood(adapter, tok, CTX, continuation)
        assert ll.n_tokens == n_words
        assert torch.isfinite(torch.tensor(float(ll)))
        enc = tok.encode(CTX + continuation)
        assert enc.input_ids[-1] == eos
        ids, chars = enc.input_ids[:-1], enc.char_ids[:-1]
        with torch.no_grad():
            out = adapter.forward({"char_ids": torch.tensor([chars]),
                                   "attention_mask": torch.ones(1, len(ids), dtype=torch.long)})
        lp = F.log_softmax(out["logits"][0], dim=-1)
        assert float(ll) == _sum_window(lp, ids, len(ids) - n_words, len(ids))
        assert len(ll.token_logprobs) == n_words
        assert abs(sum(ll.token_logprobs) - float(ll)) < 1e-6


# --------------------------------------------------------------------------
# 5. The row dump
# --------------------------------------------------------------------------

class _StubTask(LightEvalBenchmarkTask):
    def __init__(self, config: Dict[str, Any], rows: List[Dict[str, Any]]):
        super().__init__(config)
        self._rows = rows

    @classmethod
    def _default_dataset_name(cls) -> str:
        return "stub/rows"

    def _parse_example(self, raw):
        return raw

    def load_examples(self):
        return [{**ex, "_source_config": "_default"} for ex in self._rows]

    def _format_eval_context(self, ex):
        return f"{ex['question']} الإجابة:"

    def _build_continuations(self, ex):
        return [f" {c}" for c in ex["choices"]]

    def _aggregate_scores(self, ex, continuations, log_likelihoods,
                          unconditioned_log_likelihoods=None, normalization="char"):
        if normalization == "pmi":
            return [ll - u for ll, u in zip(log_likelihoods, unconditioned_log_likelihoods)]
        return [ll / max(len(c.lstrip()), 1) for c, ll in zip(continuations, log_likelihoods)]

    @property
    def name(self) -> str:
        return "stub_bench"


@pytest.mark.parametrize("specials", [True, False], ids=["appends_eos", "appends_nothing"])
def test_cont_tokens_lands_in_the_dump(tiny_llama_path, tmp_path, specials):
    tok = _WordTok(specials=specials)
    adapter = _adapter(tiny_llama_path, tok)
    rows = [
        {"question": "ذهب الولد", "choices": ["الطاولة", "المدرسة الكتاب", "أ"], "answer": 0},
        {"question": "يدرس الطلاب", "choices": ["العربية", "اللغة العربية", "ب"], "answer": 1},
    ]
    task = _StubTask({"max_length": 64}, rows=rows)
    task.evaluate(adapter, tok, row_dump_dir=tmp_path, score_normalization="char+pmi")
    f = EvalRowFile(tmp_path / "stub_bench.parquet")
    assert f.metadata["schema_version"] == SCHEMA_VERSION == 3
    recs = f.rows(range(f.n_rows))
    assert [r["cont_tokens"] for r in recs] == [[1, 2, 1], [1, 2, 1]]
    assert not any(r["hit_cap"] for r in recs)


def test_hit_cap_also_marks_a_cut_continuation():
    base = dict(row_index=0, example={"question": "q", "choices": ["a", "b"]}, prompt="p",
                continuations=[" a", " b"], log_likelihoods=[-1.0, -2.0], scores_char=None,
                scores_pmi=None, unconditioned_log_likelihoods=None, gold_idx=0, pred_idx=0,
                pred_idx_char=None, pred_idx_pmi=None, prompt_units=10, max_length=64)
    assert not build_row_record(**base)["hit_cap"]
    assert not build_row_record(**base, cont_tokens=[1, 1], cont_truncated=[False, False])["hit_cap"]
    rec = build_row_record(**base, cont_tokens=[1, 0], cont_truncated=[False, True])
    assert rec["hit_cap"] and rec["cont_tokens"] == [1, 0]


def test_schema_1_dumps_stay_readable(tmp_path):
    """A dump written before ``cont_tokens`` existed: rows, summary and the
    CSV flattening of the console all work, the missing column is absent."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = eval_row_schema({"task": "old", "schema_version": 1})
    old_schema = pa.schema([fld for fld in schema if fld.name != "cont_tokens"],
                           metadata=schema.metadata)
    rec = build_row_record(
        row_index=0, example={"question": "q", "choices": ["a", "b"]}, prompt="p",
        continuations=[" a", " b"], log_likelihoods=[-1.0, -2.0], scores_char=[-1.0, -2.0],
        scores_pmi=None, unconditioned_log_likelihoods=None, gold_idx=0, pred_idx=0,
        pred_idx_char=0, pred_idx_pmi=None, prompt_units=3, max_length=64,
    )
    rec.pop("cont_tokens")
    path = tmp_path / "old.parquet"
    pq.write_table(pa.Table.from_pylist([rec], schema=old_schema), path)
    f = EvalRowFile(path)
    assert "cont_tokens" not in f.columns
    row = f.rows([0])[0]
    assert "cont_tokens" not in row and row["ll"] == [-1.0, -2.0]
    assert f.summary()["n_rows"] == 1
    assert _flatten_for_csv(row)["cont_tokens"] is None
