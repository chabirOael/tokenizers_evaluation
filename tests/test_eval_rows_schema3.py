"""Row-dump schema 3 (2026-09-25): ``cont_token_ll``, the per-token addends of each choice's ``ll``.

  * the file says ``schema_version`` 3 and carries the column;
  * each inner list has ``cont_tokens`` entries and sums to ``ll`` (float32);
  * a scorer that returns plain floats (a stub) leaves the column null, as
    ``cont_tokens`` does;
  * the read side (``EvalRowFile``: summary, pages) works on a schema-3 file
    and does not project the new column into a page.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

import arabic_eval.models  # noqa: E402,F401 — registers adapters
from arabic_eval.evaluation.eval_rows import (  # noqa: E402
    PAGE_COLUMNS,
    SCHEMA_VERSION,
    EvalRowFile,
    build_row_record,
)
from arabic_eval.models.llama_adapter import LlamaAdapter  # noqa: E402
from arabic_eval.tasks.lighteval.base import LightEvalBenchmarkTask, LightEvalModelWrapper  # noqa: E402
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput  # noqa: E402

VOCAB = 64
PAD, BOS, EOS, UNK = 0, 1, 2, 3
WORDS = "ذهب الولد إلى الطاولة المدرسة الكتاب أ ب ج الإجابة:".split()


class _Tok(BaseTokenizer):
    def __init__(self) -> None:
        self._w2i = {w: i + 4 for i, w in enumerate(WORDS)}

    def train(self, texts, vocab_size, **kw):  # pragma: no cover
        pass

    def encode(self, text, max_length=None, padding=False, truncation=False) -> TokenizerOutput:
        ids = [BOS] + [self._w2i.get(w, UNK) for w in text.split()] + [EOS]
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


class _Task(LightEvalBenchmarkTask):
    ROWS = [
        {"question": "ذهب الولد", "choices": ["الطاولة", "المدرسة الكتاب", "أ ب ج"], "answer": 0},
        {"question": "الولد إلى", "choices": ["الكتاب", "الطاولة المدرسة", "ب"], "answer": 2},
    ]

    @classmethod
    def _default_dataset_name(cls) -> str:
        return "stub/schema3"

    @property
    def name(self) -> str:
        return "schema3_bench"

    def _parse_example(self, raw):
        return raw

    def load_examples(self):
        return [{**r, "_source_config": "_default"} for r in self.ROWS]

    def _format_eval_context(self, ex):
        return f"{ex['question']} الإجابة:"

    def _build_continuations(self, ex):
        return [f" {c}" for c in ex["choices"]]

    def _aggregate_scores(self, ex, continuations, log_likelihoods,
                          unconditioned_log_likelihoods=None, normalization="char"):
        if normalization == "pmi":
            return [ll - u for ll, u in zip(log_likelihoods, unconditioned_log_likelihoods)]
        return [ll / max(len(c.lstrip()), 1) for c, ll in zip(continuations, log_likelihoods)]


@pytest.fixture(scope="module")
def adapter(tmp_path_factory):
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, tie_word_embeddings=True,
        bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD,
    )
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_llama_schema3")
    LlamaForCausalLM(cfg).save_pretrained(path)
    a = LlamaAdapter(str(path), device="cpu", dtype="float32")
    a.adapt_to_tokenizer(_Tok())
    a.model.eval()
    return a


def test_schema_3_dump_carries_the_addends(adapter, tmp_path):
    _Task({"max_length": 64}).evaluate(adapter, _Tok(), row_dump_dir=tmp_path, score_normalization="char+pmi")
    f = EvalRowFile(tmp_path / "schema3_bench.parquet")
    assert f.metadata["schema_version"] == SCHEMA_VERSION == 3
    assert "cont_token_ll" in f.columns
    import pyarrow.parquet as pq
    recs = pq.read_table(f.path).to_pylist()
    assert [r["cont_tokens"] for r in recs] == [[1, 2, 3], [1, 2, 1]]
    for r in recs:
        assert [len(x) for x in r["cont_token_ll"]] == r["cont_tokens"]
        for addends, ll in zip(r["cont_token_ll"], r["ll"]):
            assert all(v <= 0 for v in addends)
            assert abs(float(np.float32(sum(addends))) - ll) < 1e-5


def test_plain_float_scorer_leaves_the_column_null(adapter, tmp_path, monkeypatch):
    def plain(self, requests: List) -> List[float]:
        return [-float(len(cont)) for _, cont in requests]

    monkeypatch.setattr(LightEvalModelWrapper, "loglikelihood", plain)
    _Task({"max_length": 64}).evaluate(adapter, _Tok(), row_dump_dir=tmp_path, score_normalization="char")
    import pyarrow.parquet as pq
    recs = pq.read_table(tmp_path / "schema3_bench.parquet").to_pylist()
    assert recs and all(r["cont_token_ll"] is None and r["cont_tokens"] is None for r in recs)


def test_build_row_record_passes_the_addends_through():
    base = dict(row_index=0, example={"question": "q", "choices": ["a", "b"]}, prompt="p",
                continuations=[" a", " b"], log_likelihoods=[-1.0, -2.5], scores_char=None,
                scores_pmi=None, unconditioned_log_likelihoods=None, gold_idx=0, pred_idx=0,
                pred_idx_char=None, pred_idx_pmi=None)
    assert build_row_record(**base)["cont_token_ll"] is None
    rec = build_row_record(**base, cont_tokens=[1, 2], cont_token_ll=[(-1.0,), (-2.0, -0.5)])
    assert rec["cont_token_ll"] == [[-1.0], [-2.0, -0.5]]
    assert build_row_record(**base, cont_tokens=[0, 0], cont_token_ll=[(), ()])["cont_token_ll"] == [[], []]


def test_reader_opens_a_schema_3_file(adapter, tmp_path):
    _Task({"max_length": 64}).evaluate(adapter, _Tok(), row_dump_dir=tmp_path, score_normalization="char+pmi")
    f = EvalRowFile(tmp_path / "schema3_bench.parquet")
    s = f.summary()
    assert s["n_rows"] == 2 and s["accuracy"] is not None
    assert "cont_token_ll" not in PAGE_COLUMNS
    row = f.rows([0, 1])[1]
    assert "cont_token_ll" not in row and row["cont_tokens"] == [1, 2, 1]
    assert f.describe()["metadata"]["schema_version"] == 3
