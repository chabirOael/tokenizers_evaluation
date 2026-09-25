"""``rows_file`` on the LightEval MCQ tasks (2026-09-25): score a listed subset.

The two properties that make a subset eval comparable with the full one, on a
tiny random Llama and a fixture task with two sub-configs and 2-shot prompts:

  (a) a row's few-shot demonstrations are byte-identical to the full run's —
      the full list is still loaded and cached, the demo pools and the seed
      ``self.seed + index in the cached list`` are unchanged, only the rows
      iterated change;
  (b) ``row_index`` in the dump is the index in the full list, so the subset
      dump equals the full dump's rows at those indices in every column.

Plus every validation: missing file, a file of another task, an index out of
range, a duplicate, and ``rows_file`` together with ``num_eval_samples``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

import arabic_eval.models  # noqa: E402,F401 — registers adapters
from arabic_eval.evaluation.eval_rows import EvalRowFile  # noqa: E402
from arabic_eval.models.llama_adapter import LlamaAdapter  # noqa: E402
from arabic_eval.params_spec import validate_params  # noqa: E402
from arabic_eval.tasks.lighteval.base import LightEvalBenchmarkTask  # noqa: E402
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput  # noqa: E402

VOCAB = 96
PAD, BOS, EOS, UNK = 0, 1, 2, 3
WORDS = ("س1 س2 س3 س4 س5 س6 س7 س8 س9 س10 س11 س12 أ ب ج د الإجابة: "
         "ع1 ع2 ع3 ع4 ع5 ع6 ع7 ع8").split()


def _rows() -> List[Dict[str, Any]]:
    """Twelve rows over two sub-configs, 4 choices each (gold cycling over slots)."""
    rows = []
    for i in range(12):
        rows.append({
            "question": f"س{i + 1}",
            "choices": [f"ع{(i + j) % 8 + 1}" for j in range(4)],
            "answer": i % 4,
            "_source_config": "cfg_a" if i % 2 == 0 else "cfg_b",
        })
    return rows


class _Tok(BaseTokenizer):
    """Whitespace word tokenizer that appends ``</s>`` like the from-scratch ones."""

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


class _FixtureTask(LightEvalBenchmarkTask):
    @classmethod
    def _default_dataset_name(cls) -> str:
        return "fixture/rows"

    @property
    def name(self) -> str:
        return "fixture_bench"

    def _parse_example(self, raw):
        return raw

    def load_examples(self):
        return _rows()

    def _format_eval_context(self, ex):
        opts = " ".join(f"{l} {c}" for l, c in zip("أبجد", ex["choices"]))
        return f"{ex['question']} {opts} الإجابة:"

    def _build_continuations(self, ex):
        return [f" {l}" for l in "أبجد"[: len(ex["choices"])]]

    def _aggregate_scores(self, ex, continuations, log_likelihoods,
                          unconditioned_log_likelihoods=None, normalization="char"):
        if normalization == "pmi":
            return [ll - u for ll, u in zip(log_likelihoods, unconditioned_log_likelihoods)]
        return list(log_likelihoods)


@pytest.fixture(scope="module")
def adapter_and_tok(tmp_path_factory):
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, tie_word_embeddings=True,
        bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD,
    )
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_llama_rows_file")
    LlamaForCausalLM(cfg).save_pretrained(path)
    tok = _Tok()
    adapter = LlamaAdapter(str(path), device="cpu", dtype="float32")
    adapter.adapt_to_tokenizer(tok)
    adapter.model.eval()
    return adapter, tok


def _write_rows_file(path: Path, idx: List[int], task: str = "fixture_bench") -> Path:
    path.write_text(json.dumps({"task": task, "row_index": idx, "provenance": {"test": True}}),
                    encoding="utf-8")
    return path


def _dump_by_row(path: Path) -> Dict[int, Dict[str, Any]]:
    f = EvalRowFile(path)
    import pyarrow.parquet as pq
    recs = pq.read_table(path).to_pylist()
    assert len(recs) == f.n_rows
    return {r["row_index"]: r for r in recs}


def test_subset_dump_equals_the_full_dump_at_those_rows(adapter_and_tok, tmp_path):
    adapter, tok = adapter_and_tok
    full_dir, sub_dir = tmp_path / "full", tmp_path / "sub"
    cfg = {"num_fewshot": 2, "max_length": 256}
    full = _FixtureTask(cfg).evaluate(adapter, tok, row_dump_dir=full_dir, score_normalization="char+pmi")
    assert full["num_samples"] == 12 and "rows_file" not in full

    idx = [7, 2, 10, 5]
    rf = _write_rows_file(tmp_path / "rows.json", idx)
    task = _FixtureTask({**cfg, "rows_file": str(rf)})
    sub = task.evaluate(adapter, tok, row_dump_dir=sub_dir, score_normalization="char+pmi")
    assert sub["num_samples"] == len(idx)
    assert sub["rows_file"]["n_rows"] == 4 and sub["rows_file"]["path"] == str(rf)
    # The full list is still what the task caches (few-shot pools come from it).
    assert len(task._cached_examples) == 12

    a = _dump_by_row(full_dir / "fixture_bench.parquet")
    b = _dump_by_row(sub_dir / "fixture_bench.parquet")
    assert sorted(b) == sorted(idx)
    import pyarrow.parquet as pq
    assert [r["row_index"] for r in pq.read_table(sub_dir / "fixture_bench.parquet").to_pylist()] == idx
    for i in idx:
        # Every column: prompt (the 2-shot demos included), continuations, ll,
        # cont_tokens, cont_token_ll, scores, flags.
        assert b[i] == a[i], i
        assert b[i]["prompt"].count("الإجابة:") == 3          # two demos + the row

    meta = EvalRowFile(sub_dir / "fixture_bench.parquet").metadata
    assert meta["rows_file"]["n_rows"] == 4 and len(meta["rows_file"]["sha256"]) == 64
    assert EvalRowFile(full_dir / "fixture_bench.parquet").metadata["rows_file"] is None


def test_subset_demos_are_the_full_runs(adapter_and_tok, tmp_path):
    """The seed is ``seed + index in the cached (full) list``, not a position in the subset."""
    rf = _write_rows_file(tmp_path / "rows.json", [9, 4])
    full_task = _FixtureTask({"num_fewshot": 2})
    sub_task = _FixtureTask({"num_fewshot": 2, "rows_file": str(rf)})
    full_ex = full_task.get_eval_examples()
    sub_ex, sub_idx = sub_task._select_rows(sub_task.get_eval_examples(), None)
    assert sub_idx == [9, 4]
    for ex, i in zip(sub_ex, sub_idx):
        assert ex is sub_task._cached_examples[i]
        assert sub_task._format_eval_context_with_fewshot(ex) == \
            full_task._format_eval_context_with_fewshot(full_ex[i])


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _FixtureTask({"rows_file": str(tmp_path / "nope.json")})


def test_file_of_another_task_is_an_error(tmp_path):
    rf = _write_rows_file(tmp_path / "rows.json", [0, 1], task="arabic_exam")
    with pytest.raises(ValueError, match="not 'fixture_bench'"):
        _FixtureTask({"rows_file": str(rf)})


def test_duplicate_index_is_an_error(tmp_path):
    rf = _write_rows_file(tmp_path / "rows.json", [3, 1, 3])
    with pytest.raises(ValueError, match="duplicate row_index"):
        _FixtureTask({"rows_file": str(rf)})


@pytest.mark.parametrize("bad", [[], [-1], ["2"], [1.0], [True]])
def test_malformed_index_list_is_an_error(tmp_path, bad):
    rf = _write_rows_file(tmp_path / "rows.json", bad)
    with pytest.raises(ValueError, match="row_index"):
        _FixtureTask({"rows_file": str(rf)})


def test_index_outside_the_list_is_an_error(adapter_and_tok, tmp_path):
    adapter, tok = adapter_and_tok
    rf = _write_rows_file(tmp_path / "rows.json", [0, 12])
    with pytest.raises(ValueError, match=r"row_index \[12\] outside the 12-row eval list"):
        _FixtureTask({"rows_file": str(rf)}).evaluate(adapter, tok)


def test_rows_file_with_num_eval_samples_is_an_error(adapter_and_tok, tmp_path):
    adapter, tok = adapter_and_tok
    rf = _write_rows_file(tmp_path / "rows.json", [0, 1])
    with pytest.raises(ValueError, match=r"rows_file .* and evaluation\.num_eval_samples \(5\) are both set"):
        _FixtureTask({"rows_file": str(rf)}).evaluate(adapter, tok, max_samples=5)


def test_rows_file_is_declared_on_every_lighteval_task():
    import arabic_eval.tasks  # noqa: F401 — registers the tasks
    from arabic_eval.registry import task_registry
    for key in ("acva", "alghafa", "arabic_exam", "culture_arabic_mmlu"):
        cls = task_registry.get(key)
        spec = {s.name: s for s in cls.param_spec()}
        s = spec["rows_file"]
        assert (s.type, s.default, s.nullable, s.advanced) == ("path", None, True, True)
        assert validate_params(cls.param_spec(), {"rows_file": "x.json"}, owner=key) == []
