"""End-to-end pipeline tests for the 3-phase training flow.

The full pipeline requires:
  - HuggingFace Hub access for the main Arabic corpus and the QA finetune
    corpora (Arabic-SQuAD / TyDiQA / ARCD).
  - A real Llama model checkpoint and a GPU.

Both are too heavy for unit tests, so this file mocks them out and
validates only the pipeline's *orchestration* invariants:

  1. ``run_experiment`` runs Phase 1 → Phase 2 → Phase 3 in order when all
     three are enabled, and produces an ``all_metrics.json`` with the
     expected top-level shape (config + intrinsic + training + downstream
     + mei).
  2. Setting ``sft.enabled=false`` skips ONLY Phase 3 (Phase 1 + Phase 2
     still run) — this is the "without SFT" experiment shape.
  3. Setting all three phase-enabled flags to false skips training
     entirely (pretrained-model evaluation path).
  4. Multiple LightEval tasks in ``sweep.tasks`` all get evaluated and
     each produces a downstream + MEI record.
  5. ``run_sweep`` iterates over multiple tokenizer cells and writes one
     subdirectory + ``all_metrics.json`` per cell.

The mocks are isolated to a single autouse fixture so future maintainers
don't need to repeat the patch boilerplate per test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.config import ExperimentConfig, load_config
from arabic_eval.data.finetune_corpora import QARecord
from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.pipeline.experiment import run_experiment, run_sweep
from arabic_eval.registry import model_registry, tokenizer_registry, task_registry
from arabic_eval.tasks.base import BaseTask
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput


# --------------------------------------------------------------------------
# Synthetic adapter / tokenizer / task — registered under test-only keys
# --------------------------------------------------------------------------

class _TinyLlamaLike(nn.Module):
    """Mirrors Llama's named-parameter shape (embed_tokens, layers, lm_head)."""
    def __init__(self, vocab: int = 64, hidden: int = 16) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([nn.Module() for _ in range(2)])
        for layer in self.model.layers:
            layer.self_attn = nn.Linear(hidden, hidden, bias=False)
            layer.mlp = nn.Linear(hidden, hidden, bias=False)
        self.model.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = x + layer.self_attn(x)
            x = x + layer.mlp(x)
        x = self.model.norm(x)
        return self.lm_head(x)


@model_registry.register("test_tiny")
class _TinyAdapter(BaseModelAdapter):
    """Test-only model adapter wrapping ``_TinyLlamaLike``."""
    def __init__(self, model_name_or_path: str = "tiny", device: str = "cpu", **kwargs):
        self._model = _TinyLlamaLike()
        self._device = torch.device("cpu")
        self._model.to(self._device)

    def adapt_to_tokenizer(self, tokenizer: BaseTokenizer) -> None:
        # No-op: vocab is pre-sized to 64; pipeline checks adapt_to_tokenizer fires.
        pass

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = self._model(batch["input_ids"])
        labels = batch["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return {"loss": loss, "logits": logits}

    def generate(self, *a, **k):
        raise NotImplementedError

    def get_trainable_parameters(self):
        return [p for p in self._model.parameters() if p.requires_grad]

    def save_checkpoint(self, path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), Path(path) / "model.pt")

    def load_checkpoint(self, path) -> None:
        self._model.load_state_dict(torch.load(Path(path) / "model.pt"))

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> nn.Module:
        return self._model


@tokenizer_registry.register("test_word")
class _WordTokenizer(BaseTokenizer):
    """Whitespace tokenizer with a deterministic small vocab; auto-EOS."""
    EOS = 63
    PAD = 0

    def __init__(self, **kwargs) -> None:
        self._vocab: dict[str, int] = {"<pad>": 0}

    def _enc(self, text: str) -> list[int]:
        ids = []
        for tok in text.split():
            if tok not in self._vocab and len(self._vocab) < 62:
                self._vocab[tok] = len(self._vocab)
            ids.append(self._vocab.get(tok, 1))  # 1 = unk
        ids.append(self.EOS)
        return ids

    def train(self, texts, vocab_size, **kwargs):
        for t in texts[:100]:
            self._enc(t)

    def encode(self, text, max_length=None, padding=False, truncation=False):
        ids = self._enc(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return TokenizerOutput(input_ids=ids, attention_mask=[1] * len(ids), tokens=text.split())

    def decode(self, ids):
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(i, "?") for i in ids)

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "vocab.json").write_text(json.dumps(self._vocab))

    def load(self, path):
        self._vocab = json.loads((Path(path) / "vocab.json").read_text())

    @property
    def vocab_size(self) -> int:
        return 64

    @property
    def embedding_type(self) -> str:
        return EmbeddingType.STANDARD

    @property
    def special_tokens(self) -> dict:
        return {"<pad>": 0, "</s>": 63}

    @property
    def pad_token_id(self) -> int:
        return self.PAD


# Counter accumulating evaluate() calls per task — lets tests assert "task X
# was evaluated exactly once".
_EVAL_CALL_COUNTS: Dict[str, int] = {}


def _make_lighteval_stub(task_name: str, accuracy: float):
    """Build a LightEval task stub that returns a fixed accuracy from evaluate()."""
    from arabic_eval.tasks.lighteval.base import LightEvalBenchmarkTask

    class _Stub(LightEvalBenchmarkTask):
        @property
        def name(self) -> str:
            return task_name

        @property
        def metric_names(self) -> List[str]:
            return ["accuracy"]

        def _default_dataset_name(self) -> str:
            return f"stub/{task_name}"

        def _parse_example(self, raw):
            return raw

        def load_examples(self):
            return [{"question": "س", "choices": ["A", "B"], "answer": 0,
                     "_source_config": "_default"}]

        def _format_eval_context(self, ex):
            return ex["question"]

        def _build_continuations(self, ex):
            return [" A", " B"]

        def _aggregate_scores(self, ex, continuations, log_likelihoods,
                              unconditioned_log_likelihoods=None, normalization="char"):
            return list(log_likelihoods)

        def evaluate(self, model, tokenizer, split="test", max_samples=None,
                     failure_report_dir=None, score_normalization="char"):
            _EVAL_CALL_COUNTS[task_name] = _EVAL_CALL_COUNTS.get(task_name, 0) + 1
            return {"accuracy": accuracy, "num_samples": 100}

    _Stub.__name__ = f"_StubTask_{task_name}"
    return _Stub


# Register stub tasks under fresh keys (avoid conflicts with real ACVA/etc.)
_StubAcva = _make_lighteval_stub("test_acva_stub", accuracy=0.55)
_StubAlghafa = _make_lighteval_stub("test_alghafa_stub", accuracy=0.40)
task_registry.register("test_acva_stub")(_StubAcva)
task_registry.register("test_alghafa_stub")(_StubAlghafa)


# --------------------------------------------------------------------------
# Autouse mock fixture
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patched_env(monkeypatch, tmp_path):
    """Patch all network/heavy I/O to keep tests offline + fast."""
    _EVAL_CALL_COUNTS.clear()

    # 1. Main Arabic corpus → 50 sentences
    arabic_text = ["جملة عربية قصيرة"] * 50
    fake_main = DatasetDict({
        "train": Dataset.from_dict({"text": arabic_text}),
        "eval": Dataset.from_dict({"text": arabic_text[:10]}),
    })
    monkeypatch.setattr(
        "arabic_eval.pipeline.experiment.load_arabic_dataset",
        lambda **_kw: fake_main,
    )

    # 2. Finetune corpora loaders → small synthetic QA records
    def _fake_records(corpus: str, n: int) -> List[QARecord]:
        return [
            QARecord(
                id=f"{corpus}-{i}",
                question=f"سؤال{i}",
                context=f"سياق قصير عن الموضوع رقم {i}",
                answer=f"إجابة{i}",
                source=corpus,
            )
            for i in range(n)
        ]

    def _fake_load_corpus(name, split):
        return _fake_records(name, 32 if split == "train" else 8)

    monkeypatch.setattr(
        "arabic_eval.data.finetune_corpora.load_corpus",
        _fake_load_corpus,
    )
    # ``load_corpora`` lives in pipeline imports; it calls load_corpus, so the
    # patch above propagates without further work.

    # 3. Make the intrinsic Evaluator a no-op-ish (real one is fine but slow)
    monkeypatch.setattr(
        "arabic_eval.pipeline.experiment.Evaluator",
        MagicMock(return_value=MagicMock(
            run_intrinsic=lambda **_kw: {
                "fertility": 1.0,
                "compression_ratio": 4.5,
                "unk_rate": 0.0,
                "vocab_coverage": 1.0,
                "avg_token_count": 8.0,
                "root_conservation_rate": 0.6,
                "pattern_conservation_rate": 0.5,
                "morpheme_integrity_rate": 0.9,
                "clitic_separation_accuracy": 0.85,
                "semantic_fragmentation_ratio": 1.1,
                "root_bearing_token_pct": 0.3,
                "pattern_bearing_token_pct": 0.25,
            },
        )),
    )

    yield


# --------------------------------------------------------------------------
# Helpers for building configs
# --------------------------------------------------------------------------

def _exp_config(
    output_dir: Path,
    *,
    name: str = "test",
    sft_enabled: bool = True,
    embed_enabled: bool = True,
    warmup_enabled: bool = True,
    tasks: List[str] = ["test_acva_stub"],
    tokenizer_types: List[str] = ["test_word"],
) -> ExperimentConfig:
    """Build a small ExperimentConfig wired to the test stubs."""
    cfg = load_config("configs/base.yaml")
    cfg.name = name
    cfg.output_dir = str(output_dir)
    cfg.tokenizer.type = tokenizer_types[0]
    cfg.tokenizer.vocab_size = 64
    cfg.tokenizer.save_path = str(output_dir / "tok")
    cfg.tokenizer.params = {}
    cfg.model.type = "test_tiny"
    cfg.model.device = "cpu"
    cfg.training.bf16 = False  # CPU-only test environment
    # Tiny step counts so the test runs in seconds.
    cfg.training.phases.embedding_alignment.enabled = embed_enabled
    cfg.training.phases.embedding_alignment.steps = 4
    cfg.training.phases.embedding_alignment.batch_size = 2
    cfg.training.phases.warmup.enabled = warmup_enabled
    cfg.training.phases.warmup.steps = 4
    cfg.training.phases.warmup.batch_size = 2
    cfg.training.phases.sft.enabled = sft_enabled
    cfg.training.phases.sft.steps = 4
    cfg.training.phases.sft.batch_size = 2
    cfg.training.phases.sft.early_stopping.enabled = False  # don't eval inside the tiny run
    cfg.training.phases.sft.early_stopping.eval_every_n_steps = 100  # never fires
    cfg.training.phases.sft.early_stopping.min_steps_before_stop = 100
    cfg.evaluation.morphological_metrics = False  # speed up; intrinsic mock supplies values
    cfg.evaluation.intrinsic_metrics = True

    # Build the sweep block (eval task list + tokenizer cells)
    from arabic_eval.config import SweepConfig, SweepTokenizerConfig, TaskConfig
    cfg.sweep = SweepConfig(
        tokenizers=[
            SweepTokenizerConfig(type=t, vocab_sizes=[None]) for t in tokenizer_types
        ],
        tasks=[TaskConfig(type=t, params={}) for t in tasks],
    )
    return cfg


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

class TestPhaseOrchestration:
    """Verify the 3-phase pipeline runs / skips per ``enabled`` flags."""

    def test_all_three_phases_run_when_enabled(self, tmp_path):
        cfg = _exp_config(tmp_path)
        results = run_experiment(cfg)
        training = results["training"]
        assert training["embedding_alignment"]["status"] == "ok"
        assert training["embedding_alignment"]["steps_completed"] == 4
        assert training["warmup"]["status"] == "ok"
        assert training["warmup"]["steps_completed"] == 4
        assert training["sft"]["status"] == "ok"
        assert training["sft"]["steps_completed"] == 4

    def test_sft_disabled_skips_only_phase_3(self, tmp_path):
        """The 'without SFT' experiment shape: Phase 1 + 2 run, Phase 3 skipped."""
        cfg = _exp_config(tmp_path, sft_enabled=False)
        results = run_experiment(cfg)
        training = results["training"]
        assert training["embedding_alignment"]["status"] == "ok"
        assert training["warmup"]["status"] == "ok"
        assert training["sft"]["status"] == "skipped"

    def test_all_phases_disabled_skips_training_entirely(self, tmp_path):
        """The pretrained-only path: every phase skipped."""
        cfg = _exp_config(
            tmp_path,
            embed_enabled=False, warmup_enabled=False, sft_enabled=False,
        )
        results = run_experiment(cfg)
        training = results["training"]
        for ph in ("embedding_alignment", "warmup", "sft"):
            assert training[ph]["status"] == "skipped", f"{ph} should be skipped"

    def test_phase_checkpoints_saved_when_enabled(self, tmp_path):
        cfg = _exp_config(tmp_path)
        run_experiment(cfg)
        for ph in ("embedding_alignment", "warmup", "sft"):
            assert (tmp_path / "training" / ph / "model.pt").exists(), f"missing {ph} ckpt"


class TestMultiTaskEvaluation:
    """All tasks in sweep.tasks get evaluated, each produces a downstream + MEI record."""

    def test_two_tasks_both_evaluated(self, tmp_path):
        cfg = _exp_config(tmp_path, tasks=["test_acva_stub", "test_alghafa_stub"])
        results = run_experiment(cfg)
        assert "test_acva_stub" in results["downstream"]
        assert "test_alghafa_stub" in results["downstream"]
        # Each task got exactly one evaluate() call
        assert _EVAL_CALL_COUNTS["test_acva_stub"] == 1
        assert _EVAL_CALL_COUNTS["test_alghafa_stub"] == 1
        # Per-task MEI is computed for each LightEval MCQ task
        assert "test_acva_stub" in results["mei"]
        assert "test_alghafa_stub" in results["mei"]
        assert results["mei"]["test_acva_stub"]["status"] == "ok"

    def test_mei_uses_per_task_accuracy(self, tmp_path):
        cfg = _exp_config(tmp_path, tasks=["test_acva_stub", "test_alghafa_stub"])
        results = run_experiment(cfg)
        # Stubs return distinct accuracies — MEI inputs must reflect them
        assert results["mei"]["test_acva_stub"]["inputs"]["accuracy"] == 0.55
        assert results["mei"]["test_alghafa_stub"]["inputs"]["accuracy"] == 0.40

    def test_inference_time_recorded_per_task(self, tmp_path):
        cfg = _exp_config(tmp_path, tasks=["test_acva_stub"])
        results = run_experiment(cfg)
        downstream = results["downstream"]["test_acva_stub"]
        assert "inference_time_sec" in downstream
        assert downstream["inference_time_sec"] >= 0


class TestOutputJsonShape:
    """The aggregated metrics file contains the expected top-level keys."""

    def test_all_metrics_json_top_level_keys(self, tmp_path):
        cfg = _exp_config(tmp_path)
        run_experiment(cfg)
        with open(tmp_path / "all_metrics.json") as f:
            metrics = json.load(f)
        assert "config" in metrics
        assert "intrinsic" in metrics
        assert "training" in metrics
        assert "downstream" in metrics
        assert "mei" in metrics
        # Training has all three phases keyed by name
        assert set(metrics["training"].keys()) >= {"embedding_alignment", "warmup", "sft"}

    def test_config_json_persisted(self, tmp_path):
        cfg = _exp_config(tmp_path)
        run_experiment(cfg)
        # The pipeline saves config.json before training
        assert (tmp_path / "config.json").exists()
        with open(tmp_path / "config.json") as f:
            saved = json.load(f)
        assert saved["name"] == cfg.name


class TestSweep:
    """run_sweep iterates multiple tokenizer cells and writes per-cell artifacts."""

    def test_sweep_writes_per_cell_directory(self, tmp_path):
        # Two cells: same tokenizer type but registered twice with distinct names.
        # We only have one tokenizer registered for tests, so use it twice with
        # distinct vocab_sizes — produces two cell names ("test_word" and
        # "test_word_5k" if vocab=5000).
        cfg = _exp_config(tmp_path, name="sweep_test", tokenizer_types=["test_word"])
        # Override sweep to ensure two cells (different vocab keys)
        from arabic_eval.config import SweepConfig, SweepTokenizerConfig, TaskConfig
        cfg.sweep = SweepConfig(
            tokenizers=[
                SweepTokenizerConfig(type="test_word", vocab_sizes=[None, 5000]),
            ],
            tasks=[TaskConfig(type="test_acva_stub", params={})],
        )
        results = run_sweep(cfg)
        # Two cells written; each got its own directory + all_metrics.json
        assert len(results) == 2
        for cell_name in results:
            assert (tmp_path / cell_name / "all_metrics.json").exists(), \
                f"missing all_metrics.json for cell {cell_name}"

    def test_sweep_resumes_existing_cell(self, tmp_path):
        """If all_metrics.json exists, the cell is skipped (resume semantics)."""
        cfg = _exp_config(tmp_path, name="sweep_resume", tokenizer_types=["test_word"])
        from arabic_eval.config import SweepConfig, SweepTokenizerConfig, TaskConfig
        cfg.sweep = SweepConfig(
            tokenizers=[SweepTokenizerConfig(type="test_word", vocab_sizes=[None])],
            tasks=[TaskConfig(type="test_acva_stub", params={})],
        )
        # Pre-create the cell directory + all_metrics.json
        cell_dir = tmp_path / "test_word"
        cell_dir.mkdir(parents=True)
        (cell_dir / "all_metrics.json").write_text(json.dumps({"resumed": True}))

        run_sweep(cfg)
        # eval should NOT have been called for this cell — pre-existing JSON wins
        assert _EVAL_CALL_COUNTS.get("test_acva_stub", 0) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
