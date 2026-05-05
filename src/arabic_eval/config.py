"""Experiment configuration with Pydantic validation and YAML loading."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# Registry keys understood by data/finetune_corpora.py. Adding a new
# corpus means editing both this Literal and the loader registry.
DatasetName = Literal["arabic_squad", "tydiqa_arabic", "arcd"]


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    dataset_name: str = "Jr23xd23/ArabicText-Large"
    dataset_config: Optional[str] = None
    cache_dir: str = "outputs/data_cache"
    train_split: str = "train"
    eval_split: str = "test"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = 10_000
    preprocessing: Dict[str, Any] = Field(default_factory=lambda: {
        "normalize_unicode": True,
        "remove_diacritics": False,
        "min_text_length": 10,
    })


class TokenizerConfig(BaseModel):
    type: str = "bpe"
    vocab_size: Optional[int] = 32_000
    params: Dict[str, Any] = Field(default_factory=dict)
    save_path: str = "outputs/tokenizers/bpe_32k"
    load_path: Optional[str] = None


class ModelConfig(BaseModel):
    type: str = "llama"
    name_or_path: str = "meta-llama/Llama-3.2-1B"
    dtype: str = "bfloat16"
    device: str = "auto"
    params: Dict[str, Any] = Field(default_factory=dict)


class TaskConfig(BaseModel):
    type: str = "text_generation"
    params: Dict[str, Any] = Field(default_factory=lambda: {
        "max_length": 512,
    })


class EarlyStoppingConfig(BaseModel):
    """Early-stopping policy for Phase 3 (SFT).

    Eval is run every ``eval_every_n_steps`` against the union of
    ``eval_splits`` and the answer-only causal-LM loss is tracked. Training
    stops if the loss fails to improve by at least ``min_delta`` for
    ``patience`` consecutive evaluations, but only after
    ``min_steps_before_stop`` is reached (avoids stopping during the LR
    warmup drift). When training stops (or completes normally) the
    checkpoint with the best eval_loss is restored if
    ``restore_best_at_end`` is True.
    """
    enabled: bool = True
    metric: Literal["eval_loss"] = "eval_loss"
    eval_every_n_steps: int = 200
    patience: int = 5
    min_delta: float = 5e-4
    min_steps_before_stop: int = 500
    restore_best_at_end: bool = True
    eval_splits: Dict[DatasetName, str] = Field(
        default_factory=lambda: {
            "tydiqa_arabic": "validation",
            "arcd": "validation",
        }
    )


class PhaseConfig(BaseModel):
    """One training phase.

    Phase 1 (embedding_alignment) freezes the transformer body and trains
    only the embedding + lm_head. Phases 2 and 3 unfreeze everything. Phase
    1 typically uses ``loss_target='full_sequence'``; phases 2 and 3 use
    ``'answer_only'`` (mask the question/context, only the answer span
    contributes to loss).
    """
    enabled: bool = True
    datasets: List[DatasetName]
    trainable_parameters: List[str]
    steps: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int = 1
    optimizer: Literal["adamw"] = "adamw"
    weight_decay: float = 0.0
    max_length: int = 512
    loss_target: Literal["full_sequence", "answer_only"] = "answer_only"
    lr_scheduler: Literal["cosine", "constant", "linear"] = "cosine"
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    save_checkpoint: bool = True
    early_stopping: Optional[EarlyStoppingConfig] = None

    @field_validator("datasets", mode="before")
    @classmethod
    def _coerce_datasets(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("trainable_parameters")
    @classmethod
    def _validate_trainable_parameters(cls, v):
        if not isinstance(v, list) or not v:
            raise ValueError("trainable_parameters must be a non-empty list of strings")
        for entry in v:
            if not isinstance(entry, str) or not entry:
                raise ValueError(f"trainable_parameters entries must be non-empty strings, got {entry!r}")
        if "*" in v and len(v) > 1:
            raise ValueError("trainable_parameters cannot mix '*' with other entries; use ['*'] alone for 'all parameters'")
        return v

    @field_validator("steps")
    @classmethod
    def _positive_steps(cls, v):
        if v <= 0:
            raise ValueError(f"steps must be positive, got {v}")
        return v


class PhasesConfig(BaseModel):
    """The three training phases. Each is independently toggleable via its own ``enabled`` flag."""
    embedding_alignment: PhaseConfig
    warmup: PhaseConfig
    sft: PhaseConfig

    @model_validator(mode="after")
    def _check_sft_has_early_stopping_if_enabled(self):
        if self.sft.enabled and self.sft.early_stopping is None:
            raise ValueError(
                "sft.early_stopping must be defined when sft.enabled is True "
                "(stagnation early-stop is required for Phase 3)"
            )
        return self


class TrainingConfig(BaseModel):
    """Three-phase training pipeline configuration.

    Each phase has its own ``enabled`` flag. ``embedding_alignment.enabled=True
    + warmup.enabled=True + sft.enabled=True`` is the full "with SFT" pipeline.
    Setting ``sft.enabled=False`` keeps Phase 1 + Phase 2 (the "without SFT"
    baseline). Setting all three to False skips training and evaluates the
    pretrained model directly.
    """
    phases: PhasesConfig
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 50


class EvaluationConfig(BaseModel):
    intrinsic_metrics: bool = True
    morphological_metrics: bool = True
    morph_sample_size: int = 500
    downstream_metrics: bool = True
    num_eval_samples: Optional[int] = 5_000  # None = no cap (use full eval split)
    generation_max_new_tokens: int = 128
    generation_temperature: float = 1.0
    generation_do_sample: bool = False
    failure_reports: bool = False  # If true, write per-task CSVs of failing eval cases
    # LightEval MCQ scoring normalization. ``"char"`` (default) is the
    # existing per-character-length normalization; ``"pmi"`` subtracts the
    # unconditioned per-continuation log-likelihood (LightEval's
    # ``LogProbPMINorm``) which corrects for letter / answer-text priors;
    # ``"char+pmi"`` reports both. Default stays ``"char"`` so existing
    # run JSONs remain reproducible — opt in per-experiment YAML.
    score_normalization: Literal["char", "pmi", "char+pmi"] = "char"


class TrackingConfig(BaseModel):
    use_wandb: bool = False
    wandb_project: str = "arabic-tokenizer-eval"
    wandb_entity: Optional[str] = None
    log_to_file: bool = True


class SweepTokenizerConfig(BaseModel):
    type: str
    vocab_sizes: List[Optional[int]]
    params: Dict[str, Any] = Field(default_factory=dict)


class SweepConfig(BaseModel):
    tokenizers: List[SweepTokenizerConfig] = Field(default_factory=list)
    tasks: List[TaskConfig] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    name: str = "experiment"
    description: str = ""
    output_dir: str = "outputs/experiments/default"
    seed: int = 42
    deterministic: bool = True

    data: DataConfig = Field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    sweep: Optional[SweepConfig] = None


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(
    config_path: str | Path,
    base_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """Load and validate an experiment config from YAML.

    Optionally merges a *base_path* config underneath, then applies
    *overrides* on top.
    """
    raw: dict = {}
    if base_path is not None:
        raw = load_yaml(base_path)
    experiment_raw = load_yaml(config_path)
    _deep_merge(raw, experiment_raw)
    if overrides:
        _deep_merge(raw, overrides)

    # Flatten 'experiment' key if present (some configs nest top-level fields there)
    if "experiment" in raw:
        exp = raw.pop("experiment")
        for k in ("name", "description", "output_dir", "seed", "deterministic"):
            if k in exp:
                raw.setdefault(k, exp[k])

    return ExperimentConfig(**raw)
