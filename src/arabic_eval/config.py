"""Experiment configuration with Pydantic validation and YAML loading."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


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


class TrainingConfig(BaseModel):
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 2
    early_stopping_patience: Optional[int] = 3
    early_stopping_metric: str = "eval_loss"


class EvaluationConfig(BaseModel):
    intrinsic_metrics: bool = True
    downstream_metrics: bool = True
    num_eval_samples: Optional[int] = 5_000  # None = no cap (use full eval split)
    generation_max_new_tokens: int = 128
    generation_temperature: float = 1.0
    generation_do_sample: bool = False


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
