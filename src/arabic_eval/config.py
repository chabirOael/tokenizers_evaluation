"""Experiment configuration with Pydantic validation and YAML loading."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# Registry keys understood by data/finetune_corpora.py. Adding a new
# corpus means editing both this Literal and the loader registry.
DatasetName = Literal[
    "arabic_squad",
    "tydiqa_arabic",
    "arcd",
    "arabic_squad_mcq",   # synthetic MCQ derived from arabic_squad
    "pretraining_mix",   # packed raw-text mix (training.pretraining_mix); full-sequence loss only
]


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
    # Micro-steps (one batch per step). Required unless the phase draws from
    # the pretraining mix, where it is derived from ``mix_tokens``.
    steps: Optional[int] = None
    # Tokens drawn from the packed pretraining mix (phases with
    # ``datasets: ["pretraining_mix"]`` only). Consecutive mix phases take
    # disjoint block ranges. Must equal steps × batch_size × block_size.
    mix_tokens: Optional[int] = None
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
    clean_latin_rows: bool = False
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

    @field_validator("steps", "mix_tokens")
    @classmethod
    def _positive_if_set(cls, v, info):
        if v is not None and v <= 0:
            raise ValueError(f"{info.field_name} must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def _steps_or_mix_tokens(self):
        uses_mix = "pretraining_mix" in self.datasets
        if not uses_mix and self.steps is None:
            raise ValueError("steps is required (only phases on 'pretraining_mix' may derive it from mix_tokens)")
        if not uses_mix and self.mix_tokens is not None:
            raise ValueError("mix_tokens is only valid on a phase whose datasets is ['pretraining_mix']")
        return self


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



# --------------------------------------------------------------------------
# Pretraining mix (packed raw-text corpus for Phase 1 / Phase 2)
# --------------------------------------------------------------------------

class MixSourceConfig(BaseModel):
    """One raw-text source of the pretraining mix.

    ``name`` is a key of ``data.pretraining_mix.sources.SOURCE_REGISTRY``
    (``fineweb2_arb`` | ``wikipedia_ar`` | ``arabicweb24`` | ``arabic_101b``).
    ``share`` is the target fraction of the mix *in tokens under the active
    tokenizer* (Stage B) and, tokenizer-independently, in words for the
    cached pool (Stage A). ``params`` are forwarded to the source loader.
    """
    name: str
    share: float
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("share")
    @classmethod
    def _share_in_unit_interval(cls, v):
        if not (0.0 < v <= 1.0):
            raise ValueError(f"source share must be in (0, 1], got {v}")
        return v


class MixNormalizationConfig(BaseModel):
    """Document normalization applied before every filter. Default keeps
    parity with the (un-normalized) QA phases and the eval prompts: NFKC +
    tatweel + whitespace only; alef/diacritic normalization stays opt-in."""
    nfkc: bool = True
    remove_tatweel: bool = True
    normalize_alef: bool = False
    remove_diacritics: bool = False


class MixPoolConfig(BaseModel):
    """Tokenizer-independent pool (Stage A). ``total_words`` is split across
    sources by ``share``; because every tokenizer in the panel has fertility
    >= 1, a pool of ``total_words >= token_budget`` words is sufficient for
    any cell."""
    total_words: int = 20_000_000
    stream_shuffle_buffer: int = 10_000
    # Safety valve: stop streaming a source after this many raw docs even if
    # its word target was not reached (filters rejecting ~everything).
    max_stream_docs_per_source: int = 3_000_000

    @field_validator("total_words", "max_stream_docs_per_source")
    @classmethod
    def _positive(cls, v):
        if v <= 0:
            raise ValueError(f"must be positive, got {v}")
        return v

    @field_validator("stream_shuffle_buffer")
    @classmethod
    def _non_negative(cls, v):
        if v < 0:
            raise ValueError(f"stream_shuffle_buffer must be >= 0 (0 = no shuffle), got {v}")
        return v


class MixMinHashConfig(BaseModel):
    enabled: bool = True
    num_perm: int = 128
    shingle_words: int = 5
    threshold: float = 0.80


class MixDedupConfig(BaseModel):
    """Two-level cross-source dedup. ``paragraph_exact`` removes repeated
    normalized paragraphs *from* documents (boilerplate); ``minhash`` drops
    near-duplicate *documents*. Sources are processed in ``priority`` order
    so the surviving copy of a cross-source duplicate is the higher-priority
    one; sources absent from ``priority`` come last, in YAML order."""
    paragraph_exact: bool = True
    minhash: MixMinHashConfig = Field(default_factory=MixMinHashConfig)
    priority: List[str] = Field(default_factory=list)


class MixHeuristicMsaConfig(BaseModel):
    """Closed-list dialect-marker gate (data/pretraining_mix/filters.py).
    Drop a document only when all three hold: marker density exceeds
    ``max_markers_per_1k_words``, at least ``min_markers`` markers, and at
    least ``min_distinct_markers`` *different* marker types. The count floor
    stops one stray token from sinking a 50-word document; the distinct
    floor stops a single repeated token — calibration on 2 000 docs/source
    showed every false positive was one type repeated (``شو`` as the
    Japanese name Shū ×22, ``مش`` in a quoted song title ×4, ``وش`` as the
    product word "wash" ×3) while every true positive mixed several."""
    enabled: bool = True
    max_markers_per_1k_words: float = 8.0
    min_markers: int = 3
    min_distinct_markers: int = 2


class MixCamelDidConfig(BaseModel):
    """Opt-in second signal: CAMeL Tools ``DIDModel26`` via the .venv-camel
    bridge. MADAR-trained, so it mislabels encyclopedic MSA — keep
    ``min_msa_share`` lenient. Off by default."""
    enabled: bool = False
    sentences_per_doc: int = 8
    min_sentence_words: int = 5
    min_msa_share: float = 0.30


class MixMsaFilterConfig(BaseModel):
    heuristic: MixHeuristicMsaConfig = Field(default_factory=MixHeuristicMsaConfig)
    camel_did: MixCamelDidConfig = Field(default_factory=MixCamelDidConfig)


class MixQualityConfig(BaseModel):
    """Length + boilerplate rules. Every threshold is a drop reason counted
    in the pool manifest."""
    min_words: int = 50
    max_words: int = 3000                    # truncate at a paragraph boundary (not a drop)
    strip_latin_parentheticals: bool = True  # drop "(بالإنجليزية: Name)"-style glosses before the ratio rules
    max_latin_letter_ratio: float = 0.05     # Latin letters / all letters
    min_arabic_letter_ratio: float = 0.70    # Arabic-script letters / all letters
    max_short_line_ratio: float = 0.50       # lines with < short_line_words words
    short_line_words: int = 5
    max_dup_line_ratio: float = 0.30         # repeated lines within the doc
    max_symbol_ratio: float = 0.10           # symbol chars per word (Gopher-style)
    wikipedia_strip_sections: List[str] = Field(default_factory=lambda: [
        "المراجع", "مراجع", "وصلات خارجية", "انظر أيضا", "انظر أيضًا",
        "انظر أيضاً", "مصادر", "ملاحظات", "المصادر",
    ])


class PretrainingMixConfig(BaseModel):
    """Packed raw-text mix consumed by any phase whose ``datasets`` is
    ``["pretraining_mix"]``.

    Stage A (``pool``, ``dedup``, ``msa_filter``, ``quality``,
    ``normalization``, ``sources``, ``seed``) is tokenizer-independent and
    cached under ``cache_dir/<fingerprint>``. Stage B (``block_size``)
    packs the *whole* pool once per tokenizer — at the largest budget where
    the token shares hold exactly — into ``<pool>/packed/<tokenizer_fp>/``,
    shared by every experiment. Phases then draw disjoint, consecutive
    block ranges sized by their ``mix_tokens``; nothing ever wraps.
    """
    block_size: int = 512
    seed: int = 42
    cache_dir: str = "outputs/data_cache/pretraining_mix"
    normalization: MixNormalizationConfig = Field(default_factory=MixNormalizationConfig)
    sources: List[MixSourceConfig]
    pool: MixPoolConfig = Field(default_factory=MixPoolConfig)
    dedup: MixDedupConfig = Field(default_factory=MixDedupConfig)
    msa_filter: MixMsaFilterConfig = Field(default_factory=MixMsaFilterConfig)
    quality: MixQualityConfig = Field(default_factory=MixQualityConfig)

    @field_validator("block_size")
    @classmethod
    def _positive_block(cls, v):
        if v <= 0:
            raise ValueError(f"block_size must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def _check_sources(self):
        if not self.sources:
            raise ValueError("pretraining_mix.sources must list at least one source")
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"pretraining_mix.sources has duplicate names: {names}")
        total = sum(s.share for s in self.sources)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"pretraining_mix source shares must sum to 1.0, got {total:.6f} ({names})"
            )
        unknown = [p for p in self.dedup.priority if p not in names]
        if unknown:
            raise ValueError(
                f"pretraining_mix.dedup.priority names unknown sources {unknown}; sources are {names}"
            )
        return self

    def source_order(self) -> List[MixSourceConfig]:
        """Sources in dedup-priority order (priority first, then YAML order)."""
        by_name = {s.name: s for s in self.sources}
        ordered = [by_name[n] for n in self.dedup.priority]
        ordered += [s for s in self.sources if s.name not in self.dedup.priority]
        return ordered


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
    pretraining_mix: Optional[PretrainingMixConfig] = None

    @model_validator(mode="after")
    def _check_pretraining_mix_phases(self):
        """A phase that consumes the packed mix must list it as its sole
        dataset, use full-sequence loss, match the mix block size, and
        declare ``mix_tokens`` consistent with its step budget:
        ``mix_tokens == steps × batch_size × block_size`` (``steps`` is
        derived when omitted) — so the tokens a phase *reserves* are exactly
        the tokens it *trains on*, with no silent repetition or waste."""
        for phase_name in ("embedding_alignment", "warmup", "sft"):
            phase: PhaseConfig = getattr(self.phases, phase_name)
            if "pretraining_mix" not in phase.datasets:
                continue
            if self.pretraining_mix is None:
                raise ValueError(
                    f"training.phases.{phase_name}.datasets lists 'pretraining_mix' "
                    f"but training.pretraining_mix is not configured"
                )
            if len(phase.datasets) != 1:
                raise ValueError(
                    f"training.phases.{phase_name}: 'pretraining_mix' must be the sole "
                    f"dataset of a phase (got {phase.datasets}); mixing packed raw-text "
                    f"blocks with QA records in one phase is not supported"
                )
            if phase.loss_target != "full_sequence":
                raise ValueError(
                    f"training.phases.{phase_name}: 'pretraining_mix' requires "
                    f"loss_target='full_sequence' (got {phase.loss_target!r}) — packed "
                    f"raw text has no prompt/answer span"
                )
            block = self.pretraining_mix.block_size
            if phase.max_length != block:
                raise ValueError(
                    f"training.phases.{phase_name}.max_length ({phase.max_length}) must equal "
                    f"training.pretraining_mix.block_size ({block}) — "
                    f"blocks are packed once and shared by every phase that consumes the mix"
                )
            if phase.mix_tokens is None:
                raise ValueError(
                    f"training.phases.{phase_name}: mix_tokens is required on a phase that "
                    f"draws from 'pretraining_mix' (tokens the phase takes = steps × batch_size × block_size)"
                )
            per_step = phase.batch_size * block
            if phase.mix_tokens % per_step != 0:
                raise ValueError(
                    f"training.phases.{phase_name}.mix_tokens ({phase.mix_tokens}) must be a multiple "
                    f"of batch_size × block_size ({phase.batch_size} × {block} = {per_step})"
                )
            derived = phase.mix_tokens // per_step
            if phase.steps is None:
                phase.steps = derived
            elif phase.steps != derived:
                raise ValueError(
                    f"training.phases.{phase_name}: mix_tokens ({phase.mix_tokens}) implies steps = "
                    f"{derived} (mix_tokens / (batch_size {phase.batch_size} × block_size {block})) but "
                    f"steps = {phase.steps}; set them consistently or omit steps"
                )
        return self

    def mix_blocks(self, phase_name: str) -> int:
        """Blocks a mix phase draws: ``mix_tokens / block_size``."""
        phase: PhaseConfig = getattr(self.phases, phase_name)
        assert self.pretraining_mix is not None and phase.mix_tokens is not None
        return phase.mix_tokens // self.pretraining_mix.block_size

    def phases_using_mix(self) -> List[str]:
        """Names of *enabled* phases whose datasets is ['pretraining_mix'], in run order."""
        out = []
        for phase_name in ("embedding_alignment", "warmup", "sft"):
            phase: PhaseConfig = getattr(self.phases, phase_name)
            if phase.enabled and "pretraining_mix" in phase.datasets:
                out.append(phase_name)
        return out


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
    # If true, dump the per-word list underlying ``unk_rate`` to
    # ``<output_dir>/intrinsic_unks.csv`` (one row per unique source word
    # that produced an UNK token in the intrinsic eval split). Tokenizers
    # without a usable ``unk_token`` id produce a header-only CSV.
    intrinsic_unk_report: bool = False
    # If true, scan prompts + every continuation during LightEval MCQ
    # evaluation and write per-task ``<output_dir>/unk_reports/<task>_unks.csv``
    # listing UNK occurrences. Independent from ``intrinsic_unk_report``.
    downstream_unk_report: bool = False
    # LightEval MCQ scoring normalization. ``"char"`` is the per-character-
    # length normalization; ``"pmi"`` subtracts the unconditioned per-
    # continuation log-likelihood (LightEval's ``LogProbPMINorm``) which
    # corrects for letter / answer-text priors; ``"char+pmi"`` reports both
    # and aliases ``accuracy`` to ``accuracy_pmi`` (PMI is the unbiased
    # primary metric). Default ``"char+pmi"`` since 2026-05-06.
    score_normalization: Literal["char", "pmi", "char+pmi"] = "char"

    # Number of few-shot demonstrations prepended to each LightEval MCQ
    # prompt. K demos are sampled deterministically (seeded) from the same
    # ``_source_config`` as the eval row, with the eval row excluded from
    # its own pool. ``0`` (default) preserves the existing zero-shot
    # behavior — opt in per-experiment YAML.
    num_fewshot: int = 0


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
