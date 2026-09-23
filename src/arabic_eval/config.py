"""Experiment configuration with Pydantic validation and YAML loading."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# Registry keys understood by data/finetune_corpora.py. Adding a new
# corpus means editing this Literal, ``CORPUS_CATEGORY`` below and the
# loader registry (a test pins the three in sync).
DatasetName = Literal[
    "arabic_squad",
    "tydiqa_arabic",
    "arcd",
    "arabic_squad_mcq",   # synthetic MCQ derived from arabic_squad
    "cidar",              # arbml/CIDAR — culturally-aligned Arabic instructions (free-form)
    "bactrian_x_ar",      # MBZUAI/Bactrian-X ar — translated Alpaca + Dolly instructions (free-form)
    "aya_ar",             # CohereForAI/aya_collection_language_split standard_arabic, sub-dataset allowlist (free-form)
    "pretraining_mix",   # packed raw-text mix (training.pretraining_mix); full-sequence loss only
]

# The shape of every QA corpus, fixed by the loader (its ``prompt_template``):
# ``qa`` → extractive, ``mcq_letter`` → mcq, ``instruction`` → free_form. A
# phase ``mixture`` expresses its ratio over these categories; a corpus can
# never be re-categorised from YAML. ``pretraining_mix`` has no category.
MixtureCategory = Literal["extractive", "mcq", "free_form"]
CORPUS_CATEGORY: Dict[str, str] = {
    "arabic_squad": "extractive",
    "tydiqa_arabic": "extractive",
    "arcd": "extractive",
    "arabic_squad_mcq": "mcq",
    "cidar": "free_form",
    "bactrian_x_ar": "free_form",
    "aya_ar": "free_form",
}


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
        "normalize_alef": False,
        "remove_tatweel": True,
        "min_text_length": 10,
        "join_lone_waw": True,
    })


class TokenizerConfig(BaseModel):
    type: str = "bpe"
    vocab_size: Optional[int] = 32_000
    params: Dict[str, Any] = Field(default_factory=dict)
    save_path: str = "outputs/tokenizers/bpe_32k"
    load_path: Optional[str] = None


class EmbeddingInitConfig(BaseModel):
    """How the rows of a *swapped* vocabulary's embedding matrix start (standard tokenizers).

    ``legacy`` is what every run did before 2026-09-22: ``resize_token_embeddings``
    keeps the base model's first N pretrained rows, so from-scratch token id 5
    inherits whatever the base tokenizer's id 5 meant (ASCII, bytes, English
    pieces) — an arbitrary mapping the literature treats as random init.
    ``random`` is N(0, 0.02²); ``mean`` puts the base matrix's global mean in
    every row; ``surface_avg`` builds each row from the surface strings the
    token stands for (``BaseTokenizer.token_surfaces``): every surface is
    tokenized with the *base* HF tokenizer and its pieces' pretrained rows are
    averaged (``weighting`` uniform or by character length), then averaged
    over the token's surfaces by their weights; a token with no surface gets
    the global mean. ``norm: base_mean`` rescales each new row to the mean L2
    norm of the base rows. A native tokenizer (no resize) is never touched.
    Measured on Qwen3-4B-Base + AraRooPat-17K (2026-09-22): see CLAUDE.md
    *Reinitialization behavior*.
    """
    method: Literal["legacy", "random", "mean", "surface_avg"] = "legacy"
    base_tokenizer: Optional[str] = None          # HF name/path of the base tokenizer; null = model.name_or_path
    weighting: Literal["uniform", "char_len"] = "uniform"
    norm: Literal["none", "base_mean"] = "none"
    seed: int = 42


class ModelConfig(BaseModel):
    type: str = "llama"
    name_or_path: str = "meta-llama/Llama-3.2-1B"
    dtype: str = "bfloat16"
    device: str = "auto"
    params: Dict[str, Any] = Field(default_factory=dict)
    embedding_init: EmbeddingInitConfig = Field(default_factory=EmbeddingInitConfig)


class TaskConfig(BaseModel):
    type: str = "text_generation"
    params: Dict[str, Any] = Field(default_factory=lambda: {
        "max_length": 512,
    })


class EvalMixtureConfig(BaseModel):
    """Compose the early-stop eval set the way the phase's *training* set is
    composed, from the ``dev`` split of the same corpora (added 2026-09-22).

    The default ``eval_splits`` signal is extractive only (TyDiQA + ARCD dev),
    while the Phase 3 mixture puts ~87 % of its loss tokens in free-form
    answers: on the 2026-09-22 arms the 12 % that is extractive decided when
    the 87 % stopped training, and AraRooPat restored a checkpoint that had
    seen 5 600 of 30 000 mixture examples against BPE's 27 200. With this set,
    the stop metric is the training objective measured on held-out records of
    the training composition: ``total_examples`` records drawn from the dev
    pools at the phase mixture's ``shares`` / ``within_category`` / ``weights``
    (``upsample`` is always false — a dev pool that cannot fill its quota is an
    error, not a repeat), scored as Σ NLL / Σ answer tokens over all
    categories, with the per-category numbers reported next to it.
    """
    total_examples: int = 1000
    seed: int = 42

    @field_validator("total_examples")
    @classmethod
    def _positive_total(cls, v):
        if v <= 0:
            raise ValueError(f"early_stopping.eval_mixture.total_examples must be positive, got {v}")
        return v


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
    # ``dev`` = a title-level 5 % slice of each official *train* split
    # (``finetune_corpora.is_dev_title``); the official evaluation splits
    # (``validation``) are held out and never steer training (2026-09-17).
    # ``null`` (or ``{}``) means "no split-based eval set" — the form a config
    # takes when it switches to ``eval_mixture``. It has to be ``null``, not
    # ``{}``: the YAML layers deep-merge, and an empty mapping merged over
    # base.yaml's keeps base.yaml's keys.
    eval_splits: Optional[Dict[DatasetName, str]] = Field(
        default_factory=lambda: {
            "tydiqa_arabic": "dev",
            "arcd": "dev",
        }
    )
    # The alternative to ``eval_splits``: an eval set composed like the phase's
    # training mixture, from the ``dev`` slices of the same corpora. Exactly
    # one of the two is in force; setting this requires the phase to have a
    # ``mixture`` and ``eval_splits`` to be empty.
    eval_mixture: Optional[EvalMixtureConfig] = None


class QABlendConfig(BaseModel):
    """Blend packed SFT-format QA text into a phase that trains on the
    pretraining mix.

    ``share`` is the fraction of the phase's *blocks* (hence tokens) that
    are QA records rendered with the Phase 3 / eval surface form
    (``السياق: …\\nالسؤال: …\\nالإجابة: {answer}``), packed EOS-separated into
    ``block_size`` blocks exactly like the raw-text pool and trained with
    the same full-sequence loss. The phase's token budget (``mix_tokens``)
    is unchanged — the raw-text slice shrinks by ``round(share × blocks)``
    blocks, so the with/without-blend arms of an ablation spend identical
    compute. ``split`` must not be a Phase 3 early-stop split.
    """
    datasets: List[DatasetName] = Field(default_factory=lambda: ["tydiqa_arabic", "arcd"])
    share: float = 0.07
    split: str = "train"
    seed: int = 42

    @field_validator("datasets", mode="before")
    @classmethod
    def _coerce_datasets(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("datasets")
    @classmethod
    def _qa_corpora_only(cls, v):
        if not v:
            raise ValueError("qa_blend.datasets must list at least one QA corpus")
        if "pretraining_mix" in v:
            raise ValueError("qa_blend.datasets must be QA corpora (the raw-text mix is the host, not a blend member)")
        if len(set(v)) != len(v):
            raise ValueError(f"qa_blend.datasets has duplicates: {v}")
        return v

    @field_validator("share")
    @classmethod
    def _share_open_unit_interval(cls, v):
        if not (0.0 < v < 1.0):
            raise ValueError(f"qa_blend.share must be in (0, 1), got {v}")
        return v

    @model_validator(mode="after")
    def _no_early_stop_splits(self):
        # TyDiQA / ARCD ``validation`` are the held-out evaluation splits and
        # ``dev`` steers Phase 3 early-stopping; blending either into an
        # earlier phase would contaminate evaluation or the stopping signal.
        if self.split != "train":
            for name in self.datasets:
                if name in ("tydiqa_arabic", "arcd"):
                    what = "the held-out evaluation split" if self.split == "validation" else "a Phase 3 early-stop split"
                    raise ValueError(
                        f"qa_blend: split {self.split!r} of {name!r} is {what}; "
                        f"only 'train' may be blended"
                    )
        return self


class MixtureConfig(BaseModel):
    """Ratio-controlled composition of a QA phase's training set.

    The phase draws exactly ``total_examples`` records from its ``datasets``:
    ``shares`` says how many per *category* (``extractive`` / ``mcq`` /
    ``free_form`` — the category of a corpus is fixed by its loader, see
    ``CORPUS_CATEGORY``), and inside a category the quota is split across
    the listed corpora either ``equal`` (capacity-aware water-fill: a corpus
    that cannot fill its slice hands the remainder to the others) or
    ``proportional`` to the corpus sizes; ``weights`` overrides either with
    explicit per-corpus weights. Quotas count *kept* records — the draw
    walks a seeded permutation of every corpus, tokenizes as it goes, skips
    records the answer-only masking drops (truncation ate the answer) and
    stops when the quota is met, so the ratio is exact after truncation.

    A corpus that cannot supply its quota is an error naming the numbers
    (``upsample: true`` instead repeats a second seeded permutation).
    ``drop_truncated_answers`` skips records whose full text hits
    ``max_length`` (the answer is cut) and tops up, so the model only sees
    complete answers. ``steps`` of the phase is derived as
    ``total_examples / batch_size`` (one exact pass) unless set consistently.

    Shares are in *examples*. Free-form answers carry one to two orders of
    magnitude more loss tokens than an MCQ letter, so the manifest reports
    the answer-token share per category alongside the example share.
    """
    total_examples: int
    shares: Dict[MixtureCategory, float]
    within_category: Literal["equal", "proportional"] = "equal"
    weights: Optional[Dict[DatasetName, float]] = None
    upsample: bool = False
    drop_truncated_answers: bool = False
    seed: int = 42

    @field_validator("total_examples")
    @classmethod
    def _positive_total(cls, v):
        if v <= 0:
            raise ValueError(f"mixture.total_examples must be positive, got {v}")
        return v

    @field_validator("shares")
    @classmethod
    def _shares_sum_to_one(cls, v):
        if not v:
            raise ValueError("mixture.shares must name at least one category")
        for cat, share in v.items():
            if not (0.0 < share <= 1.0):
                raise ValueError(f"mixture.shares[{cat!r}] must be in (0, 1], got {share}")
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"mixture.shares must sum to 1 (got {total:.6f} over {sorted(v)})")
        return v

    @field_validator("weights")
    @classmethod
    def _weights_positive(cls, v):
        if v is not None:
            if not v:
                raise ValueError("mixture.weights must be null or a non-empty {corpus: weight} map")
            for name, w in v.items():
                if w <= 0:
                    raise ValueError(f"mixture.weights[{name!r}] must be positive, got {w}")
        return v


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
    # Micro-steps (one batch of ``batch_size`` sequences per step; the
    # optimizer updates every ``gradient_accumulation_steps`` of them, so the
    # effective batch is ``batch_size × gradient_accumulation_steps`` and the
    # phase performs ``steps / gradient_accumulation_steps`` updates — the LR
    # scheduler's horizon). Every other step-valued knob of a phase counts the
    # same unit: ``warmup_steps``, ``early_stopping.eval_every_n_steps`` /
    # ``min_steps_before_stop``, ``mix_tokens = steps × batch_size × block_size``
    # and the mixture's ``steps = total_examples / batch_size``. Required unless
    # the phase draws from the pretraining mix, where it is derived from
    # ``mix_tokens``, or carries a ``mixture``.
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
    # Linear LR ramp over the first ``warmup_steps`` micro-steps (i.e.
    # ``warmup_steps / gradient_accumulation_steps`` optimizer updates).
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    save_checkpoint: bool = True
    clean_latin_rows: bool = False
    # Optional SFT-format QA text blended into a 'pretraining_mix' phase
    # (share of the phase's blocks). See ``QABlendConfig``.
    qa_blend: Optional[QABlendConfig] = None
    # Optional ratio-controlled composition of a QA phase (never a
    # 'pretraining_mix' phase): exact per-category example counts drawn from
    # ``datasets``. See ``MixtureConfig``. ``steps`` is derived from it.
    mixture: Optional[MixtureConfig] = None
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
        if not uses_mix and self.mix_tokens is not None:
            raise ValueError("mix_tokens is only valid on a phase whose datasets is ['pretraining_mix']")
        if not uses_mix and self.qa_blend is not None:
            raise ValueError("qa_blend is only valid on a phase whose datasets is ['pretraining_mix']")
        if self.mixture is not None:
            self._validate_mixture()
        self._validate_eval_mixture()
        if not uses_mix and self.steps is None:
            raise ValueError(
                "steps is required (only phases on 'pretraining_mix' derive it from mix_tokens, "
                "and phases with a 'mixture' from total_examples / batch_size)"
            )
        return self

    def _validate_eval_mixture(self) -> None:
        """``early_stopping.eval_mixture`` composes the stop signal from the
        phase's own corpora at the phase mixture's ratio — so it needs that
        mixture, it is the alternative to ``eval_splits`` (never both), and it
        makes no sense on a raw-text phase."""
        es = self.early_stopping
        em = getattr(es, "eval_mixture", None) if es is not None else None
        if em is None:
            return
        if "pretraining_mix" in self.datasets:
            raise ValueError(
                "early_stopping.eval_mixture is only valid on a QA phase; a 'pretraining_mix' phase "
                "has no per-category QA composition to mirror"
            )
        if self.mixture is None:
            raise ValueError(
                "early_stopping.eval_mixture needs the phase to have a 'mixture' (it reuses its shares, "
                "within_category and weights); add one or use eval_splits"
            )
        if es.eval_splits:
            raise ValueError(
                "early_stopping.eval_splits and early_stopping.eval_mixture are alternatives: the first "
                "scores held-out records of named splits, the second a mixture of the phase's own dev "
                f"pools at the training ratio. Got both (eval_splits={dict(es.eval_splits)}). base.yaml "
                "sets eval_splits and the YAML layers deep-merge, so a config that switches to eval_mixture "
                "has to clear it with 'eval_splits: null' (an empty mapping would merge away)"
            )
        if em.total_examples % self.batch_size != 0:
            lo = (em.total_examples // self.batch_size) * self.batch_size
            raise ValueError(
                f"early_stopping.eval_mixture.total_examples ({em.total_examples}) must be a multiple of "
                f"batch_size ({self.batch_size}); nearest valid values: {lo} or {lo + self.batch_size}"
            )

    def _validate_mixture(self) -> None:
        """``mixture`` needs a QA phase whose ``datasets`` cover exactly the
        share categories; ``steps`` is derived from ``total_examples``."""
        mix = self.mixture
        assert mix is not None
        if "pretraining_mix" in self.datasets:
            raise ValueError(
                "mixture is only valid on a QA phase (datasets of QA corpora); a 'pretraining_mix' "
                "phase blends QA text through qa_blend instead"
            )
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError(f"datasets has duplicates: {self.datasets}")
        listed = {name: CORPUS_CATEGORY[name] for name in self.datasets}
        for name, cat in listed.items():
            if cat not in mix.shares:
                raise ValueError(
                    f"mixture: dataset {name!r} is {cat!r} but mixture.shares has no {cat!r} share "
                    f"(shares: {sorted(mix.shares)}); drop the dataset or give its category a share"
                )
        for cat in mix.shares:
            if cat not in listed.values():
                raise ValueError(
                    f"mixture.shares names {cat!r} but no listed dataset is {cat!r} "
                    f"(datasets: {self.datasets}); add a {cat!r} corpus or drop the share"
                )
        if mix.weights is not None:
            unknown = sorted(set(mix.weights) - set(self.datasets))
            if unknown:
                raise ValueError(
                    f"mixture.weights names datasets the phase does not list: {unknown} "
                    f"(datasets: {self.datasets})"
                )
        if mix.total_examples % self.batch_size != 0:
            lo = (mix.total_examples // self.batch_size) * self.batch_size
            raise ValueError(
                f"mixture.total_examples ({mix.total_examples}) must be a multiple of batch_size "
                f"({self.batch_size}); nearest valid values: {lo} or {lo + self.batch_size}"
            )
        derived = mix.total_examples // self.batch_size
        if self.steps is None:
            self.steps = derived
        elif self.steps != derived:
            raise ValueError(
                f"mixture.total_examples ({mix.total_examples}) implies steps = {derived} "
                f"(total_examples / batch_size {self.batch_size}) but steps = {self.steps}; "
                f"set them consistently or omit steps"
            )


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


class MixHeldoutFilterConfig(BaseModel):
    """Stage A filter: drop a pool document that contains a held-out
    evaluation passage (``data.contamination``, by content, tier
    ``contaminated``). ``sets_file`` declares the held-out sets; the pool
    fingerprint includes their identity (pinned dataset revisions, the
    prompt file's hash), so a changed held-out set rebuilds the pool.
    Drop reason in the manifest: ``heldout_overlap``."""
    enabled: bool = True
    sets_file: str = "configs/contamination/heldout_sets.yaml"


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
    heldout_filter: MixHeldoutFilterConfig = Field(default_factory=MixHeldoutFilterConfig)

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
    # Per-corpus loader parameters, forwarded to ``load_corpus(name, split,
    # **params)`` wherever a phase, an early-stop split or a qa_blend loads
    # that corpus. Only ``aya_ar`` takes any today (``include_datasets``:
    # sub-dataset allowlist of the Aya collection).
    corpus_params: Dict[DatasetName, Dict[str, Any]] = Field(default_factory=dict)
    # Committed list of training record ids that contain a held-out
    # evaluation passage (``scripts/check_contamination.py``; see
    # ``data/contamination.py``). Applied to the ``train`` / ``dev`` splits
    # of every QA corpus a phase, an early-stop split or a qa_blend loads.
    # ``null`` trains without it (the pool's own filter is separate:
    # ``pretraining_mix.heldout_filter``).
    contamination_exclusions: Optional[str] = "configs/contamination/exclusions.json"

    @field_validator("corpus_params")
    @classmethod
    def _corpus_params_are_qa_corpora(cls, v):
        if "pretraining_mix" in v:
            raise ValueError("training.corpus_params: 'pretraining_mix' is configured under training.pretraining_mix")
        # ``teacher_answers`` (the distillation overlay, finetune_corpora._apply_teacher_answers)
        # replaces reference answers with a teacher's — only free-form answers are written by one.
        for name, params in v.items():
            path = (params or {}).get("teacher_answers")
            if "teacher_answers" not in (params or {}):
                continue
            if CORPUS_CATEGORY.get(name) != "free_form":
                raise ValueError(
                    f"training.corpus_params.{name}.teacher_answers: the teacher-answer overlay applies to "
                    f"free-form corpora only ({sorted(n for n, c in CORPUS_CATEGORY.items() if c == 'free_form')}); "
                    f"{name} is {CORPUS_CATEGORY.get(name)!r}")
            if path is not None:
                p = Path(path)
                if not (p.exists() or (Path(__file__).resolve().parents[2] / p).exists()):
                    raise ValueError(f"training.corpus_params.{name}.teacher_answers: {path} does not exist "
                                     f"(scripts/distill/filter_teacher_answers.py writes it)")
        return v

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
            if phase.qa_blend is not None:
                total_blocks = phase.mix_tokens // block
                qa_blocks = int(round(phase.qa_blend.share * total_blocks))
                if qa_blocks < 1:
                    raise ValueError(
                        f"training.phases.{phase_name}.qa_blend.share ({phase.qa_blend.share}) × "
                        f"{total_blocks} blocks rounds to 0 QA blocks; raise share or mix_tokens"
                    )
                if qa_blocks >= total_blocks:
                    raise ValueError(
                        f"training.phases.{phase_name}.qa_blend.share ({phase.qa_blend.share}) leaves no "
                        f"raw-text blocks in the phase ({qa_blocks} of {total_blocks})"
                    )
        return self

    def qa_blocks(self, phase_name: str) -> int:
        """QA blocks a blended mix phase draws: ``round(share × mix_tokens / block_size)``
        (0 when the phase has no ``qa_blend``)."""
        phase: PhaseConfig = getattr(self.phases, phase_name)
        if phase.qa_blend is None:
            return 0
        return int(round(phase.qa_blend.share * self.mix_blocks(phase_name)))

    def mix_blocks(self, phase_name: str) -> int:
        """Blocks a mix phase trains on: ``mix_tokens / block_size`` (raw-text
        blocks + QA-blend blocks; see ``qa_blocks``)."""
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
    # ``<output_dir>/intrinsic_unks.parquet`` (one row per unique source word
    # that produced an UNK token in the intrinsic eval split). Tokenizers
    # without a usable ``unk_token`` id produce a header-only CSV.
    intrinsic_unk_report: bool = False
    # If true, scan prompts + every continuation during LightEval MCQ
    # evaluation and write per-task
    # ``<output_dir>/unk_reports/<task>_unks.parquet`` listing UNK
    # occurrences. Independent from ``intrinsic_unk_report``.
    downstream_unk_report: bool = False
    # If true, write every scored eval row to
    # ``<output_dir>/eval_rows/<task>.parquet``: the exact prompt the model
    # received, the continuations, every per-choice score under every active
    # normalization, and the truncation diagnostics. This is what the
    # experiment console's Eval-rows tab reads. A strict superset of
    # ``failure_reports`` — when both are on, the failure report is skipped.
    # Roughly 18 MB per sweep cell across the four benchmarks.
    eval_row_dump: bool = False
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

EXPERIMENT_KEYS = ("name", "description", "output_dir", "seed", "deterministic", "created_at", "runs")
"""Top-level fields a YAML may nest under ``experiment:`` (hoisted by ``load_config``)."""


class RunStamp(BaseModel):
    """One start of the experiment, appended to ``ExperimentConfig.runs`` by
    whoever launches it (the console or ``scripts/run_experiment.py``).
    Immutable facts only — status and end time live in the run record."""
    started_at: str                       # ISO-8601 with UTC offset
    run_id: Optional[str] = None          # the console's outputs/runs/<run_id>; None for a CLI start
    source: Literal["console", "cli"] = "console"


class ExperimentConfig(BaseModel):
    name: str = "experiment"
    description: str = ""
    output_dir: str = "outputs/experiments/default"
    seed: int = 42
    deterministic: bool = True
    # Provenance, maintained by the tooling (see ``arabic_eval.config_edit``):
    # ``created_at`` is stamped once when the file is first written, ``runs``
    # gains one entry per start. Neither is read by the pipeline.
    created_at: Optional[str] = None
    runs: List[RunStamp] = Field(default_factory=list)

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
        for k in EXPERIMENT_KEYS:
            if k in exp:
                raw.setdefault(k, exp[k])

    return ExperimentConfig(**raw)
