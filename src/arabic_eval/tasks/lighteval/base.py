"""LightEval-based evaluation: dataset-agnostic core.

This module is deliberately thin and opinion-free. It contains:

  * ``LightEvalBenchmarkTask`` — abstract contract every Arabic MCQ
    benchmark must implement. The abstract methods declare exactly what
    each dataset must answer (loading, parsing, prompt format, continuations,
    SFT text, score aggregation, dataset name, registry key).
  * ``LightEvalModelWrapper`` — wraps a ``BaseModelAdapter`` to expose
    LightEval's ``loglikelihood`` request/response protocol. Truly
    generic; no dataset opinions.
  * ``_compute_loglikelihood`` — pure scoring loop. No prompt format, no
    continuation conventions, no aggregation policy.
  * ``evaluate`` — orchestrates the LightEval log-likelihood scoring over
    the full benchmark (under the 3-phase pipeline, training is
    task-agnostic so the entire benchmark is the eval set).

Anything that encodes a *choice* about prompt shape, continuation tokens,
score aggregation, or how rows are loaded from disk lives in ``utils.py``
(opt-in helpers) or in the per-dataset file. Adding a new dataset never
requires editing this module.
"""
from __future__ import annotations

import logging
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arabic_eval.data.answer_only_masking import continuation_start
from arabic_eval.evaluation.eval_rows import (
    SENTINEL_LL,
    UNIT_BY_EMBEDDING,
    EvalRowWriter,
    build_row_record,
)
from arabic_eval.evaluation.unk_reports import (
    DOWNSTREAM_UNK_FIELDS,
    WordUnkRecord,
    aggregate_occurrences,
    records_to_rows,
    scan_text,
)
from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.params_spec import ParamSpec
from arabic_eval.tasks.base import BaseTask
from arabic_eval.tasks.lighteval.utils import (
    ARABIC_CHOICE_LETTERS,
    format_mcq_context,
)
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType
from arabic_eval.tokenizers.utils.arabic_text import contains_latin_letters
from arabic_eval.utils.io import write_report_table

logger = logging.getLogger("arabic_eval.tasks.lighteval.base")

# Optional LightEval import — presence is checked so the package loads without it.
try:
    import lighteval  # noqa: F401
    LIGHTEVAL_AVAILABLE = True
    logger.info("LightEval detected; evaluations will use its log-likelihood methodology.")
except ImportError:
    LIGHTEVAL_AVAILABLE = False
    logger.warning(
        "lighteval not installed. Benchmark evaluation will use the built-in "
        "log-likelihood implementation (same methodology). "
        "Install with: pip install lighteval>=0.6.0"
    )


# ---------------------------------------------------------------------------
# Core log-likelihood computation (LightEval methodology)
# ---------------------------------------------------------------------------

class ScoredLogLikelihood(float):
    """The float ``_compute_loglikelihood`` returns, carrying how it was scored.

    Still a plain ``float`` for every consumer of LightEval's ``loglikelihood``
    protocol (arithmetic, ``np.argmax``, JSON); the two attributes let the row
    dump record the scored window without a second encode:

      * ``n_tokens`` — continuation tokens summed (0 for the sentinel);
      * ``truncated`` — the ``max_length`` cap reached the full encoding, so
        the continuation may be cut (or, with ``n_tokens == 0``, gone);
      * ``token_logprobs`` — the addends of the sum, in order (float64; empty
        for the sentinel): which token of a multi-token continuation carries
        the score (row-dump schema 3, ``cont_token_ll``).
    """

    __slots__ = ("n_tokens", "truncated", "token_logprobs")

    def __new__(cls, value: float, n_tokens: int = 0, truncated: bool = False,
                token_logprobs: Tuple[float, ...] = ()):
        obj = float.__new__(cls, value)
        obj.n_tokens = int(n_tokens)
        obj.truncated = bool(truncated)
        obj.token_logprobs = tuple(float(v) for v in token_logprobs)
        return obj


_warned_empty_prefix = False


@torch.no_grad()
def _compute_loglikelihood(
    model: BaseModelAdapter,
    tokenizer: BaseTokenizer,
    context: str,
    continuation: str,
    max_length: int = 512,
) -> ScoredLogLikelihood:
    """
    Compute log P(continuation | context) following LightEval's approach.

    ``context`` and ``context + continuation`` are encoded separately; the
    continuation is the tokens of the full encoding **after the longest common
    prefix** of the two, once a trailing ``</s>`` has been stripped from each
    (``answer_only_masking.continuation_start`` — the rule the answer-only
    training loss uses). One forward pass over the full encoding (without that
    ``</s>``) yields logits and we sum the log-probabilities of exactly those
    tokens.

    Scoring window fix (2026-09-24): the window used to be
    ``full[len(context_ids):]``, which assumed the context encoding is a token
    prefix of the full one. Every from-scratch tokenizer appends ``</s>`` to a
    standalone encoding, so that window skipped the first continuation token
    and scored the trailing ``</s>`` instead (for a single-piece letter under
    BPE, ``</s>`` alone). For a tokenizer that appends nothing (the native
    wrappers) the two windows coincide and the sum is unchanged.

    CharacterBERT (``character_cnn``) uses word-level logits: the batch is built
    from ``char_ids`` instead of ``input_ids``, but continuation scoring follows
    the same causal-LM approach using word vocabulary indices. The prefix is
    compared on ``(word id, char ids)`` pairs so two different out-of-vocabulary
    words (the same UNK word id) never count as shared.
    """
    global _warned_empty_prefix
    full_text = context + continuation
    ctx_enc = tokenizer.encode(context, max_length=max_length, truncation=True, padding=False)
    full_enc = tokenizer.encode(full_text, max_length=max_length, truncation=True, padding=False)

    raw_full_len = len(full_enc.input_ids)
    truncated = bool(max_length) and raw_full_len >= max_length
    eos_id = (getattr(tokenizer, "special_tokens", None) or {}).get("eos_token")
    char_cnn = tokenizer.embedding_type == EmbeddingType.CHARACTER_CNN

    if char_cnn:
        ctx_keys = [(i, tuple(c)) for i, c in zip(ctx_enc.input_ids, ctx_enc.char_ids)]
        full_keys = [(i, tuple(c)) for i, c in zip(full_enc.input_ids, full_enc.char_ids)]
        # The EOS word's key, from whichever encoding still ends in it (a
        # truncated full encoding has lost its EOS; the context may not have).
        eos_key = next(
            (keys[-1] for keys in (full_keys, ctx_keys) if keys and keys[-1][0] == eos_id),
            None,
        )
        kept_keys, k = continuation_start(ctx_keys, full_keys, eos_key)
        full_len = len(kept_keys)
    else:
        full_ids, k = continuation_start(ctx_enc.input_ids, full_enc.input_ids, eos_id)
        full_len = len(full_ids)

    if full_len <= k:
        # Truncation at ``max_length`` left no room for the continuation, so
        # there is nothing to score. Every choice of such a row returns this
        # same sentinel and the row's argmax is an artifact of the cap, not a
        # decision — ``eval_rows`` flags it as ``all_sentinel``.
        return ScoredLogLikelihood(SENTINEL_LL, 0, truncated)
    if k == 0:
        # No shared prefix (an empty context under a BOS-less tokenizer, or a
        # context whose only token merged with the continuation): the first
        # continuation token has no position to be predicted from.
        if not _warned_empty_prefix:
            logger.warning(
                "scorer: context and context+continuation share no token prefix "
                "(context=%r) — returning the sentinel; logged once", context[:80],
            )
            _warned_empty_prefix = True
        return ScoredLogLikelihood(SENTINEL_LL, 0, truncated)

    input_ids_list = full_enc.input_ids[:full_len]
    attention_mask = torch.tensor([full_enc.attention_mask[:full_len]], device=model.device)

    if char_cnn:
        # CharacterBERT: input is 3-D char_ids; output logits are over word vocab.
        char_ids = torch.tensor([full_enc.char_ids[:full_len]], device=model.device)
        batch = {"char_ids": char_ids, "attention_mask": attention_mask}
    else:
        input_ids = torch.tensor([input_ids_list], device=model.device)
        batch = {"input_ids": input_ids, "attention_mask": attention_mask,
                 "labels": input_ids.clone()}

    output = model.forward(batch)
    logits = output["logits"]                          # [1, seq_len, vocab_size]
    log_probs = F.log_softmax(logits[0], dim=-1)       # [seq_len, vocab_size]

    # Causal LM: position i predicts position i+1.
    # Sum log P(token_k … token_{full_len-1}) given their left contexts. The
    # addends are kept in order; the sum is accumulated exactly as before.
    total_ll = 0.0
    token_lls: List[float] = []
    for pos in range(k - 1, full_len - 1):
        next_tok = input_ids_list[pos + 1]
        tok_ll = log_probs[pos, next_tok].item()
        token_lls.append(tok_ll)
        total_ll += tok_ll

    return ScoredLogLikelihood(total_ll, full_len - k, truncated, tuple(token_lls))


# ---------------------------------------------------------------------------
# LightEval model wrapper
# ---------------------------------------------------------------------------

class LightEvalModelWrapper:
    """
    Wraps ``BaseModelAdapter`` to expose LightEval's ``loglikelihood`` interface.

    This class is dataset-agnostic. ``evaluate_mcq`` delegates prompt and
    continuation construction to the active task object; when called with
    ``task=None`` (standalone use, e.g. in tests or when wired into a vanilla
    LightEval pipeline) it falls back to ``utils.format_mcq_context`` plus the
    ``utils.ARABIC_CHOICE_LETTERS`` letter-MCQ convention.
    """

    def __init__(
        self,
        model: BaseModelAdapter,
        tokenizer: BaseTokenizer,
        max_length: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def loglikelihood(self, requests: List[Tuple[str, str]]) -> List[float]:
        """
        Compute log P(continuation | context) for each ``(context, continuation)`` pair.

        Matches LightEval's ``loglikelihood`` request/response protocol so this
        wrapper can be swapped for a ``LightevalModel`` subclass transparently.
        """
        return [
            _compute_loglikelihood(
                self.model, self.tokenizer, ctx, cont, self.max_length
            )
            for ctx, cont in requests
        ]

    def _prompt_units(self, context: str) -> Optional[int]:
        """Length of the prompt in whatever unit this tokenizer emits.

        One extra encode per row, paid only while dumping. What a "unit" means
        depends on the embedding family (tokens / words / chars / bytes), so
        the dump records the label alongside the number. Never fatal: a fake
        tokenizer in a test or an analyzer that throws just yields ``None``.
        """
        try:
            enc = self.tokenizer.encode(
                context, max_length=self.max_length, truncation=True, padding=False
            )
            return len(enc.input_ids)
        except Exception:  # noqa: BLE001 - diagnostics must never break eval
            return None

    def evaluate_mcq(
        self,
        examples: List[Dict[str, Any]],
        collect_failures: bool = False,
        task: Optional["LightEvalBenchmarkTask"] = None,
        score_normalization: str = "char",
        row_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        row_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Run LightEval-style multiple-choice accuracy evaluation.

        For each example: build one ``(context, continuation)`` pair per choice,
        call ``loglikelihood``, predict the argmax, compare to ground truth.

        Continuation building delegates to the active task via
        ``task._format_eval_context(ex)`` and ``task._build_continuations(ex)``.
        When ``task`` is ``None`` the wrapper falls back to the standalone
        letter-MCQ convention.

        ``score_normalization`` ∈ {``"char"``, ``"pmi"``, ``"char+pmi"``}.
        Under ``"pmi"`` (or ``"char+pmi"``) the wrapper additionally scores
        each continuation under ``task._unconditioned_query(ex)`` and the
        aggregator subtracts the unconditioned ll from the conditioned ll.
        Unconditioned ll values are cached per ``(unconditioned_query,
        tuple(continuations))`` key — for letter-scored MCQ the cache fires on
        every example after the first; for word-scored continuations that
        vary per row, the cache misses and we pay one extra forward pass per
        example (~2× cost on those rows, acceptable for the four current tasks).

        When ``collect_failures`` is True, also returns a list of failure
        records (one per wrong-answer example) with per-choice log-likelihoods
        for downstream reporting. Under PMI modes "failure" is defined by
        the PMI argmax (since PMI is the corrected scoring); the record also
        carries ``score_pmi_*`` and ``score_pmi_margin`` columns.

        ``row_sink``, when given, is called once per example — right *and*
        wrong — with a full dump record (``eval_rows.build_row_record``): the
        exact prompt the model was scored on, every continuation, every
        per-choice score, and the truncation diagnostics. Leaving it ``None``
        keeps this method's behaviour byte-identical to before it existed.

        ``row_indices`` names each example's position in the task's *full*
        eval list (``rows_file`` scores a subset): it is what the dump and the
        failure report record as the row's index. ``None`` = ``range(len)``.
        """
        if row_indices is not None and len(row_indices) != len(examples):
            raise ValueError(
                f"row_indices has {len(row_indices)} entries for {len(examples)} examples"
            )
        if score_normalization not in ("char", "pmi", "char+pmi"):
            raise ValueError(
                f"Unknown score_normalization={score_normalization!r}; "
                "expected one of 'char', 'pmi', 'char+pmi'."
            )
        want_char = score_normalization in ("char", "char+pmi")
        want_pmi = score_normalization in ("pmi", "char+pmi")

        correct_char = 0
        correct_pmi = 0
        total = 0
        failures: List[Dict[str, Any]] = []
        # Per-sub-config buckets: track (correct_char, correct_pmi, total)
        # keyed on ``_source_config`` (sentinel ``"_default"`` for single-config).
        per_config: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])

        # Cache keyed on (unconditioned_query, tuple(continuations)) — for
        # letter-scored MCQ this is identical across every example; for ACVA
        # / word-scored Alghafa the continuation tuple varies but the
        # unconditioned context is stable, so we still skip a redundant
        # forward when the same continuations recur.
        uncond_cache: Dict[Tuple[str, Tuple[str, ...]], List[float]] = {}

        for pos, ex in enumerate(tqdm(examples, desc="LightEval MCQ", unit="example")):
            idx = int(row_indices[pos]) if row_indices is not None else pos
            if task is not None:
                # ``_format_eval_context_with_fewshot`` collapses to the bare
                # ``_format_eval_context`` when ``task.num_fewshot == 0``, so
                # zero-shot callers see no behavior change.
                context = task._format_eval_context_with_fewshot(ex)
                continuations = task._build_continuations(ex)
            else:
                context = format_mcq_context(ex["question"], ex["choices"])
                continuations = [
                    " " + (ARABIC_CHOICE_LETTERS[i] if i < len(ARABIC_CHOICE_LETTERS) else str(i))
                    for i in range(len(ex["choices"]))
                ]
            requests: List[Tuple[str, str]] = [(context, c) for c in continuations]
            log_likelihoods = self.loglikelihood(requests)

            uncond_lls: Optional[List[float]] = None
            if want_pmi and task is not None:
                uncond_q = task._unconditioned_query(ex)
                cache_key = (uncond_q, tuple(continuations))
                cached = uncond_cache.get(cache_key)
                if cached is None:
                    cached = self.loglikelihood([(uncond_q, c) for c in continuations])
                    uncond_cache[cache_key] = cached
                uncond_lls = cached

            scores_char: Optional[List[float]] = None
            scores_pmi: Optional[List[float]] = None
            if task is not None:
                if want_char:
                    scores_char = task._aggregate_scores(
                        ex, continuations, log_likelihoods, normalization="char"
                    )
                if want_pmi:
                    scores_pmi = task._aggregate_scores(
                        ex, continuations, log_likelihoods,
                        unconditioned_log_likelihoods=uncond_lls,
                        normalization="pmi",
                    )
            else:
                # Standalone (no task) — skip aggregation; use raw lls.
                if want_char:
                    scores_char = list(log_likelihoods)
                if want_pmi and uncond_lls is not None:
                    scores_pmi = [
                        ll - u for ll, u in zip(log_likelihoods, uncond_lls)
                    ]
                elif want_pmi:
                    raise ValueError(
                        "score_normalization includes 'pmi' but task is None and "
                        "no unconditioned log-likelihoods were computed."
                    )

            # Failure-row argmax follows the *primary* scoring under the
            # selected mode. For "char+pmi" we treat PMI as primary because
            # the operator opted into that mode specifically to inspect PMI;
            # for "char" the primary is char-norm (legacy).
            primary_scores = scores_pmi if want_pmi else scores_char
            assert primary_scores is not None
            predicted = int(np.argmax(primary_scores))
            gold = ex["answer"]
            cfg_key = ex.get("_source_config", "_default")
            per_config[cfg_key][2] += 1

            pred_char: Optional[int] = None
            pred_pmi: Optional[int] = None
            if want_char:
                pred_char = int(np.argmax(scores_char))
                if pred_char == gold:
                    correct_char += 1
                    per_config[cfg_key][0] += 1
            if want_pmi:
                pred_pmi = int(np.argmax(scores_pmi))
                if pred_pmi == gold:
                    correct_pmi += 1
                    per_config[cfg_key][1] += 1

            if row_sink is not None:
                # ``_compute_loglikelihood`` returns floats that know their
                # scored window; a stubbed scorer (tests) returns plain floats
                # and the dump then leaves the two per-choice fields empty.
                cont_tokens = [getattr(v, "n_tokens", None) for v in log_likelihoods]
                cont_truncated = [getattr(v, "truncated", None) for v in log_likelihoods]
                cont_token_ll = [getattr(v, "token_logprobs", None) for v in log_likelihoods]
                row_sink(build_row_record(
                    row_index=idx,
                    example=ex,
                    prompt=context,
                    continuations=continuations,
                    log_likelihoods=log_likelihoods,
                    scores_char=scores_char,
                    scores_pmi=scores_pmi,
                    unconditioned_log_likelihoods=uncond_lls,
                    gold_idx=gold,
                    pred_idx=predicted,
                    pred_idx_char=pred_char,
                    pred_idx_pmi=pred_pmi,
                    prompt_units=self._prompt_units(context),
                    max_length=self.max_length,
                    cont_tokens=None if None in cont_tokens else cont_tokens,
                    cont_truncated=None if None in cont_truncated else cont_truncated,
                    cont_token_ll=(
                        None if any(v is None for v in cont_token_ll) else cont_token_ll
                    ),
                ))

            if predicted == gold:
                total += 1
                continue

            if collect_failures:
                gold_token = continuations[gold].lstrip()
                pred_token = continuations[predicted].lstrip()
                # Default report uses char-norm scores (or raw lls) so the
                # legacy ``score_*`` / ``score_margin`` columns stay populated
                # exactly as before. PMI columns are added separately.
                report_scores = (
                    scores_char if scores_char is not None else list(log_likelihoods)
                )
                record: Dict[str, Any] = {
                    "index": idx,
                    "question": ex["question"],
                    "gold_idx": gold,
                    "gold_letter": gold_token,
                    "pred_idx": predicted,
                    "pred_letter": pred_token,
                    "ll_margin": round(
                        float(log_likelihoods[predicted]) - float(log_likelihoods[gold]), 6
                    ),
                    "score_margin": round(
                        float(report_scores[predicted]) - float(report_scores[gold]), 6
                    ),
                }
                for i, choice in enumerate(ex["choices"]):
                    record[f"choice_{i}"] = choice
                    record[f"ll_{i}"] = round(float(log_likelihoods[i]), 6)
                    record[f"score_{i}"] = round(float(report_scores[i]), 6)
                if scores_pmi is not None:
                    record["score_pmi_margin"] = round(
                        float(scores_pmi[predicted]) - float(scores_pmi[gold]), 6
                    )
                    for i in range(len(ex["choices"])):
                        record[f"score_pmi_{i}"] = round(float(scores_pmi[i]), 6)
                failures.append(record)
            total += 1

        # Build the metrics dict. Under "char" mode the shape is byte-identical
        # to the pre-PMI version (no ``accuracy_char_norm`` / ``accuracy_pmi``
        # keys at all). Under "char+pmi" mode (the new default), ``accuracy``
        # aliases ``accuracy_pmi`` because PMI is the unbiased metric — char-norm
        # is dominated by class-collapse on weak-signal MCQ tasks.
        metrics: Dict[str, Any] = {"num_samples": total}
        char_acc = round(correct_char / max(total, 1), 4) if want_char else None
        pmi_acc = round(correct_pmi / max(total, 1), 4) if want_pmi else None
        if want_char and not want_pmi:
            metrics["accuracy"] = char_acc
        elif want_pmi and not want_char:
            metrics["accuracy"] = pmi_acc
            metrics["accuracy_pmi"] = pmi_acc
        else:  # char+pmi — PMI is primary
            metrics["accuracy"] = pmi_acc
            metrics["accuracy_char_norm"] = char_acc
            metrics["accuracy_pmi"] = pmi_acc

        per_sub: Dict[str, Dict[str, Any]] = {}
        for k, (c_char, c_pmi, t) in sorted(per_config.items()):
            entry: Dict[str, Any] = {"num_samples": t}
            char_a = round(c_char / t, 4) if (want_char and t) else 0.0
            pmi_a = round(c_pmi / t, 4) if (want_pmi and t) else 0.0
            if want_char and not want_pmi:
                entry["accuracy"] = char_a
            elif want_pmi and not want_char:
                entry["accuracy"] = pmi_a
                entry["accuracy_pmi"] = pmi_a
            else:  # char+pmi — PMI is primary
                entry["accuracy"] = pmi_a
                entry["accuracy_char_norm"] = char_a
                entry["accuracy_pmi"] = pmi_a
            per_sub[k] = entry
        metrics["per_subconfig_accuracy"] = per_sub

        return metrics, failures


# ---------------------------------------------------------------------------
# Base benchmark task — abstract contract
# ---------------------------------------------------------------------------

class LightEvalBenchmarkTask(BaseTask):
    """
    Abstract base class for LightEval-based Arabic MCQ benchmarks.

    Every subclass MUST implement SEVEN things:

      * ``_default_dataset_name()``    — default HuggingFace dataset path
      * ``name``                       — registry key string
      * ``_parse_example(raw)``        — normalise a raw row into
                                         ``{"question", "choices", "answer", ...}``
      * ``load_examples()``            — return the full parsed, ``_source_config``-
                                         stamped list of examples. Typically
                                         delegates to
                                         ``utils.load_huggingface_mcq``.
      * ``_format_eval_context(ex)``   — context string fed to the model at eval time
      * ``_build_continuations(ex)``   — list of continuations to score, in
                                         answer-index order; each typically begins
                                         with a leading space (LightEval convention)
      * ``_aggregate_scores(ex, conts, lls)`` — combine per-continuation
                                         log-likelihoods into per-choice scores
                                         for argmax (e.g. char-norm)

    No prompt format, continuation shape, score aggregation, or data-loading
    strategy is hard-coded on this class. Datasets that want LightEval's
    standard letter-MCQ + char-norm + HF-loader conventions opt into them by
    importing helpers from ``arabic_eval.tasks.lighteval.utils``. A future
    dataset that loads from somewhere else (local files, S3, …) just
    implements its own ``load_examples``.

    Under the 3-phase pipeline, training is task-agnostic (Phase 3 SFT uses
    TyDiQA-Arabic + ARCD), so the benchmark contributes 0 rows to training
    and 100% of rows to evaluation. ``get_eval_examples`` returns the full
    list (after the optional Latin-script filter).
    """

    # Defaults of the parameters every LightEval MCQ task reads from
    # ``sweep.tasks[].params`` (``param_spec`` declares them; ``__init__`` reads them).
    DEFAULT_CACHE_DIR = "outputs/data_cache"
    DEFAULT_MAX_LENGTH = 512
    DEFAULT_SEED = 42
    DEFAULT_CLEAN_LATIN_ROWS = False
    DEFAULT_NUM_FEWSHOT = 0
    DEFAULT_ROWS_FILE: Optional[str] = None

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        default_ds = config.get("dataset_name")
        self.dataset_name: str = default_ds if default_ds is not None else self._default_dataset_name()
        self.dataset_config: Optional[str] = config.get("dataset_config", None)
        self.cache_dir: str = config.get("cache_dir", self.DEFAULT_CACHE_DIR)
        self.max_length: int = config.get("max_length", self.DEFAULT_MAX_LENGTH)
        self.seed: int = config.get("seed", self.DEFAULT_SEED)
        self.clean_latin_rows: bool = bool(config.get("clean_latin_rows", self.DEFAULT_CLEAN_LATIN_ROWS))
        # Number of few-shot demonstrations to prepend (per-row) to each eval
        # prompt. Demos are sampled deterministically from the same
        # ``_source_config`` as the eval row, with the eval row excluded.
        # 0 = pure zero-shot (existing default).
        self.num_fewshot: int = int(config.get("num_fewshot", self.DEFAULT_NUM_FEWSHOT))
        # Score only these rows of the full eval list (``rows_file``). The full
        # list is still loaded and cached, so the few-shot pools and seeds — and
        # therefore every prompt — are those of the full benchmark.
        self.rows_file: Optional[str] = config.get("rows_file", self.DEFAULT_ROWS_FILE)
        self._rows_subset: Optional[Dict[str, Any]] = (
            self._read_rows_file(self.rows_file) if self.rows_file is not None else None
        )
        self._cached_examples: Optional[List[Dict]] = None
        # Sub-config -> list of indices, populated lazily on first few-shot use.
        self._fewshot_pool_by_config: Optional[Dict[str, List[int]]] = None

    @classmethod
    def param_spec(cls) -> List[ParamSpec]:
        """The seven parameters every LightEval MCQ task accepts. ``dataset_name``
        defaults to the class's ``_default_dataset_name()``; ``num_fewshot`` is
        the one the pipeline fills from ``evaluation.num_fewshot`` when the YAML
        does not set it. Subclasses inherit the list and may re-word an entry
        (ACVA does for ``num_fewshot``)."""
        try:
            default_ds: Optional[str] = cls._default_dataset_name()
        except TypeError:           # a subclass (a test stub) still defining it as an instance method
            default_ds = None
        return [
            ParamSpec("dataset_name", "str", default_ds, nullable=default_ds is None,
                      help="HuggingFace dataset path of the benchmark; override only if it moves on the Hub."),
            ParamSpec("dataset_config", "str", None, nullable=True,
                      help="One sub-config of the dataset (a topic / subject) instead of every config merged; null = all."),
            ParamSpec("cache_dir", "path", cls.DEFAULT_CACHE_DIR, advanced=True,
                      help="Where the HuggingFace datasets cache of this benchmark lives on disk."),
            ParamSpec("max_length", "int", cls.DEFAULT_MAX_LENGTH, min=16,
                      help="Token cap of prompt + continuation at scoring time; a row whose continuation the cap "
                           "eats scores the sentinel for every choice (all_sentinel in the row dump). 1024 for "
                           "3-shot prompts under long-sequence tokenizers."),
            ParamSpec("seed", "int", cls.DEFAULT_SEED, min=0, advanced=True,
                      help="Seed of the few-shot demonstration sampling (deterministic per row and sub-config)."),
            ParamSpec("clean_latin_rows", "bool", cls.DEFAULT_CLEAN_LATIN_ROWS,
                      help="Drop rows whose question / choices / context contain Latin letters before scoring "
                           "(logs the count, warns on a wiped sub-config)."),
            ParamSpec("num_fewshot", "int", cls.DEFAULT_NUM_FEWSHOT, min=0,
                      help="In-context demonstrations prepended to each prompt, sampled from the same sub-config "
                           "with the row itself excluded. Absent = the pipeline injects evaluation.num_fewshot "
                           "(3 in the reference configs); set it here to override for this task only."),
            ParamSpec("rows_file", "path", cls.DEFAULT_ROWS_FILE, nullable=True, advanced=True,
                      help="score only the rows listed in this JSON (full-list `row_index`); few-shot pools and "
                           "seeds stay those of the full benchmark."),
        ]

    # ------------------------------------------------------------------
    # Abstract hooks (every subclass MUST implement)
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def _default_dataset_name(cls) -> str:
        """Default HuggingFace dataset identifier. A classmethod so the default
        is readable without an instance (``param_spec``, the console)."""
        ...

    @abstractmethod
    def _parse_example(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse one raw dataset row into ``{"question": str, "choices": List[str],
        "answer": int}`` (additional dataset-specific keys allowed). Return
        ``None`` to skip malformed rows."""
        ...

    @abstractmethod
    def load_examples(self) -> List[Dict[str, Any]]:
        """Return the full parsed list of examples, each stamped with a
        ``_source_config`` key (sentinel ``"_default"`` for single-config
        datasets). Typically delegates to ``utils.load_huggingface_mcq``."""
        ...

    @abstractmethod
    def _format_eval_context(self, ex: Dict[str, Any]) -> str:
        """Context string fed to the model at eval time (everything before the
        continuation)."""
        ...

    @abstractmethod
    def _build_continuations(self, ex: Dict[str, Any]) -> List[str]:
        """Continuation strings to score, one per choice in answer-index order.
        Each typically begins with a leading space to match LightEval's
        tokenisation convention."""
        ...

    @abstractmethod
    def _aggregate_scores(
        self,
        ex: Dict[str, Any],
        continuations: List[str],
        log_likelihoods: List[float],
        unconditioned_log_likelihoods: Optional[List[float]] = None,
        normalization: str = "char",
    ) -> List[float]:
        """Combine per-continuation log-likelihoods into per-choice scores for
        argmax.

        ``normalization``:
          * ``"char"`` (default) — divide each ll by the continuation's
            character length (LightEval ``LogProbCharNorm`` equivalent). Most
            letter-MCQ datasets delegate to ``utils.char_norm_aggregator``.
          * ``"pmi"`` — subtract the unconditioned per-continuation ll
            (LightEval ``LogProbPMINorm`` equivalent). Removes letter / answer-
            text priors that otherwise dominate weak-signal MCQ decisions.
            Requires ``unconditioned_log_likelihoods`` to be supplied; raises
            otherwise so a missing wiring is loud.
        """
        ...

    # ------------------------------------------------------------------
    # PMI hook: per-task unconditioned context (override-friendly)
    # ------------------------------------------------------------------

    def _unconditioned_query(self, ex: Dict[str, Any]) -> str:
        """Return the unconditioned query used to score each continuation under
        PMI normalization (``log P(c | unconditioned) − log P(c | full)``).

        Default: the bare answer prefix ``"الإجابة:"`` — matches the trailing
        line every LightEval-official benchmark prompt ends with (no ``###``
        marker). Tasks whose prompt uses a different answer-prefix convention
        can override this; the default is the right answer for all four
        current tasks (acva / alghafa / culture_arabic_mmlu / arabic_exam).
        """
        return "الإجابة:"

    # ------------------------------------------------------------------
    # Few-shot prompt helpers
    # ------------------------------------------------------------------

    def _format_eval_context_with_fewshot(self, ex: Dict[str, Any]) -> str:
        """Return ``ex``'s eval context, optionally prepended with K few-shot
        demonstrations.

        K = ``self.num_fewshot``. Demonstrations are sampled deterministically
        (seeded with ``self.seed`` + the eval row's index in the cached list)
        from rows that share ``ex["_source_config"]``, with the eval row
        itself excluded from its own pool. Each demo is rendered as
        ``<demo_eval_context> <demo_gold_continuation>`` followed by a blank
        line; the eval prompt is appended last.

        When ``num_fewshot == 0`` this collapses to the inherited
        ``_format_eval_context``.
        """
        base_ctx = self._format_eval_context(ex)
        if self.num_fewshot <= 0:
            return base_ctx
        demos = self._build_fewshot_examples(ex)
        if not demos:
            return base_ctx
        rendered_demos: List[str] = []
        for demo in demos:
            demo_ctx = self._format_eval_context(demo)
            demo_conts = self._build_continuations(demo)
            gold_idx = demo["answer"]
            if not (0 <= gold_idx < len(demo_conts)):
                # Defensive: malformed demo — skip.
                continue
            # Continuations conventionally have a leading space; strip it for
            # the demo since we emit ``<ctx> <answer>`` ourselves.
            rendered_demos.append(f"{demo_ctx}{demo_conts[gold_idx]}")
        if not rendered_demos:
            return base_ctx
        return "\n\n".join(rendered_demos) + "\n\n" + base_ctx

    def _build_fewshot_examples(self, ex: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sample ``num_fewshot`` demonstration rows for ``ex``.

        Selection rules:
          * Demos come from the same ``_source_config`` as ``ex`` (when
            ``_source_config`` is missing on either side, defaults to
            ``"_default"``). Same-source demos give the model the most
            representative format for its sub-task.
          * The eval row itself is excluded by identity (``id(...)``); even
            if no global ID is available, we never use the same dict object.
          * Selection is deterministic — seeded with ``self.seed + index``
            where ``index`` is ``ex``'s position in the cached eval list,
            so re-running the eval pass produces byte-identical demos.
          * Returns fewer than K demos if the pool can't supply that many
            (rare; happens only on tiny sub-configs).
        """
        import random

        if self.num_fewshot <= 0:
            return []
        if self._cached_examples is None:
            return []  # Eval set not loaded yet.

        # Build per-sub-config index pools once (lazy).
        if self._fewshot_pool_by_config is None:
            pool: Dict[str, List[int]] = {}
            for i, row in enumerate(self._cached_examples):
                key = row.get("_source_config", "_default")
                pool.setdefault(key, []).append(i)
            self._fewshot_pool_by_config = pool

        cfg_key = ex.get("_source_config", "_default")
        candidates = self._fewshot_pool_by_config.get(cfg_key, [])
        if not candidates:
            return []

        # Find ex's index for deterministic seeding + self-exclusion.
        ex_index: Optional[int] = None
        for i, row in enumerate(self._cached_examples):
            if row is ex:
                ex_index = i
                break
        seed = self.seed + (ex_index if ex_index is not None else 0)
        rng = random.Random(seed)

        # Exclude ex's index from candidates if present.
        eligible = [i for i in candidates if i != ex_index]
        if not eligible:
            return []
        k = min(self.num_fewshot, len(eligible))
        chosen_indices = rng.sample(eligible, k)
        return [self._cached_examples[i] for i in chosen_indices]

    # ------------------------------------------------------------------
    # clean_latin_rows hook: which fields to inspect when filtering Latin-
    # contaminated rows before the 10/90 split.
    # ------------------------------------------------------------------

    def _text_fields(self, ex: Dict[str, Any]) -> List[str]:
        """Return the text fields of a parsed example to inspect for the
        ``clean_latin_rows`` filter. Default covers the standard MCQ shape:
        question + every choice + optional context (Arabic_Exam ships ~5%
        of rows with a ``context`` field). Subclasses MAY override to add
        additional task-specific text fields.
        """
        fields: List[str] = []
        q = ex.get("question")
        if isinstance(q, str):
            fields.append(q)
        for c in ex.get("choices", []) or []:
            if isinstance(c, str):
                fields.append(c)
        ctx = ex.get("context")
        if isinstance(ctx, str) and ctx:
            fields.append(ctx)
        return fields

    def _row_has_latin(self, ex: Dict[str, Any]) -> bool:
        """Return True if any inspected text field of ``ex`` contains
        Latin-script letters. Used as the row-drop predicate when
        ``clean_latin_rows`` is enabled.
        """
        return any(contains_latin_letters(t) for t in self._text_fields(ex))

    # ------------------------------------------------------------------
    # Eval examples (no SFT split; the 3-phase pipeline trains on
    # task-agnostic corpora — TyDiQA-Arabic + ARCD — so the benchmark
    # contributes 0 rows to training and 100% to eval).
    # ------------------------------------------------------------------

    def _read_rows_file(self, path: str) -> Dict[str, Any]:
        """Load and check a ``rows_file``: ``{"task", "row_index": [int, …],
        "provenance": {…}}``. The task name must be this task's (a file never
        crosses benchmarks); the indices must be distinct non-negative ints.
        Their range is checked against the full list at evaluate time."""
        import hashlib
        import json

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"{self.name}: rows_file {str(p)!r} does not exist")
        raw = p.read_bytes()
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict) or "row_index" not in data or "task" not in data:
            raise ValueError(f"{self.name}: rows_file {str(p)!r} must be a JSON object with 'task' and 'row_index'")
        if data["task"] != self.name:
            raise ValueError(
                f"{self.name}: rows_file {str(p)!r} lists rows of task {data['task']!r}, not {self.name!r}"
            )
        idx = data["row_index"]
        if not isinstance(idx, list) or not idx or any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in idx
        ):
            raise ValueError(f"{self.name}: rows_file {str(p)!r}: row_index must be a non-empty list of ints ≥ 0")
        if len(set(idx)) != len(idx):
            dup = sorted({i for i in idx if idx.count(i) > 1})[:5]
            raise ValueError(f"{self.name}: rows_file {str(p)!r}: duplicate row_index {dup}")
        return {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "row_index": list(idx),
                "n_rows": len(idx)}

    def _select_rows(
        self, examples: List[Dict[str, Any]], max_samples: Optional[int],
    ) -> Tuple[List[Dict[str, Any]], Optional[List[int]]]:
        """The rows to score and their full-list indices (``None`` = all rows in order)."""
        if self._rows_subset is None:
            return (examples[:max_samples] if max_samples else examples), None
        if max_samples:
            raise ValueError(
                f"{self.name}: rows_file ({self.rows_file}) and evaluation.num_eval_samples "
                f"({max_samples}) are both set — a rows_file already names the rows; drop one of them"
            )
        idx = self._rows_subset["row_index"]
        bad = [i for i in idx if i >= len(examples)]
        if bad:
            raise ValueError(
                f"{self.name}: rows_file {self.rows_file!r} lists row_index {bad[:5]} outside the "
                f"{len(examples)}-row eval list"
            )
        return [examples[i] for i in idx], list(idx)

    def get_eval_examples(self) -> List[Dict[str, Any]]:
        """Return all examples for evaluation, after the optional Latin filter."""
        if self._cached_examples is None:
            examples = self.load_examples()
            if self.clean_latin_rows:
                n_before = len(examples)
                examples = [ex for ex in examples if not self._row_has_latin(ex)]
                pct = 100.0 * (n_before - len(examples)) / max(n_before, 1)
                logger.info(
                    "%s clean_latin_rows: dropped %d/%d rows (%.1f%% removed)",
                    self.name, n_before - len(examples), n_before, pct,
                )
                if not examples:
                    raise RuntimeError(
                        f"{self.name}: clean_latin_rows dropped every row (was {n_before})."
                    )
            self._cached_examples = examples
            logger.info("%s eval set: %d rows (full benchmark)", self.name, len(examples))
        return self._cached_examples

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _compute_downstream_unk_records(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: BaseTokenizer,
    ) -> Dict[str, WordUnkRecord]:
        """Scan each example's prompt + every continuation for UNK tokens
        and aggregate occurrences per unique source word.

        Uses the *same* prompt that the model was scored on
        (``_format_eval_context_with_fewshot`` — picks up few-shot demos),
        so the report reflects what the tokenizer actually saw. ``source_field``
        is ``"prompt"`` for the context and ``"continuation_<i>"`` for the
        i-th continuation. Returns an empty dict when the tokenizer has no
        usable ``unk_token`` id — the caller still writes a header-only CSV.
        """
        occs_by_example = []
        for ex in examples:
            prompt = self._format_eval_context_with_fewshot(ex)
            continuations = self._build_continuations(ex)
            ex_occs = list(scan_text(tokenizer, prompt, source_field="prompt"))
            for i, c in enumerate(continuations):
                ex_occs.extend(
                    scan_text(tokenizer, c, source_field=f"continuation_{i}")
                )
            occs_by_example.append(ex_occs)
        return aggregate_occurrences(occs_by_example)

    def _open_row_writer(
        self,
        row_dump_dir: Path,
        tokenizer: BaseTokenizer,
        score_normalization: str,
        n_examples: int,
    ) -> EvalRowWriter:
        """Open the per-row dump for this task, stamped with its run context.

        Only facts available here go into the metadata: what the file says
        about itself must be true of the file. The sweep cell and the model
        checkpoint are recorded by the pipeline in ``config.json`` next to it.
        """
        embedding = getattr(tokenizer, "embedding_type", EmbeddingType.STANDARD)
        try:
            vocab_size = int(tokenizer.vocab_size)
        except Exception:  # noqa: BLE001 - a fake tokenizer in a test
            vocab_size = None
        meta = {
            "task": self.name,
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "n_examples": int(n_examples),
            "score_normalization": score_normalization,
            "primary": "pmi" if score_normalization in ("pmi", "char+pmi") else "char",
            "max_length": int(self.max_length),
            "num_fewshot": int(self.num_fewshot),
            "clean_latin_rows": bool(self.clean_latin_rows),
            "label_rotation": (
                int(self.label_rotation) if hasattr(self, "label_rotation") else None
            ),
            "rows_file": (
                None if self._rows_subset is None
                else {k: self._rows_subset[k] for k in ("path", "sha256", "n_rows")}
            ),
            "tokenizer_class": type(tokenizer).__name__,
            "vocab_size": vocab_size,
            "embedding_type": embedding,
            "unit": UNIT_BY_EMBEDDING.get(embedding, "tokens"),
        }
        path = Path(row_dump_dir) / f"{self.name}.parquet"
        return EvalRowWriter(path, metadata=meta)

    @torch.no_grad()
    def evaluate(
        self,
        model: BaseModelAdapter,
        tokenizer: BaseTokenizer,
        split: str = "test",
        max_samples: Optional[int] = None,
        failure_report_dir: Optional[Path] = None,
        score_normalization: str = "char",
        unk_report_dir: Optional[Path] = None,
        row_dump_dir: Optional[Path] = None,
    ) -> Dict[str, float]:
        """
        Evaluate on the **90 % held-out split** using LightEval's log-likelihood
        multiple-choice methodology.

        If ``row_dump_dir`` is given, ``<task_name>.parquet`` is written there
        with **every** scored row: the exact prompt the model received, the
        continuations, every per-choice score under every active normalization,
        and the truncation diagnostics. This is a strict superset of the
        failure report, so when both are requested the failure report is
        skipped and the dump stands in for it.

        If ``failure_report_dir`` is given (and no row dump is being written),
        a ``<task_name>_accuracy_failures.parquet`` is written there with one
        row per wrong-answer example.

        If ``unk_report_dir`` is given, a ``<task_name>_unks.parquet`` is
        written there listing the unique source words that produced UNK tokens
        in either the prompt or any continuation. Empty (schema-only) when no
        UNK was seen or the tokenizer has no ``unk_token`` id.

        ``score_normalization`` selects the aggregation policy:
          * ``"char"`` (default) — char-length normalization (existing behavior;
            backward-compatible).
          * ``"pmi"`` — subtract unconditioned per-continuation log-likelihoods.
            ``accuracy`` and the failure CSV reflect the PMI argmax.
          * ``"char+pmi"`` — compute both. ``accuracy`` aliases ``accuracy_char_norm``
            for backward compat with comparison-report consumers.
        """
        if score_normalization not in ("char", "pmi", "char+pmi"):
            raise ValueError(
                f"Unknown score_normalization={score_normalization!r}; "
                "expected one of 'char', 'pmi', 'char+pmi'."
            )
        if not LIGHTEVAL_AVAILABLE:
            logger.warning(
                "lighteval package not found. Running built-in log-likelihood "
                "evaluation (identical methodology to LightEval)."
            )

        # The full list stays cached (few-shot pools and seeds are drawn from
        # it); ``rows_file`` only chooses which rows are iterated.
        examples, row_indices = self._select_rows(self.get_eval_examples(), max_samples)

        logger.info(
            "%s: evaluating %d examples (%s, normalization=%s)",
            self.name, len(examples),
            f"rows_file {self.rows_file}" if row_indices is not None else "full benchmark",
            score_normalization,
        )
        model.model.eval()
        wrapper = LightEvalModelWrapper(model, tokenizer, max_length=self.max_length)
        # The dump carries every failure row and then some, so asking for both
        # would write the same wrong answers twice in two shapes.
        collect = failure_report_dir is not None and row_dump_dir is None
        if failure_report_dir is not None and row_dump_dir is not None:
            logger.info(
                "%s: row dump enabled — skipping the failure report "
                "(it is the dump filtered to correct == False)", self.name,
            )
        writer = (
            self._open_row_writer(row_dump_dir, tokenizer, score_normalization, len(examples))
            if row_dump_dir is not None else None
        )
        # Pass `self` so the wrapper uses this task's prompt/continuation/scoring hooks.
        try:
            metrics, failures = wrapper.evaluate_mcq(
                examples,
                collect_failures=collect,
                task=self,
                score_normalization=score_normalization,
                row_sink=writer.write if writer is not None else None,
                row_indices=row_indices,
            )
        finally:
            if writer is not None:
                writer.close()
        logger.info("%s metrics: %s", self.name, metrics)

        if collect:
            max_choices = max((len(ex["choices"]) for ex in examples), default=0)
            fieldnames: List[str] = [
                "index", "question",
                *[f"choice_{i}" for i in range(max_choices)],
                "gold_idx", "gold_letter", "pred_idx", "pred_letter",
                *[f"ll_{i}" for i in range(max_choices)],
                "ll_margin",
                *[f"score_{i}" for i in range(max_choices)],
                "score_margin",
            ]
            if score_normalization in ("pmi", "char+pmi"):
                fieldnames.extend(
                    [*[f"score_pmi_{i}" for i in range(max_choices)], "score_pmi_margin"]
                )
            out_path = Path(failure_report_dir) / f"{self.name}_accuracy_failures.parquet"
            n_written = write_report_table(
                out_path, failures, fieldnames,
                metadata={"task": self.name, "kind": "accuracy_failures",
                          "score_normalization": score_normalization},
            )
            logger.info(
                "%s: wrote %d failure rows to %s", self.name, n_written, out_path
            )

        if unk_report_dir is not None:
            udir = Path(unk_report_dir)
            udir.mkdir(parents=True, exist_ok=True)
            records = self._compute_downstream_unk_records(examples, tokenizer)
            rows = records_to_rows(records.values(), DOWNSTREAM_UNK_FIELDS)
            out_path = udir / f"{self.name}_unks.parquet"
            n_unk_written = write_report_table(
                out_path, rows, DOWNSTREAM_UNK_FIELDS,
                metadata={"task": self.name, "kind": "downstream_unks"},
            )
            logger.info(
                "%s: wrote %d UNK word rows to %s",
                self.name, n_unk_written, out_path,
            )

        # Stamp the eval-preprocessing flag into the metrics dict so downstream
        # comparison-report consumers can detect mixed runs (clean vs unclean).
        metrics["clean_latin_rows"] = self.clean_latin_rows
        if hasattr(self, "label_rotation"):
            metrics["label_rotation"] = int(self.label_rotation)
        if self._rows_subset is not None:
            metrics["rows_file"] = {k: self._rows_subset[k] for k in ("path", "sha256", "n_rows")}

        return metrics

    @property
    def metric_names(self) -> List[str]:
        return ["accuracy", "num_samples"]
