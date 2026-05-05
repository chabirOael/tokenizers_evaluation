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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.tasks.base import BaseTask
from arabic_eval.tasks.lighteval.utils import (
    ARABIC_CHOICE_LETTERS,
    format_mcq_context,
)
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType
from arabic_eval.tokenizers.utils.arabic_text import contains_latin_letters
from arabic_eval.utils.io import write_failure_csv

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

@torch.no_grad()
def _compute_loglikelihood(
    model: BaseModelAdapter,
    tokenizer: BaseTokenizer,
    context: str,
    continuation: str,
    max_length: int = 512,
) -> float:
    """
    Compute log P(continuation | context) following LightEval's approach.

    The full sequence ``context + continuation`` is encoded once; the model's
    forward pass yields logits; we sum the log-probabilities of the continuation
    tokens only (the tokens after ``len(context_tokens)``).

    CharacterBERT (``character_cnn``) uses word-level logits: the batch is built
    from ``char_ids`` instead of ``input_ids``, but continuation scoring follows
    the same causal-LM approach using word vocabulary indices.
    """
    full_text = context + continuation
    ctx_enc = tokenizer.encode(context, max_length=max_length, truncation=True, padding=False)
    full_enc = tokenizer.encode(full_text, max_length=max_length, truncation=True, padding=False)

    ctx_len = len(ctx_enc.input_ids)
    full_len = len(full_enc.input_ids)

    if full_len <= ctx_len:
        # Continuation was completely truncated — should not happen for single letters.
        return -1e9

    attention_mask = torch.tensor([full_enc.attention_mask], device=model.device)

    if tokenizer.embedding_type == EmbeddingType.CHARACTER_CNN:
        # CharacterBERT: input is 3-D char_ids; output logits are over word vocab.
        char_ids = torch.tensor([full_enc.char_ids], device=model.device)
        batch = {"char_ids": char_ids, "attention_mask": attention_mask}
    else:
        input_ids = torch.tensor([full_enc.input_ids], device=model.device)
        batch = {"input_ids": input_ids, "attention_mask": attention_mask,
                 "labels": input_ids.clone()}

    output = model.forward(batch)
    logits = output["logits"]                          # [1, seq_len, vocab_size]
    log_probs = F.log_softmax(logits[0], dim=-1)       # [seq_len, vocab_size]

    # Causal LM: position i predicts position i+1.
    # Sum log P(token_{ctx_len} … token_{full_len-1}) given their left contexts.
    total_ll = 0.0
    for pos in range(ctx_len - 1, full_len - 1):
        next_tok = full_enc.input_ids[pos + 1]
        total_ll += log_probs[pos, next_tok].item()

    return total_ll


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

    def evaluate_mcq(
        self,
        examples: List[Dict[str, Any]],
        collect_failures: bool = False,
        task: Optional["LightEvalBenchmarkTask"] = None,
        score_normalization: str = "char",
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
        for downstream CSV reporting. Under PMI modes "failure" is defined by
        the PMI argmax (since PMI is the corrected scoring); the record also
        carries ``score_pmi_*`` and ``score_pmi_margin`` columns.
        """
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

        for idx, ex in enumerate(tqdm(examples, desc="LightEval MCQ", unit="example")):
            if task is not None:
                context = task._format_eval_context(ex)
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

        # Build the metrics dict. Under default ``"char"`` mode the shape is
        # byte-identical to the pre-PMI version (no ``accuracy_char_norm`` /
        # ``accuracy_pmi`` keys at all). Those keys appear *only* when the
        # operator opts into a PMI mode.
        metrics: Dict[str, Any] = {"num_samples": total}
        char_acc = round(correct_char / max(total, 1), 4) if want_char else None
        pmi_acc = round(correct_pmi / max(total, 1), 4) if want_pmi else None
        if want_char and not want_pmi:
            metrics["accuracy"] = char_acc
        elif want_pmi and not want_char:
            metrics["accuracy"] = pmi_acc
            metrics["accuracy_pmi"] = pmi_acc
        else:  # char+pmi
            metrics["accuracy"] = char_acc
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
            else:  # char+pmi
                entry["accuracy"] = char_a
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

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.dataset_name: str = config.get("dataset_name", self._default_dataset_name())
        self.dataset_config: Optional[str] = config.get("dataset_config", None)
        self.cache_dir: str = config.get("cache_dir", "outputs/data_cache")
        self.max_length: int = config.get("max_length", 512)
        self.seed: int = config.get("seed", 42)
        self.clean_latin_rows: bool = bool(config.get("clean_latin_rows", False))
        self._cached_examples: Optional[List[Dict]] = None

    # ------------------------------------------------------------------
    # Abstract hooks (every subclass MUST implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def _default_dataset_name(self) -> str:
        """Default HuggingFace dataset identifier."""
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

        Default: the bare answer prefix ``"### الإجابة:"`` — matches the
        trailing line every existing benchmark prompt ends with under the
        ``###``-block formatting. Tasks whose prompt uses a different
        answer-prefix convention can override this; the default is the right
        answer for all four current tasks (acva / alghafa /
        culture_arabic_mmlu / arabic_exam).
        """
        return "### الإجابة:"

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

    @torch.no_grad()
    def evaluate(
        self,
        model: BaseModelAdapter,
        tokenizer: BaseTokenizer,
        split: str = "test",
        max_samples: Optional[int] = None,
        failure_report_dir: Optional[Path] = None,
        score_normalization: str = "char",
    ) -> Dict[str, float]:
        """
        Evaluate on the **90 % held-out split** using LightEval's log-likelihood
        multiple-choice methodology.

        If ``failure_report_dir`` is given, a ``<task_name>_accuracy_failures.csv``
        is written there with one row per wrong-answer example.

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

        examples = self.get_eval_examples()
        if max_samples:
            examples = examples[:max_samples]

        logger.info(
            "%s: evaluating %d examples (90 %% LightEval split, normalization=%s)",
            self.name, len(examples), score_normalization,
        )
        model.model.eval()
        wrapper = LightEvalModelWrapper(model, tokenizer, max_length=self.max_length)
        collect = failure_report_dir is not None
        # Pass `self` so the wrapper uses this task's prompt/continuation/scoring hooks.
        metrics, failures = wrapper.evaluate_mcq(
            examples,
            collect_failures=collect,
            task=self,
            score_normalization=score_normalization,
        )
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
            csv_path = Path(failure_report_dir) / f"{self.name}_accuracy_failures.csv"
            n_written = write_failure_csv(csv_path, failures, fieldnames)
            logger.info(
                "%s: wrote %d failure rows to %s", self.name, n_written, csv_path
            )

        # Stamp the eval-preprocessing flag into the metrics dict so downstream
        # comparison-report consumers can detect mixed runs (clean vs unclean).
        metrics["clean_latin_rows"] = self.clean_latin_rows

        return metrics

    @property
    def metric_names(self) -> List[str]:
        return ["accuracy", "num_samples"]
