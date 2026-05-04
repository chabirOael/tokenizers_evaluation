"""Opt-in helpers for LightEval benchmark dataset implementations.

Datasets in this package import what they need from here. Nothing in
``base.py`` reaches into this module — the base class is dataset-agnostic
and these helpers are the conventions a *dataset file* may opt into.

Conventions covered:
  * Letter-MCQ prompt format (``format_mcq_context``, ``format_mcq_full``)
  * Latin / Arabic choice-letter constants
  * Char-length normalization aggregator (LightEval ``LogProbCharNorm``
    equivalent)
  * Generic A/B/C/D-style row parser (``parse_mcq_generic``)
  * HuggingFace MCQ loader with multi-config auto-detection
    (``load_huggingface_mcq``)

If a future dataset needs none of these (e.g. a free-form-answer benchmark
or one loaded from a non-HF source), it simply implements its own
``load_examples`` and imports nothing from here.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

logger = logging.getLogger("arabic_eval.tasks.lighteval.utils")


# Latin letters — used ONLY for dataset field-name lookups and answer-key parsing
# (datasets store choices as columns "A"/"B"/"C"/"D" and answer keys as "A"–"D").
CHOICE_LETTERS: List[str] = ["A", "B", "C", "D", "E"]

# Arabic letters — used in user-facing prompt text (SFT training + evaluation)
# for letter-MCQ datasets.
ARABIC_CHOICE_LETTERS: List[str] = ["أ", "ب", "ج", "د", "هـ"]


def format_mcq_context(question: str, choices: List[str]) -> str:
    """Letter-labelled MCQ prompt: question + lettered choice list + ``الإجابة:``."""
    lines = [f"السؤال: {question}", ""]
    for i, choice in enumerate(choices):
        letter = ARABIC_CHOICE_LETTERS[i] if i < len(ARABIC_CHOICE_LETTERS) else str(i)
        lines.append(f"{letter}. {choice}")
    lines.append("الإجابة:")
    return "\n".join(lines)


def format_mcq_full(question: str, choices: List[str], answer_idx: int) -> str:
    """Letter-labelled MCQ prompt followed by the correct-answer letter (for SFT)."""
    context = format_mcq_context(question, choices)
    letter = ARABIC_CHOICE_LETTERS[answer_idx] if answer_idx < len(ARABIC_CHOICE_LETTERS) else str(answer_idx)
    return f"{context} {letter}"


def char_norm_aggregator(
    continuations: List[str],
    log_likelihoods: List[float],
) -> List[float]:
    """LightEval ``LogProbCharNorm`` equivalent: divide each log-likelihood by
    the character length of its continuation (leading whitespace stripped).

    For 1-character continuations (standard letter-based MCQ) this is a no-op
    (divide by 1). It only meaningfully shifts picks when continuations
    differ in length — e.g. ACVA (``صح`` 2 chars vs ``خطأ`` 3 chars), Alghafa's
    word-scored sub-configs.
    """
    return [
        ll / max(len(c.lstrip()), 1)
        for ll, c in zip(log_likelihoods, continuations)
    ]


def pmi_aggregator(
    log_likelihoods: List[float],
    unconditioned_log_likelihoods: List[float],
) -> List[float]:
    """LightEval ``LogProbPMINorm`` equivalent:
    ``score_i = ll_i − unconditioned_ll_i``.

    Removes the per-continuation prior (e.g. unigram letter prior on
    Arabic-MCQ tasks) that otherwise dominates argmax decisions when the
    question-conditioned signal is weak. The unconditioned context is the
    bare answer prefix (``"الإجابة:"`` for the four current tasks; supplied
    by ``task._unconditioned_query``).
    """
    if len(log_likelihoods) != len(unconditioned_log_likelihoods):
        raise ValueError(
            f"PMI aggregator: ll length ({len(log_likelihoods)}) must match "
            f"unconditioned_ll length ({len(unconditioned_log_likelihoods)})"
        )
    return [
        ll - u for ll, u in zip(log_likelihoods, unconditioned_log_likelihoods)
    ]


def select_aggregator(
    continuations: List[str],
    log_likelihoods: List[float],
    unconditioned_log_likelihoods: Optional[List[float]],
    normalization: str,
) -> List[float]:
    """Dispatch to the right aggregator based on ``normalization``.

    Used by every concrete task's ``_aggregate_scores`` so the per-task
    override only controls which dispatcher is reached (and any task-specific
    pre-processing) — the actual aggregation policy stays centralised.
    """
    if normalization == "char":
        return char_norm_aggregator(continuations, log_likelihoods)
    if normalization == "pmi":
        if unconditioned_log_likelihoods is None:
            raise ValueError(
                "normalization='pmi' requires unconditioned_log_likelihoods"
            )
        return pmi_aggregator(log_likelihoods, unconditioned_log_likelihoods)
    raise ValueError(
        f"Unknown normalization={normalization!r}; expected 'char' or 'pmi'."
    )


def parse_mcq_generic(
    raw: Dict[str, Any],
    question_keys: Tuple[str, ...] = ("question",),
    answer_keys: Tuple[str, ...] = ("answer", "label", "answerKey"),
) -> Optional[Dict[str, Any]]:
    """
    Generic parser that handles the most common MCQ dataset schemas:

    * Separate A/B/C/D columns + answer letter/index
    * ``choices`` or ``options`` list + answer index
    * PIQA-style ``sol1``/``sol2`` + label

    Returns ``{"question", "choices", "answer"}`` or ``None`` if the row
    doesn't match any supported shape.
    """
    question = ""
    for key in question_keys:
        question = str(raw.get(key, "")).strip()
        if question:
            break
    if not question:
        return None

    choices: List[str] = []
    if "A" in raw and "B" in raw:
        for letter in CHOICE_LETTERS:
            val = raw.get(letter, "")
            if val:
                choices.append(str(val).strip())
    elif "choices" in raw:
        choices = [str(c).strip() for c in raw["choices"]]
    elif "options" in raw:
        choices = [str(c).strip() for c in raw["options"]]
    elif "sol1" in raw and "sol2" in raw:
        choices = [str(raw["sol1"]).strip(), str(raw["sol2"]).strip()]

    if not choices:
        return None

    answer_raw = None
    for key in answer_keys:
        answer_raw = raw.get(key)
        if answer_raw is not None:
            break
    if answer_raw is None:
        return None

    if isinstance(answer_raw, str) and answer_raw.strip().upper() in CHOICE_LETTERS:
        answer_idx = CHOICE_LETTERS.index(answer_raw.strip().upper())
    elif isinstance(answer_raw, (int, np.integer)):
        answer_idx = int(answer_raw)
    else:
        try:
            answer_idx = int(answer_raw)
        except (ValueError, TypeError):
            return None

    if answer_idx < 0 or answer_idx >= len(choices):
        return None

    return {"question": question, "choices": choices, "answer": answer_idx}


# ---------------------------------------------------------------------------
# HuggingFace MCQ loader
# ---------------------------------------------------------------------------

ParseFn = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]


def _parse_and_stamp(
    combined,
    parse_fn: ParseFn,
    source_config: str,
    *,
    quiet: bool = False,
    context: str = "",
) -> List[Dict[str, Any]]:
    """Iterate rows, call ``parse_fn``, stamp ``_source_config`` on accepted rows.

    The ``_source_config`` key is what the base class's stratified split keys
    on. Single-config datasets use the sentinel ``"_default"``.
    """
    examples: List[Dict[str, Any]] = []
    for raw in combined:
        parsed = parse_fn(dict(raw))
        if parsed is not None:
            parsed["_source_config"] = source_config
            examples.append(parsed)

    if not examples and not quiet:
        raise RuntimeError(
            f"No valid MCQ examples found in {context!r}. "
            "Check that the parse function matches the dataset schema."
        )
    if not quiet:
        logger.info("Loaded %d valid MCQ examples from %s", len(examples), context)
    return examples


def _load_multi_config(
    dataset_name: str,
    parse_fn: ParseFn,
    cache_dir: str,
    excluded_configs: FrozenSet[str],
) -> List[Dict[str, Any]]:
    """Enumerate every sub-config of a multi-config dataset, parse + stamp each.

    Used by ``load_huggingface_mcq`` when the initial ``load_dataset(name, None)``
    raises ``ValueError: Config name is missing``.
    """
    try:
        config_names = get_dataset_config_names(dataset_name)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot enumerate configs for '{dataset_name}': {exc}. "
            "Pass a specific dataset_config to load_huggingface_mcq instead."
        ) from exc

    excluded = [c for c in config_names if c in excluded_configs]
    if excluded:
        logger.info(
            "Excluding %d/%d sub-config(s) of '%s': %s",
            len(excluded), len(config_names), dataset_name, excluded,
        )
        config_names = [c for c in config_names if c not in excluded_configs]

    logger.info(
        "Multi-config dataset '%s': merging %d sub-configs (%s … %s)",
        dataset_name, len(config_names),
        config_names[0] if config_names else "",
        config_names[-1] if config_names else "",
    )
    all_examples: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for config_name in config_names:
        try:
            raw_ds = load_dataset(dataset_name, config_name, cache_dir=cache_dir)
            combined = concatenate_datasets(list(raw_ds.values()))
            n_before = len(all_examples)
            all_examples.extend(
                _parse_and_stamp(combined, parse_fn, config_name, quiet=True)
            )
            logger.debug("Config '%s': +%d examples", config_name, len(all_examples) - n_before)
        except Exception as exc:
            logger.warning("Skipping config '%s' of '%s': %s", config_name, dataset_name, exc)
            skipped.append(config_name)

    if skipped:
        logger.warning(
            "Skipped %d/%d sub-configs of '%s': %s%s",
            len(skipped), len(config_names), dataset_name,
            skipped[:3], " …" if len(skipped) > 3 else "",
        )

    if not all_examples:
        raise RuntimeError(
            f"No valid MCQ examples found across all {len(config_names)} "
            f"sub-configs of '{dataset_name}'. "
            "Check that the parse function matches the dataset schema."
        )

    logger.info(
        "Loaded %d total examples from %d/%d sub-configs of '%s'",
        len(all_examples), len(config_names) - len(skipped), len(config_names), dataset_name,
    )
    return all_examples


def load_huggingface_mcq(
    dataset_name: str,
    parse_fn: ParseFn,
    cache_dir: str,
    *,
    dataset_config: Optional[str] = None,
    excluded_configs: FrozenSet[str] = frozenset(),
) -> List[Dict[str, Any]]:
    """Load a HuggingFace MCQ dataset and return parsed, ``_source_config``-stamped rows.

    Auto-detects single vs multi-config: if ``load_dataset(name, None)`` raises
    ``ValueError: Config name is missing``, falls through to enumerating
    sub-configs via ``get_dataset_config_names`` and merging them. ``excluded_configs``
    is honoured only on the multi-config path (e.g. ``frozenset({"All"})`` for
    ``MBZUAI/ArabicMMLU``, whose ``All`` config is a strict union of the others
    and would otherwise duplicate every row).

    All predefined splits in the loaded ``DatasetDict`` are concatenated so
    callers can do their own train/eval split downstream without depending on
    the dataset author's predefined splits.
    """
    try:
        raw_ds = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
    except ValueError as exc:
        if "Config name is missing" in str(exc) and dataset_config is None:
            return _load_multi_config(dataset_name, parse_fn, cache_dir, excluded_configs)
        raise RuntimeError(
            f"Failed to load '{dataset_name}' (config={dataset_config!r}): {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load '{dataset_name}' (config={dataset_config!r}): {exc}"
        ) from exc

    combined = concatenate_datasets(list(raw_ds.values()))
    return _parse_and_stamp(
        combined,
        parse_fn,
        dataset_config or "_default",
        context=f"{dataset_name} (config={dataset_config!r})",
    )
