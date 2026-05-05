"""Alghafa — AlGhafa Native Arabic benchmark
(OALL/AlGhafa-Arabic-LLM-Benchmark-Native).

Heterogeneous: 9 sub-configs spanning 2/3/4/5-way MCQ shapes. Every sub-config
is merged into one pool and a single aggregate accuracy is reported (per-
sub-config breakdown is also emitted via ``per_subconfig_accuracy`` on the
metrics dict).

Schema: ``query``, ``sol1`` … ``sol5``, ``label``. **``label`` is 0-indexed**
(``"0"`` → first option correct), matching LightEval's ``alghafa_adapter``.
``sol5`` is only present in two grounded-statement configs.

Topic mix (rough share of the eval pool):

  ~35 % T/F + 2-way binary sentiment → word-scored (see overrides below).
  ~34 % 3-way sentiment             → word-scored.
  ~30 % 4-way MCQ                   → letter-scored.
  ~ 1 % 5-way grounded statement    → letter-scored.

The word-scored sub-configs use the same protocol as ACVA: drop the
letter-listing prompt, score the answer text directly with char-normalization.
Letter-scored on these binary configs hits the ACVA letter-prior pathology
(decisions dominated by unigram letter prior, near-tie margins, accuracy
clusters near majority class). Per-config dispatch is keyed on
``ex["_source_config"]`` populated by the loader.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from arabic_eval.registry import task_registry
from arabic_eval.tasks.lighteval.base import LightEvalBenchmarkTask
from arabic_eval.tasks.lighteval.utils import (
    ARABIC_CHOICE_LETTERS,
    format_mcq_context,
    load_huggingface_mcq,
    select_aggregator,
)


@task_registry.register("alghafa")
class AlghafaTask(LightEvalBenchmarkTask):
    # Sub-configs that should be scored word-wise rather than letter-wise.
    # Single source of truth for the per-row dispatch in the four
    # prompt/scoring hooks below.
    WORD_SCORED_CONFIGS: frozenset = frozenset({
        "multiple_choice_facts_truefalse_balanced_task",
        "multiple_choice_rating_sentiment_no_neutral_task",
        "multiple_choice_rating_sentiment_task",
        "multiple_choice_sentiment_task",
    })

    @property
    def name(self) -> str:
        return "alghafa"

    def _default_dataset_name(self) -> str:
        return "OALL/AlGhafa-Arabic-LLM-Benchmark-Native"

    def _parse_example(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        question = str(raw.get("query", "")).strip()
        if not question:
            return None
        # Enumerate sol1..sol5: two grounded-statement sub-configs ship sol5
        # and a label up to "4". Smaller sub-configs (T/F, sentiment) only
        # populate sol1..sol3 — the per-key existence check trims them.
        choices: List[str] = []
        for key in ("sol1", "sol2", "sol3", "sol4", "sol5"):
            val = raw.get(key)
            if val is not None and str(val).strip():
                choices.append(str(val).strip())
        if not choices:
            return None
        label_raw = raw.get("label")
        if label_raw is None:
            return None
        try:
            answer_idx = int(label_raw)   # 0-indexed (matches LightEval adapter)
        except (ValueError, TypeError):
            return None
        if answer_idx < 0 or answer_idx >= len(choices):
            return None
        return {"question": question, "choices": choices, "answer": answer_idx}

    def load_examples(self) -> List[Dict[str, Any]]:
        return load_huggingface_mcq(
            self.dataset_name,
            parse_fn=self._parse_example,
            cache_dir=self.cache_dir,
            dataset_config=self.dataset_config,
        )

    # ---- Per-topic prompt / continuation dispatch ----

    def _is_word_scored(self, ex: Dict[str, Any]) -> bool:
        return ex.get("_source_config") in self.WORD_SCORED_CONFIGS

    def _format_eval_context(self, ex: Dict[str, Any]) -> str:
        if self._is_word_scored(ex):
            # No "أ. <choice>" listing — the answer is the choice text itself,
            # not a letter referring to it. Mirrors ACVA's eval prompt shape.
            return f"### السؤال:\n{ex['question']}\n\n### الإجابة:"
        return format_mcq_context(ex["question"], ex["choices"])

    def _build_continuations(self, ex: Dict[str, Any]) -> List[str]:
        if self._is_word_scored(ex):
            # Score the answer text directly. Char-normalization in
            # _aggregate_scores rescues the length bias that would otherwise
            # favor the shorter answer.
            return [f" {choice}" for choice in ex["choices"]]
        n = len(ex["choices"])
        return [
            " " + (ARABIC_CHOICE_LETTERS[i] if i < len(ARABIC_CHOICE_LETTERS) else str(i))
            for i in range(n)
        ]

    def _aggregate_scores(
        self,
        ex: Dict[str, Any],
        continuations: List[str],
        log_likelihoods: List[float],
        unconditioned_log_likelihoods: Optional[List[float]] = None,
        normalization: str = "char",
    ) -> List[float]:
        # Letter-scored sub-configs all have 1-char continuations → char-norm
        # is a mathematical no-op. Word-scored sub-configs need it. PMI is
        # also valid for both shapes (subtracts the per-letter / per-text
        # prior under the bare answer-prefix context).
        return select_aggregator(
            continuations, log_likelihoods,
            unconditioned_log_likelihoods, normalization,
        )
