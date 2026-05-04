"""Culture Arabic MMLU — Arabic MMLU (OALL/Arabic_MMLU, AceGPT Arabic MMLU).

Schema: question, A, B, C, D, answer (letter), subject.
Single-config dataset. Uses the standard LightEval letter-MCQ conventions:
letter-listed prompt, single Arabic-letter continuations, char-norm
aggregation (no-op for 1-char continuations but kept for symmetry).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from arabic_eval.registry import task_registry
from arabic_eval.tasks.lighteval.base import LightEvalBenchmarkTask
from arabic_eval.tasks.lighteval.utils import (
    ARABIC_CHOICE_LETTERS,
    format_mcq_context,
    format_mcq_full,
    load_huggingface_mcq,
    parse_mcq_generic,
    select_aggregator,
)


@task_registry.register("culture_arabic_mmlu")
class CultureArabicMMLUTask(LightEvalBenchmarkTask):
    @property
    def name(self) -> str:
        return "culture_arabic_mmlu"

    def _default_dataset_name(self) -> str:
        return "OALL/Arabic_MMLU"

    def _parse_example(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return parse_mcq_generic(
            raw,
            question_keys=("question",),
            answer_keys=("answer", "label"),
        )

    def load_examples(self) -> List[Dict[str, Any]]:
        return load_huggingface_mcq(
            self.dataset_name,
            parse_fn=self._parse_example,
            cache_dir=self.cache_dir,
            dataset_config=self.dataset_config,
        )

    def _format_eval_context(self, ex: Dict[str, Any]) -> str:
        return format_mcq_context(ex["question"], ex["choices"])

    def _build_continuations(self, ex: Dict[str, Any]) -> List[str]:
        n = len(ex["choices"])
        return [
            " " + (ARABIC_CHOICE_LETTERS[i] if i < len(ARABIC_CHOICE_LETTERS) else str(i))
            for i in range(n)
        ]

    def _format_sft_text(self, ex: Dict[str, Any]) -> str:
        return format_mcq_full(ex["question"], ex["choices"], ex["answer"])

    def _aggregate_scores(
        self,
        ex: Dict[str, Any],
        continuations: List[str],
        log_likelihoods: List[float],
        unconditioned_log_likelihoods: Optional[List[float]] = None,
        normalization: str = "char",
    ) -> List[float]:
        return select_aggregator(
            continuations, log_likelihoods,
            unconditioned_log_likelihoods, normalization,
        )
