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
    LABEL_ROTATION_SPEC,
    choice_letters,
    format_mcq_context_letter_official,
    load_huggingface_mcq,
    parse_mcq_generic,
    read_label_rotation,
    select_aggregator,
)


@task_registry.register("culture_arabic_mmlu")
class CultureArabicMMLUTask(LightEvalBenchmarkTask):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        # Letters rotated over the slots (diagnostic; 0 = the official prompt).
        self.label_rotation: int = read_label_rotation(config)

    @classmethod
    def param_spec(cls):
        return [*super().param_spec(), LABEL_ROTATION_SPEC]

    @property
    def name(self) -> str:
        return "culture_arabic_mmlu"

    @classmethod
    def _default_dataset_name(cls) -> str:
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
        # Official LightEval letter-MCQ format (no ``###`` markers, instruction
        # prefix + Arabic-letter listings).
        return format_mcq_context_letter_official(
            ex["question"], ex["choices"], rotation=self.label_rotation
        )

    def _build_continuations(self, ex: Dict[str, Any]) -> List[str]:
        return [" " + letter for letter in choice_letters(len(ex["choices"]), self.label_rotation)]

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
