"""ACVA — Arabic Culture and Values Assessment (OALL/ACVA).

Schema: id, question, answer — where answer is "صح" (True) or "خطأ" (False).
True/False dataset; choices presented as ``["صح", "خطأ"]``.

ACVA-specific scoring decision: instead of LightEval's standard letter-based
multiple-choice scoring (where the model is asked to emit " أ" or " ب"),
ACVA scores the continuation strings ``" صح"`` / ``" خطأ"`` directly.
Single-letter continuations carry almost no signal — every Arabic letter
has roughly the same unigram prior, so the choice is dominated by noise.
Scoring the actual T/F words gives the model lexically meaningful options.
SFT formatting matches: training examples end with the full word, not a letter.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from arabic_eval.registry import task_registry
from arabic_eval.tasks.lighteval.base import LightEvalBenchmarkTask
from arabic_eval.tasks.lighteval.utils import (
    load_huggingface_mcq,
    select_aggregator,
)


@task_registry.register("acva")
class ACVATask(LightEvalBenchmarkTask):
    # Single source of truth for the dataset's True/False label strings.
    # Used by `_parse_example` (matching incoming raw rows), by
    # `_build_continuations` (eval-time scoring), and by `_format_sft_text`
    # (training-time supervision). To switch to e.g. صحيح/خاطئ, change here only.
    LABEL_TRUE = "صح"
    LABEL_FALSE = "خطأ"
    LABELS = (LABEL_TRUE, LABEL_FALSE)  # index 0 = TRUE, 1 = FALSE

    @property
    def name(self) -> str:
        return "acva"

    def _default_dataset_name(self) -> str:
        return "OALL/ACVA"

    def _parse_example(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        question = str(raw.get("question", "")).strip()
        if not question:
            return None
        answer = str(raw.get("answer", "")).strip()
        if answer == self.LABEL_TRUE:
            answer_idx = 0
        elif answer == self.LABEL_FALSE:
            answer_idx = 1
        else:
            return None
        return {
            "question": question,
            "choices": list(self.LABELS),
            "answer": answer_idx,
        }

    def load_examples(self) -> List[Dict[str, Any]]:
        return load_huggingface_mcq(
            self.dataset_name,
            parse_fn=self._parse_example,
            cache_dir=self.cache_dir,
            dataset_config=self.dataset_config,
        )

    def _format_eval_context(self, ex: Dict[str, Any]) -> str:
        # ACVA is True/False — no need to display "أ. صح / ب. خطأ" choice
        # lines because the answer is the word itself, not a letter.
        return f"السؤال: {ex['question']}\nالإجابة:"

    def _build_continuations(self, ex: Dict[str, Any]) -> List[str]:
        # Score the words themselves, in the same index order as `choices`
        # (so argmax over log-likelihoods → answer index, unchanged).
        return [f" {label}" for label in self.LABELS]

    def _format_sft_text(self, ex: Dict[str, Any]) -> str:
        # SFT supervision must match eval scoring: train on the word, not
        # the letter, otherwise the model never sees " صح"/" خطأ" during
        # fine-tuning.
        return f"{self._format_eval_context(ex)} {self.LABELS[ex['answer']]}"

    def _aggregate_scores(
        self,
        ex: Dict[str, Any],
        continuations: List[str],
        log_likelihoods: List[float],
        unconditioned_log_likelihoods: Optional[List[float]] = None,
        normalization: str = "char",
    ) -> List[float]:
        # Char-norm matters here: " صح" is 2 chars, " خطأ" is 3 chars; without
        # normalization the longer answer is systematically penalised.
        # PMI is also valid — it subtracts the per-word prior under the bare
        # answer-prefix context (``الإجابة:``).
        return select_aggregator(
            continuations, log_likelihoods,
            unconditioned_log_likelihoods, normalization,
        )
