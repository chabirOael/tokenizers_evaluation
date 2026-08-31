"""Alghafa — AlGhafa Native Arabic benchmark
(OALL/AlGhafa-Arabic-LLM-Benchmark-Native).

Heterogeneous: 9 sub-configs spanning 2/3/4/5-way MCQ shapes. Every sub-config
is merged into one pool and a single aggregate accuracy is reported (per-
sub-config breakdown is also emitted via ``per_subconfig_accuracy`` on the
metrics dict).

Schema: ``query``, ``sol1`` … ``sol5``, ``label``. **``label`` is 0-indexed**
(``"0"`` → first option correct), matching LightEval's ``alghafa_adapter``.
``sol5`` is only present in two grounded-statement configs.

**Scoring (LightEval-official format, post 2026-05-06).** Every sub-config
uses the same numeric-list prompt and scores the **choice text** directly.
The per-sub-config word/letter dispatch was removed — LightEval's official
``alghafa_prompt`` always shows choices as ``0) {c}\\n1) {c}\\n…`` and scores
``log P(choice_text | prompt)``, regardless of whether the choices are
binary words (T/F, sentiment) or longer phrases. PMI normalization handles
the per-continuation prior; char-norm handles continuation length variance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from arabic_eval.registry import task_registry
from arabic_eval.tasks.lighteval.base import LightEvalBenchmarkTask
from arabic_eval.tasks.lighteval.utils import (
    format_mcq_context_numeric_official,
    load_huggingface_mcq,
    select_aggregator,
)


@task_registry.register("alghafa")
class AlghafaTask(LightEvalBenchmarkTask):

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

    def _format_eval_context(self, ex: Dict[str, Any]) -> str:
        # Official LightEval prompt: numeric list (`0) ...`, `1) ...`) with
        # the standard Arabic instruction prefix. Continuations score the
        # choice text directly (numbers in the prompt are display-only).
        return format_mcq_context_numeric_official(ex["question"], ex["choices"])

    def _build_continuations(self, ex: Dict[str, Any]) -> List[str]:
        # Score the choice text strings directly (LightEval convention). The
        # numeric markers in the prompt are display-only; the model picks the
        # answer by predicting which choice text is most likely.
        return [f" {choice}" for choice in ex["choices"]]

    def _aggregate_scores(
        self,
        ex: Dict[str, Any],
        continuations: List[str],
        log_likelihoods: List[float],
        unconditioned_log_likelihoods: Optional[List[float]] = None,
        normalization: str = "char",
    ) -> List[float]:
        # Continuations vary in character length per row (since they're choice
        # text), so char-norm is essential to avoid favoring shorter answers.
        # PMI removes the per-continuation prior under the bare answer prefix
        # ``الإجابة:`` (supplied by `_unconditioned_query` on the base class).
        return select_aggregator(
            continuations, log_likelihoods,
            unconditioned_log_likelihoods, normalization,
        )
