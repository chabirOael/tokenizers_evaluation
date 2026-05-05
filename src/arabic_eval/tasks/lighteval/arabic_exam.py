"""Arabic Exam — MBZUAI ArabicMMLU (school exam MCQ).

Schema: Question, Context (optional, ~5 % of rows), Option 1–5,
Answer Key (letter A–E), is_few_shot (filtered), plus subject metadata.

Multi-config (one per subject); 40 subject configs are loaded. The 41st config
``All`` is excluded because it is a strict union of the other 40 — including
it would double every row and leak SFT-eval pairs across the stratified split.

When the row has a non-empty ``Context`` (Arabic poem, prose excerpt, or
math word-problem setup), it is prepended to the prompt — the question
typically refers to it explicitly ("based on the two verses…").
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from arabic_eval.registry import task_registry
from arabic_eval.tasks.lighteval.base import LightEvalBenchmarkTask
from arabic_eval.tasks.lighteval.utils import (
    ARABIC_CHOICE_LETTERS,
    CHOICE_LETTERS,
    format_mcq_context,
    load_huggingface_mcq,
    select_aggregator,
)


# OALL/ArabicMMLU's "All" config is a verbatim union of the other 40 subject
# configs (verified: |All| = sum(|other 40|) = 14575). Loading it alongside
# the subjects duplicates every row and contaminates the 10/90 SFT/eval split.
ARABIC_EXAM_EXCLUDED_CONFIGS: frozenset = frozenset({"All"})


@task_registry.register("arabic_exam")
class ArabicExamTask(LightEvalBenchmarkTask):
    @property
    def name(self) -> str:
        return "arabic_exam"

    def _default_dataset_name(self) -> str:
        return "MBZUAI/ArabicMMLU"

    def _parse_example(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # ``is_few_shot=1`` flags the dataset's own dev-split rows that the
        # original benchmark intended as in-context demonstrations (3 per
        # subject, ~120 total). They are loaded into our pool because we
        # concatenate every split, but they shouldn't be evaluated alongside
        # the test rows — we already do our own 10/90 SFT split.
        if raw.get("is_few_shot"):
            return None
        question = str(raw.get("Question", raw.get("question", ""))).strip()
        if not question:
            return None
        choices: List[str] = []
        # Option 5 exists on ~344 rows (e.g. Driving Test) and 141 rows have
        # ``Answer Key=E``. Both ``CHOICE_LETTERS`` and ``ARABIC_CHOICE_LETTERS``
        # already include the 5th entry, so iterating Options 1–5 is sufficient.
        for key in ("Option 1", "Option 2", "Option 3", "Option 4", "Option 5"):
            val = raw.get(key)
            if val is not None and str(val).strip() and str(val).lower() != "nan":
                choices.append(str(val).strip())
        if not choices:
            return None
        answer_key = str(raw.get("Answer Key", raw.get("answer", ""))).strip().upper()
        if answer_key not in CHOICE_LETTERS:
            return None
        answer_idx = CHOICE_LETTERS.index(answer_key)
        if answer_idx >= len(choices):
            return None
        # ``Context`` carries the supporting passage for ~5 % of rows. The
        # question is often unanswerable without it; store on the parsed
        # dict so the prompt builder can prepend it.
        ctx_raw = raw.get("Context")
        context = ""
        if ctx_raw is not None:
            ctx_str = str(ctx_raw).strip()
            if ctx_str and ctx_str.lower() != "nan":
                context = ctx_str
        return {
            "question": question,
            "choices": choices,
            "answer": answer_idx,
            "context": context,
        }

    def load_examples(self) -> List[Dict[str, Any]]:
        return load_huggingface_mcq(
            self.dataset_name,
            parse_fn=self._parse_example,
            cache_dir=self.cache_dir,
            dataset_config=self.dataset_config,
            excluded_configs=ARABIC_EXAM_EXCLUDED_CONFIGS,
        )

    def _format_eval_context(self, ex: Dict[str, Any]) -> str:
        ctx = ex.get("context", "")
        base = format_mcq_context(ex["question"], ex["choices"])
        if ctx:
            return f"### السياق:\n{ctx}\n\n{base}"
        return base

    def _build_continuations(self, ex: Dict[str, Any]) -> List[str]:
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
        return select_aggregator(
            continuations, log_likelihoods,
            unconditioned_log_likelihoods, normalization,
        )
