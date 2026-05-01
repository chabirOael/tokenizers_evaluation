"""Evaluation orchestrator: runs intrinsic + downstream evaluation."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.tasks.base import BaseTask
from arabic_eval.tokenizers.base import BaseTokenizer
from arabic_eval.tokenizers.intrinsic_metrics import compute_intrinsic_metrics
from arabic_eval.utils.io import save_json

logger = logging.getLogger("arabic_eval.evaluation")


class Evaluator:
    """Runs intrinsic tokenizer evaluation and downstream task evaluation."""

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        model: Optional[BaseModelAdapter] = None,
        tasks: Optional[List[BaseTask]] = None,
        eval_texts: Optional[List[str]] = None,
        output_dir: str = "outputs/results",
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.tasks = tasks or []
        self.eval_texts = eval_texts or []
        self.output_dir = output_dir

    def run_intrinsic(
        self,
        num_samples: Optional[int] = 5000,
        morphological_metrics: bool = True,
        morph_sample_size: int = 500,
    ) -> Dict[str, float]:
        """Compute intrinsic tokenizer metrics."""
        texts = (
            self.eval_texts
            if num_samples is None
            else self.eval_texts[:num_samples]
        )
        if not texts:
            logger.warning("No texts provided for intrinsic evaluation")
            return {}

        logger.info("Running intrinsic evaluation on %d texts", len(texts))
        metrics = compute_intrinsic_metrics(
            self.tokenizer,
            texts,
            morphological_metrics=morphological_metrics,
            morph_sample_size=morph_sample_size,
        )
        save_json(metrics, f"{self.output_dir}/intrinsic_metrics.json")
        return metrics

    def run_downstream(
        self, split: str = "test", max_samples: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """Run all downstream tasks."""
        if self.model is None:
            logger.warning("No model provided for downstream evaluation")
            return {}

        results = {}
        for task in tqdm(self.tasks, desc="Downstream tasks", unit="task"):
            logger.info("Evaluating task: %s", task.name)
            task_metrics = task.evaluate(
                self.model, self.tokenizer, split=split, max_samples=max_samples
            )
            results[task.name] = task_metrics
            save_json(task_metrics, f"{self.output_dir}/{task.name}_metrics.json")

        return results

    def run_all(
        self,
        intrinsic: bool = True,
        downstream: bool = True,
        num_intrinsic_samples: int = 5000,
        downstream_split: str = "test",
        downstream_max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run all evaluations and return combined results."""
        results: Dict[str, Any] = {}

        if intrinsic:
            results["intrinsic"] = self.run_intrinsic(num_intrinsic_samples)

        if downstream:
            results["downstream"] = self.run_downstream(
                split=downstream_split, max_samples=downstream_max_samples
            )

        save_json(results, f"{self.output_dir}/all_metrics.json")
        logger.info("All evaluation results saved to %s", self.output_dir)
        return results
