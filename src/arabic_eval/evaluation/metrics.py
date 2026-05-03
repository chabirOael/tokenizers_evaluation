"""Metric computation utilities."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional


def compute_perplexity(avg_loss: float) -> float:
    """Compute perplexity from average cross-entropy loss."""
    return math.exp(min(avg_loss, 100))


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670]", "", text)  # Remove diacritics
    return text


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_mei(
    accuracy: Optional[float],
    rps: Optional[float],
    compression: Optional[float],
    inference_time_sec: Optional[float],
    is_lighteval_mcq: bool,
) -> Dict[str, Any]:
    """Morphological Efficiency Index.

    MEI = (accuracy * RPS * compression) / inference_time_sec

    Asks whether high downstream accuracy is aligned with high root preservation
    and high compression, *per unit of inference time*. Defined only for the
    LightEval MCQ task family (acva / alghafa / culture_arabic_mmlu / arabic_exam)
    where ``accuracy`` is the natural primary metric.

    Args:
        accuracy: LightEval MCQ accuracy in [0, 1].
        rps: ``root_conservation_rate`` from the intrinsic morphological metrics.
        compression: ``compression_ratio`` (avg chars per token) from intrinsic
            metrics.
        inference_time_sec: wall-clock seconds for the evaluation pass.
        is_lighteval_mcq: True iff the active task is a LightEval MCQ benchmark.

    Returns:
        ``{"mei": float|None, "status": str, "inputs": {...}}``. The ``status``
        is ``"ok"`` on success, otherwise one of:
        ``"task_not_mcq"``, ``"missing_accuracy"``, ``"missing_rps"``,
        ``"missing_compression"``, ``"missing_time"``, ``"zero_time"``.
    """
    inputs = {
        "accuracy": accuracy,
        "rps": rps,
        "compression": compression,
        "inference_time_sec": inference_time_sec,
    }
    if not is_lighteval_mcq:
        return {"mei": None, "status": "task_not_mcq", "inputs": inputs}
    if accuracy is None:
        return {"mei": None, "status": "missing_accuracy", "inputs": inputs}
    if rps is None:
        return {"mei": None, "status": "missing_rps", "inputs": inputs}
    if compression is None:
        return {"mei": None, "status": "missing_compression", "inputs": inputs}
    if inference_time_sec is None:
        return {"mei": None, "status": "missing_time", "inputs": inputs}
    if inference_time_sec <= 0:
        return {"mei": None, "status": "zero_time", "inputs": inputs}

    mei = (accuracy * rps * compression) / inference_time_sec
    return {"mei": round(mei, 6), "status": "ok", "inputs": inputs}


def aggregate_qa_metrics(
    predictions: List[str], references: List[List[str]]
) -> Dict[str, float]:
    """Compute aggregate F1 and EM over a list of predictions and reference sets."""
    all_f1 = []
    all_em = []
    for pred, refs in zip(predictions, references):
        best_f1 = max(compute_f1(pred, ref) for ref in refs) if refs else 0.0
        best_em = max(compute_exact_match(pred, ref) for ref in refs) if refs else 0.0
        all_f1.append(best_f1)
        all_em.append(best_em)

    return {
        "f1": round(sum(all_f1) / max(len(all_f1), 1), 4),
        "exact_match": round(sum(all_em) / max(len(all_em), 1), 4),
    }
