"""Metric computation utilities."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List


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
