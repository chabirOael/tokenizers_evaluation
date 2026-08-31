"""Analyze failure CSVs from the three experiments.

Diagnostic angles:
  1) Per-task: distribution of predicted letters/words vs gold (class-collapse?)
  2) Per-task: distribution of ll_margin and score_margin (confidently-wrong vs near-tie)
  3) Per-task: how often the model agrees with the majority class
  4) Per-task PMI: does PMI flip the same answer back to gold?
  5) Compute predicted-letter histogram for the FULL eval set (gold + failures combined)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/s3user/tokenizers_evaluation")
EXPERIMENTS = [
    "native_llama_3phase_with_sft",
    "native_llama_3phase_no_sft",
    "araroopat_3phase_with_sft",
]
TASKS = ["acva", "alghafa", "arabic_exam"]


def analyze_failure_csv(path: Path) -> dict:
    """Read a failure CSV and compute diagnostic statistics."""
    if not path.exists():
        return {"error": f"missing: {path}"}

    pred_letters = Counter()
    gold_letters = Counter()
    pred_pmi_letters = Counter()
    margins_score = []
    margins_score_pmi = []
    margins_ll = []
    n_pmi_flipped_to_gold = 0
    n_pmi_flipped_away = 0
    n_total = 0

    # We also need to know when score and score_pmi differ in their argmax.
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            gold_letter = row["gold_letter"]
            pred_letter = row["pred_letter"]
            pred_letters[pred_letter] += 1
            gold_letters[gold_letter] += 1
            try:
                margins_ll.append(float(row["ll_margin"]))
                margins_score.append(float(row["score_margin"]))
            except (ValueError, KeyError):
                pass
            # PMI columns may not exist
            if "score_pmi_margin" in row:
                try:
                    margins_score_pmi.append(float(row["score_pmi_margin"]))
                except (ValueError, KeyError):
                    pass
                # Recompute predicted under PMI by argmax of score_pmi_*
                pmi_scores = []
                for k in sorted(row.keys()):
                    if k.startswith("score_pmi_") and k != "score_pmi_margin":
                        try:
                            pmi_scores.append(float(row[k]))
                        except (ValueError, TypeError):
                            pmi_scores.append(float("-inf"))
                if pmi_scores:
                    pmi_pred_idx = max(range(len(pmi_scores)), key=lambda i: pmi_scores[i])
                    gold_idx = int(row["gold_idx"])
                    # Find the letter at this index by looking up choice_<i>
                    # But for letter-MCQ the pred letter == arabic letter at that idx.
                    # Just record whether PMI matches gold here (= flip to correct)
                    if pmi_pred_idx == gold_idx:
                        n_pmi_flipped_to_gold += 1
                    pmi_letter = ""
                    # Try to find a "score_pmi_<idx>" column whose index matches
                    pred_pmi_letters[f"idx_{pmi_pred_idx}"] += 1

    return {
        "n_failures": n_total,
        "gold_letter_dist": dict(gold_letters.most_common()),
        "pred_letter_dist": dict(pred_letters.most_common()),
        "pred_pmi_idx_dist": dict(pred_pmi_letters.most_common()),
        "n_pmi_would_flip_to_gold": n_pmi_flipped_to_gold,
        # Ratio of failures recovered if we used PMI
        "pmi_recovery_rate": (
            n_pmi_flipped_to_gold / n_total if n_total else 0
        ),
        "score_margin_stats": _stats(margins_score),
        "ll_margin_stats": _stats(margins_ll),
        "score_pmi_margin_stats": _stats(margins_score_pmi) if margins_score_pmi else None,
    }


def _stats(values):
    if not values:
        return None
    n = len(values)
    s = sorted(values)
    mean = sum(values) / n
    median = s[n // 2]
    return {
        "n": n,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "min": round(s[0], 4),
        "p25": round(s[n // 4], 4),
        "p75": round(s[3 * n // 4], 4),
        "max": round(s[-1], 4),
        "near_tie_pct": round(
            100.0 * sum(1 for v in values if abs(v) < 0.05) / n, 2
        ),
    }


def main():
    results = {}
    for exp in EXPERIMENTS:
        results[exp] = {}
        for task in TASKS:
            csv_path = ROOT / "outputs" / "experiments" / exp / "failure_reports" / f"{task}_accuracy_failures.csv"
            results[exp][task] = analyze_failure_csv(csv_path)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
