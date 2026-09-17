#!/usr/bin/env python
"""Scan Phase 3 corpora for overlap with the four eval benchmarks.

One-off diagnostic to run once per new SFT corpus (it is not part of the
pipeline). For every (corpus, benchmark) pair it reports:

* ``exact``  — corpus records whose normalized question / instruction equals
  a normalized benchmark question;
* ``ngram``  — corpus records that contain at least one word 8-gram of a
  benchmark question (in their question, context or answer), the usual
  test-set-decontamination criterion.

Normalization: NFKC, diacritics + tatweel stripped, alef / ة / ى folded,
punctuation removed, whitespace collapsed. Hits are written as JSON so the
offending record ids can be excluded if the overlap is real.

    .venv/bin/python scripts/scan_sft_contamination.py --corpora cidar bactrian_x_ar aya_ar
    .venv/bin/python scripts/scan_sft_contamination.py --corpora tydiqa_arabic arcd arabic_squad_mcq --ngram 10
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.data.contamination import normalize_text as normalize   # one normalizer for every scan
from arabic_eval.data.finetune_corpora import load_corpus
from arabic_eval.registry import task_registry
from arabic_eval.utils.logging import setup_logger
import arabic_eval.tasks  # noqa: F401  (registers the benchmarks)

BENCHMARKS = ("acva", "alghafa", "culture_arabic_mmlu", "arabic_exam")


def ngrams(words: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    for i in range(len(words) - n + 1):
        yield tuple(words[i:i + n])


def benchmark_questions(name: str, corpus_params: Dict) -> List[Tuple[str, str]]:
    task = task_registry.get(name)(corpus_params.get(name, {}))
    out = []
    for ex in task.get_eval_examples():
        q = ex.get("question") or ""
        ctx = ex.get("context") or ""
        out.append((str(ex.get("_source_config", "_default")), normalize(f"{ctx} {q}" if ctx else q)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlap between Phase 3 corpora and the eval benchmarks")
    parser.add_argument("--corpora", nargs="+", default=["cidar", "bactrian_x_ar", "aya_ar"])
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    parser.add_argument("--ngram", type=int, default=8, help="Word n-gram size for the containment test")
    parser.add_argument("--min-words", type=int, default=8, help="Ignore benchmark questions shorter than this")
    parser.add_argument("--out", default="outputs/data_cache/sft_corpora/contamination_scan.json")
    parser.add_argument("--aya-include", nargs="*", default=None, help="aya_ar include_datasets override")
    args = parser.parse_args()
    setup_logger("arabic_eval")

    bench: Dict[str, List[Tuple[str, str]]] = {b: benchmark_questions(b, {}) for b in args.benchmarks}
    exact_index: Dict[str, Dict[str, List[str]]] = {}
    ngram_index: Dict[str, Dict[Tuple[str, ...], Set[str]]] = {}
    for b, rows in bench.items():
        exact_index[b] = defaultdict(list)
        ngram_index[b] = defaultdict(set)
        for i, (cfg, q) in enumerate(rows):
            key = f"{cfg}#{i}"
            exact_index[b][q].append(key)
            words = q.split()
            if len(words) >= args.min_words:
                for g in ngrams(words, args.ngram):
                    ngram_index[b][g].add(key)
        print(f"{b}: {len(rows)} questions, {len(ngram_index[b])} distinct {args.ngram}-grams")

    report: Dict[str, Dict] = {"ngram": args.ngram, "benchmarks": {b: len(r) for b, r in bench.items()}, "corpora": {}}
    for corpus in args.corpora:
        params = {"include_datasets": args.aya_include} if corpus == "aya_ar" and args.aya_include else {}
        records = load_corpus(corpus, "train", **params)
        summary = {"records": len(records), "per_benchmark": {}}
        for b in args.benchmarks:
            exact_hits, ngram_hits = [], []
            for rec in records:
                q = normalize(rec.question)
                if q in exact_index[b]:
                    exact_hits.append({"id": rec.id, "matches": exact_index[b][q][:3], "text": rec.question[:120]})
                full = normalize(f"{rec.question} {rec.context} {rec.answer}")
                words = full.split()
                seen: Set[str] = set()
                for g in ngrams(words, args.ngram):
                    hit = ngram_index[b].get(g)
                    if hit:
                        seen.update(hit)
                        if len(seen) >= 3:
                            break
                if seen:
                    ngram_hits.append({"id": rec.id, "matches": sorted(seen)[:3], "text": rec.question[:120]})
            summary["per_benchmark"][b] = {
                "exact": len(exact_hits), "ngram": len(ngram_hits),
                "exact_hits": exact_hits[:50], "ngram_hits": ngram_hits[:50],
            }
            print(f"{corpus:16s} vs {b:20s}: exact {len(exact_hits):5d}   {args.ngram}-gram {len(ngram_hits):5d}   (of {len(records)})")
        report["corpora"][corpus] = summary

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nreport → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
