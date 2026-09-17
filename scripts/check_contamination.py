#!/usr/bin/env python
"""Contamination check: every training passage against the held-out sets.

Held-out sets (``configs/contamination/heldout_sets.yaml``): TyDiQA-AR
``validation`` (the public dev split; TyDi's test is hidden), ARCD
``validation`` (the paper's test split) and, once its ``path`` is set, the
free-form eval prompt file. Training side: the Phase 3 QA corpora (the
``train`` *and* ``dev`` slices of TyDiQA / ARCD, ``train`` of the others),
the cached pretraining-mix pool of a config (by document) and, opt-in,
the tokenizer-training corpus.

Every hit is one row of ``hits.parquet`` (corpus, record, held-out set and
record, exact-hash kind, shared n-grams, coverage, longest shared run,
tier). ``report.json`` summarizes per (corpus, set). ``--write-exclusions``
writes the ``contaminated`` tier's training record ids to
``configs/contamination/exclusions.json`` — the list every train / dev
loader applies (``training.contamination_exclusions``). Pool documents are
not listed there: the pool builder re-applies the same filter by content
(``pretraining_mix.heldout_filter``), so rebuild the pool after a change.

    .venv/bin/python scripts/check_contamination.py --write-exclusions
    .venv/bin/python scripts/check_contamination.py --corpora tydiqa_arabic arcd --no-pool
    .venv/bin/python scripts/check_contamination.py --include-tokenizer-corpus --max-samples 50000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.config import load_config  # noqa: E402
from arabic_eval.data.contamination import (  # noqa: E402
    DEFAULT_EXCLUSIONS_FILE, DEFAULT_SETS_FILE, HIT_FIELDS, TIER_CONTAMINATED, Exclusions, HeldoutIndex,
    new_exclusions_meta, scan_records, scan_texts, summarize_hits,
)
from arabic_eval.data.finetune_corpora import _LOADERS, load_corpus  # noqa: E402
from arabic_eval.utils.io import write_report_table  # noqa: E402
from arabic_eval.utils.logging import setup_logger  # noqa: E402

ALL_CORPORA = tuple(_LOADERS)
# The two corpora whose official train split is partitioned into train + dev:
# both slices are training-side (dev steers early-stopping), so both are scanned.
TRAIN_DEV_CORPORA = ("tydiqa_arabic", "arcd")
DEFAULT_POOL_CONFIG = "configs/experiments/all_tokenizers_sweep_pretrain_mix.yaml"


def _splits_for(corpus: str) -> List[str]:
    return ["train", "dev"] if corpus in TRAIN_DEV_CORPORA else ["train"]


def _pool_dir(config_path: str, base_path: Path):
    from arabic_eval.data.pretraining_mix.pool import load_pool_manifest, pool_dir
    config = load_config(config_path, base_path=str(base_path) if base_path.exists() else None)
    mix = config.training.pretraining_mix
    if mix is None:
        return None, None, "no training.pretraining_mix in the config"
    d = pool_dir(mix)
    manifest = load_pool_manifest(d)
    if not manifest or not manifest.get("complete"):
        # The current fingerprint (held-out filter on) has no pool yet; the
        # scan still wants to see what the previous, unfiltered pool held.
        cache = Path(mix.cache_dir)
        candidates = sorted((p for p in cache.iterdir() if (p / "manifest.json").exists() and not p.name.startswith("_")),
                            key=lambda p: (p / "manifest.json").stat().st_mtime) if cache.exists() else []
        if not candidates:
            return None, mix, f"no built pool under {cache}"
        d = candidates[-1]
        manifest = load_pool_manifest(d)
    return d, mix, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Training passages vs the held-out evaluation sets")
    ap.add_argument("--sets", default=str(DEFAULT_SETS_FILE), help="held-out declaration YAML")
    ap.add_argument("--corpora", nargs="+", default=list(ALL_CORPORA), help=f"QA corpora to scan (default: all: {' '.join(ALL_CORPORA)})")
    ap.add_argument("--pool-config", default=DEFAULT_POOL_CONFIG, help="experiment YAML whose pretraining-mix pool is scanned")
    ap.add_argument("--no-pool", action="store_true", help="skip the pool")
    ap.add_argument("--include-tokenizer-corpus", action="store_true", help="also scan Jr23xd23/ArabicText-Large (tokenizer training only)")
    ap.add_argument("--max-samples", type=int, default=None, help="cap on tokenizer-corpus texts")
    ap.add_argument("--out", default=None, help="artifact directory (default outputs/data_cache/contamination/<timestamp>)")
    ap.add_argument("--write-exclusions", nargs="?", const=str(DEFAULT_EXCLUSIONS_FILE), default=None,
                    metavar="PATH", help=f"write the exclusion list (default path {DEFAULT_EXCLUSIONS_FILE})")
    ap.add_argument("--strict", action="store_true", help="exclude the overlap tier too (any shared n-gram)")
    ap.add_argument("--aya-include", nargs="*", default=None, help="aya_ar include_datasets override")
    args = ap.parse_args()
    setup_logger("arabic_eval")
    t_all = time.perf_counter()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else Path("outputs/data_cache/contamination") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    index = HeldoutIndex.from_sets_file(args.sets)
    hs = index.summary()
    print(f"held-out: {hs['records']} records, {hs['ngrams']:,} n-grams; thresholds coverage ≥ {hs['thresholds']['coverage']}, "
          f"run ≥ {hs['thresholds']['run_words']} words")
    for name, s in hs["sets"].items():
        print(f"  {name:28s} {s['status']:8s} {s['records']:>6,} records  n={s['ngram']}"
              + (f"  revision {s['revision'][:12]}" if s.get("revision") else "")
              + (f"  sha256 {s['sha256'][:12]}" if s.get("sha256") else ""))

    rows: List[Dict[str, Any]] = []
    scanned: Dict[str, Dict[str, int]] = {}
    for corpus in args.corpora:
        params = {"include_datasets": args.aya_include} if corpus == "aya_ar" and args.aya_include else {}
        for split in _splits_for(corpus):
            t0 = time.perf_counter()
            records = load_corpus(corpus, split, **params)     # no exclusions: this is what produces them
            hits = scan_records(index, corpus, split, records)
            rows.extend(hits)
            scanned.setdefault(corpus, {})[split] = len(records)
            n_cont = len({r["record_id"] for r in hits if r["tier"] == TIER_CONTAMINATED})
            n_over = len({r["record_id"] for r in hits})
            print(f"{corpus:16s} {split:5s} {len(records):>7,} records  overlap {n_over:>6,}  contaminated {n_cont:>6,}"
                  f"  ({time.perf_counter() - t0:.0f} s)")

    pool_info: Dict[str, Any] = {"scanned": False}
    if not args.no_pool:
        base = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
        d, mix, why = _pool_dir(args.pool_config, base)
        if d is None:
            print(f"pool: skipped — {why}")
            pool_info["skipped"] = why
        else:
            from arabic_eval.data.pretraining_mix.pool import iter_pool_docs, load_pool_manifest
            manifest = load_pool_manifest(d)
            pool_info.update({"scanned": True, "dir": str(d), "fingerprint": manifest.get("fingerprint"),
                              "heldout_filter_at_build": manifest.get("heldout_filter", {"enabled": False}),
                              "sources": {}})
            for src in manifest.get("sources", []):
                name = src["name"]
                t0 = time.perf_counter()
                docs = 0

                def _docs():
                    nonlocal docs
                    for doc_id, text in iter_pool_docs(d, name):
                        docs += 1
                        yield doc_id, text
                hits = scan_texts(index, f"pool:{name}", "pool", _docs())
                rows.extend(hits)
                n_cont = len({r["record_id"] for r in hits if r["tier"] == TIER_CONTAMINATED})
                n_over = len({r["record_id"] for r in hits})
                pool_info["sources"][name] = {"docs": docs, "overlap": n_over, "contaminated": n_cont}
                print(f"{'pool:' + name:16s} {'':5s} {docs:>7,} docs     overlap {n_over:>6,}  contaminated {n_cont:>6,}"
                      f"  ({time.perf_counter() - t0:.0f} s)")

    tok_info: Dict[str, Any] = {"scanned": False}
    if args.include_tokenizer_corpus:
        from arabic_eval.config import DataConfig, load_yaml
        from arabic_eval.data.loader import extract_texts, load_arabic_dataset
        base = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
        preprocessing = DataConfig(**(load_yaml(base).get("data") or {})).preprocessing
        t0 = time.perf_counter()
        ds = load_arabic_dataset(dataset_name="Jr23xd23/ArabicText-Large", max_train_samples=args.max_samples,
                                 preprocessing_config=preprocessing, seed=42)
        texts = extract_texts(ds["train"])
        hits = scan_texts(index, "tokenizer:ArabicText-Large", "train", ((str(i), t) for i, t in enumerate(texts)))
        rows.extend(hits)
        n_cont = len({r["record_id"] for r in hits if r["tier"] == TIER_CONTAMINATED})
        tok_info = {"scanned": True, "texts": len(texts), "overlap": len({r["record_id"] for r in hits}), "contaminated": n_cont}
        print(f"{'tokenizer corpus':16s} {'':5s} {len(texts):>7,} texts    overlap {tok_info['overlap']:>6,}  contaminated {n_cont:>6,}"
              f"  ({time.perf_counter() - t0:.0f} s)")

    hits_path = out_dir / "hits.parquet"
    write_report_table(hits_path, rows, list(HIT_FIELDS), metadata={"sets": str(args.sets), "strict": str(args.strict)})
    summary = summarize_hits(rows, strict=args.strict)
    tiers = Counter(r["tier"] for r in rows)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sets_file": str(args.sets), "heldout": hs, "strict": args.strict,
        "corpora_scanned": scanned, "pool": pool_info, "tokenizer_corpus": tok_info,
        "hits": {"rows": len(rows), "by_tier": dict(tiers)},
        "summary": summary,
        "wall_sec": round(time.perf_counter() - t_all, 1),
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nper (corpus | held-out set): records with overlap / contaminated / exact, held-out records hit / contaminated")
    for key, d in summary["pairs"].items():
        print(f"  {key:44s} {d['records_overlap']:>6,} / {d['records_contaminated']:>6,} / {d['exact']:>5,}    "
              f"{d['heldout_hit']:>5,} / {d['heldout_contaminated']:>5,}")
    print(f"\nexcluded per corpus ({'overlap + contaminated' if args.strict else 'contaminated tier'}): "
          f"{summary['excluded_per_corpus'] or 'none'}")

    if args.write_exclusions:
        corpus_rows = [r for r in rows if not r["corpus"].startswith(("pool:", "tokenizer:"))]
        exc = Exclusions.from_rows(corpus_rows, strict=args.strict, **new_exclusions_meta(
            index, sets_file=str(args.sets), corpora_scanned=scanned, report_dir=str(out_dir),
            pool={k: v for k, v in pool_info.items() if k != "sources"} | {"contaminated_docs": sum(
                s["contaminated"] for s in pool_info.get("sources", {}).values())},
        ))
        path = exc.save(args.write_exclusions)
        print(f"exclusions → {path} ({exc.total()} record ids; digest {exc.digest() or 'empty'})")
    print(f"artifacts → {out_dir}  ({report['wall_sec']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
