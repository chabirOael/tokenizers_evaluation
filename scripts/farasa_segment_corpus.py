#!/usr/bin/env python
"""Farasa morpheme boundaries for the tokenizer-training corpus, cached (2026-11-08).

Loads the same 668 959 texts as ``bpe_16k_v3`` (``Jr23xd23/ArabicText-Large``
through the pipeline's loader and ``base.yaml``'s ``data.preprocessing`` — the
path of ``scripts/train_tokenizer.py``), segments every **distinct** whitespace
word once with Farasa (one word per line, one JVM per worker), aligns each
segmentation to the raw word (``farasa_segmentation.align_word``) and writes

    outputs/data_cache/farasa_segmented/<fingerprint>/
        boundary_table.parquet   word → kept boundary offsets (words with ≥ 1 boundary)
        farasa_raw.parquet       every word type, its count and Farasa's raw output (audit)
        manifest.json            fingerprint inputs, joiner check, counts, wall times, context check

``farasa_bpe.train`` reads the table by fingerprint, so a second build never
re-segments. ``--context-check N`` compares, on N sampled texts, Farasa's
in-context segmentation with the per-word table (the equivalence this relies on).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arabic_eval.config import DataConfig, load_yaml  # noqa: E402
from arabic_eval.data.loader import extract_texts, load_arabic_dataset  # noqa: E402
from arabic_eval.tokenizers import farasa_segmentation as fs  # noqa: E402
from arabic_eval.utils.logging import setup_logger  # noqa: E402

DEFAULT_CACHE = ROOT / "outputs/data_cache/farasa_segmented"


def load_training_texts(dataset: str, base_config: Path, seed: int):
    base_data = DataConfig(**(load_yaml(base_config).get("data") or {}))
    pre = base_data.preprocessing
    ds = load_arabic_dataset(dataset_name=dataset, cache_dir=base_data.cache_dir, max_train_samples=None,
                             preprocessing_config=pre, seed=seed)
    return extract_texts(ds["train"]), pre, base_data.cache_dir


def context_check(texts, farasa_by_word, n: int, seed: int):
    """Farasa on whole texts vs the per-word outputs: share of words segmented identically."""
    from farasa.segmenter import FarasaSegmenter
    seg = FarasaSegmenter(interactive=True)
    rng = random.Random(seed)
    sample = rng.sample(range(len(texts)), min(n, len(texts)))
    same_texts = words = same_words = 0
    examples = []
    for i in sample:
        t = " ".join(texts[i].split())          # one line, single spaces
        ctx = seg.segment(t).split()
        iso_parts = [farasa_by_word[w] for w in texts[i].split()]
        iso = " ".join(iso_parts).split()
        words += len(texts[i].split())
        if ctx == iso:
            same_texts += 1
            same_words += len(texts[i].split())
            continue
        # word-level agreement through a token-sequence diff
        import difflib
        sm = difflib.SequenceMatcher(None, ctx, iso, autojunk=False)
        eq = sum(b.size for b in sm.get_matching_blocks())
        same_words += len(texts[i].split()) * eq / max(len(iso), 1)
        if len(examples) < 5:
            for op, a0, a1, b0, b1 in sm.get_opcodes():
                if op != "equal":
                    examples.append({"text_index": i, "in_context": " ".join(ctx[a0:a1]), "per_word": " ".join(iso[b0:b1])})
                    break
    return {"texts": len(sample), "texts_identical": same_texts,
            "word_agreement_approx": round(same_words / max(words, 1), 5), "examples": examples, "seed": seed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="Jr23xd23/ArabicText-Large")
    ap.add_argument("--base-config", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context-check", type=int, default=500)
    ap.add_argument("--max-texts", type=int, default=None, help="debug: first N texts only (fingerprint changes)")
    args = ap.parse_args()
    setup_logger("arabic_eval")

    t0 = time.time()
    texts, pre, ds_cache = load_training_texts(args.dataset, Path(args.base_config), args.seed)
    if args.max_texts:
        texts = texts[: args.max_texts]
    t_load = time.time() - t0
    print(f"{len(texts)} training texts loaded in {t_load:.0f} s (preprocessing {pre})", flush=True)

    versions = fs.farasa_versions()
    digest = fs.texts_digest(texts)
    fp = fs.corpus_fingerprint(digest, pre, versions)
    out = Path(args.cache_dir) / fp
    if (out / "manifest.json").exists():
        print(f"cache exists: {out}")
        return
    out.mkdir(parents=True, exist_ok=True)

    joiner_hits = {"U+2060": sum(1 for t in texts if fs.JOINER in t),
                   "U+E000": sum(1 for t in texts if fs.FALLBACK_JOINER in t)}
    joiner = fs.JOINER if joiner_hits["U+2060"] == 0 else fs.FALLBACK_JOINER
    if joiner_hits["U+2060"] and joiner_hits["U+E000"]:
        raise SystemExit(f"both joiner candidates occur in the corpus: {joiner_hits}")
    print(f"joiner {'U+2060' if joiner == fs.JOINER else 'U+E000'} (texts containing: {joiner_hits})", flush=True)

    t1 = time.time()
    counts = fs.collect_word_counts(texts)
    words = sorted(counts)
    t_count = time.time() - t1
    print(f"{len(words)} distinct words, {sum(counts.values())} occurrences ({t_count:.0f} s)", flush=True)

    farasa_out, seg_stats = fs.segment_words(words, workers=args.workers)
    print(f"Farasa done: {seg_stats['wall_sec']} s, {len(seg_stats['errors'])} batch errors", flush=True)

    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"word": words, "count": [counts[w] for w in words], "farasa": farasa_out}),
                   out / "farasa_raw.parquet", compression="zstd")
    table, stats = fs.build_boundary_table(counts, farasa_out, words)
    fs.save_table(out / "boundary_table.parquet", table)

    by_word = dict(zip(words, farasa_out))
    t2 = time.time()
    ctx = context_check(texts, by_word, args.context_check, args.seed) if args.context_check else None
    t_ctx = time.time() - t2

    manifest = {
        "fingerprint": fp, "dataset": args.dataset, "preprocessing": pre, "datasets_cache": ds_cache,
        "num_texts": len(texts), "texts_sha256": digest, "max_texts": args.max_texts, **versions,
        "joiner": "U+2060" if joiner == fs.JOINER else "U+E000", "joiner_check_texts_containing": joiner_hits,
        "segmentation_unit": "distinct whitespace word, one per line (see context_check)",
        "alignment_rule": "farasa_segmentation.align_word: join == raw → cumulative lengths; else SequenceMatcher "
                          "(autojunk off), keep a cut whose two neighbouring characters lie in one equal block",
        "stats": stats, "segmentation": seg_stats, "context_check": ctx,
        "wall_sec": {"load": round(t_load, 1), "count": round(t_count, 1), "farasa": seg_stats["wall_sec"],
                     "context_check": round(t_ctx, 1), "total": round(time.time() - t0, 1)},
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("fingerprint", "joiner", "stats", "context_check", "wall_sec")},
                     indent=2, ensure_ascii=False)[:4000])
    print(f"written: {out}")


if __name__ == "__main__":
    main()
