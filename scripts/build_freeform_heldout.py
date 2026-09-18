#!/usr/bin/env python
"""Build the held-out free-form evaluation set from CIDAR.

Writes the JSONL that ``configs/contamination/heldout_sets.yaml`` points at
(``freeform_prompts``, ``kind: file``) plus a manifest with every filter's
count next to it. Afterwards run ``scripts/check_contamination.py
--write-exclusions`` so the CIDAR rows drawn (and everything overlapping them)
leave the training splits; the pretraining-mix pool rebuilds by itself (its
fingerprint includes the prompt file's hash).

    .venv/bin/python scripts/build_freeform_heldout.py                 # 250 rows, E5 gate calibrated
    .venv/bin/python scripts/build_freeform_heldout.py --embed-threshold 0.9
    .venv/bin/python scripts/build_freeform_heldout.py --no-embed --n 300
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.data.contamination import DEFAULT_EXCLUSIONS_FILE, Exclusions  # noqa: E402
from arabic_eval.data.finetune_corpora import PINNED_REVISIONS, load_corpus  # noqa: E402
from arabic_eval.data.freeform_heldout import (  # noqa: E402
    E5Embedder, SelectionConfig, select_heldout, write_heldout_jsonl,
)

DEFAULT_OUT = Path("configs/contamination/freeform_cidar_heldout_v1.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-ref-chars", type=int, default=20)
    ap.add_argument("--max-ref-chars", type=int, default=1500)
    ap.add_argument("--embed-threshold", default="auto", help="cosine cut for the E5 gate; 'auto' calibrates on the n-gram twins")
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large")
    ap.add_argument("--no-embed", action="store_true", help="skip the embedding gate (n-gram twins only)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--training-corpora", nargs="+", default=["bactrian_x_ar", "aya_ar"])
    ap.add_argument("--aya-include", nargs="*", default=None, help="aya_ar include_datasets override")
    ap.add_argument("--exclusions", default=str(DEFAULT_EXCLUSIONS_FILE), help="committed contamination list; '' = none")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = SelectionConfig(
        n_examples=args.n, seed=args.seed, min_ref_chars=args.min_ref_chars, max_ref_chars=args.max_ref_chars,
        embed_threshold=None if args.embed_threshold == "auto" else float(args.embed_threshold),
    )
    candidates = load_corpus("cidar", "train")
    training = []
    corpora_info = {}
    for name in args.training_corpora:
        kw = {}
        if name == "aya_ar" and args.aya_include is not None:
            kw["include_datasets"] = args.aya_include
        recs = load_corpus(name, "train", **kw)
        training.extend(recs)
        corpora_info[name] = {"records": len(recs), "revision": PINNED_REVISIONS.get(name), **kw}
    excluded = []
    if args.exclusions:
        excluded = sorted(Exclusions.load(args.exclusions).ids.get("cidar", ()))
    embedder = None if args.no_embed else E5Embedder(args.embed_model, device=args.device)

    rows, manifest = select_heldout(candidates, training, cfg, embedder=embedder, excluded_ids=excluded)
    manifest["source"] = {"corpus": "cidar", "revision": PINNED_REVISIONS["cidar"], "split": "train"}
    manifest["training_corpora"] = corpora_info
    manifest["embedding"]["model"] = None if args.no_embed else args.embed_model
    manifest["exclusions_file"] = args.exclusions or None

    out = Path(args.out)
    write_heldout_jsonl(rows, out)
    mpath = out.with_suffix(".manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    c = manifest["counts"]
    print(f"\ncandidates {c['candidates']:>6}   (cidar train; training records for the twin check: {c['training_records']})")
    for k in ("excluded_id", "latin", "context_nonempty", "reference_length", "prompt_length",
              "ngram_prompt_twin", "ngram_reference_twin", "embedding_near_duplicate"):
        print(f"  - {k:<26} {c.get(k, 0):>6}")
    print(f"survivors  {c['survivors']:>6}")
    e = manifest["embedding"]
    if e.get("applied"):
        print(f"embedding gate: threshold {e['threshold']} ({'calibrated' if e.get('calibrated') else 'fixed'}) "
              f"on {e['twins_for_calibration']} n-gram twins; twin max-cos p05 {e['twin_max_cosine_quantiles']['p05']}, "
              f"p50 {e['twin_max_cosine_quantiles']['p50']}; alive p50 {e['alive_max_cosine_quantiles']['p50']}, "
              f"p95 {e['alive_max_cosine_quantiles']['p95']}")
    s = manifest["strata"]
    print(f"strata (ref chars ≤ {s['boundaries_ref_chars']}): available {s['available']} → selected {s['selected']}")
    print(f"selected   {c['selected']:>6}  → {out}  (+ {mpath.name})")
    return 0 if c["selected"] == cfg.n_examples else 1


if __name__ == "__main__":
    sys.exit(main())
