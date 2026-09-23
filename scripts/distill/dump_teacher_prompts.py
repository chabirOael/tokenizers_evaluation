#!/usr/bin/env python
"""The training prompts the chosen teacher answers (P1.5 of the 2026-09-23 distillation brief).

``cidar``, ``bactrian_x_ar`` and ``aya_ar`` are loaded for ``train`` and ``dev`` through
``load_corpus`` with the committed contamination exclusions (``aya_ar`` with the
campaign's ``include_datasets``), and the selection is:

  * every ``dev`` record of the three corpora — the early-stop eval mixture reads
    the dev slices, and with the overlay it scores teacher text there too;
  * every ``cidar/train`` record;
  * a seeded subset of ``bactrian_x_ar/train`` and ``aya_ar/train``: sorted ids,
    ``numpy.random.default_rng(42).permutation``, the first ``--train-subset``
    (14 000 each by default). Extending a subset later (``--train-subset
    bactrian_x_ar=18000``) keeps the first 14 000 — the generation resumes by id.

Writes ``outputs/data_cache/distill/<slug>_v1/prompts.jsonl`` (``id, corpus, split,
instruction, context, reference, ref_chars``) and ``prompts_manifest.json``
(exclusions digest, ``TEMPLATE_VERSION``, counts per corpus × split, the selection
rule, the system prompt and its sha256, the cap from the bake-off manifest, the
held-out overlap check — zero hits or the script stops).

    .venv/bin/python scripts/distill/dump_teacher_prompts.py --slug <winner>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arabic_eval.data.contamination import load_exclusions  # noqa: E402
from arabic_eval.data.finetune_corpora import TEMPLATE_VERSION, load_corpus  # noqa: E402
from arabic_eval.distill.prompts import SEED, heldout_overlap, seeded_subset  # noqa: E402
from arabic_eval.distill.teacher import SYSTEM_PROMPT, file_sha256, sha256_text, write_jsonl  # noqa: E402

log = logging.getLogger("dump_teacher_prompts")

CORPORA = ("cidar", "bactrian_x_ar", "aya_ar")
CORPUS_PARAMS: Dict[str, Dict[str, Any]] = {"aya_ar": {"include_datasets": ["Aya-Dataset", "Dolly-v2 (T)"]}}
DEFAULT_SUBSETS = {"bactrian_x_ar": 14000, "aya_ar": 14000}
HELDOUT = "configs/contamination/freeform_cidar_heldout_v1.jsonl"
EXCLUSIONS = "configs/contamination/exclusions.json"
BAKEOFF_MANIFEST = "outputs/data_cache/distill/bakeoff/bakeoff_manifest.json"


def parse_subsets(items: Sequence[str]) -> Dict[str, int]:
    out = dict(DEFAULT_SUBSETS)
    for it in items:
        name, _, n = it.partition("=")
        if name not in DEFAULT_SUBSETS or not n.isdigit():
            raise SystemExit(f"--train-subset takes corpus=N for {sorted(DEFAULT_SUBSETS)}, got {it!r}")
        out[name] = int(n)
    return out


def select(exclusions, subsets: Dict[str, int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for corpus in CORPORA:
        for split in ("dev", "train"):
            recs = load_corpus(corpus, split, exclusions=exclusions, **CORPUS_PARAMS.get(corpus, {}))
            n = subsets.get(corpus) if split == "train" else None
            picked = seeded_subset(recs, n, SEED)
            log.info("%s/%s: %d records after exclusions → %d selected", corpus, split, len(recs), len(picked))
            rows.extend({"id": r.id, "corpus": corpus, "split": split, "instruction": r.question, "context": r.context,
                         "reference": r.answer, "ref_chars": len(r.answer), "_available": len(recs)} for r in picked)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", required=True, help="the bake-off winner (names the output folder <slug>_v1)")
    ap.add_argument("--train-subset", nargs="*", default=[], help="corpus=N overrides (default 14000 for bactrian_x_ar and aya_ar)")
    ap.add_argument("--exclusions", default=EXCLUSIONS)
    ap.add_argument("--out-dir", default=None, help="default outputs/data_cache/distill/<slug>_v1")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    subsets = parse_subsets(args.train_subset)
    exclusions = load_exclusions(REPO_ROOT / args.exclusions)
    rows = select(exclusions, subsets)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate record ids across corpora / splits")
    check = heldout_overlap(rows, REPO_ROOT / HELDOUT, REPO_ROOT)
    if not check["passed"]:
        raise SystemExit(f"held-out overlap check FAILED: {check}")

    counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    for r in rows:
        c = counts.setdefault(r["corpus"], {}).setdefault(r["split"], {"available": r["_available"], "selected": 0})
        c["selected"] += 1
    for r in rows:
        del r["_available"]

    out_dir = REPO_ROOT / (args.out_dir or f"outputs/data_cache/distill/{args.slug}_v1")
    path = out_dir / "prompts.jsonl"
    write_jsonl(path, rows)
    bake = json.loads((REPO_ROOT / BAKEOFF_MANIFEST).read_text(encoding="utf-8"))
    cap = (bake.get("caps") or {}).get(args.slug)
    manifest = {
        "slug": args.slug, "prompts_file": str(path.relative_to(REPO_ROOT)), "prompts_sha256": file_sha256(path),
        "n": len(rows), "counts": counts,
        "selection": {"dev": "every record of cidar / bactrian_x_ar / aya_ar dev", "cidar_train": "every record",
                      "subset_rule": f"sorted ids, numpy.random.default_rng({SEED}).permutation, first N",
                      "train_subsets": subsets},
        "corpus_params": CORPUS_PARAMS,
        "exclusions": str(args.exclusions), "exclusions_digest": exclusions.digest() if exclusions else None,
        "exclusions_dropped": dict(exclusions.dropped) if exclusions else {},
        "template_version": TEMPLATE_VERSION,
        "system_prompt": SYSTEM_PROMPT, "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "cap": cap, "heldout_check": check,
    }
    with open(out_dir / "prompts_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "system_prompt"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
