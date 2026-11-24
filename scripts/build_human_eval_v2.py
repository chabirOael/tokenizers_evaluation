#!/usr/bin/env python3
"""Build the human-evaluation campaign v2 (2026-11-12): two blind rating sets for the console's Rate tab.

Design (the supervisor's TODO of ARA_version1.pdf §4.6): 120 items sampled without replacement from the 250 held-out
CIDAR prompts, stratified across arms and base models, three native-speaker annotators, blind to the arm, the same
five-point rubric as the LLM judge.

Concretely: for each of Qwen3-4B-Base and Llama-3.2-3B (the two models where the judged AraRooPat − BPE-16K effect is
significant), 20 prompts are drawn at random (seeded) in balance over the held-out set's three reference-length strata,
disjoint between the two models (40 distinct prompts), and every drawn prompt is rated under the three adapted arms
AraRooPat, BPE-16K and Farasa-BPE-16K — 60 items per model, 120 in all, 20 per arm–model cell. Items are shuffled and
carry opaque ids (h2q-001 …), so neither the page nor the file name tells the rater which arm an answer comes from; the
arm sits only under ``hidden``, which the Rate tab never sends to the browser.

Writes, per model folder, ``freeform_rating/human_v2.json`` (schema 1, readable by the Rate tab, ``agreement()`` and
``scripts/human_eval_v2_report.py``), and under ``outputs/experiments/_human_eval_v2/`` a manifest plus one blind CSV
per set for annotators who rate offline (``scripts/import_human_ratings.py`` brings a filled CSV back).

    .venv/bin/python scripts/build_human_eval_v2.py            # refuses to overwrite a set that already has ratings
    .venv/bin/python scripts/build_human_eval_v2.py --overwrite
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arabic_eval.tools.freeform_rows_browser import _cell_summary, load_cell  # noqa: E402

NAME = "human_v2"
SEED = 7
MODELS = {"qwen": "qwen_native_vs_araroopat", "llama3b": "llama3b_tokenizers"}
CELLS = ["araroopat_3phase_v5_distill", "bpe_16k_3phase_v5_distill", "farasa_bpe16k_3phase_v5_distill"]
PROMPTS_PER_MODEL = 20
QUOTAS = {"qwen": {"short": 7, "medium": 7, "long": 6}, "llama3b": {"short": 6, "medium": 7, "long": 7}}
HELDOUT = ROOT / "configs/contamination/freeform_cidar_heldout_v1.jsonl"
OUT = ROOT / "outputs/experiments/_human_eval_v2"
CSV_COLUMNS = ["item_id", "position", "instruction", "context", "reference", "generation", "score", "flags", "note"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--overwrite", action="store_true", help="rebuild even if the sets exist (refused when ratings exist)")
    args = ap.parse_args()

    strata = {}
    for line in HELDOUT.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        strata[r["id"]] = r["stratum"]

    # one seeded order per stratum, shared by both models so that the draws are disjoint
    rng = random.Random(SEED)
    by_stratum = {s: sorted(i for i, st in strata.items() if st == s) for s in ("short", "medium", "long")}
    for s in by_stratum:
        rng.shuffle(by_stratum[s])
    cursor = {s: 0 for s in by_stratum}

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"name": NAME, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": SEED, "design": __doc__.split("\n\n")[1],
                "cells": CELLS, "prompts_per_model": PROMPTS_PER_MODEL, "quotas": QUOTAS, "sets": {}}
    for model, folder in MODELS.items():
        exp_dir = ROOT / "outputs/experiments" / folder
        out = exp_dir / "freeform_rating" / f"{NAME}.json"
        ratings_dir = exp_dir / "freeform_rating" / f"{NAME}.ratings"
        if ratings_dir.is_dir() and any(ratings_dir.glob("*.json")):
            sys.exit(f"{out} already has ratings under {ratings_dir}; not rebuilding")
        if out.exists() and not args.overwrite:
            sys.exit(f"{out} exists; pass --overwrite to rebuild")
        loaded = {c: load_cell(ROOT, f"outputs/experiments/{folder}/{c}") for c in CELLS}
        common = set.intersection(*[{r["id"] for r in d["rows"]} for d in loaded.values()])
        by_cell = {c: {r["id"]: r for r in d["rows"]} for c, d in loaded.items()}
        judges = sorted({j["name"] for d in loaded.values() for j in d["judges"]})
        prompts = []
        for s, q in QUOTAS[model].items():
            pool = by_stratum[s]
            while q:
                pid = pool[cursor[s]]
                cursor[s] += 1
                if pid in common:
                    prompts.append(pid)
                    q -= 1
        items = []
        for pid in prompts:
            for cell in CELLS:
                r = by_cell[cell][pid]
                items.append({"prompt_id": pid, "instruction": r.get("instruction"), "context": r.get("context") or "",
                              "reference": r.get("reference"), "generation": r.get("generation"),
                              "hidden": {"cell": cell, "tokenizer": _cell_summary(exp_dir / cell).get("tokenizer"), "stratum": strata[pid]}})
        item_rng = random.Random(SEED * 100 + len(folder))
        item_rng.shuffle(items)
        tag = "h2q" if model == "qwen" else "h2l"
        for pos, it in enumerate(items):
            it["item_id"] = f"{tag}-{pos + 1:03d}"
            it["position"] = pos
        doc = {"schema": 1, "name": NAME, "created_at": manifest["created_at"], "seed": SEED,
               "experiment": f"outputs/experiments/{folder}", "baseline": "bpe_16k_3phase_v5_distill", "cells": CELLS,
               "judges": judges, "n_prompts": len(prompts), "variants_per_prompt": len(CELLS),
               "selection": {"random_stratified": {"quotas": QUOTAS[model], "prompts": prompts}, "disagreement": [], "random": prompts},
               "items": items}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        # blind CSV for offline annotators (no cell anywhere in it)
        csv_path = OUT / f"{NAME}_{model}_blind.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            for it in items:
                w.writerow({"item_id": it["item_id"], "position": it["position"], "instruction": it["instruction"], "context": it["context"],
                            "reference": it["reference"], "generation": it["generation"], "score": "", "flags": "", "note": ""})
        manifest["sets"][model] = {"experiment": folder, "set_file": str(out.relative_to(ROOT)), "blind_csv": str(csv_path.relative_to(ROOT)),
                                   "n_items": len(items), "n_prompts": len(prompts), "prompts": prompts,
                                   "strata": {s: sum(1 for p in prompts if strata[p] == s) for s in ("short", "medium", "long")}, "judges": judges}
        print(f"{model:8s} {len(items)} items over {len(prompts)} prompts × {len(CELLS)} arms → {out.relative_to(ROOT)}; blind CSV {csv_path.relative_to(ROOT)}")
    assert not set(manifest["sets"]["qwen"]["prompts"]) & set(manifest["sets"]["llama3b"]["prompts"]), "prompt draws overlap"
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("manifest:", (OUT / "manifest.json").relative_to(ROOT))


if __name__ == "__main__":
    main()
