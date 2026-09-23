#!/usr/bin/env python
"""Teacher bake-off prompt set (P1.1 of the 2026-09-23 distillation brief) and the
per-candidate token caps.

250 ``cidar`` **dev** records — ``load_corpus("cidar", "dev")`` with the committed
contamination exclusions loaded as the pipeline loads them — sorted by id, permuted
with ``numpy.random.default_rng(42)``, the first 250 kept. Each row carries the
held-out set's reference-length stratum (its tercile boundaries, read from its
manifest), and ``prompt`` next to ``instruction`` so the file is also a valid
``freeform_cidar`` ``heldout_path`` (the ``native_qwen3_base_dev`` reference row runs
the pipeline's own free-form task on it). The script asserts zero id overlap and
zero normalized-text matches (``contamination.normalize_text``) with the 250 test
prompts of ``configs/contamination/freeform_cidar_heldout_v1.jsonl`` and records
the check in ``bakeoff_manifest.json``.

``--caps`` adds, per candidate of ``configs/distill/teacher_candidates.yaml``, the
generation token cap: ``derive_token_cap`` (2 400 characters, margin 1.15, + 8) over
``measure_chars_per_token`` on the 250 dev references with the candidate's own
tokenizer (``AutoTokenizer`` at the pinned revision; special ids included, as the
free-form task measures its own tokenizers).

    .venv/bin/python scripts/distill/dump_bakeoff_prompts.py --caps
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
from arabic_eval.distill.teacher import (  # noqa: E402
    SYSTEM_PROMPT, file_sha256, load_candidates, sha256_text, stratum_of, write_jsonl,
)

log = logging.getLogger("dump_bakeoff_prompts")

HELDOUT = "configs/contamination/freeform_cidar_heldout_v1.jsonl"
HELDOUT_MANIFEST = "configs/contamination/freeform_cidar_heldout_v1.manifest.json"
EXCLUSIONS = "configs/contamination/exclusions.json"       # TrainingConfig.contamination_exclusions default
CANDIDATES = "configs/distill/teacher_candidates.yaml"
OUT_DIR = "outputs/data_cache/distill/bakeoff"
N_PROMPTS = 250


def strata_bounds() -> List[float]:
    with open(REPO_ROOT / HELDOUT_MANIFEST, encoding="utf-8") as f:
        return list(json.load(f)["strata"]["boundaries_ref_chars"])


class _HFTok:
    """``.encode(text).input_ids`` over a Hugging Face tokenizer (default special
    tokens), so ``measure_chars_per_token`` measures a teacher exactly as it
    measures the pipeline's tokenizers."""

    def __init__(self, tok) -> None:
        self.tok = tok

    def encode(self, text: str):
        class _Out:
            pass
        o = _Out()
        o.input_ids = self.tok(text).input_ids
        return o


def teacher_caps(candidates_path: Path, references: Sequence[str]) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    from arabic_eval.tasks.freeform.generation import DecodingConfig, derive_token_cap, measure_chars_per_token
    _defaults, cands = load_candidates(candidates_path)
    cfg = DecodingConfig()
    out: Dict[str, Any] = {}
    for c in cands:
        tok = AutoTokenizer.from_pretrained(c.model, revision=c.revision)
        cpt = measure_chars_per_token(_HFTok(tok), references)
        out[c.slug] = {"repo": c.repo, "revision": c.revision, "chars_per_token": round(cpt, 4),
                       "token_cap": derive_token_cap(cfg, cpt), "max_output_chars": cfg.max_output_chars,
                       "tokenizer_class": type(tok).__name__, "vocab": len(tok)}
        log.info("%s: %.3f chars/token → cap %d", c.slug, cpt, out[c.slug]["token_cap"])
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--n", type=int, default=N_PROMPTS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--exclusions", default=EXCLUSIONS)
    ap.add_argument("--caps", action="store_true", help="also compute every candidate's token cap")
    ap.add_argument("--candidates", default=CANDIDATES)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    exclusions = load_exclusions(REPO_ROOT / args.exclusions)
    recs = load_corpus("cidar", "dev", exclusions=exclusions)
    picked = seeded_subset(recs, args.n, args.seed)
    bounds = strata_bounds()
    rows = [{"id": r.id, "instruction": r.question, "prompt": r.question, "context": r.context,
             "reference": r.answer, "ref_chars": len(r.answer), "stratum": stratum_of(len(r.answer), bounds)}
            for r in picked]
    check = heldout_overlap(rows, REPO_ROOT / HELDOUT, REPO_ROOT)
    if not check["passed"]:
        raise SystemExit(f"held-out overlap check FAILED: {check}")

    out_dir = REPO_ROOT / args.out_dir
    path = out_dir / "prompts_dev.jsonl"
    write_jsonl(path, rows)
    manifest: Dict[str, Any] = {
        "corpus": "cidar", "split": "dev", "dev_records_after_exclusions": len(recs), "n": len(rows),
        "selection": f"sorted ids, numpy.random.default_rng({args.seed}).permutation, first {args.n}",
        "exclusions": str(args.exclusions), "exclusions_digest": exclusions.digest() if exclusions else None,
        "exclusions_dropped": dict(exclusions.dropped) if exclusions else {},
        "template_version": TEMPLATE_VERSION, "strata_bounds_ref_chars": bounds,
        "strata": {s: sum(1 for r in rows if r["stratum"] == s) for s in ("short", "medium", "long")},
        "mean_ref_chars": round(sum(r["ref_chars"] for r in rows) / len(rows), 2),
        "system_prompt": SYSTEM_PROMPT, "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "prompts_file": str(path.relative_to(REPO_ROOT)), "prompts_sha256": file_sha256(path),
        "heldout_check": check,
    }
    if args.caps:
        manifest["caps"] = teacher_caps(REPO_ROOT / args.candidates, [r["reference"] for r in rows])
    with open(out_dir / "bakeoff_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("system_prompt",)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
