#!/usr/bin/env python
"""How many tokens a cell's Phase 3 trained on (brief v6 §8, the tables' T6 compute rows).

    .venv/bin/python scripts/count_phase3_tokens.py --cell outputs/experiments/qwen_native_vs_araroopat/bpe_16k_3phase_v5_distill \
        --out outputs/experiments/_comparative_tables/m10/phase3_tokens.json

Phase 3 trains on a fixed number of instruction examples, not a fixed token budget, so its token count differs per
tokenizer. This rebuilds the cell's mixture exactly: the pools are loaded as the run loaded them
(``sft_mixture.load_mixture_pools`` with the cell's ``training.corpus_params`` — the teacher overlay — and its
contamination exclusions), the kept record ids are read from ``<cell>/data/sft_mixture_manifest.json`` (repeats
included), and every record goes through the training path's own ``tokenize_record`` at the phase's ``max_length`` and
``loss_target``. Reports per corpus and in all: examples, input tokens (prompt + answer + EOS, as fed) and loss
tokens; then the examples the run actually processed (``steps_completed × batch_size`` from ``all_metrics.json``)
and the input tokens that implies (processed examples × the mixture's mean — the loader's shuffle order is not
replayed, so this is an estimate of the processed share, exact for the full mixture).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def count(cell: Path) -> Dict[str, Any]:
    import arabic_eval.tokenizers  # noqa: F401
    from arabic_eval.config import PhaseConfig, TokenizerConfig
    from arabic_eval.data.contamination import load_exclusions
    from arabic_eval.data.finetune_corpora import tokenize_record
    from arabic_eval.data.sft_mixture import load_mixture_pools
    from arabic_eval.registry import tokenizer_registry

    raw = json.loads((cell / "config.json").read_text())
    tr = raw["training"]
    sft = PhaseConfig(**tr["phases"]["sft"])
    tc = TokenizerConfig(**raw["tokenizer"])
    tok = tokenizer_registry.get(tc.type)(**(tc.params or {}))
    path = tc.load_path or tc.save_path
    if path and (REPO / path).exists():
        tok.load(str(REPO / path))
    manifest = json.loads((cell / "data" / "sft_mixture_manifest.json").read_text())
    exclusions = load_exclusions(tr.get("contamination_exclusions", "configs/contamination/exclusions.json"))
    datasets = list(manifest["datasets"])
    pools, _ = load_mixture_pools(datasets, corpus_params=tr.get("corpus_params") or {},
                                  clean_latin_rows=bool(manifest.get("clean_latin_rows")), exclusions=exclusions)
    t0 = time.perf_counter()
    per: Dict[str, Any] = {}
    tot_ex = tot_in = tot_loss = 0
    for name in datasets:
        by_id = {r.id: r for r in pools[name]}
        ids = manifest["datasets"][name]["ids"]
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"{cell.name}/{name}: {len(missing)} manifest ids not in the rebuilt pool, e.g. {missing[:3]}")
        n_in = n_loss = 0
        for i in ids:
            entry, loss_tokens, _trunc = tokenize_record(by_id[i], tok, sft.max_length, sft.loss_target)
            n_in += len(entry["input_ids"]) if entry else 0
            n_loss += loss_tokens
        per[name] = {"examples": len(ids), "input_tokens": n_in, "loss_tokens": n_loss,
                     "manifest_loss_tokens": manifest["datasets"][name].get("loss_tokens")}
        tot_ex += len(ids)
        tot_in += n_in
        tot_loss += n_loss
    am = json.loads((cell / "all_metrics.json").read_text())
    steps = int(am["training"]["sft"]["steps_completed"])
    processed = steps * int(sft.batch_size)
    return {"cell": str(cell.relative_to(REPO)), "tokenizer": tc.type, "max_length": sft.max_length,
            "examples": tot_ex, "input_tokens": tot_in, "loss_tokens": tot_loss,
            "manifest_loss_tokens": manifest.get("loss_tokens"),
            "mean_input_tokens_per_example": round(tot_in / tot_ex, 2),
            "steps_completed": steps, "batch_size": sft.batch_size, "examples_processed": processed,
            "input_tokens_processed_est": round(processed * tot_in / tot_ex),
            "per_corpus": per, "seconds": round(time.perf_counter() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_p = Path(args.out)
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    for c in args.cell:
        cell = (REPO / c).resolve()
        r = count(cell)
        res[r["cell"]] = r
        check = ("= manifest" if r["loss_tokens"] == r["manifest_loss_tokens"]
                 else f"manifest says {r['manifest_loss_tokens']}")
        print(f"{r['cell']}: {r['examples']} examples, {r['input_tokens']:,} input tokens "
              f"({r['mean_input_tokens_per_example']} / example), loss tokens {r['loss_tokens']:,} ({check}); "
              f"processed {r['examples_processed']} examples ≈ {r['input_tokens_processed_est']:,} tokens "
              f"({r['seconds']} s)", flush=True)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=1))
    print(f"→ {out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
