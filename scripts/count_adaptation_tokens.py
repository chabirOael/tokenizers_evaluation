#!/usr/bin/env python
"""How much of a vocabulary the adaptation actually showed the model (brief v6 §4.1, R-128-4; the tables' T1).

    .venv/bin/python scripts/count_adaptation_tokens.py --cell outputs/experiments/qwen_native_vs_araroopat/bpe_16k_3phase_v3 \
        [--cell …] --out outputs/experiments/_comparative_tables/m5/adaptation_token_counts.json

For a cell whose Phases 1 + 2 trained on the packed pretraining mix, read the blocks those phases consumed — the
ranges ``all_metrics.json["training"][<phase>]["data"]`` records (raw-text ``[block_start, block_end)`` of the
packed pool, and the QA-blend prefix of the packed QA corpus) — from the packed ``.npy`` arrays the cell's
``data/pretraining_mix/{packed,qa_blend}_manifest.json`` point at, and count every token id. Reports, over the
tokenizer's vocabulary (special ids included in the denominator, listed apart): the number and share of ids seen 0,
1–9 and ≥ 10 times, the tokens counted, and the most frequent unseen-adjacent facts. Phase 3 (instruction records,
not packed) is not counted: the question is what the raw-text adaptation that teaches the new vocabulary covered.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _arr(d: Path) -> np.ndarray:
    return np.load(d / "packed_input_ids.npy", mmap_mode="r")


def count(cell: Path) -> Dict[str, Any]:
    am = json.loads((cell / "all_metrics.json").read_text())
    pm = json.loads((cell / "data/pretraining_mix/packed_manifest.json").read_text())
    vocab = int(pm["tokenizer"]["vocab_size"])
    specials = sorted({int(v) for v in (pm["tokenizer"].get("special_tokens") or {}).values() if isinstance(v, int)})
    pool_arr = _arr(Path(pm["packed_dir"]) if Path(pm["packed_dir"]).is_absolute() else REPO / pm["packed_dir"])
    counts = np.zeros(vocab, dtype=np.int64)
    ranges = []
    for ph in ("embedding_alignment", "warmup"):
        d = ((am.get("training") or {}).get(ph) or {}).get("data") or {}
        if d.get("dataset") != "pretraining_mix":
            continue
        a, b = int(d["block_start"]), int(d["block_end"])
        for s in range(a, b, 20000):
            blk = np.asarray(pool_arr[s:min(b, s + 20000)]).ravel()
            counts += np.bincount(blk, minlength=vocab)[:vocab]
        ranges.append({"phase": ph, "source": "pool", "blocks": [a, b]})
        qb = d.get("qa_blend")
        if qb:
            qm = json.loads((cell / "data/pretraining_mix/qa_blend_manifest.json").read_text())
            qdir = Path(qm["packed_dir"]) if Path(qm["packed_dir"]).is_absolute() else REPO / qm["packed_dir"]
            qa = np.asarray(_arr(qdir)[int(qb["block_start"]):int(qb["block_end"])]).ravel()
            counts += np.bincount(qa, minlength=vocab)[:vocab]
            ranges.append({"phase": ph, "source": "qa_blend", "blocks": [int(qb["block_start"]), int(qb["block_end"])]})
    regular = np.ones(vocab, dtype=bool)
    regular[[s for s in specials if s < vocab]] = False
    c = counts[regular]
    n = int(regular.sum())
    return {"cell": str(cell.relative_to(REPO)), "tokenizer": pm["tokenizer"].get("type"), "vocab_size": vocab,
            "special_ids": specials, "ranges": ranges, "tokens_counted": int(counts.sum()),
            "regular_ids": n,
            "seen_0": int((c == 0).sum()), "seen_1_9": int(((c >= 1) & (c <= 9)).sum()), "seen_ge_10": int((c >= 10).sum()),
            "share_seen_0": round(float((c == 0).mean()), 6), "share_seen_1_9": round(float(((c >= 1) & (c <= 9)).mean()), 6),
            "share_seen_ge_10": round(float((c >= 10).mean()), 6),
            "median_count": float(np.median(c)), "p10_count": float(np.percentile(c, 10)),
            # the never-seen regular ids (for R-128-4: do the model's answers use pieces the adaptation never showed?)
            "unseen_ids": [int(i) for i in np.nonzero((counts == 0) & regular)[0]]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_p = Path(args.out)
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    for c in args.cell:
        r = count((REPO / c).resolve() if not Path(c).is_absolute() else Path(c))
        res[r["cell"]] = r
        print(f"{r['cell']}: vocab {r['vocab_size']}, {r['tokens_counted']:,} tokens; never seen {r['seen_0']} "
              f"({100 * r['share_seen_0']:.2f} %), 1–9× {r['seen_1_9']} ({100 * r['share_seen_1_9']:.2f} %), "
              f"≥ 10× {r['seen_ge_10']} ({100 * r['share_seen_ge_10']:.2f} %)")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=1))
    print(f"→ {out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
