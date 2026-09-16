#!/usr/bin/env python3
"""Discover closed-class function-word candidates for AraRooPat's ``[FUNC_*]`` group.

Runs CAMeL (through the araroopat bridge) over every distinct alpha chunk
of the tokenizer-training corpus — the same chunking, preprocessing and
batching as ``AraRooPatTokenizer._corpus_prepass`` — and reports every
chunk whose top reading carries a closed-class POS tag, with occurrence
counts, lemma, clitics and how the *current* tokenizer handles the word
(LIT / ROOT+PAT / PREP, read from an existing pre-pass cache).

The requested POS set is matched on CAMeL's *actual* tag names: CAMeL emits
``pron_interrog`` / ``adv_interrog``, not ``interrog_pron`` / ``interrog_adv``
(both spellings are accepted here and mapped). Neighbouring closed-class
tags outside the requested set (``adv`` for ثم / هنا, ``adv_rel`` for كيف,
``verb_pseudo`` for إنّ, ``part_focus`` for أما, ``part_voc`` for يا, ...)
are reported too, flagged ``neighbour``, so the curation happens on the
full picture rather than inside the filter.

Outputs (``--output``, default ``outputs/tokenizers/araroopat_func_discovery/``):

* ``func_candidates.csv`` — one row per surface chunk (UTF-8 BOM for Excel).
* ``func_by_lemma.csv`` — one row per bare lemma with all its surfaces.
* ``summary.json`` — tag histograms (types and occurrences), coverage, timing.

Usage:
    .venv/bin/python scripts/discover_araroopat_func_words.py
    .venv/bin/python scripts/discover_araroopat_func_words.py --max-samples 20000 --top 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.config import DataConfig, load_yaml
from arabic_eval.data.loader import extract_texts, load_arabic_dataset
from arabic_eval.tokenizers.araroopat import _extract_alpha_chunks
from arabic_eval.tokenizers.araroopat_backend import _alef_norm, _strip_diac, clitic_surface, _norm_clitic
from arabic_eval.tokenizers.araroopat_bridge import CamelBridge
from arabic_eval.utils.logging import setup_logger

# The set the user asked for, in CAMeL's tag spelling. ``interrog_pron`` /
# ``interrog_adv`` are not CAMeL tags; they map to the two below.
REQUESTED_POS = frozenset({
    "pron_dem", "pron_rel", "pron", "conj", "conj_sub",
    "pron_interrog", "adv_interrog",
    "part_neg", "part_verb", "part_interrog", "part_fut", "part",
})
POS_ALIASES = {"interrog_pron": "pron_interrog", "interrog_adv": "adv_interrog"}
# Closed-class tags CAMeL uses that sit next to the requested set.
NEIGHBOUR_POS = frozenset({
    "adv", "adv_rel", "part_det", "part_focus", "part_restrict", "part_voc",
    "pron_exclam", "verb_pseudo", "interj",
})

CSV_FIELDS = (
    "surface", "occurrences", "top_pos", "lemma", "lemma_bare", "proclitics", "enclitics",
    "bucket", "in_requested_set", "alt_pos", "alt_lemmas", "current_path", "current_tokens",
)


def _parse_pos(arg: Optional[str]) -> frozenset:
    if not arg:
        return REQUESTED_POS
    out = set()
    for raw in arg.split(","):
        t = raw.strip()
        if t:
            out.add(POS_ALIASES.get(t, t))
    return frozenset(out)


def _load_current_paths(cache_pkl: Path) -> Dict[str, tuple]:
    """word → (path, token-ish summary) from an existing corpus_analysis.pkl (alef-normalised keys)."""
    import pickle
    if not cache_pkl.exists():
        return {}
    with cache_pkl.open("rb") as f:
        payload = pickle.load(f)
    entries = payload["entries"] if isinstance(payload, dict) else payload
    out: Dict[str, tuple] = {}
    for e in entries:
        if getattr(e, "particle", None):
            summ = f"[PREP_{e.particle}]"
            path = "PREP"
        elif e.analyzed and e.root:
            summ = f"[ROOT_{e.root}] [PAT_{e.pattern}]"
            path = "ROOT+PAT"
        else:
            summ = "[LIT_BEGIN] … [LIT_END]"
            path = "LIT"
        pr = "".join(e.proclitics or ())
        en = "".join(e.enclitics or ())
        out[e.word] = (path, (f"{pr}+" if pr else "") + summ + (f"+{en}" if en else ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="Jr23xd23/ArabicText-Large")
    ap.add_argument("--base-config", default=None, help="YAML with data.preprocessing (default configs/base.yaml)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--top", type=int, default=3, help="readings per chunk to fetch (1..32)")
    ap.add_argument("--pos", default=None, help="comma-separated POS tags (default: the requested set)")
    ap.add_argument("--min-count", type=int, default=1, help="drop surfaces below this occurrence count")
    ap.add_argument("--cache", default="outputs/tokenizers/araroopat_cache/corpus_analysis.pkl",
                    help="existing pre-pass cache used to label each word's current path")
    ap.add_argument("--output", default="outputs/tokenizers/araroopat_func_discovery")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_logger("arabic_eval")
    requested = _parse_pos(args.pos)
    report_pos = requested | NEIGHBOUR_POS
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    base_path = Path(args.base_config) if args.base_config else \
        Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    preprocessing = DataConfig(**(load_yaml(base_path).get("data") or {})).preprocessing
    print(f"[discover] preprocessing ({base_path}): {preprocessing}", flush=True)

    t0 = time.perf_counter()
    ds = load_arabic_dataset(dataset_name=args.dataset, max_train_samples=args.max_samples,
                             preprocessing_config=preprocessing, seed=args.seed)
    texts = extract_texts(ds["train"])
    load_s = time.perf_counter() - t0
    print(f"[discover] loaded {len(texts):,} texts in {load_s:.0f} s", flush=True)

    # Same chunking as _corpus_prepass.
    t0 = time.perf_counter()
    counts: Counter = Counter()
    for t in texts:
        t = unicodedata.normalize("NFKC", t)
        for w in t.split():
            for chunk in _extract_alpha_chunks(w):
                counts[chunk] += 1
    unique = list(counts.keys())
    chunk_s = time.perf_counter() - t0
    print(f"[discover] {len(unique):,} unique alpha chunks ({sum(counts.values()):,} occurrences) in {chunk_s:.0f} s",
          flush=True)

    current = _load_current_paths(Path(args.cache))
    print(f"[discover] current-path cache: {len(current):,} entries from {args.cache}", flush=True)

    bridge = CamelBridge()
    rows: List[dict] = []
    tag_types: Counter = Counter()
    tag_occ: Counter = Counter()
    by_lemma: Dict[str, dict] = defaultdict(lambda: {"pos": Counter(), "surfaces": Counter(), "occurrences": 0,
                                                     "bucket": "neighbour"})
    batch = 256
    t0 = time.perf_counter()
    done = 0
    for start in range(0, len(unique), batch):
        words = unique[start:start + batch]
        results = bridge.analyze(words, top=max(1, min(args.top, 32)))
        for w, cands in zip(words, results):
            if not cands:
                continue
            top = cands[0]
            pos = top.get("pos") or ""
            tag_types[pos] += 1
            tag_occ[pos] += counts[w]
            if pos not in report_pos or counts[w] < args.min_count:
                continue
            lemma = top.get("lex") or ""
            lemma_bare = _strip_diac(lemma)
            prc = [clitic_surface(_norm_clitic(top.get(k))) for k in ("prc3", "prc2", "prc1", "prc0")]
            prc = [c for c in prc if c]
            enc = clitic_surface(_norm_clitic(top.get("enc0")))
            alt = [(c.get("pos") or "", _strip_diac(c.get("lex") or "")) for c in cands[1:]]
            cur = current.get(_alef_norm(w)) or current.get(w) or ("(not in cache)", "")
            bucket = "requested" if pos in requested else "neighbour"
            rows.append({
                "surface": w, "occurrences": counts[w], "top_pos": pos, "lemma": lemma,
                "lemma_bare": lemma_bare, "proclitics": "+".join(prc), "enclitics": enc or "",
                "bucket": bucket, "in_requested_set": pos in requested,
                "alt_pos": "|".join(p for p, _ in alt), "alt_lemmas": "|".join(l for _, l in alt),
                "current_path": cur[0], "current_tokens": cur[1],
            })
            L = by_lemma[lemma_bare]
            L["pos"][pos] += counts[w]
            L["surfaces"][w] += counts[w]
            L["occurrences"] += counts[w]
            if bucket == "requested":
                L["bucket"] = "requested"
        done += len(words)
        if (start // batch) % 200 == 0:
            rate = done / max(time.perf_counter() - t0, 1e-9)
            print(f"[discover] {done:,}/{len(unique):,} chunks  {rate:.0f} w/s  "
                  f"eta {(len(unique) - done) / max(rate, 1e-9) / 60:.1f} min", flush=True)
    analyze_s = time.perf_counter() - t0

    rows.sort(key=lambda r: (-r["occurrences"], r["surface"]))
    with (out_dir / "func_candidates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wr.writeheader()
        wr.writerows(rows)

    lemma_rows = []
    for lemma_bare, L in by_lemma.items():
        lemma_rows.append({
            "lemma_bare": lemma_bare, "occurrences": L["occurrences"], "bucket": L["bucket"],
            "pos": "|".join(f"{p}:{n}" for p, n in L["pos"].most_common()),
            "num_surfaces": len(L["surfaces"]),
            "surfaces": " ".join(f"{s}:{n}" for s, n in L["surfaces"].most_common(40)),
        })
    lemma_rows.sort(key=lambda r: (-r["occurrences"], r["lemma_bare"]))
    with (out_dir / "func_by_lemma.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=("lemma_bare", "occurrences", "bucket", "pos", "num_surfaces", "surfaces"))
        wr.writeheader()
        wr.writerows(lemma_rows)

    total_occ = sum(counts.values())
    req_rows = [r for r in rows if r["in_requested_set"]]
    summary = {
        "dataset": args.dataset, "preprocessing": preprocessing, "max_samples": args.max_samples,
        "top": args.top, "requested_pos": sorted(requested), "neighbour_pos": sorted(NEIGHBOUR_POS),
        "texts": len(texts), "unique_chunks": len(unique), "occurrences": total_occ,
        "requested": {
            "types": len(req_rows), "occurrences": sum(r["occurrences"] for r in req_rows),
            "share_of_occurrences": sum(r["occurrences"] for r in req_rows) / max(total_occ, 1),
            "lemmas": sum(1 for r in lemma_rows if r["bucket"] == "requested"),
            "current_path_occurrences": {},   # filled below
        },
        "neighbour": {"types": len(rows) - len(req_rows),
                      "occurrences": sum(r["occurrences"] for r in rows if not r["in_requested_set"])},
        "pos_histogram_types": dict(tag_types.most_common()),
        "pos_histogram_occurrences": dict(tag_occ.most_common()),
        "top_requested": [(r["surface"], r["occurrences"], r["top_pos"], r["current_path"]) for r in req_rows[:60]],
        "timing_s": {"load": round(load_s, 1), "chunk": round(chunk_s, 1), "analyze": round(analyze_s, 1),
                     "total": round(time.perf_counter() - t_all, 1)},
    }
    cp: Counter = Counter()
    for r in req_rows:
        cp[r["current_path"]] += r["occurrences"]
    summary["requested"]["current_path_occurrences"] = dict(cp)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[discover] requested set: {summary['requested']['types']:,} surface types, "
          f"{summary['requested']['occurrences']:,} occurrences "
          f"({100 * summary['requested']['share_of_occurrences']:.2f} % of the corpus), "
          f"{summary['requested']['lemmas']:,} lemmas; by current path {dict(cp)}", flush=True)
    print(f"[discover] wrote {out_dir}/func_candidates.csv, func_by_lemma.csv, summary.json "
          f"in {summary['timing_s']['total']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
