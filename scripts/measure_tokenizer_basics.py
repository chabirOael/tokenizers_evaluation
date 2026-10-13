#!/usr/bin/env python
"""Basic, model-free measurements of saved tokenizers on fixed texts (brief v6 §4.1, the supervisor tables' T1).

    .venv/bin/python scripts/measure_tokenizer_basics.py \
        --tok bpe_16k=bpe:outputs/tokenizers/bpe_16k_v3 --tok bpe_128k=bpe:outputs/tokenizers/bpe_128k_v1 \
        --tok native_qwen3=native_qwen3 --out outputs/experiments/_comparative_tables/m5/tokenizer_basics.json

Per tokenizer:
  * eval split — the first ``--n-eval`` texts of the ArabicText-Large eval split (the pipeline's loader, base.yaml
    preprocessing, seed 42): tokens per whitespace word (non-special tokens) and characters per token
    (all ids, the free-form eval's ``measure_chars_per_token`` definition);
  * the 250 held-out references — characters per token (same definition, the one the token cap is derived from),
    reference round-trip chrF (``decode(encode(ref))`` against ``ref``, the eval's ``reference_roundtrip_chrf``),
    the same on the references with their diacritics stripped first, and the exact round-trip share;
  * words — exact ``decode(encode(w)) == w`` on the 2 000 distinct corpus words of
    ``scripts/measure_araroopat_roundtrip.py`` (its cached list, seed 42);
  * pool — ``--n-pool`` documents drawn from a pool's per-source parquets at the source shares (seed 42):
    tokens per pool word (Σ tokens of ``encode_document``-style encodes ÷ Σ ``n_words``), per source and pooled —
    the number that sizes a pool for a token budget;
  * the choice letter ``" أ"`` — its non-special token ids, the window the MCQ scorer sums.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_eval.registry import tokenizer_registry  # noqa: E402
import arabic_eval.tokenizers  # noqa: E402,F401  (registers every tokenizer)
from arabic_eval.tokenizers.utils.arabic_text import strip_diacritics  # noqa: E402
from arabic_eval.utils.io import save_json  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HELDOUT = REPO / "configs/contamination/freeform_cidar_heldout_v1.jsonl"
# the 2 000 distinct undiacritized corpus words of scripts/measure_araroopat_roundtrip.py (seed 42, cached there)
WORDS = REPO / "outputs/data_cache/araroopat_roundtrip_words_2000_42.json"
DEFAULT_POOL = REPO / "outputs/data_cache/pretraining_mix/51e0a74a3d05b6cf"
SHARES = {"fineweb2_arb": 0.7, "wikipedia_ar": 0.2, "arabicweb24": 0.1}


def build(spec: str):
    name, _, rest = spec.partition("=")
    typ, _, path = rest.partition(":")
    params = {}
    if "@" in typ:                      # type@model_name_or_path, e.g. native_llama@meta-llama/Llama-3.2-3B
        typ, model = typ.split("@", 1)
        params = {"model_name_or_path": model}
    tok = tokenizer_registry.get(typ)(**params)
    if path:
        tok.load(str(REPO / path))
    return name, typ, path or None, tok


def specials(tok) -> set:
    ids = set()
    st = tok.special_tokens or {}
    for v in st.values():
        if isinstance(v, int):
            ids.add(v)
    return ids


def eval_texts(n: int) -> List[str]:
    from arabic_eval.config import DataConfig, load_yaml
    from arabic_eval.data.loader import extract_texts, load_arabic_dataset
    data = DataConfig(**(load_yaml(REPO / "configs/base.yaml").get("data") or {}))
    ds = load_arabic_dataset(dataset_name=data.dataset_name, cache_dir=data.cache_dir, max_eval_samples=n,
                             preprocessing_config=data.preprocessing, seed=42)
    return extract_texts(ds["eval"])[:n]


def pool_docs(pool: Path, n: int, seed: int = 42) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    out = []
    for src, share in SHARES.items():
        t = pq.read_table(pool / f"{src}.parquet", columns=["id", "text", "n_words"])
        k = round(n * share)
        idx = sorted(rng.sample(range(t.num_rows), k))
        sub = t.take(idx).to_pylist()
        out += [{**r, "source": src} for r in sub]
    return out


def measure(tok, ev: List[str], refs: List[str], docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    from arabic_eval.tasks.freeform import metrics as M
    sp = specials(tok)
    t0 = time.perf_counter()
    words = toks = chars = all_ids = 0
    for t in ev:
        ids = tok.encode(t).input_ids
        all_ids += len(ids)
        toks += sum(1 for i in ids if i not in sp)
        words += len(t.split())
        chars += len(t)
    ev_out = {"n_texts": len(ev), "words": words, "tokens_nonspecial": toks, "fertility": round(toks / words, 4),
              "chars_per_token": round(chars / all_ids, 4), "seconds": round(time.perf_counter() - t0, 2)}
    rc = rt_ids = 0
    chrfs, exact = [], 0
    for r in refs:
        ids = tok.encode(r).input_ids
        rc += len(r)
        rt_ids += len(ids)
        back = M.roundtrip(tok, r)
        chrfs.append(M.chrf_sentence(back, r))
        exact += int(back == r)
    stripped = [strip_diacritics(r) for r in refs]
    chrfs_stripped = [M.chrf_sentence(M.roundtrip(tok, r), r) for r in stripped]
    ref_out = {"n": len(refs), "chars": rc, "tokens": rt_ids, "chars_per_token": round(rc / rt_ids, 4),
               "roundtrip_chrf": round(sum(chrfs) / len(chrfs), 4), "roundtrip_exact": round(exact / len(refs), 4),
               "roundtrip_chrf_diacritics_stripped": round(sum(chrfs_stripped) / len(chrfs_stripped), 4)}
    words = json.loads(WORDS.read_text(encoding="utf-8"))
    word_exact = sum(1 for w in words if tok.decode(tok.encode(w).input_ids) == w)
    word_out = {"n": len(words), "file": str(WORDS.relative_to(REPO)), "exact": round(word_exact / len(words), 4)}
    per: Dict[str, Dict[str, int]] = {}
    for d in docs:
        ids = tok.encode(d["text"]).input_ids
        s = per.setdefault(d["source"], {"docs": 0, "words": 0, "tokens": 0})
        s["docs"] += 1
        s["words"] += int(d["n_words"])
        s["tokens"] += len(ids)
    for s in per.values():
        s["tokens_per_word"] = round(s["tokens"] / s["words"], 4)
    tw = sum(s["words"] for s in per.values())
    tt = sum(s["tokens"] for s in per.values())
    pool_out = {"per_source": per, "words": tw, "tokens": tt, "tokens_per_word": round(tt / tw, 4) if tw else None}
    letter = [i for i in tok.encode(" أ").input_ids if i not in sp]
    return {"vocab_size": tok.vocab_size, "eval_split": ev_out, "references": ref_out, "words": word_out, "pool": pool_out,
            "letter_alef": {"ids": letter, "n_tokens": len(letter),
                            "pieces": [tok.decode([i]) for i in letter]}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tok", action="append", required=True, help="name=type[:saved_dir]")
    ap.add_argument("--n-eval", type=int, default=20000)
    ap.add_argument("--n-pool", type=int, default=1800)
    ap.add_argument("--pool", default=str(DEFAULT_POOL))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    refs = [json.loads(line)["reference"] for line in HELDOUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    ev = eval_texts(args.n_eval)
    docs = pool_docs(Path(args.pool), args.n_pool)
    out: Dict[str, Any] = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_eval": len(ev), "n_refs": len(refs),
                           "pool": str(Path(args.pool).relative_to(REPO)), "n_pool_docs": len(docs),
                           "pool_sample": {s: sum(1 for d in docs if d["source"] == s) for s in SHARES},
                           "tokenizers": {}}
    for spec in args.tok:
        name, typ, path, tok = build(spec)
        print(f"measuring {name} ({typ} {path or ''}) …", flush=True)
        out["tokenizers"][name] = {"type": typ, "path": path, **measure(tok, ev, refs, docs)}
        r = out["tokenizers"][name]
        print(f"  vocab {r['vocab_size']}  fertility {r['eval_split']['fertility']}  chars/tok eval "
              f"{r['eval_split']['chars_per_token']} refs {r['references']['chars_per_token']}  "
              f"round-trip chrF {r['references']['roundtrip_chrf']} / stripped {r['references']['roundtrip_chrf_diacritics_stripped']} "
              f"words exact {r['words']['exact']}  "
              f"pool tok/word {r['pool']['tokens_per_word']}  ' أ' → {r['letter_alef']['n_tokens']} token(s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists():   # keep the tokenizers measured earlier into the same file (same texts, same seeds)
        prev = json.loads(Path(args.out).read_text(encoding="utf-8"))
        out["tokenizers"] = {**prev.get("tokenizers", {}), **out["tokenizers"]}
    save_json(out, args.out)
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
