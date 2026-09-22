#!/usr/bin/env python
"""Decode fidelity of a saved AraRooPat tokenizer: ``decode(encode(x))`` against ``x``.

Three numbers, the ones the free-form eval's ``reference_roundtrip_chrf`` ceiling
is made of (2026-09-22):

* round-trip chrF on the 250 held-out CIDAR references, raw;
* the same with the references' diacritics stripped first (the tokenizer stores
  undiacritized stems, so tashkeel in a reference is lost by design);
* exact-match rate of ``decode(encode(w)) == w`` on N distinct undiacritized
  corpus words — a seeded sample of the tokenizer-training corpus under the
  pipeline's own preprocessing — plus a breakdown of the mismatches (alef seat,
  ة/ه, ى/ي, other) and the first examples of each.

Usage::

    .venv/bin/python scripts/measure_araroopat_roundtrip.py --tokenizer-path outputs/tokenizers/araroopat_maxpat40k_v2
    .venv/bin/python scripts/measure_araroopat_roundtrip.py --tokenizer-path <dir> --out <json> --words 2000

The corpus sample is cached under ``outputs/data_cache/araroopat_roundtrip_words_<n>_<seed>.json``
so before/after runs score the same words.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from arabic_eval.tokenizers.araroopat import AraRooPatTokenizer  # noqa: E402
from arabic_eval.tokenizers.utils.arabic_text import ARABIC_DIACRITICS, ARABIC_LETTERS, strip_diacritics  # noqa: E402

HELDOUT = REPO / "configs/contamination/freeform_cidar_heldout_v1.jsonl"
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})


def _classify(word: str, back: str) -> str:
    if back == word:
        return "exact"
    if "?" in back:
        return "unk"
    if back.translate(_ALEF) == word.translate(_ALEF):
        return "alef"
    if back.replace("ة", "ه") == word.replace("ة", "ه"):
        return "ta_ha"
    if back.replace("ى", "ي") == word.replace("ى", "ي"):
        return "ya"
    if " " in back:
        return "split"
    return "other"


def sample_corpus_words(n: int, seed: int) -> list[str]:
    cache = REPO / "outputs/data_cache" / f"araroopat_roundtrip_words_{n}_{seed}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    import yaml

    from arabic_eval.data.loader import extract_texts, load_arabic_dataset

    base = yaml.safe_load((REPO / "configs/base.yaml").read_text(encoding="utf-8"))
    data_cfg = base["data"]
    ds = load_arabic_dataset(
        dataset_name=data_cfg["dataset_name"], dataset_config=data_cfg.get("dataset_config"),
        cache_dir=data_cfg.get("cache_dir", "outputs/data_cache"),
        preprocessing_config=data_cfg.get("preprocessing"), seed=seed,
    )
    texts = list(extract_texts(ds["train"]))
    rng = random.Random(seed)
    rng.shuffle(texts)
    seen: set[str] = set()
    words: list[str] = []
    for t in texts:
        for w in unicodedata.normalize("NFKC", t).split():
            # undiacritized, letters only, ≥ 2 chars, distinct
            if len(w) < 2 or any(c in ARABIC_DIACRITICS for c in w) or not all(c in ARABIC_LETTERS for c in w):
                continue
            if w in seen:
                continue
            seen.add(w)
            words.append(w)
            if len(words) >= n:
                break
        if len(words) >= n:
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    return words


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--words", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="write the numbers + examples to this JSON")
    args = ap.parse_args()

    from arabic_eval.tasks.freeform.metrics import chrf_sentence

    tok = AraRooPatTokenizer()
    tok.load(args.tokenizer_path)
    rt = lambda s: tok.decode(tok.encode(s).input_ids)  # noqa: E731

    rows = [json.loads(l) for l in HELDOUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    t0 = time.perf_counter()
    raw = [chrf_sentence(rt(r["reference"]), r["reference"]) for r in rows]
    stripped_refs = [strip_diacritics(r["reference"]) for r in rows]
    stripped = [chrf_sentence(rt(ref), ref) for ref in stripped_refs]
    ref_wall = time.perf_counter() - t0

    words = sample_corpus_words(args.words, args.seed)
    t0 = time.perf_counter()
    kinds: Counter = Counter()
    examples: dict[str, list] = {}
    for w in words:
        back = rt(w)
        k = _classify(w, back)
        kinds[k] += 1
        if k != "exact" and len(examples.setdefault(k, [])) < 15:
            examples[k].append([w, back])
    word_wall = time.perf_counter() - t0

    result = {
        "tokenizer_path": str(args.tokenizer_path),
        "vocab_size": tok.vocab_size,
        "reconstruction_entries": len(tok._reconstruction),
        "reference_roundtrip_chrf_raw": round(sum(raw) / len(raw), 2),
        "reference_roundtrip_chrf_diacritics_stripped": round(sum(stripped) / len(stripped), 2),
        "references": len(rows),
        "word_sample": {"n": len(words), "seed": args.seed},
        "word_exact_match_rate": round(kinds["exact"] / max(len(words), 1), 4),
        "word_mismatch_kinds": dict(kinds),
        "word_mismatch_examples": examples,
        "wall_sec": {"references": round(ref_wall, 1), "words": round(word_wall, 1)},
    }
    print(json.dumps({k: v for k, v in result.items() if k != "word_mismatch_examples"}, ensure_ascii=False, indent=2))
    for k, ex in examples.items():
        print(f"-- {k}: " + ", ".join(f"{w}→{b}" for w, b in ex[:8]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
