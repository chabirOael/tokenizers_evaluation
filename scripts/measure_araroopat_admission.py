#!/usr/bin/env python3
"""Admission of saved AraRooPat tokenizers on real eval text.

*Admission* = share of word occurrences (alpha chunks) that reach each path
when the tokenizer encodes real Arabic-Exam questions: ``ROOT+PAT``,
``PREP`` ([PREP_*]), ``FUNC`` ([FUNC_*]), ``CLITIC`` (pronoun-hosted
prepositions emitted as clitic tokens only: له, بها) and ``LIT`` (the
character fallback). Plus fertility (tokens per whitespace word) and compression
(characters per token) on the same text. The numbers quoted in CLAUDE.md
(48.2 % ROOT+PAT at the balanced tier, 2026-08-30) were measured this way
on 300 seeded questions.

Usage:
    .venv/bin/python scripts/measure_araroopat_admission.py \\
        outputs/tokenizers/araroopat_hashfix outputs/tokenizers/araroopat_func
    .venv/bin/python scripts/measure_araroopat_admission.py <dir>... --questions 300 --seed 42
"""
from __future__ import annotations

import argparse
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabic_eval.tokenizers.araroopat import (  # noqa: E402
    PFX_CLITICE,
    PFX_CLITICP,
    PFX_FUNC,
    PFX_PREP,
    PFX_ROOT,
    TOK_LIT_BEGIN,
    TOK_PROP_BEGIN,
    AraRooPatTokenizer,
    _extract_alpha_chunks,
)

PATHS = ("ROOT+PAT", "PREP", "FUNC", "CLITIC", "PROP", "LIT")


def _eval_texts(n: int, seed: int) -> List[str]:
    import arabic_eval.tasks  # noqa: F401  (registry)
    from arabic_eval.registry import task_registry
    task = task_registry.get("arabic_exam")({})
    examples = task.get_eval_examples()
    rng = random.Random(seed)
    picked = rng.sample(examples, min(n, len(examples)))
    return [" ".join([ex["question"], *ex["choices"]]) for ex in picked]


def _chunk_path(tok: AraRooPatTokenizer, chunk: str) -> str:
    ids: List[int] = []
    toks: List[str] = []
    tok._emit_alpha(chunk, ids, toks)
    names = [tok._reverse_vocab.get(i, "") for i in ids]
    if any(t.startswith(PFX_FUNC) for t in names):
        return "FUNC"
    if any(t.startswith(PFX_PREP) for t in names):
        return "PREP"
    if any(t.startswith(PFX_ROOT) for t in names):
        return "ROOT+PAT"
    if TOK_PROP_BEGIN in names:
        return "PROP"
    if TOK_LIT_BEGIN in names:
        return "LIT"
    if names and all(t.startswith((PFX_CLITICP, PFX_CLITICE)) for t in names):
        return "CLITIC"
    return "LIT"


def measure(tok_dir: Path, texts: List[str]) -> Dict[str, float]:
    tok = AraRooPatTokenizer()
    tok.load(tok_dir)
    tok.encode("نص قصير للإحماء")   # warm the bridge
    counts: Counter = Counter()
    n_tokens = n_words = n_chars = 0
    for text in texts:
        out = tok.encode(text)
        n_tokens += len(out.input_ids)
        words = text.split()
        n_words += len(words)
        n_chars += sum(len(w) for w in words)
        for w in unicodedata.normalize("NFKC", text).split():
            for chunk in _extract_alpha_chunks(w):
                counts[_chunk_path(tok, chunk)] += 1
    total = sum(counts.values()) or 1
    res = {p: 100.0 * counts[p] / total for p in PATHS}
    res.update({"chunks": total, "vocab_size": tok.vocab_size,
                "fertility": n_tokens / max(n_words, 1), "compression": n_chars / max(n_tokens, 1),
                "func_inventory": len(tok.func_words), "prep_inventory": len(tok.prepositions),
                "proper_nouns": getattr(tok, "proper_nouns", "-")})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tokenizers", nargs="+", help="saved AraRooPat directories")
    ap.add_argument("--questions", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    texts = _eval_texts(args.questions, args.seed)
    print(f"{len(texts)} Arabic-Exam questions (seed {args.seed}), question + choices text\n")
    header = (f"{'tokenizer':34s} {'vocab':>6s} {'ROOT+PAT':>9s} {'PREP':>6s} {'FUNC':>6s} {'CLITIC':>7s} "
              f"{'PROP':>6s} {'LIT':>6s} {'fert':>6s} {'compr':>6s}  inventories")
    print(header)
    print("-" * len(header))
    for d in args.tokenizers:
        r = measure(Path(d), texts)
        print(f"{Path(d).name:34s} {r['vocab_size']:>6d} {r['ROOT+PAT']:>8.1f}% {r['PREP']:>5.1f}% {r['FUNC']:>5.1f}% "
              f"{r['CLITIC']:>6.1f}% {r['PROP']:>5.1f}% {r['LIT']:>5.1f}% {r['fertility']:>6.2f} {r['compression']:>6.2f}  "
              f"prep={r['prep_inventory']} func={r['func_inventory']} proper_nouns={r['proper_nouns']}")


if __name__ == "__main__":
    main()
