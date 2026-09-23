#!/usr/bin/env python
"""Add the per-character answer NLL to ``diag_heldout_*.json`` files written
before ``scripts/diag_heldout_loss.py`` emitted it (2026-09-23). No GPU.

For every file: ``answer_nll_total`` = the recorded ``nll_per_answer_token``
(4 decimals) × ``answer_tokens`` — the relative error of the reconstruction
is ≤ 0.00005 / NLL-per-token, i.e. a few 1e-5 —, then ``reference_chars`` (Σ
``len`` of the kept references after the tokenizer's own normalization, the
cell's tokenizer rebuilt from its ``config.json``), ``reference_chars_raw``,
``nll_per_answer_char`` and ``nll_per_answer_char_raw_chars`` — exactly the
fields a fresh run writes (``per_char_fields``). A file whose total was
*measured* by a fresh run keeps it; only the derived fields are recomputed.
Idempotent: a second pass rewrites the same values and prints ``=``.

    .venv/bin/python scripts/diag_backfill_per_char.py                 # the Qwen experiment folder
    .venv/bin/python scripts/diag_backfill_per_char.py --root outputs/experiments/<exp> --dry-run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = "outputs/experiments/qwen_native_vs_araroopat"

_spec = importlib.util.spec_from_file_location("diag_heldout_loss", REPO_ROOT / "scripts" / "diag_heldout_loss.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)  # type: ignore[union-attr]

FIELDS = ("answer_nll_total", "reference_chars", "reference_chars_raw",
          "nll_per_answer_char", "nll_per_answer_char_raw_chars")


def backfill_answers(ans: Dict[str, Any], references: Sequence[str], tokenizer) -> Dict[str, Any]:
    """The heldout_answers block with the per-character fields added (a new dict)."""
    out = dict(ans)
    if out.get("answer_nll_total_source") != "measured":
        out["answer_nll_total"] = round(float(ans["nll_per_answer_token"]) * int(ans["answer_tokens"]), 6)
        out["answer_nll_total_source"] = "backfill: nll_per_answer_token (4 d.p.) x answer_tokens"
    out.update(D.per_char_fields(out["answer_nll_total"], references, tokenizer))
    return out


def kept_references(res: Dict[str, Any], tokenizer) -> List[str]:
    """The references the file's run scored: all of them unless truncation
    dropped some, in which case the encoding is replayed to know which."""
    recs = D.heldout_records(REPO_ROOT / res.get("heldout", D.DEFAULT_HELDOUT))
    if not int(res["heldout_answers"].get("references_dropped_truncation") or 0):
        return [r.answer for r in recs]
    kept: List = []
    D.answer_encodings(recs, tokenizer, int(res.get("max_length", 2048)),
                       v1=int(res.get("template_version", 2)) == 1, kept=kept)
    return [r.answer for r in kept]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", help=f"folder searched recursively (default {DEFAULT_ROOT})")
    ap.add_argument("--dry-run", action="store_true", help="print old -> new, write nothing")
    args = ap.parse_args(argv)

    files = sorted(p for root in (args.root or [DEFAULT_ROOT])
                   for p in (REPO_ROOT / root).rglob("diag_heldout_*.json"))
    if not files:
        print("no diag_heldout_*.json found")
        return 1
    tokenizers: Dict[Tuple[str, str], Any] = {}
    for path in files:
        res = json.loads(path.read_text(encoding="utf-8"))
        cell = Path(res["cell"])
        cell = cell if cell.is_absolute() else REPO_ROOT / cell
        if not (cell / "config.json").exists():          # a cell moved to _superseded/ after the run
            cell = path.parent
        tok_cfg, _model_cfg, _raw = D.read_cell(cell)
        key = (tok_cfg.type, str(tok_cfg.load_path or tok_cfg.save_path))
        if key not in tokenizers:
            tokenizers[key] = D.build_tokenizer(tok_cfg)
        tok = tokenizers[key]
        old = res["heldout_answers"].get("nll_per_answer_char")
        new_ans = backfill_answers(res["heldout_answers"], kept_references(res, tok), tok)
        res["heldout_answers"] = new_ans
        rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        mark = "=" if old is not None and abs(float(old) - new_ans["nll_per_answer_char"]) < 1e-9 else "->"
        print(f"{rel}: nll_per_answer_char {old} {mark} {new_ans['nll_per_answer_char']:.6f} "
              f"(÷ {new_ans['reference_chars']} as encoded; ÷ raw {new_ans['reference_chars_raw']}: "
              f"{new_ans['nll_per_answer_char_raw_chars']:.6f}; total {new_ans['answer_nll_total']:.2f}, "
              f"{new_ans['answer_nll_total_source'].split(':')[0]})")
        if not args.dry_run:
            path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
