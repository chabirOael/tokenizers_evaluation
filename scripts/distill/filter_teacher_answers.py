#!/usr/bin/env python
"""Filter the teacher's raw answers into the training file the overlay reads (P1.6).

Rule 3 of the distillation brief — don't copy the teacher's mistakes. Every answer is
checked with ``arabic_eval.distill.postprocess.drop_reason`` (the free-form eval's own
text rules, in this order): not EOS-terminated (``length``), longer than 2 400
characters (``char_truncated`` — dropped, not cut), a repetition loop at its tail
(``loop``, rule and period recorded), degenerate by the density rules
(``degenerate``), empty or under 20 characters (``empty``), a Latin letter
(``latin``, the eval's row flag), our template's section label or the instruction
itself (``echo``). A prompt skipped for length at generation time counts as
``skipped_prompt_length``. Answers that open with a stock preamble are kept and
counted (``has_preamble``). Dropped prompts are **not** regenerated.

Writes ``teacher_answers_v1.jsonl`` (``id, corpus, split, answer, n_tokens,
answer_chars``; ``answer`` = the teacher's text, stripped) and ``manifest.json``:
model + snapshot, dtype / quantization, sampling, system prompt, cap, per corpus ×
split the prompts, every drop reason, the kept count and share, the preamble share,
teacher vs reference characters for the kept records, and the sha256 of the answers
file (the configs and the report cite it).

    .venv/bin/python scripts/distill/filter_teacher_answers.py --dir outputs/data_cache/distill/<slug>_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arabic_eval.distill.postprocess import DROP_REASONS, drop_reason, has_preamble  # noqa: E402
from arabic_eval.distill.teacher import file_sha256, iter_jsonl, read_jsonl, write_jsonl  # noqa: E402
from arabic_eval.tasks.freeform.metrics import detect_loop  # noqa: E402

log = logging.getLogger("filter_teacher_answers")
ANSWERS_FILE = "teacher_answers_v1.jsonl"


def _stats(xs: Sequence[int]) -> Dict[str, Optional[float]]:
    return {"mean": round(statistics.fmean(xs), 1) if xs else None, "median": statistics.median(xs) if xs else None}


def filter_answers(prompts: Sequence[Dict[str, Any]], raw: Dict[str, Dict[str, Any]]):
    """``(kept rows, per-(corpus, split) report)``."""
    kept: List[Dict[str, Any]] = []
    report: Dict[str, Dict[str, Any]] = {}
    loops: Counter = Counter()
    for p in prompts:
        key = f"{p['corpus']}/{p['split']}"
        rep = report.setdefault(key, {"prompts": 0, "answered": 0, "skipped_prompt_length": 0,
                                      **{f"drop_{r}": 0 for r in DROP_REASONS}, "kept": 0, "preamble_answered": 0,
                                      "preamble_kept": 0, "_teacher_chars": [], "_ref_chars": [], "_raw_chars": []})
        rep["prompts"] += 1
        a = raw.get(p["id"])
        if a is None:
            rep["skipped_prompt_length"] += 1
            continue
        rep["answered"] += 1
        text = (a.get("text") or "").strip()
        rep["_raw_chars"].append(len(text))
        pre = has_preamble(text)
        rep["preamble_answered"] += pre
        reason = drop_reason(a.get("text"), a.get("finish_reason") or "", p["instruction"])
        if reason is not None:
            rep[f"drop_{reason}"] += 1
            if reason == "loop":
                lp = detect_loop(text)
                loops[f"{lp.rule}:{lp.period}"] += 1
            continue
        rep["kept"] += 1
        rep["preamble_kept"] += pre
        rep["_teacher_chars"].append(len(text))
        rep["_ref_chars"].append(p["ref_chars"])
        kept.append({"id": p["id"], "corpus": p["corpus"], "split": p["split"], "answer": text,
                     "n_tokens": a.get("n_tokens"), "answer_chars": len(text)})
    for rep in report.values():
        rep["kept_share"] = round(rep["kept"] / rep["prompts"], 4) if rep["prompts"] else None
        rep["preamble_share_answered"] = round(rep["preamble_answered"] / rep["answered"], 4) if rep["answered"] else None
        rep["preamble_share_kept"] = round(rep["preamble_kept"] / rep["kept"], 4) if rep["kept"] else None
        rep["teacher_chars_kept"] = _stats(rep.pop("_teacher_chars"))
        rep["reference_chars_kept"] = _stats(rep.pop("_ref_chars"))
        rep["teacher_chars_answered"] = _stats(rep.pop("_raw_chars"))
    return kept, report, dict(loops)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="outputs/data_cache/distill/<slug>_v1 (prompts.jsonl, teacher_raw.jsonl, generation_meta.json)")
    ap.add_argument("--raw", default="teacher_raw.jsonl")
    ap.add_argument("--meta", default="generation_meta.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    d = Path(args.dir)
    prompts = read_jsonl(d / "prompts.jsonl")
    raw = {r["id"]: r for r in iter_jsonl(d / args.raw)}
    meta = json.loads((d / args.meta).read_text(encoding="utf-8"))
    pman = json.loads((d / "prompts_manifest.json").read_text(encoding="utf-8"))
    kept, report, loops = filter_answers(prompts, raw)
    out = d / ANSWERS_FILE
    write_jsonl(out, kept)
    totals = {k: sum(r[k] for r in report.values())
              for k in ("prompts", "answered", "skipped_prompt_length", *(f"drop_{x}" for x in DROP_REASONS), "kept",
                        "preamble_answered", "preamble_kept")}
    manifest = {
        "answers_file": str(out.relative_to(REPO_ROOT)) if out.is_absolute() else str(out),
        "answers_sha256": file_sha256(out), "records": len(kept),
        "teacher": {k: meta.get(k) for k in ("slug", "repo", "loaded_from", "revision", "dtype", "quantization",
                                              "max_model_len", "vllm_version", "sampling", "chat_template_kwargs",
                                              "system_role", "system_prompt", "system_prompt_sha256", "eos_token")},
        "cap": (meta.get("sampling") or {}).get("max_tokens"),
        "prompts_file": pman.get("prompts_file"), "prompts_sha256": pman.get("prompts_sha256"),
        "exclusions_digest": pman.get("exclusions_digest"), "template_version": pman.get("template_version"),
        "drop_rules": list(DROP_REASONS), "totals": totals,
        "kept_share": round(totals["kept"] / totals["prompts"], 4) if totals["prompts"] else None,
        "preamble_share_answered": round(totals["preamble_answered"] / totals["answered"], 4) if totals["answered"] else None,
        "loop_drops_by_rule_period": loops,
        "per_corpus_split": report,
    }
    with open(d / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("teacher",)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
