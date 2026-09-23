#!/usr/bin/env python
"""Write a teacher's answers as a free-form **pseudo-cell** — a folder the judge, the
paired comparison and the console's Free-form tab read like any trained cell.

A pseudo-cell holds no model: ``eval_rows/freeform_cidar.parquet`` with the columns of a
real dump (schema 3 of ``FreeformCidarTask``; ``prompt_text`` is the exact rendered chat
the teacher answered, ``reference_roundtrip_chrf`` is None — there is no student
tokenizer), ``all_metrics.json`` whose ``downstream.freeform_cidar`` is
``metrics.summarize`` of those rows plus ``"kind": "teacher_pseudo_cell"``, and a
``config.json`` with ``tokenizer.type = "teacher:<slug>"``. The per-row fields come
from ``arabic_eval.distill.postprocess.eval_row`` — the eval's own text rules with no
stop markers (the teacher never saw our template) — then chrF and the in-house
BERTScore against the reference.

Two kinds (``--kind``):
  * ``teacher_bakeoff`` — 250 cidar-dev prompts, under
    ``outputs/experiments/teacher_bakeoff/<slug>/`` (P1.3);
  * ``teacher_ceiling`` — the winner on the 250 held-out test prompts, a cell of the
    main experiment folder (P1.4). Its answers stay in this dump; nothing under
    ``outputs/data_cache/`` is written.

    .venv/bin/python scripts/distill/build_pseudo_cell.py --kind teacher_bakeoff \\
        --prompts outputs/data_cache/distill/bakeoff/prompts_dev.jsonl \\
        --raw outputs/data_cache/distill/bakeoff/jais2_8b/teacher_raw.jsonl \\
        --meta outputs/data_cache/distill/bakeoff/jais2_8b/generation_meta.json \\
        --cell outputs/experiments/teacher_bakeoff/jais2_8b
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

from arabic_eval.distill.postprocess import MAX_OUTPUT_CHARS, eval_row  # noqa: E402
from arabic_eval.distill.teacher import file_sha256, read_jsonl  # noqa: E402
from arabic_eval.tasks.freeform import metrics as M  # noqa: E402
from arabic_eval.tasks.freeform.cidar import (  # noqa: E402
    DEFAULT_BERTSCORE_BATCH_SIZE, DEFAULT_BERTSCORE_LAYER, DEFAULT_BERTSCORE_MODEL, ROW_FIELDS, TASK_NAME,
)
from arabic_eval.utils.io import save_json, write_report_table  # noqa: E402

log = logging.getLogger("build_pseudo_cell")
KINDS = ("teacher_bakeoff", "teacher_ceiling")


def build_rows(prompts: Sequence[Dict[str, Any]], raw: Dict[str, Dict[str, Any]], gen_wall: float,
               max_output_chars: int = MAX_OUTPUT_CHARS) -> List[Dict[str, Any]]:
    """One dump row per prompt, in prompt order; a prompt without an answer is an error."""
    missing = [p["id"] for p in prompts if p["id"] not in raw]
    if missing:
        raise SystemExit(f"{len(missing)} prompts have no teacher answer (first: {missing[:5]}) — "
                         f"skipped for length or an incomplete run")
    per_row = gen_wall / len(prompts) if prompts else 0.0
    rows = []
    for p in prompts:
        a = raw[p["id"]]
        ref = p.get("reference") or ""
        ev = eval_row(a.get("text"), a.get("finish_reason") or "", max_output_chars)
        rows.append({
            "id": p["id"], "stratum": p.get("stratum") or "", "instruction": p.get("instruction") or p.get("prompt"),
            "context": p.get("context") or "", "prompt_text": a.get("prompt_text") or "", "reference": ref,
            **{k: ev[k] for k in ("generation", "generation_raw")},
            "prompt_tokens": a.get("prompt_tokens"), "gen_tokens": a.get("n_tokens"),
            "gen_chars": len(ev["generation"]), "ref_chars": len(ref),
            **{k: ev[k] for k in ("stop_reason", "hit_cap", "hit_loop", "loop_rule", "loop_period", "char_truncated",
                                  "empty", "degenerate", "latin", "arabic_letter_ratio")},
            "chrf": round(M.chrf_sentence(ev["generation"], ref), 4),
            "bertscore_p": None, "bertscore_r": None, "bertscore_f1": None,
            "reference_roundtrip_chrf": None, "gen_time_sec": round(per_row, 4),
        })
    return rows


def add_bertscore(rows: List[Dict[str, Any]], device: str) -> Optional[Dict[str, Any]]:
    from arabic_eval.tasks.freeform.bertscore import BertScorer
    scorer = BertScorer(DEFAULT_BERTSCORE_MODEL, DEFAULT_BERTSCORE_LAYER, device=device, batch_size=DEFAULT_BERTSCORE_BATCH_SIZE)
    P, R, F = scorer.score([r["generation"] for r in rows], [r["reference"] for r in rows])
    for r, p, rr, f in zip(rows, P, R, F):
        r["bertscore_p"], r["bertscore_r"], r["bertscore_f1"] = round(p, 4), round(rr, 4), round(f, 4)
    return {"model": DEFAULT_BERTSCORE_MODEL, "layer": DEFAULT_BERTSCORE_LAYER}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--prompts", required=True, help="JSONL: id, instruction (or prompt), context, reference, stratum")
    ap.add_argument("--raw", required=True, help="teacher_raw.jsonl of generate_teacher_answers.py")
    ap.add_argument("--meta", required=True, help="generation_meta.json of the same run")
    ap.add_argument("--cell", required=True, help="output cell directory")
    ap.add_argument("--bertscore-device", default="cuda")
    ap.add_argument("--no-bertscore", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    prompts = read_jsonl(args.prompts)
    raw = {r["id"]: r for r in read_jsonl(args.raw)}
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    cell = Path(args.cell)
    if (cell / "eval_rows" / f"{TASK_NAME}.parquet").exists():
        raise SystemExit(f"{cell} already holds a dump — move it to _superseded/ first; nothing is overwritten")
    gen_wall = float(meta.get("gen_wall_sec") or 0.0)
    rows = build_rows(prompts, raw, gen_wall)
    bs = None if args.no_bertscore else add_bertscore(rows, args.bertscore_device)

    slug = meta["slug"]
    prompts_sha = file_sha256(args.prompts)
    teacher = {k: meta.get(k) for k in ("slug", "repo", "loaded_from", "revision", "dtype", "quantization", "max_model_len",
                                         "vllm_version", "sampling", "chat_template_kwargs", "system_role",
                                         "system_prompt_sha256", "eos_token", "load_wall_sec", "gen_wall_sec",
                                         "gen_tokens", "tok_per_sec")}
    token_cap = (meta.get("sampling") or {}).get("max_tokens")
    rel_prompts = str(Path(args.prompts).resolve().relative_to(REPO_ROOT))
    write_report_table(cell / "eval_rows" / f"{TASK_NAME}.parquet", rows, ROW_FIELDS, metadata={
        "task": TASK_NAME, "kind": args.kind, "schema_version": 3,
        "heldout_path": rel_prompts, "heldout_sha256": prompts_sha,
        "teacher": teacher, "token_cap": token_cap, "max_output_chars": MAX_OUTPUT_CHARS,
        "decoding": {"sampling": "greedy", "stop_markers": [], "loop_stop": True, "max_output_chars": MAX_OUTPUT_CHARS,
                     "repetition_penalty": 1.0, "applied": "post hoc on the full answer (the teacher ran to EOS or the cap)"},
        "bertscore": bs, "prompt_template": "teacher_chat_template", "num_rows": len(rows),
    })
    summary = M.summarize(rows, gen_wall)
    summary.update({"chars_per_token": None, "token_cap": token_cap, "max_output_chars": MAX_OUTPUT_CHARS,
                    "status": "ok", "kind": "teacher_pseudo_cell", "pseudo_cell_kind": args.kind})
    save_json({"config": {"tokenizer": f"teacher:{slug}", "vocab_size": None, "model": meta.get("repo"),
                          "tasks": [TASK_NAME]},
               "kind": "teacher_pseudo_cell", "pseudo_cell_kind": args.kind, "teacher": teacher,
               "prompts": {"path": rel_prompts, "sha256": prompts_sha, "n": len(rows)},
               "downstream": {TASK_NAME: summary}}, cell / "all_metrics.json")
    save_json({"name": cell.name, "kind": args.kind,
               "description": f"teacher pseudo-cell ({args.kind}): {meta.get('repo')} on {rel_prompts}",
               "output_dir": str(cell), "tokenizer": {"type": f"teacher:{slug}", "vocab_size": None},
               "model": {"type": "teacher", "name_or_path": meta.get("repo"), "revision": meta.get("revision"),
                         "dtype": meta.get("dtype"), "quantization": meta.get("quantization")},
               "teacher": teacher}, cell / "config.json")
    print(json.dumps({k: summary[k] for k in ("num_samples", "chrf", "bertscore_f1", "loop_stop_rate", "eos_rate",
                                              "hit_cap_rate", "degenerate_rate", "latin_rate", "empty_rate",
                                              "char_truncated_rate", "mean_gen_chars", "mean_ref_chars")},
                     ensure_ascii=False, indent=2))
    print(f"wrote {cell}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
