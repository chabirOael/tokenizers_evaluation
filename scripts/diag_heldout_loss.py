#!/usr/bin/env python
"""Held-out loss diagnostic for one cell: raw-text LM loss + free-form answer NLL.

Two numbers an SFT run must not get wrong, measured on data no phase trained on:

  (a) **Pretraining-mix loss** — causal LM loss per token (and per character)
      on the last 120 documents of the pool's ``fineweb2_arb.parquet``,
      concatenated with EOS and chunked into 512-token blocks. A rise over the
      untouched base is damage to the language model (the 2e-4 native run of
      2026-09-18 went from 2.27 to 3.4 nats).
  (b) **Held-out CIDAR answers** — the 250 references of the free-form held-out
      set rendered with the *current* Phase 3 ``instruction`` template
      (``tokenize_record``: LCP answer-only masking, EOS appended when the
      tokenizer emits none — exactly the training path): NLL per answer token,
      mean / median P(EOS) at the reference's end and the share of references
      where EOS is the top-1 prediction there. SFT has to lower the NLL below
      the base's and raise P(EOS); if it does not, it did not learn the task.

    # the SFT checkpoint of a cell (reads config.json, rebuilds the tokenizer)
    .venv/bin/python scripts/diag_heldout_loss.py --cell outputs/experiments/<exp>/<cell>
    # the untouched base model with the cell's tokenizer / model name
    .venv/bin/python scripts/diag_heldout_loss.py --cell <cell> --base
    # the v1 flat template (``السؤال: … الإجابة: …``), to reproduce pre-2026-09-18 numbers
    .venv/bin/python scripts/diag_heldout_loss.py --cell <cell> --base --v1-template

The model is loaded with ``AutoModelForCausalLM.from_pretrained(<checkpoint>)``,
which is right for **standard-embedding** tokenizers only (native_*, bpe,
wordpiece, morpho_bpe, araroopat): the CharCNN / char-JABER / Charformer cells
replace the embedding and output head with modules a vanilla load cannot
rebuild, and the script refuses them. The pool directory comes from the
cell's ``data/pretraining_mix/packed_manifest.json`` (a Phase-3-only cell has
none: ``--pool`` or the default pool). Prints a table and writes the same
numbers as JSON next to it (``--out``; default ``<cell>/diag_heldout_loss[_base][_v1].json``).
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arabic_eval.config import ModelConfig, TokenizerConfig  # noqa: E402
from arabic_eval.data.answer_only_masking import compute_answer_only_labels  # noqa: E402
from arabic_eval.data.finetune_corpora import (  # noqa: E402
    TEMPLATE_VERSION, QARecord, _format_qa_full, _format_qa_prompt, tokenize_record,
)
from arabic_eval.data.freeform_heldout import load_freeform_heldout  # noqa: E402
from arabic_eval.registry import tokenizer_registry  # noqa: E402
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType  # noqa: E402
from arabic_eval.utils.io import load_json  # noqa: E402

import arabic_eval.tokenizers  # noqa: E402,F401 — populate the registry

log = logging.getLogger("diag_heldout_loss")

DEFAULT_POOL = "outputs/data_cache/pretraining_mix/9b02b0462005e107"
DEFAULT_HELDOUT = "configs/contamination/freeform_cidar_heldout_v1.jsonl"
N_DOCS = 120
BLOCK = 512


# --------------------------------------------------------------------------
# pure helpers (tested)
# --------------------------------------------------------------------------

def strip_specials(ids: Sequence[int], bos: Optional[int], eos: Optional[int]) -> List[int]:
    """Drop a leading BOS and a trailing EOS the tokenizer added on its own."""
    out = list(ids)
    if bos is not None and out and out[0] == bos:
        out = out[1:]
    if eos is not None and out and out[-1] == eos:
        out = out[:-1]
    return out


def blocks_from(doc_ids: Sequence[Sequence[int]], eos: int, block: int = BLOCK) -> List[List[int]]:
    """Concatenate documents with an EOS separator and chunk into full blocks
    (the tail shorter than ``block`` is dropped, as the packer does)."""
    flat: List[int] = []
    for ids in doc_ids:
        flat.extend(ids)
        flat.append(eos)
    return [flat[i:i + block] for i in range(0, len(flat) - block + 1, block)]


def v1_format_prompt(rec: QARecord) -> str:
    """The flat pre-2026-09-18 ``instruction`` prompt (TEMPLATE_VERSION 1)."""
    head = f"السياق: {rec.context}\n" if rec.context else ""
    return f"{head}السؤال: {rec.question}\nالإجابة:"


def v1_format_full(rec: QARecord) -> str:
    return f"{v1_format_prompt(rec)} {rec.answer}"


def heldout_records(path: Path) -> List[QARecord]:
    return [
        QARecord(id=r["id"], question=r["prompt"], context=r.get("context") or "", answer=r["reference"],
                 source="cidar", prompt_template="instruction")
        for r in load_freeform_heldout(path)
    ]


def answer_encodings(recs: Sequence[QARecord], tokenizer: BaseTokenizer, max_length: int,
                     v1: bool = False) -> Tuple[List[Tuple[List[int], List[int]]], int]:
    """``(input_ids, labels)`` per reference under answer-only masking, the EOS
    appended when the tokenizer emits none (``tokenize_record``'s rule); the
    number of references dropped because truncation ate the answer."""
    encs: List[Tuple[List[int], List[int]]] = []
    dropped = 0
    eos = (tokenizer.special_tokens or {}).get("eos_token")
    for rec in recs:
        if v1:
            full_enc = tokenizer.encode(v1_format_full(rec), max_length=max_length, truncation=True)
            ids = list(full_enc.input_ids)
            truncated = len(ids) >= max_length
            if eos is not None and full_enc.char_ids is None and not truncated and (not ids or ids[-1] != eos):
                ids.append(int(eos))
            prompt_ids = tokenizer.encode(v1_format_prompt(rec), max_length=max_length, truncation=True).input_ids
            labels = compute_answer_only_labels(prompt_ids, ids)
            entry = None if labels is None else {"input_ids": ids, "labels": labels}
        else:
            entry, _, _ = tokenize_record(rec, tokenizer, max_length, "answer_only")
        if entry is None:
            dropped += 1
            continue
        encs.append((list(entry["input_ids"]), list(entry["labels"])))
    return encs, dropped


def summarize_eos(p_eos: Sequence[float], ranks: Sequence[int]) -> Dict[str, Any]:
    return {
        "p_eos_at_end_mean": round(statistics.fmean(p_eos), 4) if p_eos else None,
        "p_eos_at_end_median": round(statistics.median(p_eos), 4) if p_eos else None,
        "eos_rank1_share": round(sum(1 for r in ranks if r == 1) / len(ranks), 4) if ranks else None,
        "n": len(p_eos),
    }


# --------------------------------------------------------------------------
# model-side measurements
# --------------------------------------------------------------------------

def lm_loss(model, blocks: Sequence[Sequence[int]], device, batch_size: int = 8) -> Dict[str, Any]:
    import torch
    tot = n = 0.0
    with torch.inference_mode():
        for i in range(0, len(blocks), batch_size):
            x = torch.tensor(blocks[i:i + batch_size], device=device)
            out = model(input_ids=x, labels=x)
            ntok = x.numel() - x.shape[0]           # one shifted-off position per row
            tot += float(out.loss.item()) * ntok
            n += ntok
    return {"loss_per_token": round(tot / n, 4) if n else None, "tokens_scored": int(n), "blocks": len(blocks)}


def answer_stats(model, encs: Sequence[Tuple[List[int], List[int]]], pad_id: int, device,
                 batch_size: int = 8) -> Dict[str, Any]:
    """Answer-only NLL per token (EOS included as in training) and the EOS
    probability / rank at the position that predicts the final EOS."""
    import torch
    tot = n = 0.0
    p_eos: List[float] = []
    ranks: List[int] = []
    with torch.inference_mode():
        for i in range(0, len(encs), batch_size):
            chunk = encs[i:i + batch_size]
            w = max(len(e[0]) for e in chunk)
            x = torch.full((len(chunk), w), pad_id, dtype=torch.long, device=device)
            y = torch.full((len(chunk), w), -100, dtype=torch.long, device=device)
            m = torch.zeros((len(chunk), w), dtype=torch.long, device=device)
            for r, (ids, lab) in enumerate(chunk):
                x[r, :len(ids)] = torch.tensor(ids, device=device)
                y[r, :len(ids)] = torch.tensor(lab, device=device)
                m[r, :len(ids)] = 1
            logits = model(input_ids=x, attention_mask=m).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1)
            tgt = y[:, 1:]
            mask = tgt != -100
            nll = -lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            tot += float((nll * mask).sum().item())
            n += float(mask.sum().item())
            for r, (ids, lab) in enumerate(chunk):
                last = len(ids) - 2                      # predicts the final token (the EOS)
                p = lp[r, last].exp()
                p_end = float(p[ids[-1]].item())
                p_eos.append(p_end)
                ranks.append(int((p > p_end).sum().item()) + 1)
    return {"nll_per_answer_token": round(tot / n, 4) if n else None, "answer_tokens": int(n),
            **summarize_eos(p_eos, ranks)}


# --------------------------------------------------------------------------
# cell plumbing
# --------------------------------------------------------------------------

def read_cell(cell_dir: Path) -> Tuple[TokenizerConfig, ModelConfig, Dict[str, Any]]:
    raw = load_json(cell_dir / "config.json")
    return TokenizerConfig(**(raw.get("tokenizer") or {})), ModelConfig(**(raw.get("model") or {})), raw


def build_tokenizer(tok_cfg: TokenizerConfig) -> BaseTokenizer:
    tok_cls = tokenizer_registry.get(tok_cfg.type)
    tokenizer = tok_cls(**(tok_cfg.params or {}))
    path = tok_cfg.load_path or tok_cfg.save_path
    if path:
        full = REPO_ROOT / path
        if not full.exists():
            raise SystemExit(f"tokenizer directory {path} is gone — it is needed to rebuild the tokenizer")
        tokenizer.load(str(full))
    if tokenizer.embedding_type != EmbeddingType.STANDARD:
        raise SystemExit(
            f"tokenizer {tok_cfg.type} has embedding_type={tokenizer.embedding_type!r}: its cells replace the "
            f"embedding / output head with custom modules that AutoModelForCausalLM.from_pretrained cannot "
            f"rebuild (see scripts/dump_eval_rows.py for the adapted-then-loaded route). Standard only."
        )
    return tokenizer


def pool_dir_for(cell_dir: Path, override: Optional[str]) -> Path:
    if override:
        return REPO_ROOT / override
    manifest = cell_dir / "data" / "pretraining_mix" / "packed_manifest.json"
    if manifest.exists():
        return REPO_ROOT / load_json(manifest)["pool_dir"]
    log.info("no packed_manifest.json in %s (Phase-3-only cell?) — using the default pool %s", cell_dir, DEFAULT_POOL)
    return REPO_ROOT / DEFAULT_POOL


def checkpoint_for(cell_dir: Path, base: bool, override: Optional[str], model_cfg: ModelConfig) -> str:
    if override:
        return override
    if base:
        return model_cfg.name_or_path
    ckpt = cell_dir / "training" / "sft"
    if not (ckpt / "model.safetensors").exists():
        raise SystemExit(f"no SFT checkpoint at {ckpt} (use --base for the untouched model or --checkpoint <dir>)")
    return str(ckpt)


def run(cell_dir: Path, *, base: bool, checkpoint: Optional[str], pool: Optional[str], heldout: str,
        v1_template: bool, n_docs: int, block: int, max_length: int, device: str, dtype: str) -> Dict[str, Any]:
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM

    tok_cfg, model_cfg, _raw = read_cell(cell_dir)
    tokenizer = build_tokenizer(tok_cfg)
    specials = tokenizer.special_tokens or {}
    eos = specials.get("eos_token")
    if eos is None:
        raise SystemExit("the tokenizer has no eos_token; the diagnostic needs one for the EOS statistics")
    bos = specials.get("bos_token")
    pad = int(specials.get("pad_token", eos))
    ckpt = checkpoint_for(cell_dir, base, checkpoint, model_cfg)

    # (a) raw text
    pool_path = pool_dir_for(cell_dir, pool) / "fineweb2_arb.parquet"
    docs = pq.read_table(pool_path, columns=["text"]).column("text").to_pylist()[-n_docs:]
    t0 = time.perf_counter()
    doc_ids = [strip_specials(tokenizer.encode(d).input_ids, bos, eos) for d in docs]
    chars = sum(len(d) for d in docs)
    toks = sum(len(x) for x in doc_ids)
    blocks = blocks_from(doc_ids, int(eos), block)
    log.info("pool: %d docs, %d chars, %d tokens (%.2f chars/token), %d blocks of %d — encoded in %.0fs",
             len(docs), chars, toks, chars / max(toks, 1), len(blocks), block, time.perf_counter() - t0)

    # (b) held-out answers
    recs = heldout_records(REPO_ROOT / heldout)
    encs, dropped = answer_encodings(recs, tokenizer, max_length, v1=v1_template)
    log.info("held-out: %d references, %d encoded, %d dropped (truncation), template %s",
             len(recs), len(encs), dropped, "v1" if v1_template else f"v{TEMPLATE_VERSION}")

    log.info("loading %s", ckpt)
    t0 = time.perf_counter()
    from arabic_eval.models.llama_adapter import configure_sdpa_backends
    configure_sdpa_backends()          # the cuDNN SDPA per-shape compile would bill every new batch width
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=getattr(torch, dtype)).to(device).eval()
    log.info("loaded in %.0fs", time.perf_counter() - t0)
    mix = lm_loss(model, blocks, device)
    cpt = chars / max(toks, 1)
    mix["chars_per_token"] = round(cpt, 4)
    mix["loss_per_char"] = round(mix["loss_per_token"] / cpt, 4) if mix["loss_per_token"] is not None else None
    ans = answer_stats(model, encs, pad, device)
    ans["references_dropped_truncation"] = dropped

    return {
        "cell": str(cell_dir), "checkpoint": ckpt, "base": base, "tokenizer": tok_cfg.type,
        "template_version": 1 if v1_template else TEMPLATE_VERSION,
        "prompt_example": (v1_format_prompt if v1_template else _format_qa_prompt)(recs[0]) if recs else None,
        "pool": str(pool_path), "n_docs": len(docs), "block_size": block, "max_length": max_length,
        "heldout": heldout, "dtype": dtype,
        "mix": mix, "heldout_answers": ans,
    }


def print_table(res: Dict[str, Any]) -> None:
    m, a = res["mix"], res["heldout_answers"]
    rows = [
        ("checkpoint", res["checkpoint"]),
        ("tokenizer", res["tokenizer"]),
        ("template", f"v{res['template_version']}"),
        ("mix loss / token", m["loss_per_token"]),
        ("mix loss / char", m["loss_per_char"]),
        ("mix chars / token", m["chars_per_token"]),
        ("mix tokens scored", m["tokens_scored"]),
        ("answer NLL / token", a["nll_per_answer_token"]),
        ("answer tokens", a["answer_tokens"]),
        ("P(EOS @ end) mean", a["p_eos_at_end_mean"]),
        ("P(EOS @ end) median", a["p_eos_at_end_median"]),
        ("EOS rank-1 share", a["eos_rank1_share"]),
        ("references (dropped)", f"{a['n']} ({a['references_dropped_truncation']})"),
    ]
    w = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"{k:<{w}}  {v}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True, help="experiment cell directory (has config.json)")
    ap.add_argument("--base", action="store_true", help="measure the untouched model.name_or_path instead of training/sft")
    ap.add_argument("--checkpoint", help="an explicit checkpoint directory (e.g. <cell>/training/warmup)")
    ap.add_argument("--pool", help="pool directory holding fineweb2_arb.parquet (default: the cell's packed manifest, else the default pool)")
    ap.add_argument("--heldout", default=DEFAULT_HELDOUT)
    ap.add_argument("--v1-template", action="store_true", help="render the references with the flat v1 template")
    ap.add_argument("--n-docs", type=int, default=N_DOCS)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--max-length", type=int, default=2048, help="truncation length for the held-out references")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", help="JSON output path (default <cell>/diag_heldout_loss[_base][_v1].json)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cell_dir = Path(args.cell)
    res = run(cell_dir, base=args.base, checkpoint=args.checkpoint, pool=args.pool, heldout=args.heldout,
              v1_template=args.v1_template, n_docs=args.n_docs, block=args.block, max_length=args.max_length,
              device=args.device, dtype=args.dtype)
    print_table(res)
    out = Path(args.out) if args.out else cell_dir / (
        "diag_heldout_loss" + ("_base" if args.base else "") + ("_v1" if args.v1_template else "") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
