#!/usr/bin/env python
"""Uncertainty diagnostic for one cell: next-token entropy and top-1 margin on the
held-out references and along the cell's own generations, and at loop onset.

The hypothesis it tests (docs/report.md §3.7 *Verification*): at equal held-out
likelihood of the references, the from-scratch vocabularies (rows learned in
131 M tokens that, under tied embeddings, are also the output head) decode from
flatter next-token distributions than the native model, and under greedy
decoding a flat distribution falls into the copy attractor (Ivgi et al. 2024).
Two teacher-forced passes, per answer token: the entropy ``H = −Σ p log p``
(nats, over the full vocabulary), the top-1 probability ``p1``, the margin
``p1 − p2`` and ``−log p(actual token)``.

  (a) **References** — the 250 held-out CIDAR references under the current
      template, encoded exactly as ``diag_heldout_loss.answer_encodings`` does
      (prompt + answer + EOS, answer tokens = the LCP-masked span, EOS included),
      scored in the same order and batches; ``Σ −log p ÷ reference chars as
      encoded`` must reproduce the cell's ``nll_per_answer_char`` (read from the
      matching ``diag_heldout_rawtext_v1*.json`` of this experiment folder, or
      measured in-process with the existing helper when none exists) to three
      decimals — the script stops otherwise.
  (b) **Generations** — every row of the cell's free-form dump:
      ``prompt_text`` encoded as the generation did (512-token truncation, a
      trailing EOS stripped) followed by ``generation_raw`` re-encoded with the
      cell's tokenizer (its own BOS / EOS stripped); answer tokens = the span
      after the prompt. Re-encoding decoded text is not the token sequence the
      model emitted (AraRooPat's word round trip is exact on 99.3 % of words),
      so the share of rows with ``decode(encode(generation_raw)) ==
      generation_raw`` is reported.

Per cell: on the references ``entropy_per_char`` (Σ H ÷ reference chars as
encoded), ``entropy_per_token``, mean ``p1``, mean margin, the share of
positions with ``p1 < 0.5`` and ``nll_per_answer_char``; on the generations the
same five over the generated text (chars = ``generation_raw`` as encoded); the
**loop contrast** — for loop-stopped rows (``stop_reason == "loop"``,
``loop_rule == "period"``) the onset token is the first answer token that emits
a character of the first copy of the repeated unit (the unit is the last
``loop_period`` words of ``generation``, which the loop stop cut right after
that first copy), the ``--window`` tokens before it are averaged, the control
is the same absolute token window in the EOS-terminated rows that have at least
``onset`` answer tokens (averaged per looped row, then over looped rows), and
inside the loop (onset to the end of ``generation_raw``, at most 64 tokens) the
mean ``p1`` — self-reinforcement shows as ``p1`` rising toward 1; and the mean
``H`` by answer-position decile (per-row decile means, averaged over rows with
at least 10 answer tokens).

Per-token entropy and ``p1`` are comparable *within* a cell only (a ``[PAT_*]``
after a ``[ROOT_*]`` is predictable in a way no BPE piece is); across
vocabularies read the per-character numbers.

    .venv/bin/python scripts/diag_uncertainty.py --cell outputs/experiments/<exp>/<cell>
    # eval-only cells: the model is model.name_or_path
    .venv/bin/python scripts/diag_uncertainty.py --cell <exp>/native_qwen3_sft_m2_c2400 --base

Standard-embedding tokenizers only (``diag_heldout_loss.build_tokenizer``
refuses the others); bf16, ``model.eval()``, no grad. Writes
``<cell>/diag_uncertainty_v1[_base].json`` (``--out``).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import diag_heldout_loss as DH  # noqa: E402 — read_cell / build_tokenizer / checkpoint_for / answer_encodings / heldout_records

from arabic_eval.data.finetune_corpora import TEMPLATE_VERSION  # noqa: E402
from arabic_eval.tasks.freeform.generation import strip_trailing_eos  # noqa: E402
from arabic_eval.tokenizers.base import BaseTokenizer  # noqa: E402

log = logging.getLogger("diag_uncertainty")

WINDOW = 32                 # tokens before a loop onset
IN_LOOP_MAX = 64            # tokens after the onset averaged for the in-loop p1
MAX_PROMPT_TOKENS = 512     # the generation's prompt truncation (DecodingConfig.max_prompt_tokens)
CROSSCHECK_TOL = 5e-4       # nll_per_answer_char must agree to three decimals
DECILES = 10
_WORD_RE = re.compile(r"\S+")

Seq = Tuple[List[int], List[int]]          # (input_ids, labels with -100 outside the answer)
Stats = Dict[str, List[float]]             # per answer token: H, p1, margin, nll


# --------------------------------------------------------------------------
# pure helpers (tested)
# --------------------------------------------------------------------------

def token_stats(logits, targets, chunk: int = 4096) -> Tuple[Any, Any, Any, Any]:
    """Entropy (nats), top-1 probability, top-1 − top-2 margin and NLL of the
    target, per row of ``logits`` ``[N, V]``. Computed in float32 (bf16 /
    fp16 logits are upcast, as ``answer_stats`` does), ``chunk`` rows at a time."""
    import torch
    hs, p1s, mgs, nlls = [], [], [], []
    for s in range(0, logits.shape[0], chunk):
        x = logits[s:s + chunk]
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()
        lp = torch.log_softmax(x, -1)
        p = lp.exp()
        hs.append(-(p * lp).nan_to_num(0.0).sum(-1))
        top2 = p.topk(2, dim=-1).values
        p1s.append(top2[:, 0])
        mgs.append(top2[:, 0] - top2[:, 1])
        nlls.append(-lp.gather(-1, targets[s:s + chunk].unsqueeze(-1)).squeeze(-1))
        del lp, p
    cat = lambda xs: torch.cat(xs) if xs else torch.zeros(0)  # noqa: E731
    return cat(hs), cat(p1s), cat(mgs), cat(nlls)


def onset_char_offset(generation_raw: str, generation: str, period: int) -> Optional[int]:
    """Character offset in ``generation_raw`` where the first copy of the repeated
    unit starts. The loop stop cut ``generation`` right after that first copy
    (``text_stop`` then ``strip``), so the unit is the last ``period`` words of
    ``generation`` and its first copy is the tail of ``generation``. ``None``
    when ``generation`` is not a prefix of the (left-stripped) raw text or has
    fewer than ``period`` words."""
    stripped = generation_raw.lstrip()
    if not generation or not stripped.startswith(generation):
        return None
    spans = [m.start() for m in _WORD_RE.finditer(generation)]
    if period < 1 or len(spans) < period:
        return None
    lead = len(generation_raw) - len(stripped)
    return lead + spans[len(spans) - period]


def onset_token_index(ids: Sequence[int], onset_char: int, decode: Callable[[List[int]], str]) -> Optional[int]:
    """The first token whose decoded prefix ``decode(ids[:k + 1])`` is longer than
    ``onset_char`` — the token that emits the first character of the unit.
    Binary search (prefix lengths grow with k up to decoder artefacts), then a
    step back while the previous prefix already reaches past the offset."""
    n = len(ids)
    if n == 0:
        return None
    reach = lambda k: len(decode(list(ids[:k + 1]))) > onset_char  # noqa: E731
    if not reach(n - 1):
        return None
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if reach(mid):
            hi = mid
        else:
            lo = mid + 1
    while lo > 0 and reach(lo - 1):
        lo -= 1
    return lo


def _mean(xs: Sequence[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def window_means(stats: Stats, start: int, end: int) -> Optional[Dict[str, float]]:
    """Mean H / p1 / margin over answer positions ``[start, end)`` (``None`` when empty)."""
    if end <= start:
        return None
    return {k: statistics.fmean(stats[k][start:end]) for k in ("H", "p1", "margin")}


def position_matched_control(onset: int, window: int, eos_rows: Sequence[Stats]) -> Tuple[Optional[Dict[str, float]], int]:
    """The looped row's window ``[onset − window, onset)`` read in every
    EOS-terminated row with at least ``onset`` answer tokens, averaged over
    those rows; returns ``(means, number of control rows)``."""
    start = max(0, onset - window)
    picked = [window_means(r, start, onset) for r in eos_rows if len(r["H"]) >= onset]
    picked = [p for p in picked if p is not None]
    if not picked:
        return None, 0
    return {k: statistics.fmean(p[k] for p in picked) for k in ("H", "p1", "margin")}, len(picked)


def decile_curve(rows: Sequence[Stats], key: str = "H", min_tokens: int = DECILES) -> List[Optional[float]]:
    """Mean ``key`` by answer-position decile: per row, the mean over the
    positions ``j`` with ``floor(10 j / n) == d``; then the mean over rows with
    at least ``min_tokens`` answer tokens."""
    per_d: List[List[float]] = [[] for _ in range(DECILES)]
    for r in rows:
        xs = r[key]
        n = len(xs)
        if n < min_tokens:
            continue
        buckets: List[List[float]] = [[] for _ in range(DECILES)]
        for j, v in enumerate(xs):
            buckets[min(DECILES - 1, DECILES * j // n)].append(v)
        for d in range(DECILES):
            if buckets[d]:
                per_d[d].append(statistics.fmean(buckets[d]))
    return [round(statistics.fmean(v), 4) if v else None for v in per_d]


def summarize_positions(rows: Sequence[Stats], chars: int) -> Dict[str, Any]:
    """The five pooled numbers over every answer token of ``rows`` (+ Σ NLL)."""
    H = [v for r in rows for v in r["H"]]
    p1 = [v for r in rows for v in r["p1"]]
    mg = [v for r in rows for v in r["margin"]]
    nll = [v for r in rows for v in r["nll"]]
    n = len(H)
    return {
        "tokens": n, "chars": chars,
        "entropy_total": round(sum(H), 4),
        "entropy_per_char": round(sum(H) / chars, 6) if chars else None,
        "entropy_per_token": round(sum(H) / n, 6) if n else None,
        "mean_p1": round(statistics.fmean(p1), 6) if n else None,
        "mean_margin": round(statistics.fmean(mg), 6) if n else None,
        "share_p1_lt_0_5": round(sum(1 for v in p1 if v < 0.5) / n, 6) if n else None,
        "nll_total": round(sum(nll), 6),
    }


def generation_sequence(prompt_text: str, generation_raw: str, tokenizer: BaseTokenizer,
                        max_prompt_tokens: int = MAX_PROMPT_TOKENS) -> Tuple[Seq, List[int]]:
    """``(input_ids, labels)`` of prompt + re-encoded generation, and the
    generation ids alone. The prompt is encoded the way ``generate_freeform``
    encoded it (truncated, trailing EOS stripped); the generation's own BOS /
    EOS are stripped (``diag_heldout_loss.strip_specials``)."""
    specials = tokenizer.special_tokens or {}
    eos = specials.get("eos_token")
    bos = specials.get("bos_token")
    p_ids = list(tokenizer.encode(prompt_text, max_length=max_prompt_tokens, truncation=True).input_ids)
    p_ids = strip_trailing_eos(p_ids, None if eos is None else int(eos))
    g_ids = DH.strip_specials(tokenizer.encode(generation_raw).input_ids, bos, eos) if generation_raw else []
    return (p_ids + g_ids, [-100] * len(p_ids) + g_ids), g_ids


# --------------------------------------------------------------------------
# model side
# --------------------------------------------------------------------------

def score_sequences(model, seqs: Sequence[Seq], pad_id: int, device, batch_size: int = 8,
                    order: Optional[Sequence[int]] = None) -> List[Stats]:
    """Per answer token (``labels != -100``) of every sequence: H, p1, margin,
    NLL. Right-padded batches of ``batch_size`` in ``order`` (default: input
    order — the references keep ``answer_stats``'s batches so the NLL matches)."""
    import torch
    order = list(range(len(seqs))) if order is None else list(order)
    out: List[Optional[Stats]] = [None] * len(seqs)
    with torch.inference_mode():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            chunk = [seqs[i] for i in idx]
            w = max(len(c[0]) for c in chunk)
            x = torch.full((len(chunk), w), pad_id, dtype=torch.long, device=device)
            y = torch.full((len(chunk), w), -100, dtype=torch.long, device=device)
            m = torch.zeros((len(chunk), w), dtype=torch.long, device=device)
            for r, (ids, lab) in enumerate(chunk):
                x[r, :len(ids)] = torch.tensor(ids, device=device)
                y[r, :len(ids)] = torch.tensor(lab, device=device)
                m[r, :len(ids)] = 1
            logits = model(input_ids=x, attention_mask=m).logits
            tgt = y[:, 1:]
            mask = tgt != -100
            H, p1, mg, nll = token_stats(logits[:, :-1][mask], tgt[mask])
            del logits
            counts = mask.sum(1).tolist()
            vals = [t.double().cpu().tolist() for t in (H, p1, mg, nll)]
            off = 0
            for r, i in enumerate(idx):
                k = counts[r]
                out[i] = {"H": vals[0][off:off + k], "p1": vals[1][off:off + k],
                          "margin": vals[2][off:off + k], "nll": vals[3][off:off + k]}
                off += k
    return out  # type: ignore[return-value]


# --------------------------------------------------------------------------
# cross-check
# --------------------------------------------------------------------------

def _same_checkpoint(a: str, b: str) -> bool:
    if a == b:
        return True
    pa, pb = Path(a), Path(b)
    if not pa.is_absolute():
        pa = REPO_ROOT / pa
    if not pb.is_absolute():
        pb = REPO_ROOT / pb
    return pa.exists() and pb.exists() and pa.resolve() == pb.resolve()


def find_crosscheck(cell_dir: Path, ckpt: str, max_length: int) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """The held-out diagnostic JSON of the same checkpoint, template and
    truncation in this cell or a sibling cell (the eval-only reference cells
    re-use their source cell's checkpoint and have no diagnostic of their own)."""
    cands = sorted(cell_dir.glob("diag_heldout_rawtext_v1*.json")) + sorted(
        p for p in cell_dir.parent.glob("*/diag_heldout_rawtext_v1*.json") if p.parent != cell_dir)
    for p in cands:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        a = d.get("heldout_answers") or {}
        if (d.get("template_version") == TEMPLATE_VERSION and d.get("max_length") == max_length
                and a.get("nll_per_answer_char") is not None and _same_checkpoint(str(d.get("checkpoint")), ckpt)):
            return p, d
    return None


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def run(cell_dir: Path, *, base: bool, checkpoint: Optional[str], dump: Optional[str], heldout: str, window: int,
        max_length: int, device: str, dtype: str, batch_size: int) -> Dict[str, Any]:
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM

    tok_cfg, model_cfg, _raw = DH.read_cell(cell_dir)
    tokenizer = DH.build_tokenizer(tok_cfg)
    specials = tokenizer.special_tokens or {}
    eos = specials.get("eos_token")
    if eos is None:
        raise SystemExit("the tokenizer has no eos_token")
    pad = int(specials.get("pad_token", eos))
    ckpt = DH.checkpoint_for(cell_dir, base, checkpoint, model_cfg)

    # (a) references — the diagnostic's encodings, order and batches
    recs = DH.heldout_records(REPO_ROOT / heldout)
    kept = []
    encs, dropped = DH.answer_encodings(recs, tokenizer, max_length, kept=kept)
    ref_chars = sum(len(DH.encoded_text(tokenizer, r.answer)) for r in kept)
    ref_chars_raw = sum(len(r.answer) for r in kept)

    # (b) generations
    dump_path = Path(dump) if dump else cell_dir / "eval_rows" / "freeform_cidar.parquet"
    table = pq.read_table(dump_path)
    rows = table.to_pylist()
    gen_seqs: List[Seq] = []
    gen_ids: List[List[int]] = []
    roundtrip_exact = 0
    t0 = time.perf_counter()
    for r in rows:
        seq, g = generation_sequence(r["prompt_text"], r["generation_raw"] or "", tokenizer)
        gen_seqs.append(seq)
        gen_ids.append(g)
        if tokenizer.decode(g) == (r["generation_raw"] or ""):
            roundtrip_exact += 1
    gen_chars = sum(len(DH.encoded_text(tokenizer, r["generation_raw"] or "")) for r in rows)
    log.info("generations: %d rows re-encoded in %.0fs, decode(encode(raw)) exact on %d", len(rows),
             time.perf_counter() - t0, roundtrip_exact)

    log.info("loading %s", ckpt)
    from arabic_eval.models.llama_adapter import configure_sdpa_backends
    configure_sdpa_backends()
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=getattr(torch, dtype)).to(device).eval()

    t0 = time.perf_counter()
    ref_stats = score_sequences(model, encs, pad, device, batch_size=8)       # answer_stats' batching
    t_ref = time.perf_counter() - t0
    gen_order = sorted(range(len(gen_seqs)), key=lambda i: -len(gen_seqs[i][0]))
    t0 = time.perf_counter()
    gen_stats = score_sequences(model, gen_seqs, pad, device, batch_size=batch_size, order=gen_order)
    t_gen = time.perf_counter() - t0
    log.info("scored %d references in %.0fs and %d generations in %.0fs", len(encs), t_ref, len(rows), t_gen)

    refs = summarize_positions(ref_stats, ref_chars)
    nll_total = refs["nll_total"]
    refs["nll_per_answer_char"] = round(nll_total / ref_chars, 6) if ref_chars else None
    refs["nll_per_answer_char_raw_chars"] = round(nll_total / ref_chars_raw, 6) if ref_chars_raw else None
    refs["reference_chars"] = ref_chars
    refs["reference_chars_raw"] = ref_chars_raw
    refs["references_dropped_truncation"] = dropped
    refs["decile_H"] = decile_curve(ref_stats)

    # cross-check against the held-out diagnostic of the same checkpoint
    found = find_crosscheck(cell_dir, ckpt, max_length)
    if found is not None:
        cc_path, cc = found
        cc_value = float(cc["heldout_answers"]["nll_per_answer_char"])
        cc_source = str(cc_path.relative_to(REPO_ROOT)) if cc_path.is_absolute() else str(cc_path)
    else:
        cc_value = DH.answer_stats(model, encs, pad, device)["answer_nll_total"] / ref_chars
        cc_source = "measured in-process with diag_heldout_loss.answer_stats"
    diff = abs(refs["nll_per_answer_char"] - cc_value)
    refs["crosscheck"] = {"source": cc_source, "nll_per_answer_char": round(cc_value, 6), "abs_diff": round(diff, 6)}
    print(f"cross-check: nll_per_answer_char {refs['nll_per_answer_char']:.4f} (this pass) vs "
          f"{cc_value:.4f} ({cc_source}); |Δ| = {diff:.2e}")
    if diff > CROSSCHECK_TOL:
        raise SystemExit(f"cross-check FAILED: the reference pass gives {refs['nll_per_answer_char']:.6f} nats/char, "
                         f"{cc_source} gives {cc_value:.6f} (|Δ| {diff:.2e} > {CROSSCHECK_TOL})")

    live = [s for s, r in zip(gen_stats, rows) if s["H"]]
    gens = summarize_positions(live, gen_chars)
    gens.pop("nll_total")
    gens["rows"] = len(rows)
    gens["rows_scored"] = len(live)
    gens["roundtrip_exact_rows"] = roundtrip_exact
    gens["roundtrip_exact_share"] = round(roundtrip_exact / len(rows), 4) if rows else None
    gens["decile_H"] = decile_curve(gen_stats)

    # loop contrast
    eos_rows = [s for s, r in zip(gen_stats, rows) if r["stop_reason"] == "eos" and s["H"]]
    decode = lambda ids: tokenizer.decode(ids)  # noqa: E731
    per_loop: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}
    for r, s, g in zip(rows, gen_stats, gen_ids):
        if r["stop_reason"] != "loop" or r.get("loop_rule") != "period":
            continue
        if r.get("char_truncated"):
            skipped["char_truncated"] = skipped.get("char_truncated", 0) + 1
            continue
        oc = onset_char_offset(r["generation_raw"], r["generation"], int(r["loop_period"]))
        k = onset_token_index(g, oc, decode) if oc is not None else None
        if k is None:
            skipped["no_onset"] = skipped.get("no_onset", 0) + 1
            continue
        if k == 0:
            skipped["onset_at_first_token"] = skipped.get("onset_at_first_token", 0) + 1
            continue
        start = max(0, k - window)
        before = window_means(s, start, k)
        control, n_ctrl = position_matched_control(k, window, eos_rows)
        in_loop = window_means(s, k, min(len(s["H"]), k + IN_LOOP_MAX))
        per_loop.append({"id": r["id"], "loop_period": r["loop_period"], "onset_char": oc, "onset_token": k,
                         "answer_tokens": len(s["H"]), "window": k - start, "before": before,
                         "control": control, "control_rows": n_ctrl, "in_loop": in_loop})
    paired = [p for p in per_loop if p["control"] is not None]
    agg = lambda key, part, xs: round(statistics.fmean(p[part][key] for p in xs), 6) if xs else None  # noqa: E731
    loop_block = {
        "window": window, "in_loop_max": IN_LOOP_MAX,
        "looped_rows": sum(1 for r in rows if r["stop_reason"] == "loop" and r.get("loop_rule") == "period"),
        "rows_measured": len(per_loop), "rows_with_control": len(paired), "skipped": skipped,
        "eos_rows": len(eos_rows),
        "before": {k: agg(k, "before", paired) for k in ("H", "p1", "margin")},
        "control": {k: agg(k, "control", paired) for k in ("H", "p1", "margin")},
        "before_all_rows": {k: agg(k, "before", per_loop) for k in ("H", "p1", "margin")},
        "in_loop": {k: agg(k, "in_loop", [p for p in per_loop if p["in_loop"]]) for k in ("H", "p1", "margin")},
        "margin_lower_before_onset_share": round(sum(1 for p in paired if p["before"]["margin"] < p["control"]["margin"])
                                                 / len(paired), 4) if paired else None,
        "median_onset_token": statistics.median([p["onset_token"] for p in per_loop]) if per_loop else None,
        "per_row": per_loop,
    }

    return {
        "cell": str(cell_dir), "checkpoint": ckpt, "base": base, "tokenizer": tok_cfg.type,
        "template_version": TEMPLATE_VERSION, "max_length": max_length, "dtype": dtype,
        "heldout": heldout, "dump": str(dump_path),
        "dump_rows": len(rows), "vocab_rows": int(model.get_output_embeddings().weight.shape[0]),
        "references": refs, "generations": gens, "loop": loop_block,
        "seconds": {"references": round(t_ref, 1), "generations": round(t_gen, 1)},
    }


def print_table(res: Dict[str, Any]) -> None:
    r, g, lp = res["references"], res["generations"], res["loop"]
    f = lambda v, d=4: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    lines = [
        ("checkpoint", res["checkpoint"]),
        ("tokenizer / vocab rows", f"{res['tokenizer']} / {res['vocab_rows']}"),
        ("refs: entropy / char", f(r["entropy_per_char"])),
        ("refs: entropy / token", f(r["entropy_per_token"])),
        ("refs: mean p1 / margin", f"{f(r['mean_p1'])} / {f(r['mean_margin'])}"),
        ("refs: share p1 < 0.5", f(r["share_p1_lt_0_5"])),
        ("refs: NLL / char (cross-check)", f"{f(r['nll_per_answer_char'])} ({f(r['crosscheck']['nll_per_answer_char'])})"),
        ("gens: entropy / char", f(g["entropy_per_char"])),
        ("gens: entropy / token", f(g["entropy_per_token"])),
        ("gens: mean p1 / margin", f"{f(g['mean_p1'])} / {f(g['mean_margin'])}"),
        ("gens: share p1 < 0.5", f(g["share_p1_lt_0_5"])),
        ("gens: round-trip exact", f"{g['roundtrip_exact_rows']} / {g['rows']}"),
        ("loop rows measured (with control)", f"{lp['rows_measured']} of {lp['looped_rows']} ({lp['rows_with_control']})"),
        ("before onset: H / p1 / margin", " / ".join(f(lp["before"][k]) for k in ("H", "p1", "margin"))),
        ("control:      H / p1 / margin", " / ".join(f(lp["control"][k]) for k in ("H", "p1", "margin"))),
        ("in loop: mean p1", f(lp["in_loop"]["p1"])),
        ("refs decile H", " ".join(f(v, 2) for v in r["decile_H"])),
        ("gens decile H", " ".join(f(v, 2) for v in g["decile_H"])),
    ]
    w = max(len(k) for k, _ in lines)
    for k, v in lines:
        print(f"{k:<{w}}  {v}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True, help="experiment cell directory (has config.json)")
    ap.add_argument("--base", action="store_true", help="score model.name_or_path instead of training/sft")
    ap.add_argument("--checkpoint", help="an explicit checkpoint directory")
    ap.add_argument("--dump", help="free-form dump (default <cell>/eval_rows/freeform_cidar.parquet)")
    ap.add_argument("--heldout", default=DH.DEFAULT_HELDOUT)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--max-length", type=int, default=2048, help="truncation length for the references (the diagnostic's)")
    ap.add_argument("--batch-size", type=int, default=4, help="batch size of the generation pass")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", help="JSON output (default <cell>/diag_uncertainty_v1[_base].json)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cell_dir = Path(args.cell)
    res = run(cell_dir, base=args.base, checkpoint=args.checkpoint, dump=args.dump, heldout=args.heldout,
              window=args.window, max_length=args.max_length, device=args.device, dtype=args.dtype,
              batch_size=args.batch_size)
    print_table(res)
    out = Path(args.out) if args.out else cell_dir / ("diag_uncertainty_v1" + ("_base" if args.base else "") + ".json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
