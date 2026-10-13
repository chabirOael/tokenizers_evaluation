#!/usr/bin/env python
"""Response time per tokenizer (brief v6 §6, the supervisor tables' T4): how long the *model* takes to answer
under a cell's tokenizer, under identical conditions — eval only, one cell per invocation, the GPU exclusive.

    .venv/bin/python scripts/measure_response_time.py --cell outputs/experiments/qwen_native_vs_araroopat/araroopat_3phase_v5_distill

The cell's tokenizer and checkpoint are loaded the way ``scripts/dump_eval_rows.py`` does (adapt to the tokenizer,
then the state dict strictly); the adapter turns the cuDNN SDPA backend off as in every eval; every timed region is
bracketed by ``torch.cuda.synchronize()`` (when the model is on the GPU). What is measured:

1. **Answer latency at batch size 1** on the 250 held-out prompts, through the free-form task's own
   ``generate_freeform`` (greedy, the cell's dumped prompt strings, its decoding rules — stop markers, loop stop, the
   2 400-character budget — and its token cap, read from the dump metadata). Per prompt: wall time of the answer,
   the tokenizer's time inside it (``encode`` / ``decode`` wrapped on this process's tokenizer — the prompt encode and
   the stop checks every 16 steps), generated tokens and characters, stop reason, and whether the answer equals the
   cell's dumped one. **Time to first piece** is a separate ``generate`` of one new token on the same prompt
   (a streamer would force a GPU sync per step and slow the full answer). Repeat 1 covers all 250 prompts; repeats
   2..R a seeded, length-stratified subset (``--subset``) for the spread.
2. **Batched throughput** at batch 16 on the same prompts (the eval's own setting, warm-up included, untimed):
   characters and tokens per second, to set against the cell's recorded ``gen_chars_per_sec``.
3. **Multiple-choice scoring latency**: ``--mcq-rows`` rows per benchmark (Arabic-Exam and Culture-MMLU: the first
   rows of the 1 000-row letter-slot subsets; ACVA and Alghafa: drawn with seed 0), their exact dumped prompts and
   continuations, scored by the task's own ``_compute_loglikelihood`` at ``max_length 4096`` (3-shot prompts as
   dumped). Per row: encode time and forward time (the rest); ms per row, ms per 1 000 prompt characters, prompt
   tokens per row. The PMI rule's unconditioned call is not included (cached across rows in the eval).
4. **Energy**: ``nvidia-smi`` sampled every 500 ms in the background; mean watts per section, peak memory,
   Wh per 1 000 answers.

Output ``<cell>/timing_v1.json`` with the per-prompt and per-row numbers, the summaries, GPU, driver, torch version,
commit and date.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

log = logging.getLogger("response_time")
TASKS = ("arabic_exam", "culture_arabic_mmlu", "acva", "alghafa")
ROWS_FILES = {"arabic_exam": REPO / "configs/experiments/mcq_letter_slot/rows_arabic_exam.json",
              "culture_arabic_mmlu": REPO / "configs/experiments/mcq_letter_slot/rows_culture_arabic_mmlu.json"}
DEFAULT_MCQ_PROMPTS = REPO / "outputs/experiments/qwen_native_vs_araroopat/native_qwen3_sft_v5_distill_mcq4096"


_ON_GPU = True


def _sync() -> None:
    if _ON_GPU:
        import torch
        torch.cuda.synchronize()


def _reset_peak() -> None:
    if _ON_GPU:
        import torch
        torch.cuda.reset_peak_memory_stats()


def _peak_gib():
    if not _ON_GPU:
        return None
    import torch
    return round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)


def _dump_module():
    spec = importlib.util.spec_from_file_location("dump_eval_rows", REPO / "scripts" / "dump_eval_rows.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# --------------------------------------------------------------------------------------------------------------
# GPU power sampler
# --------------------------------------------------------------------------------------------------------------

class PowerSampler:
    def __init__(self, every_ms: int = 500) -> None:
        self.samples: List[Tuple[float, float, float, float]] = []      # (t, watts, util %, memory MiB)
        self._p = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits", f"-lms={every_ms}"],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()

    def _read(self) -> None:
        assert self._p.stdout is not None
        for line in self._p.stdout:
            try:
                w, u, m = (float(x) for x in line.strip().split(","))
            except ValueError:
                continue
            self.samples.append((time.time(), w, u, m))

    def window(self, t0: float, t1: float) -> Dict[str, Any]:
        xs = [s for s in self.samples if t0 <= s[0] <= t1]
        if not xs:
            return {"n_samples": 0}
        return {"n_samples": len(xs), "mean_watts": round(statistics.fmean(s[1] for s in xs), 1),
                "max_watts": round(max(s[1] for s in xs), 1), "mean_util_pct": round(statistics.fmean(s[2] for s in xs), 1),
                "peak_memory_mib": round(max(s[3] for s in xs))}

    def stop(self) -> None:
        self._p.terminate()


def gpu_facts() -> Dict[str, Any]:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().split(",")
    return {"gpu": q[0].strip() if q else None, "driver": q[1].strip() if len(q) > 1 else None,
            "memory_used_mib_at_start": int(q[2]) if len(q) > 2 else None}


# --------------------------------------------------------------------------------------------------------------
# tokenizer timing wrapper
# --------------------------------------------------------------------------------------------------------------

class TokTimer:
    """Wraps ``encode`` / ``decode`` of one tokenizer instance and accumulates their wall time."""

    def __init__(self, tok) -> None:
        self.tok = tok
        self.encode_s = 0.0
        self.decode_s = 0.0
        self._enc, self._dec = tok.encode, tok.decode

        def enc(*a, **k):
            t = time.perf_counter()
            try:
                return self._enc(*a, **k)
            finally:
                self.encode_s += time.perf_counter() - t

        def dec(*a, **k):
            t = time.perf_counter()
            try:
                return self._dec(*a, **k)
            finally:
                self.decode_s += time.perf_counter() - t

        tok.encode, tok.decode = enc, dec

    def reset(self) -> Tuple[float, float]:
        e, d = self.encode_s, self.decode_s
        self.encode_s = self.decode_s = 0.0
        return e, d


# --------------------------------------------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------------------------------------------

def stratified_subset(rows: List[Dict[str, Any]], n: int, seed: int = 0) -> List[int]:
    rng = random.Random(seed)
    by: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        by.setdefault(r.get("stratum") or "", []).append(i)
    out: List[int] = []
    k = max(1, n // max(1, len(by)))
    for s in sorted(by):
        out += rng.sample(by[s], min(k, len(by[s])))
    return sorted(out)[:n]


def freeform_latency(adapter, tok, timer: TokTimer, rows: List[Dict[str, Any]], cfg, token_cap: int,
                     idx: List[int], sampler: PowerSampler) -> Dict[str, Any]:
    import torch
    from arabic_eval.tasks.freeform.generation import generate_freeform
    from arabic_eval.tasks.freeform.generation import strip_trailing_eos
    specials = tok.special_tokens
    pad_id = int(specials.get("pad_token", 0))
    eos_id = specials.get("eos_token")
    eos_id = None if eos_id is None else int(eos_id)
    per: List[Dict[str, Any]] = []
    t_start = time.time()
    for i in idx:
        r = rows[i]
        prompt = r["prompt_text"]
        # time to first piece: prefill + one decode step, a call of its own
        ids = strip_trailing_eos(list(timer._enc(prompt, max_length=cfg.max_prompt_tokens, truncation=True).input_ids), eos_id)
        x = torch.tensor([ids], dtype=torch.long, device=adapter.device)
        am = torch.ones_like(x)
        _sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            adapter.generate(x, attention_mask=am, max_new_tokens=1, do_sample=False, num_beams=1,
                             pad_token_id=pad_id, eos_token_id=eos_id, use_cache=True)
        _sync()
        ttft = time.perf_counter() - t0
        timer.reset()
        _sync()
        t0 = time.perf_counter()
        g = generate_freeform(adapter, tok, [prompt], cfg, token_cap, warmup=False)[0]
        _sync()
        wall = time.perf_counter() - t0
        enc_s, dec_s = timer.reset()
        per.append({"id": r["id"], "stratum": r.get("stratum"), "wall_sec": round(wall, 4), "ttft_sec": round(ttft, 4),
                    "tokenizer_encode_sec": round(enc_s, 5), "tokenizer_decode_sec": round(dec_s, 5),
                    "prompt_tokens": len(ids), "gen_tokens": g.gen_tokens, "gen_chars": len(g.generation),
                    "stop_reason": g.stop_reason, "equal_to_dump": g.generation == r["generation"]})
    t_end = time.time()
    walls = [p["wall_sec"] for p in per]
    toks = sum(p["gen_tokens"] for p in per)
    chars = sum(p["gen_chars"] for p in per)
    tz = sum(p["tokenizer_encode_sec"] + p["tokenizer_decode_sec"] for p in per)
    power = sampler.window(t_start, t_end)
    summ = {"n": len(per), "sec_per_answer_median": round(statistics.median(walls), 4),
            "sec_per_answer_mean": round(statistics.fmean(walls), 4),
            "ms_per_generated_token": round(1000 * sum(walls) / toks, 3) if toks else None,
            "ms_per_100_generated_chars": round(1000 * sum(walls) / chars * 100, 2) if chars else None,
            "ttft_ms_median": round(1000 * statistics.median(p["ttft_sec"] for p in per), 2),
            "tokenizer_share_of_answer_time": round(tz / sum(walls), 4),
            "equal_to_dump_share": round(sum(p["equal_to_dump"] for p in per) / len(per), 4),
            "mean_gen_tokens": round(toks / len(per), 1), "mean_gen_chars": round(chars / len(per), 1),
            "power": power}
    if power.get("mean_watts"):
        summ["wh_per_1000_answers"] = round(power["mean_watts"] * statistics.fmean(walls) * 1000 / 3600, 2)
    return {"summary": summ, "per_prompt": per}


def freeform_batched(adapter, tok, rows, cfg, token_cap: int, sampler: PowerSampler) -> Dict[str, Any]:
    import dataclasses
    import torch
    from arabic_eval.tasks.freeform.generation import generate_freeform
    c16 = dataclasses.replace(cfg, batch_size=16)
    prompts = [r["prompt_text"] for r in rows]
    _sync()
    t0w = time.time()
    out = generate_freeform(adapter, tok, prompts, c16, token_cap, warmup=True)
    _sync()
    t1w = time.time()
    wall = sum(g.gen_time_sec for g in out)              # Σ per-batch walls (batch wall / size per row)
    toks = sum(g.gen_tokens for g in out)
    chars = sum(len(g.generation) for g in out)
    return {"batch_size": 16, "generation_wall_sec": round(wall, 2), "chars_per_sec": round(chars / wall, 1),
            "tokens_per_sec": round(toks / wall, 1),
            "equal_to_dump_share": round(sum(g.generation == r["generation"] for g, r in zip(out, rows)) / len(rows), 4),
            "power": sampler.window(t0w, t1w)}


def mcq_rows(prompts_cell: Path, n: int) -> Dict[str, List[Dict[str, Any]]]:
    import pyarrow.parquet as pq
    out: Dict[str, List[Dict[str, Any]]] = {}
    for task in TASKS:
        p = prompts_cell / "eval_rows" / f"{task}.parquet"
        t = pq.read_table(p, columns=["row_index", "prompt", "continuations"]).to_pylist()
        by = {r["row_index"]: r for r in t}
        if task in ROWS_FILES:
            want = json.loads(ROWS_FILES[task].read_text())["row_index"][:n]
        else:
            want = sorted(random.Random(0).sample(sorted(by), min(n, len(by))))
        out[task] = [by[i] for i in want if i in by]
    return out


def mcq_latency(adapter, tok, timer: TokTimer, rows: Dict[str, List[Dict[str, Any]]], sampler: PowerSampler,
                max_length: int = 4096) -> Dict[str, Any]:
    import torch
    from arabic_eval.tasks.lighteval.base import _compute_loglikelihood
    res: Dict[str, Any] = {}
    for task, rr in rows.items():
        # warm-up on two rows (the first forward of a new length pays kernel selection)
        for r in rr[:2]:
            for c in r["continuations"]:
                _compute_loglikelihood(adapter, tok, r["prompt"], c, max_length=max_length)
        per = []
        t0w = time.time()
        for r in rr:
            timer.reset()
            _sync()
            t0 = time.perf_counter()
            for c in r["continuations"]:
                _compute_loglikelihood(adapter, tok, r["prompt"], c, max_length=max_length)
            _sync()
            wall = time.perf_counter() - t0
            enc_s, _ = timer.reset()
            ptoks = len(timer._enc(r["prompt"], max_length=max_length, truncation=True).input_ids)
            per.append({"row_index": r["row_index"], "wall_sec": wall, "encode_sec": enc_s, "forward_sec": wall - enc_s,
                        "prompt_chars": len(r["prompt"]), "prompt_tokens": ptoks, "n_choices": len(r["continuations"])})
        t1w = time.time()
        W = sum(p["wall_sec"] for p in per)
        res[task] = {"n_rows": len(per), "ms_per_row": round(1000 * W / len(per), 2),
                     "encode_ms_per_row": round(1000 * sum(p["encode_sec"] for p in per) / len(per), 3),
                     "forward_ms_per_row": round(1000 * sum(p["forward_sec"] for p in per) / len(per), 2),
                     "ms_per_1000_prompt_chars": round(1000 * W / sum(p["prompt_chars"] for p in per) * 1000, 3),
                     "prompt_tokens_per_row": round(statistics.fmean(p["prompt_tokens"] for p in per), 1),
                     "choices_per_row": round(statistics.fmean(p["n_choices"] for p in per), 2),
                     "power": sampler.window(t0w, t1w)}
    return res


# --------------------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--subset", type=int, default=50, help="prompts in repeats 2..R (length-stratified, seed 0)")
    ap.add_argument("--mcq-rows", type=int, default=500)
    ap.add_argument("--mcq-repeats", type=int, default=3)
    ap.add_argument("--mcq-prompts-cell", default=str(DEFAULT_MCQ_PROMPTS),
                    help="a cell whose eval_rows/<task>.parquet holds the prompt strings (identical across cells)")
    ap.add_argument("--skip-mcq", action="store_true")
    ap.add_argument("--force", action="store_true", help="run although the GPU is not idle")
    # functional-test knobs (never for a reported number): fewer prompts / rows, another device, a small token cap
    ap.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--device", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--token-cap", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    cell = Path(args.cell).resolve()
    facts = gpu_facts()
    if (facts.get("memory_used_mib_at_start") or 0) > 1000 and not args.force:
        print(f"GPU not idle ({facts['memory_used_mib_at_start']} MiB in use) — refusing to time (use --force to override)")
        return 2

    import torch
    from arabic_eval.registry import model_registry
    import arabic_eval.models  # noqa: F401
    import arabic_eval.tokenizers  # noqa: F401
    from arabic_eval.tasks.freeform.generation import DecodingConfig
    from arabic_eval.utils.reproducibility import set_seed
    import pyarrow.parquet as pq

    dm = _dump_module()
    spec = dm.CellSpec(cell)
    set_seed(spec.seed)
    tok_cfg = spec.tokenizer
    path = tok_cfg.load_path or tok_cfg.save_path
    from arabic_eval.registry import tokenizer_registry
    tok = tokenizer_registry.get(tok_cfg.type)(**(tok_cfg.params or {}))
    if path and (REPO / path).exists():
        tok.load(str(REPO / path))
    adapter = model_registry.get(spec.model.type)(model_name_or_path=spec.model.name_or_path, device=args.device or spec.model.device,
                                                  dtype=spec.model.dtype, **spec.model.params)
    adapter.adapt_to_tokenizer(tok)
    ckpt = dm.pick_checkpoint(cell, spec, None)
    if ckpt is not None:
        dm.load_state_into(adapter, ckpt)
    adapter.model.eval()
    global _ON_GPU
    _ON_GPU = adapter.device.type == "cuda"
    timer = TokTimer(tok)

    gp = cell / "eval_rows" / "freeform_cidar.parquet"
    t = pq.read_table(gp)
    meta = json.loads(t.schema.metadata[b"arabic_eval"])
    rows = t.select(["id", "stratum", "prompt_text", "generation"]).to_pylist()
    if args.limit:
        rows = rows[: args.limit]
    dec = meta.get("decoding") or {}
    cfg = DecodingConfig(**{k: (tuple(v) if k == "stop_markers" else v) for k, v in dec.items()
                            if k in DecodingConfig.__dataclass_fields__})
    cfg = __import__("dataclasses").replace(cfg, batch_size=1)
    token_cap = int(args.token_cap or meta["token_cap"])
    out: Dict[str, Any] = {"cell": str(cell.relative_to(REPO)), "checkpoint": str(ckpt.relative_to(REPO)) if ckpt else None,
                           "tokenizer": {"type": tok_cfg.type, "path": path, "vocab_size": tok.vocab_size},
                           "token_cap": token_cap, "decoding": cfg.to_json(), **facts, "torch": torch.__version__,
                           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                                                    text=True).stdout.strip(),
                           "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "repeats": args.repeats, "subset": args.subset,
                           "test_mode": bool(args.limit or args.device or args.token_cap)}
    sampler = PowerSampler()
    try:
        # warm-up: one untimed short answer
        from arabic_eval.tasks.freeform.generation import generate_freeform
        generate_freeform(adapter, tok, [rows[0]["prompt_text"]], cfg, 8, warmup=False)
        _reset_peak()
        sub = stratified_subset(rows, min(args.subset, len(rows)))
        reps = []
        for k in range(args.repeats):
            idx = list(range(len(rows))) if k == 0 else sub
            log.info("batch-1 latency, repeat %d/%d over %d prompts", k + 1, args.repeats, len(idx))
            reps.append(freeform_latency(adapter, tok, timer, rows, cfg, token_cap, idx, sampler))
        out["batch1"] = {"repeat_1_all_prompts": reps[0]["summary"],
                         "repeats_on_subset": [r["summary"] for r in reps[1:]],
                         "repeat_1_on_subset": {"sec_per_answer_median": round(statistics.median(
                             reps[0]["per_prompt"][i]["wall_sec"] for i in sub), 4)},
                         "per_prompt": reps[0]["per_prompt"]}
        log.info("batched throughput at batch 16")
        out["batch16"] = freeform_batched(adapter, tok, rows, cfg, token_cap, sampler)
        out["peak_memory_allocated_gib_freeform"] = _peak_gib()
        if not args.skip_mcq:
            mrows = mcq_rows(Path(args.mcq_prompts_cell), args.limit or args.mcq_rows)
            _reset_peak()
            mreps = []
            for k in range(args.mcq_repeats):
                log.info("MCQ scoring latency, repeat %d/%d", k + 1, args.mcq_repeats)
                mreps.append(mcq_latency(adapter, tok, timer, mrows, sampler))
            out["mcq"] = {task: {"repeats": [m[task] for m in mreps],
                                 "ms_per_row_median": round(statistics.median(m[task]["ms_per_row"] for m in mreps), 2)}
                          for task in mrows}
            out["mcq_prompts_cell"] = str(Path(args.mcq_prompts_cell).resolve().relative_to(REPO))
            out["peak_memory_allocated_gib_mcq"] = _peak_gib()
    finally:
        sampler.stop()
    dst = Path(args.out) if args.out else cell / "timing_v1.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    b1 = out["batch1"]["repeat_1_all_prompts"]
    print(f"{cell.name}: {b1['sec_per_answer_median']} s/answer (median), {b1['ms_per_100_generated_chars']} ms per 100 chars, "
          f"TTFT {b1['ttft_ms_median']} ms, tokenizer share {b1['tokenizer_share_of_answer_time']}, equal to dump "
          f"{b1['equal_to_dump_share']}; batch 16 {out['batch16']['chars_per_sec']} chars/s → {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
