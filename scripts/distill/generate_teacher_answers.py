#!/usr/bin/env python
"""Teacher answers under vLLM (run from ``.venv-judge`` through ``scripts/distill/run_teacher.sh``).

One candidate of ``configs/distill/teacher_candidates.yaml`` answers every prompt of
a JSONL file (``id``, ``instruction``, ``context``): its own chat template through
``LLM.chat`` with the fixed system prompt (``arabic_eval.distill.teacher``; a template
without a system role gets it prepended to the user turn — detected by rendering),
thinking off (``CHAT_TEMPLATE_KWARGS``), greedy (temperature 0), seed 0, repetition
penalty 1.0, stop at the model's own EOS / end of turn (its generation config), **no
stop markers**, ``--max-tokens`` = the cap ``dump_bakeoff_prompts.py --caps`` derived.
A prompt whose rendered chat is longer than ``max_model_len − cap`` tokens is skipped
and counted. Output: one JSONL row per prompt ``{id, text, finish_reason, n_tokens,
prompt_tokens, prompt_text}`` (``finish_reason`` ``stop`` = EOS, ``length`` = the cap;
``prompt_text`` = the exact rendered chat), appended chunk by chunk so a restarted run
resumes; and ``--meta`` JSON: snapshot, dtype / quantization actually loaded, sampling
parameters, system prompt + sha256, system-role handling, cap, load / generation wall,
tokens per second.

    scripts/distill/run_teacher.sh --slug jais2_8b --prompts outputs/data_cache/distill/bakeoff/prompts_dev.jsonl \
        --max-tokens 1100 --out <dir>/teacher_raw.jsonl --meta <dir>/generation_meta.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arabic_eval.distill.teacher import (  # noqa: E402
    CHAT_TEMPLATE_KWARGS, SYSTEM_PROMPT, build_conversation, candidate, iter_jsonl, read_jsonl, sha256_text,
)

log = logging.getLogger("generate_teacher_answers")
CANDIDATES = "configs/distill/teacher_candidates.yaml"


def _render(tok, conv: List[Dict[str, str]]) -> str:
    return tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False, **CHAT_TEMPLATE_KWARGS)


def system_role_supported(tok) -> bool:
    """True when the template renders a system turn and keeps its text."""
    marker = "SYSTEM-ROLE-PROBE-7c1f"
    try:
        text = _render(tok, [{"role": "system", "content": marker}, {"role": "user", "content": "x"}])
    except Exception:  # noqa: BLE001 — templates raise on an unsupported role
        return False
    return marker in text


def _count_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False).input_ids)


def _load_llm(cand, attempt: Dict[str, Any], max_model_len: int, seed: int):
    from vllm import LLM
    kw: Dict[str, Any] = dict(model=attempt.get("repo", cand.model), revision=attempt.get("revision", cand.revision),
                              tokenizer_revision=attempt.get("revision", cand.revision),
                              dtype=attempt.get("dtype", cand.dtype), max_model_len=max_model_len,
                              gpu_memory_utilization=attempt.get("gpu_memory_utilization", cand.gpu_memory_utilization),
                              seed=seed)
    q = attempt.get("quantization", cand.quantization)
    if q:
        kw["quantization"] = q
    # scheduler / capacity knobs (e.g. max_num_seqs — a Mamba hybrid's per-sequence state sizes the profiling
    # cache); they change throughput, never the text a greedy request produces from its own prompt
    kw.update({**(cand.engine_kwargs or {}), **(attempt.get("engine_kwargs") or {})})
    log.info("loading %s", {k: v for k, v in kw.items()})
    return LLM(**kw), kw


def _model_config(llm) -> Any:
    eng = llm.llm_engine
    for path in ("model_config", "vllm_config.model_config"):
        obj: Any = eng
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--candidates", default=CANDIDATES)
    ap.add_argument("--prompts", required=True, help="JSONL with id / instruction (or prompt, the held-out file's field) / context")
    ap.add_argument("--max-tokens", type=int, required=True, help="the candidate's token cap")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (the determinism check)")
    ap.add_argument("--chunk", type=int, default=4000, help="prompts per LLM.chat call (the file is appended per chunk)")
    ap.add_argument("--attempt", type=int, default=None, help="start at this load attempt (0 = the main entry, 1.. = fallbacks)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    defaults, cand = candidate(REPO_ROOT / args.candidates, args.slug)
    max_model_len = int(cand.max_model_len)
    seed = int(defaults.get("seed", 0))
    prompts = read_jsonl(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    # load: the main entry, then each fallback in order
    attempts: List[Dict[str, Any]] = [{}] + list(cand.fallbacks or [])
    start = args.attempt or 0
    t0 = time.perf_counter()
    llm = kw = None
    errors: List[str] = []
    for i in range(start, len(attempts)):
        try:
            llm, kw = _load_llm(cand, attempts[i], max_model_len, seed)
            used_attempt = i
            break
        except Exception as e:  # noqa: BLE001 — record, try the next fallback
            msg = f"attempt {i} {attempts[i] or 'main'}: {type(e).__name__}: {str(e)[:2000]}"
            log.error("load failed — %s", msg)
            errors.append(msg)
            try:
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
    if llm is None:
        Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
        with open(args.meta, "w", encoding="utf-8") as f:
            json.dump({"slug": cand.slug, "repo": cand.repo, "loaded": False, "load_errors": errors}, f, ensure_ascii=False, indent=2)
        print("LOAD FAILED:\n" + "\n".join(errors))
        return 2
    load_wall = time.perf_counter() - t0

    from vllm import SamplingParams
    tok = llm.get_tokenizer()
    sys_role = system_role_supported(tok)
    sp = SamplingParams(temperature=0.0, seed=seed, repetition_penalty=1.0, max_tokens=args.max_tokens)
    budget = max_model_len - args.max_tokens

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in iter_jsonl(out_path)} if out_path.exists() else set()
    if done:
        log.info("resuming: %d of %d prompts already answered in %s", len(done), len(prompts), out_path)

    convs, ids, texts, skipped = [], [], [], []
    prompt_tokens: Dict[str, int] = {}
    for p in prompts:
        if p["id"] in done:
            continue
        conv = build_conversation(p.get("instruction") or p["prompt"], p.get("context") or "", system_role=sys_role)
        text = _render(tok, conv)
        n = _count_tokens(tok, text)
        if n > budget:
            skipped.append({"id": p["id"], "prompt_tokens": n})
            continue
        convs.append(conv)
        ids.append(p["id"])
        texts.append(text)
        prompt_tokens[p["id"]] = n
    log.info("%s: %d prompts to answer (%d already done, %d skipped: rendered chat > %d tokens), cap %d, "
             "system role %s", cand.slug, len(convs), len(done), len(skipped), budget, args.max_tokens,
             "native" if sys_role else "prepended to the user turn")

    gen_wall = 0.0
    gen_tokens = 0
    n_written = 0
    finish: Dict[str, int] = {}
    with open(out_path, "a", encoding="utf-8") as f:
        for s in range(0, len(convs), args.chunk):
            t1 = time.perf_counter()
            res = llm.chat(convs[s:s + args.chunk], sp, use_tqdm=True, chat_template_kwargs=dict(CHAT_TEMPLATE_KWARGS))
            dt = time.perf_counter() - t1
            gen_wall += dt
            chunk_tokens = 0
            for pid, ptext, r in zip(ids[s:s + args.chunk], texts[s:s + args.chunk], res):
                o = r.outputs[0]
                n_tok = len(o.token_ids)
                chunk_tokens += n_tok
                finish[str(o.finish_reason)] = finish.get(str(o.finish_reason), 0) + 1
                f.write(json.dumps({"id": pid, "text": o.text, "finish_reason": o.finish_reason, "n_tokens": n_tok,
                                    "prompt_tokens": len(r.prompt_token_ids or []) or prompt_tokens[pid],
                                    "prompt_text": ptext}, ensure_ascii=False) + "\n")
                n_written += 1
            f.flush()
            gen_tokens += chunk_tokens
            log.info("%s: %d / %d answered (%d tokens in %.0fs = %.0f tok/s)", cand.slug, min(s + args.chunk, len(convs)),
                     len(convs), chunk_tokens, dt, chunk_tokens / dt if dt > 0 else 0.0)

    mc = _model_config(llm)
    gen_cfg = {}
    try:
        gen_cfg = {k: v for k, v in (llm.llm_engine.get_model_config().get_diff_sampling_param() or {}).items()}
    except Exception:  # noqa: BLE001
        pass
    import vllm
    meta = {
        "slug": cand.slug, "repo": cand.repo, "loaded_from": kw["model"], "revision": kw["revision"],
        "loaded": True, "load_attempt": used_attempt, "load_errors": errors,
        "load_kwargs": {k: v for k, v in kw.items()},
        "dtype": str(getattr(mc, "dtype", kw.get("dtype"))), "quantization": getattr(mc, "quantization", kw.get("quantization")),
        "max_model_len": max_model_len, "gpu_memory_utilization": kw["gpu_memory_utilization"],
        "vllm_version": vllm.__version__,
        "sampling": {"temperature": 0.0, "seed": seed, "repetition_penalty": 1.0, "max_tokens": args.max_tokens,
                     "stop": None, "generation_config_defaults": gen_cfg},
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS, "system_role": "native" if sys_role else "prepended_to_user_turn",
        "system_prompt": SYSTEM_PROMPT, "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "eos_token": getattr(tok, "eos_token", None), "eos_token_id": getattr(tok, "eos_token_id", None),
        "prompts_file": str(args.prompts), "n_prompts": len(prompts), "n_answered_this_run": n_written,
        "n_already_done": len(done), "skipped_length": skipped, "prompt_token_budget": budget,
        "finish_reasons_this_run": finish,
        "load_wall_sec": round(load_wall, 1), "gen_wall_sec": round(gen_wall, 1), "gen_tokens": gen_tokens,
        "tok_per_sec": round(gen_tokens / gen_wall, 1) if gen_wall > 0 else None,
        "prompt_tokens_total": sum(prompt_tokens.values()),
        "example_prompt_text": texts[0] if texts else None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("system_prompt", "example_prompt_text", "skipped_length")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
