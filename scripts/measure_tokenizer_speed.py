#!/usr/bin/env python
"""The tokenizers' own speed (brief v6 §7, the supervisor tables' T5): encode / decode throughput and latency on
fixed texts, cold and warm, with memory — CPU only, one fresh process per measured case.

    .venv/bin/python scripts/measure_tokenizer_speed.py                     # every tokenizer that loads
    .venv/bin/python scripts/measure_tokenizer_speed.py --only bpe_16k bpe_128k --repeats 3

Texts (fixed, seeded, saved next to the results as ``texts.jsonl``): (a) ``--n-eval`` texts of the ArabicText-Large
eval split (the pipeline's loader, base.yaml preprocessing, seed 42), (b) the 250 held-out free-form references,
(c) the 1 000 Arabic-Exam 3-shot prompts of the letter-slot subset (``configs/experiments/mcq_letter_slot/
rows_arabic_exam.json``) as the native v5 MCQ dump holds them.

Per tokenizer and set, in a fresh subprocess (so load time, memory and AraRooPat's cold CAMeL start are real):
  * load — wall seconds and process RSS before / after, vocabulary files on disk;
  * encode — one untimed warm-up pass, then ``--repeats`` timed passes through the wrapper's own ``encode`` (the call
    the pipeline makes): median and min / max wall, characters / words / tokens per second, microseconds per word,
    per-text latency p50 / p95 (median over repeats);
  * decode — the same on the produced ids, plus the exact-match share of ``decode(encode(text)) == text``;
  * Rust tokenizers (native wrappers, BPE): the same encode pass through ``encode_batch`` too, once with
    ``TOKENIZERS_PARALLELISM=false`` and once with the library default (two subprocesses);
  * AraRooPat, honestly: *cold* (a fresh bridge and empty caches, the first half of each set, timed — the bridge
    start-up measured apart), *warm* (the same first half again, ``--repeats`` times), *steady* (the second half,
    texts it has not seen, once); bridge round trips, words sent to CAMeL per 1 000 words, analysis-cache hit rate,
    the CAMeL server's RSS, and decode-tier shares (lookup / generator / naive) — the counters wrap this process's own
    bridge and analyzer, nothing in the package changes.
Machine facts (CPU model, cores, Python / tokenizers versions, commit, date) go into the JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

HELDOUT = REPO / "configs/contamination/freeform_cidar_heldout_v1.jsonl"
ROWS = REPO / "configs/experiments/mcq_letter_slot/rows_arabic_exam.json"
EXAM_DUMP = REPO / "outputs/experiments/qwen_native_vs_araroopat/native_qwen3_sft_v5_distill_mcq4096/eval_rows/arabic_exam.parquet"

# name → (registry type, saved dir or None, constructor params)
TOKENIZERS: Dict[str, Any] = {
    "native_qwen3": ("native_qwen3", None, {}),
    "native_llama3b": ("native_llama", None, {"model_name_or_path": "meta-llama/Llama-3.2-3B"}),
    "native_gemma3": ("native_gemma", None, {"model_name_or_path": "google/gemma-3-4b-pt"}),
    "bpe_16k": ("bpe", "outputs/tokenizers/bpe_16k_v3", {}),
    "bpe_128k": ("bpe", "outputs/tokenizers/bpe_128k_v1", {}),
    "araroopat": ("araroopat", "outputs/tokenizers/araroopat_maxpat40k_v3", {}),
}
RUST_TYPES = {"bpe", "native_qwen3", "native_llama", "native_gemma"}


def rss_mb(pid: Optional[int] = None) -> Optional[float]:
    try:
        with open(f"/proc/{pid or 'self'}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


def dir_mb(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    p = REPO / path
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2 ** 20, 2) if p.exists() else None


def machine() -> Dict[str, Any]:
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    import tokenizers
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    return {"cpu": cpu, "cores": os.cpu_count(), "loadavg_at_start": list(os.getloadavg()),
            "python": platform.python_version(), "tokenizers": tokenizers.__version__,
            "commit": commit + ("+uncommitted" if dirty else ""), "date": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


# --------------------------------------------------------------------------------------------------------------
# texts
# --------------------------------------------------------------------------------------------------------------

def build_texts(n_eval: int) -> Dict[str, List[str]]:
    from arabic_eval.config import DataConfig, load_yaml
    from arabic_eval.data.loader import extract_texts, load_arabic_dataset
    import pyarrow.parquet as pq
    data = DataConfig(**(load_yaml(REPO / "configs/base.yaml").get("data") or {}))
    ds = load_arabic_dataset(dataset_name=data.dataset_name, cache_dir=data.cache_dir, max_eval_samples=n_eval,
                             preprocessing_config=data.preprocessing, seed=42)
    ev = extract_texts(ds["eval"])[:n_eval]
    refs = [json.loads(l)["reference"] for l in HELDOUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = set(json.loads(ROWS.read_text())["row_index"])
    t = pq.read_table(EXAM_DUMP, columns=["row_index", "prompt"]).to_pylist()
    exam = [r["prompt"] for r in sorted(t, key=lambda r: r["row_index"]) if r["row_index"] in rows]
    return {"eval_split": ev, "references": refs, "exam_prompts": exam}


# --------------------------------------------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------------------------------------------

def _pctl(xs: List[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def time_pass(fn, items: List[Any]) -> Dict[str, Any]:
    lat = []
    out = []
    t0 = time.perf_counter()
    for x in items:
        a = time.perf_counter_ns()
        out.append(fn(x))
        lat.append((time.perf_counter_ns() - a) / 1e6)
    return {"wall": time.perf_counter() - t0, "lat_ms": lat, "out": out}


def summarize(passes: List[Dict[str, Any]], texts: List[str], n_tokens: int) -> Dict[str, Any]:
    walls = [p["wall"] for p in passes]
    w = statistics.median(walls)
    chars = sum(len(t) for t in texts)
    words = sum(len(t.split()) for t in texts)
    return {"repeats": len(passes), "wall_sec_median": round(w, 4), "wall_sec_min": round(min(walls), 4),
            "wall_sec_max": round(max(walls), 4), "chars_per_sec": round(chars / w, 1), "words_per_sec": round(words / w, 1),
            "tokens_per_sec": round(n_tokens / w, 1), "us_per_word": round(1e6 * w / words, 3),
            "latency_ms_p50": round(statistics.median(_pctl(p["lat_ms"], 0.5) for p in passes), 4),
            "latency_ms_p95": round(statistics.median(_pctl(p["lat_ms"], 0.95) for p in passes), 4),
            "texts": len(texts), "chars": chars, "words": words, "tokens": n_tokens}


def worker(case: Dict[str, Any]) -> Dict[str, Any]:
    import arabic_eval.tokenizers  # noqa: F401
    from arabic_eval.registry import tokenizer_registry
    texts: Dict[str, List[str]] = json.loads(Path(case["texts_json"]).read_text(encoding="utf-8"))
    typ, path, params = case["type"], case["path"], case["params"]
    repeats = case["repeats"]
    res: Dict[str, Any] = {"name": case["name"], "type": typ, "path": path, "mode": case["mode"],
                           "rss_mb_before_load": rss_mb(), "vocab_files_mb": dir_mb(path)}
    t0 = time.perf_counter()
    tok = tokenizer_registry.get(typ)(**params)
    if path:
        tok.load(str(REPO / path))
    res["load_sec"] = round(time.perf_counter() - t0, 3)
    res["rss_mb_after_load"] = rss_mb()
    res["vocab_size"] = tok.vocab_size
    enc = lambda s: tok.encode(s).input_ids  # noqa: E731
    dec = lambda ids: tok.decode(ids)  # noqa: E731
    res["sets"] = {}
    if typ != "araroopat":
        batch = None
        if case["mode"].startswith("batch"):
            backend = getattr(tok, "_tokenizer", None) or getattr(getattr(tok, "_hf_tokenizer", None), "backend_tokenizer", None)
            if backend is None:
                res["error"] = "no encode_batch backend"
                return res
            batch = backend.encode_batch
        for set_name, items in texts.items():
            if batch is not None:
                batch(items)                                         # warm-up
                passes = []
                for _ in range(repeats):
                    t1 = time.perf_counter()
                    encs = batch(items)
                    passes.append({"wall": time.perf_counter() - t1, "lat_ms": [0.0]})
                ntok = sum(len(e.ids) for e in encs)
                s = summarize(passes, items, ntok)
                s.pop("latency_ms_p50"), s.pop("latency_ms_p95")
                res["sets"][set_name] = {"encode_batch": s}
                continue
            time_pass(enc, items)                                    # untimed warm-up
            ep = [time_pass(enc, items) for _ in range(repeats)]
            ids = ep[-1]["out"]
            ntok = sum(len(x) for x in ids)
            time_pass(dec, ids)
            dp = [time_pass(dec, ids) for _ in range(repeats)]
            back = dp[-1]["out"]
            res["sets"][set_name] = {"encode": summarize(ep, items, ntok), "decode": summarize(dp, items, ntok),
                                     "roundtrip_exact": round(sum(1 for a, b in zip(back, items) if a == b) / len(items), 4)}
        res["rss_mb_end"] = rss_mb()
        return res

    # ---- AraRooPat: cold / warm / steady with counters --------------------------------------------------------
    import arabic_eval.tokenizers.araroopat as ar_mod
    backend = tok._ensure_backend()
    bridge = backend._bridge
    counters = {"round_trips": 0, "analyze_calls": 0, "words_sent": 0, "generate_calls": 0}
    orig_analyze, orig_generate = bridge.analyze, bridge.generate

    def analyze(words, top=1):
        counters["round_trips"] += 1
        counters["analyze_calls"] += 1
        counters["words_sent"] += len(words)
        return orig_analyze(words, top=top)

    def generate(root, pattern):
        counters["round_trips"] += 1
        counters["generate_calls"] += 1
        return orig_generate(root, pattern)

    bridge.analyze, bridge.generate = analyze, generate
    tiers = {"lookup": 0, "generator": 0, "naive": 0}
    orig_reconstruct = tok._reconstruct
    orig_naive = ar_mod.naive_pattern_fill

    def naive(*a, **k):
        tiers["naive"] += 1
        return orig_naive(*a, **k)

    def reconstruct(root_id, pat_id, root, pattern, backend_):
        if tok._reconstruction.get((root_id, pat_id)):
            tiers["lookup"] += 1
            return orig_reconstruct(root_id, pat_id, root, pattern, backend_)
        before = tiers["naive"]
        out = orig_reconstruct(root_id, pat_id, root, pattern, backend_)
        if tiers["naive"] == before:
            tiers["generator"] += 1
        return out

    tok._reconstruct = reconstruct
    ar_mod.naive_pattern_fill = naive
    t1 = time.perf_counter()
    bridge._ensure_started()
    res["bridge_startup_sec"] = round(time.perf_counter() - t1, 3)
    res["rss_mb_after_bridge"] = rss_mb()

    def snap() -> Dict[str, int]:
        return dict(counters)

    def delta(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
        return {k: b[k] - a[k] for k in a}

    for set_name, items in texts.items():
        half = len(items) // 2
        A, B = items[:half], items[half:]
        words_a = sum(len(t.split()) for t in A)
        words_b = sum(len(t.split()) for t in B)
        rec: Dict[str, Any] = {}
        c0 = snap()
        cold = time_pass(enc, A)
        c1 = snap()
        ntok_a = sum(len(x) for x in cold["out"])
        rec["cold"] = {**summarize([cold], A, ntok_a), "bridge": delta(c0, c1),
                       "words_sent_per_1000_words": round(1000 * (c1["words_sent"] - c0["words_sent"]) / words_a, 2)}
        warm = [time_pass(enc, A) for _ in range(repeats)]
        c2 = snap()
        rec["warm"] = {**summarize(warm, A, ntok_a), "bridge": delta(c1, c2),
                       "words_sent_per_1000_words": round(1000 * (c2["words_sent"] - c1["words_sent"]) / (words_a * repeats), 2)}
        steady = time_pass(enc, B)
        c3 = snap()
        ntok_b = sum(len(x) for x in steady["out"])
        rec["steady"] = {**summarize([steady], B, ntok_b), "bridge": delta(c2, c3),
                         "words_sent_per_1000_words": round(1000 * (c3["words_sent"] - c2["words_sent"]) / words_b, 2)}
        ids = cold["out"]
        t0d = dict(tiers)
        dp = [time_pass(dec, ids) for _ in range(repeats)]
        back = dp[-1]["out"]
        tsum = {k: tiers[k] - t0d[k] for k in tiers}
        tot = sum(tsum.values()) or 1
        rec["decode"] = {**summarize(dp, A, ntok_a), "tiers": tsum,
                         "tier_shares": {k: round(v / tot, 4) for k, v in tsum.items()}}
        rec["roundtrip_exact"] = round(sum(1 for a, b in zip(back, A) if a == b) / len(A), 4)
        res["sets"][set_name] = rec
    res["analysis_cache_entries"] = len(getattr(backend, "_analyze_cache", {}))
    res["camel_server_pid"] = getattr(getattr(bridge, "_proc", None), "pid", None)
    res["camel_server_rss_mb"] = rss_mb(res["camel_server_pid"]) if res["camel_server_pid"] else None
    res["rss_mb_end"] = rss_mb()
    res["counters_total"] = counters
    return res


# --------------------------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(REPO / "outputs/experiments/_tokenizer_speed" / time.strftime("%Y-%m-%d")))
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--n-eval", type=int, default=20000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        case = json.loads(args.worker)
        print("@@RESULT@@" + json.dumps(worker(case), ensure_ascii=False))
        return 0
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tj = out / "texts.json"
    if not tj.exists():
        texts = build_texts(args.n_eval)
        tj.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        with open(out / "texts.jsonl", "w", encoding="utf-8") as f:
            for k, v in texts.items():
                for i, t in enumerate(v):
                    f.write(json.dumps({"set": k, "i": i, "text": t}, ensure_ascii=False) + "\n")
    texts = json.loads(tj.read_text(encoding="utf-8"))
    result: Dict[str, Any] = {"machine": machine(), "repeats": args.repeats,
                              "sets": {k: {"texts": len(v), "chars": sum(len(t) for t in v),
                                           "words": sum(len(t.split()) for t in v)} for k, v in texts.items()},
                              "texts_file": str(out / "texts.jsonl"), "cases": []}
    names = args.only or list(TOKENIZERS)
    prev = out / "speed_v1.json"
    if args.only and prev.exists():
        # --only re-measures the named tokenizers and keeps every other case of the existing file
        old = json.loads(prev.read_text(encoding="utf-8"))
        result["cases"] = [c for c in old.get("cases", []) if c.get("name") not in names]
        result["machine_previous_runs"] = old.get("machine_previous_runs", []) + [old.get("machine")]
    for name in names:
        typ, path, params = TOKENIZERS[name]
        modes = ["encode"] + (["batch_parallel_off", "batch_parallel_default"] if typ in RUST_TYPES else [])
        if typ in RUST_TYPES:
            modes.insert(1, "encode_parallel_off")
        for mode in modes:
            env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
            if "parallel_off" in mode:
                env["TOKENIZERS_PARALLELISM"] = "false"
            else:
                env.pop("TOKENIZERS_PARALLELISM", None)
            case = {"name": name, "type": typ, "path": path, "params": params, "repeats": args.repeats, "mode": mode,
                    "texts_json": str(tj)}
            print(f"[speed] {name} / {mode} …", flush=True)
            t0 = time.perf_counter()
            p = subprocess.run([sys.executable, __file__, "--worker", json.dumps(case)], env=env, cwd=REPO,
                               capture_output=True, text=True)
            line = next((l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")), None)
            if p.returncode != 0 or line is None:
                err = (p.stderr or "").strip().splitlines()[-3:]
                print(f"[speed] {name} / {mode}: FAILED ({p.returncode}): {' | '.join(err)}", flush=True)
                result["cases"].append({"name": name, "mode": mode, "error": err, "returncode": p.returncode})
            else:
                r = json.loads(line[len("@@RESULT@@"):])
                r["subprocess_wall_sec"] = round(time.perf_counter() - t0, 1)
                r["loadavg_after"] = list(os.getloadavg())
                result["cases"].append(r)
                ev = (r.get("sets") or {}).get("eval_split") or {}
                e = ev.get("encode") or ev.get("encode_batch") or (ev.get("warm") or {})
                print(f"[speed] {name} / {mode}: load {r.get('load_sec')} s, eval encode {e.get('us_per_word')} µs/word, "
                      f"{e.get('chars_per_sec')} chars/s ({r['subprocess_wall_sec']} s)", flush=True)
            (out / "speed_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out / 'speed_v1.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
