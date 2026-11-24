#!/usr/bin/env python3
"""The report of the human-evaluation campaign v2: human–LLM agreement, human–human agreement, and the arm ranking
under human scores — the three numbers the paper's §4.6 / §6.5 ask for.

    .venv/bin/python scripts/human_eval_v2_report.py                 # progress only until every item has 3 ratings
    .venv/bin/python scripts/human_eval_v2_report.py --min-raters 3   # the default
    .venv/bin/python scripts/human_eval_v2_report.py --force          # unblind early (says so in the report; avoid)

**The blind guard.** Until every item of both sets carries a rating from at least ``--min-raters`` annotators, the script
prints each rater's progress and nothing else: no per-arm mean, no per-item cell. Reading the arm means while the
rating is under way would let a rater's later scores drift toward the known ranking.

What it computes once the guard lifts (per model and pooled):
- human–LLM agreement: for each rater and each judge, quadratic-weighted κ, Spearman and exact agreement on the rated
  items, pooled and per arm; the same for the human consensus (mean of the raters, rounded to the scale);
- human–human agreement: pairwise QWK and Fleiss' κ over the items every rater scored;
- the ranking: per-arm human mean ± SE, and the paired differences AraRooPat − BPE-16K, Farasa-BPE-16K − BPE-16K and
  AraRooPat − Farasa-BPE-16K under the human consensus, each with a 95 % bootstrap interval (10 000 resamples, seed 0),
  next to the same differences under every judge on the same prompts.
Writes outputs/experiments/_human_eval_v2/report.{md,json}.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arabic_eval.judge.freeform_judge import exact_agreement, paired_bootstrap, quadratic_weighted_kappa, spearman  # noqa: E402
from arabic_eval.tools.freeform_rows_browser import load_cell  # noqa: E402

NAME = "human_v2"
MODELS = {"qwen": "qwen_native_vs_araroopat", "llama3b": "llama3b_tokenizers"}
ARMS = {"araroopat_3phase_v5_distill": "AraRooPat", "bpe_16k_3phase_v5_distill": "BPE-16K", "farasa_bpe16k_3phase_v5_distill": "Farasa-BPE-16K"}
PAIRS = [("araroopat_3phase_v5_distill", "bpe_16k_3phase_v5_distill"), ("farasa_bpe16k_3phase_v5_distill", "bpe_16k_3phase_v5_distill"),
         ("araroopat_3phase_v5_distill", "farasa_bpe16k_3phase_v5_distill")]
OUT = ROOT / "outputs/experiments/_human_eval_v2"
COMPLETE_SHARE = 0.95     # a rater counts as complete from 95 % of the items (57 of 60); every metric uses only items all raters scored


def fleiss_kappa(rows: list[list[int]], categories=(1, 2, 3, 4, 5)) -> float | None:
    """rows = per item, the list of integer scores (equal length k per item)."""
    if not rows:
        return None
    k = len(rows[0])
    if k < 2 or any(len(r) != k for r in rows):
        return None
    n = len(rows)
    p_j = {c: sum(r.count(c) for r in rows) / (n * k) for c in categories}
    p_i = [(sum(r.count(c) ** 2 for c in categories) - k) / (k * (k - 1)) for r in rows]
    p_bar = sum(p_i) / n
    p_e = sum(v * v for v in p_j.values())
    return None if p_e == 1 else round((p_bar - p_e) / (1 - p_e), 4)


def stats(x, y):
    if not x:
        return {"n": 0, "qwk": None, "spearman": None, "exact": None}
    return {"n": len(x), "qwk": quadratic_weighted_kappa(x, y), "spearman": spearman(x, y), "exact": exact_agreement(x, y)}


def load(model: str):
    folder = MODELS[model]
    exp = ROOT / "outputs/experiments" / folder
    doc = json.loads((exp / "freeform_rating" / f"{NAME}.json").read_text(encoding="utf-8"))
    rdir = exp / "freeform_rating" / f"{NAME}.ratings"
    raters = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (sorted(rdir.glob("*.json")) if rdir.is_dir() else [])}
    cells = {c: load_cell(ROOT, f"outputs/experiments/{folder}/{c}") for c in doc["cells"]}
    judge_of = {}        # item_id -> judge -> score
    for it in doc["items"]:
        row = next(r for r in cells[it["hidden"]["cell"]]["rows"] if r["id"] == it["prompt_id"])
        judge_of[it["item_id"]] = {j: int(v["score"]) for j, v in (row.get("judges") or {}).items()
                                   if v and v.get("score") is not None and not j.endswith("_repeat")}   # repeats are reproducibility runs, not judges
    return doc, raters, judge_of


def main() -> None:
    global NAME, OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--name", default=NAME, help="set name (default human_v2; a probe copy for testing the chain)")
    ap.add_argument("--out", type=Path, default=OUT, help="where report.md / report.json go")
    args = ap.parse_args()
    NAME, OUT = args.name, args.out

    data = {m: load(m) for m in MODELS}
    # ---- the guard
    progress, complete = {}, True
    for m, (doc, raters, _) in data.items():
        n = len(doc["items"])
        progress[m] = {r: f"{len(v)}/{n}" for r, v in raters.items()}
        full = [r for r, v in raters.items() if len(v) >= COMPLETE_SHARE * n]   # a rater who missed an item or two still counts; the metrics use only items every rater scored
        if len(full) < args.min_raters:
            complete = False
    print("progress:", json.dumps(progress, ensure_ascii=False))
    if not complete and not args.force:
        print(f"the sets are not complete (need {args.min_raters} raters with every item rated per set); nothing is unblinded. "
              f"Pass --force only if you accept reading the ranking before the rating ends.")
        return

    report = {"name": NAME, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "forced": bool(args.force and not complete), "models": {}}
    pooled_h, pooled_j = defaultdict(list), defaultdict(lambda: defaultdict(list))
    pooled_pairs = defaultdict(lambda: ({}, {}))
    for m, (doc, raters, judge_of) in data.items():
        n = len(doc["items"])
        full = {r: v for r, v in raters.items() if len(v) >= COMPLETE_SHARE * n} if not args.force else raters
        ids = [it["item_id"] for it in doc["items"] if all(i in v for v in full.values() for i in [it["item_id"]])]
        cell_of = {it["item_id"]: it["hidden"]["cell"] for it in doc["items"]}
        pid_of = {it["item_id"]: it["prompt_id"] for it in doc["items"]}
        consensus = {i: statistics.mean(int(full[r][i]["score"]) for r in full) for i in ids}
        rounded = {i: int(round(consensus[i])) for i in ids}
        judges = sorted({j for i in ids for j in judge_of[i]})
        M = {"n_items": n, "n_rated_by_all": len(ids), "raters": sorted(full), "judges": judges, "rater_vs_judge": {}, "consensus_vs_judge": {},
             "rater_vs_rater": {}, "fleiss_kappa": None, "per_arm": {}, "pairs": {}}
        for r in full:
            M["rater_vs_judge"][r] = {}
            for j in judges:
                sel = [i for i in ids if j in judge_of[i]]
                M["rater_vs_judge"][r][j] = {"pooled": stats([int(full[r][i]["score"]) for i in sel], [judge_of[i][j] for i in sel])}
                for cell, label in ARMS.items():
                    s2 = [i for i in sel if cell_of[i] == cell]
                    M["rater_vs_judge"][r][j][label] = stats([int(full[r][i]["score"]) for i in s2], [judge_of[i][j] for i in s2])
        for j in judges:
            sel = [i for i in ids if j in judge_of[i]]
            M["consensus_vs_judge"][j] = stats([rounded[i] for i in sel], [judge_of[i][j] for i in sel])
            pooled_h[j] += [rounded[i] for i in sel]
            pooled_j[j]["judge"] += [judge_of[i][j] for i in sel]
        names = sorted(full)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                M["rater_vs_rater"][f"{names[a]}|{names[b]}"] = stats([int(full[names[a]][i]["score"]) for i in ids], [int(full[names[b]][i]["score"]) for i in ids])
        M["fleiss_kappa"] = fleiss_kappa([[int(full[r][i]["score"]) for r in names] for i in ids])
        for cell, label in ARMS.items():
            s2 = [i for i in ids if cell_of[i] == cell]
            vals = [consensus[i] for i in s2]
            M["per_arm"][label] = {"n": len(vals), "human_mean": round(statistics.mean(vals), 3) if vals else None,
                                   "human_se": round(statistics.pstdev(vals) / len(vals) ** 0.5, 3) if len(vals) > 1 else None,
                                   "judges": {j: (lambda v: round(statistics.mean(v), 3) if v else None)([judge_of[i][j] for i in s2 if j in judge_of[i]]) for j in judges}}
        M["rater_scale"] = {r: {"mean": round(statistics.mean(int(full[r][i]["score"]) for i in ids), 3),
                                "std": round(statistics.pstdev([int(full[r][i]["score"]) for i in ids]), 3),
                                "share_of_5": round(sum(1 for i in ids if int(full[r][i]["score"]) == 5) / len(ids), 3),
                                "share_of_1": round(sum(1 for i in ids if int(full[r][i]["score"]) == 1) / len(ids), 3)} for r in names}
        for a, b in PAIRS:
            key = f"{ARMS[a]} − {ARMS[b]}"
            ha = {pid_of[i]: consensus[i] for i in ids if cell_of[i] == a}
            hb = {pid_of[i]: consensus[i] for i in ids if cell_of[i] == b}
            M["pairs"][key] = {"human": paired_bootstrap(hb, ha, n_boot=10000, seed=0)}
            for r in names:   # the same difference under each rater alone
                ra = {pid_of[i]: int(full[r][i]["score"]) for i in ids if cell_of[i] == a}
                rb = {pid_of[i]: int(full[r][i]["score"]) for i in ids if cell_of[i] == b}
                M["pairs"][key][f"rater:{r}"] = paired_bootstrap(rb, ra, n_boot=10000, seed=0)
            for j in judges:
                ja = {pid_of[i]: judge_of[i][j] for i in ids if cell_of[i] == a and j in judge_of[i]}
                jb = {pid_of[i]: judge_of[i][j] for i in ids if cell_of[i] == b and j in judge_of[i]}
                M["pairs"][key][j] = paired_bootstrap(jb, ja, n_boot=10000, seed=0)
            pa, pb = pooled_pairs[key]
            pa.update({f"{m}:{k}": v for k, v in ha.items()}); pb.update({f"{m}:{k}": v for k, v in hb.items()})
        report["models"][m] = M
    report["pooled"] = {"consensus_vs_judge": {j: stats(pooled_h[j], pooled_j[j]["judge"]) for j in pooled_h},
                        "pairs_human": {k: paired_bootstrap(b, a, n_boot=10000, seed=0) for k, (a, b) in pooled_pairs.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    # ---- markdown
    L = [f"# Human evaluation v2 — report ({report['created_at']}){'  **FORCED: the sets were not complete**' if report['forced'] else ''}", ""]
    for m, M in report["models"].items():
        L += [f"## {m}: {M['n_rated_by_all']} items rated by all of {', '.join(M['raters'])}; judges {', '.join(M['judges'])}", "",
              f"Fleiss' κ (all raters, 5 categories): **{M['fleiss_kappa']}**; pairwise QWK: " +
              ", ".join(f"{k} {v['qwk']}" for k, v in M["rater_vs_rater"].items()), "", "| arm | human mean ± SE | " + " | ".join(M["judges"]) + " |",
              "|---|---|" + "---|" * len(M["judges"])]
        for label, v in M["per_arm"].items():
            L.append(f"| {label} | {v['human_mean']} ± {v['human_se']} | " + " | ".join(str(v["judges"].get(j)) for j in M["judges"]) + " |")
        L += ["", "Rater scales (mean, std, share of 5s, share of 1s): " + "; ".join(f"{r} {v['mean']} / {v['std']} / {v['share_of_5']} / {v['share_of_1']}" for r, v in M["rater_scale"].items())]
        fmt = lambda s: f"{s['delta_mean']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]" if s.get("delta_mean") is not None else "—"
        L += ["", "| pair | human consensus Δ [95 % CI] | " + " | ".join(M["judges"]) + " |", "|---|---|" + "---|" * len(M["judges"])]
        for k, v in M["pairs"].items():
            L.append(f"| {k} | {fmt(v['human'])} | " + " | ".join(fmt(v[j]) for j in M["judges"]) + " |")
        L += ["", "| pair | " + " | ".join(f"under {r} alone" for r in M["raters"]) + " |", "|---|" + "---|" * len(M["raters"])]
        for k, v in M["pairs"].items():
            L.append(f"| {k} | " + " | ".join(fmt(v[f"rater:{r}"]) for r in M["raters"]) + " |")
        L += ["", "Consensus vs judge (QWK / Spearman / exact): " + "; ".join(f"{j}: {s['qwk']} / {s['spearman']} / {s['exact']}" for j, s in M["consensus_vs_judge"].items()), ""]
        for r, per in M["rater_vs_judge"].items():
            L.append(f"- {r} vs judges, QWK pooled then per arm: " + "; ".join(
                f"{j}: {d['pooled']['qwk']} (" + ", ".join(f"{a} {d[a]['qwk']}" for a in ARMS.values()) + ")" for j, d in per.items()))
        L.append("")
    L += ["## Pooled over both models", "", "Consensus vs judge: " + "; ".join(f"{j}: QWK {s['qwk']}, Spearman {s['spearman']}, exact {s['exact']}" for j, s in report["pooled"]["consensus_vs_judge"].items()), ""]
    for k, s in report["pooled"]["pairs_human"].items():
        L.append(f"- {k} under the human consensus, {s['n']} prompts: {s['delta_mean']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}], win / tie / loss {s['win_rate']} / {s['tie_rate']} / {s['loss_rate']}")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
