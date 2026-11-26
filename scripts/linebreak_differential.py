#!/usr/bin/env python
"""The line-break differential (docs/report.md §3.20; brief ``opus_prompt_paper_finish_2026-11-24.md`` §4).

Our four trained tokenizers cannot write a line break (§3.18). This script asks whether that costs them judge points
on the prompts whose answer should be laid out in paragraphs, from the existing dumps and verdicts only (files in,
files out, no model, no GPU; idempotent).

* **Subsets.** A prompt is *layout-demanding* when its held-out reference contains a ``\\n`` (58 of 250); the
  sensitivity subset needs two or more (40). The complement is the rest.
* **Pairs** (per base model). Shipped versus ours: Jais-2 − AraRooPat, Fanar-1 − AraRooPat, the model as shipped −
  AraRooPat; the control: AraRooPat − BPE-16K (neither arm can write a line break).
* **Δ(S)** = ``paired_bootstrap`` on subset S (10 000 resamples, seed 0). **D** = Δ(layout) − Δ(complement), with a
  percentile interval from resampling the two subsets independently (one ``RandomState(0)``, at each draw the layout
  indices first, then the complement's).
* **Layout from the text alone**: paragraphs (non-empty blocks between line breaks), a list marker at a line start,
  a heading line, and the layout match with the reference (both ≥ 2 paragraphs or both < 2).
* **Rationale scan**: the share of verdicts on the layout-demanding prompts whose rationale mentions formatting.
* **Layout-only judge** (``gptoss_120b_layout_v1``, rubric ``layout_v1``): used when its files exist.
* **The reading rule** of §3.20 (written before any number existed) is applied and its outcome written.

    .venv/bin/python scripts/linebreak_differential.py [--out outputs/experiments/_paper_finish/linebreak]

Outputs: ``per_prompt.parquet`` (one row per prompt × cell), ``linebreak_differential.json`` (every number),
``linebreak_differential.md`` (the tables).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

HELDOUT = REPO / "configs/contamination/freeform_cidar_heldout_v1.jsonl"
EXP = REPO / "outputs/experiments"
MODELS = {
    "qwen3_4b": {"label": "Qwen3-4B-Base", "dir": "qwen_native_vs_araroopat",
                 "native_base": "native_qwen3_base_m2_c2400", "native_sft": "native_qwen3_sft_v5_distill"},
    "llama3b": {"label": "Llama-3.2-3B", "dir": "llama3b_tokenizers",
                "native_base": "native_base_m2_c2400", "native_sft": "native_sft_v5_distill"},
    "gemma3_4b": {"label": "Gemma-3-4B", "dir": "gemma3_4b_tokenizers",
                  "native_base": "native_base_m2_c2400", "native_sft": "native_sft_v5_distill"},
}
TRAINED = {"araroopat": "araroopat_3phase_v5_distill", "bpe_16k": "bpe_16k_3phase_v5_distill",
           "bpe_128k": "bpe_128k_3phase_v5_distill", "farasa_bpe16k": "farasa_bpe16k_3phase_v5_distill",
           "jais2": "jais2_3phase_v5_distill", "fanar1": "fanar1_3phase_v5_distill"}
ARM_LABEL = {"native_base": "model as shipped", "native_sft": "native vocabulary, adapted", "araroopat": "AraRooPat",
             "bpe_16k": "BPE-16K", "bpe_128k": "BPE-128K", "farasa_bpe16k": "Farasa-BPE-16K",
             "jais2": "Jais-2 vocabulary", "fanar1": "Fanar-1 vocabulary", "teacher": "Aya-Expanse-32B (teacher)"}
ARM_ORDER = ["native_base", "native_sft", "araroopat", "bpe_16k", "bpe_128k", "farasa_bpe16k", "jais2", "fanar1",
             "teacher"]
TEACHER = EXP / "qwen_native_vs_araroopat/teacher_aya_expanse_32b_m2_c2400"
OURS = ("araroopat", "bpe_16k", "bpe_128k", "farasa_bpe16k")
JUDGES = ("gptoss_120b", "falcon_h1_34b")
LAYOUT_JUDGE = "gptoss_120b_layout_v1"
PRIMARY = "gptoss_120b"
# (name, other arm, base arm, kind): Δ = other − base
PAIRS = [("jais2-araroopat", "jais2", "araroopat", "shipped_vs_ours"),
         ("fanar1-araroopat", "fanar1", "araroopat", "shipped_vs_ours"),
         ("shipped-araroopat", "native_base", "araroopat", "shipped_vs_ours"),
         ("araroopat-bpe16k", "araroopat", "bpe_16k", "control")]
PAIR_LABEL = {"jais2-araroopat": "Jais-2 − AraRooPat", "fanar1-araroopat": "Fanar-1 − AraRooPat",
              "shipped-araroopat": "model as shipped − AraRooPat", "araroopat-bpe16k": "AraRooPat − BPE-16K (control)"}
N_BOOT = 10_000
SEED = 0

_LIST_RE = re.compile(r"^\s*(?:[-•*]|[0-9٠-٩۰-۹]+[.)])\s*\S")
_FORMAT_AR = ("تنسيق", "فقرات", "فقرة", "قائمة", "نقاط", "ترقيم", "سطر")
_FORMAT_EN = re.compile(r"\b(?:format(?:s|ted|ting)?|paragraphs?|lists?|bullets?|bullet-points?|structure[sd]?|"
                        r"structuring|layouts?)\b", re.IGNORECASE)
_LAYOUT_AR_STRICT = ("تنسيق", "فقرات", "ترقيم", "سطر")
_LAYOUT_EN_STRICT = re.compile(r"\b(?:format(?:s|ted|ting)?|paragraph(?:s|ing|ed)?|bullet(?:s|ed)?|bullet-points?|"
                               r"layouts?|line[- ]breaks?|newlines?)\b", re.IGNORECASE)


# ----------------------------------------------------------------------------------------------------------------
# text rules
# ----------------------------------------------------------------------------------------------------------------

def paragraphs(text: str) -> int:
    """Non-empty blocks separated by one or more line breaks."""
    return sum(1 for b in re.split(r"\n+", text or "") if b.strip())


def has_list(text: str) -> bool:
    return any(_LIST_RE.match(line) for line in (text or "").split("\n"))


def has_heading(text: str) -> bool:
    for line in (text or "").split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            return True
        if s.endswith(":") and len(s.split()) <= 6:
            return True
    return False


def mentions_format(rationale: Optional[str]) -> bool:
    """Broad scan (the brief's word list): also catches content uses such as "a list of facts"."""
    r = rationale or ""
    return any(w in r for w in _FORMAT_AR) or bool(_FORMAT_EN.search(r))


def mentions_layout_strict(rationale: Optional[str]) -> bool:
    """Strict scan: words that only name visual layout (format…, paragraph, bullet, layout, line break, newline)."""
    r = rationale or ""
    return any(w in r for w in _LAYOUT_AR_STRICT) or bool(_LAYOUT_EN_STRICT.search(r))


# ----------------------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------------------

def delta(base: Dict[str, float], other: Dict[str, float], ids: List[str]) -> Dict[str, Any]:
    from arabic_eval.judge.freeform_judge import paired_bootstrap
    b = {i: base[i] for i in ids if i in base and i in other}
    o = {i: other[i] for i in ids if i in base and i in other}
    return paired_bootstrap(b, o, n_boot=N_BOOT, seed=SEED)


def differential(base: Dict[str, float], other: Dict[str, float], ids_a: List[str], ids_b: List[str]) -> Dict[str, Any]:
    import numpy as np
    da = np.array([other[i] - base[i] for i in sorted(ids_a) if i in base and i in other], dtype=float)
    db = np.array([other[i] - base[i] for i in sorted(ids_b) if i in base and i in other], dtype=float)
    rng = np.random.RandomState(SEED)
    boots = np.empty(N_BOOT)
    for k in range(N_BOOT):
        boots[k] = da[rng.randint(0, len(da), len(da))].mean() - db[rng.randint(0, len(db), len(db))].mean()
    return {"n_layout": int(len(da)), "n_complement": int(len(db)),
            "D": round(float(da.mean() - db.mean()), 4),
            "ci_low": round(float(np.percentile(boots, 2.5)), 4), "ci_high": round(float(np.percentile(boots, 97.5)), 4)}


def excludes_zero_positive(d: Dict[str, Any]) -> bool:
    return d["ci_low"] is not None and d["ci_low"] > 0


def excludes_zero(d: Dict[str, Any]) -> bool:
    return d["ci_low"] is not None and (d["ci_low"] > 0 or d["ci_high"] < 0)


# ----------------------------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------------------------

def cells_of(model: str) -> Dict[str, Path]:
    m = MODELS[model]
    d = EXP / m["dir"]
    out = {"native_base": d / m["native_base"], "native_sft": d / m["native_sft"]}
    out.update({k: d / v for k, v in TRAINED.items()})
    out["teacher"] = TEACHER
    return out


def load_generations(cell: Path) -> Dict[str, str]:
    import pyarrow.parquet as pq
    t = pq.read_table(cell / "eval_rows" / "freeform_cidar.parquet", columns=["id", "generation"]).to_pylist()
    return {r["id"]: r["generation"] or "" for r in t}


def load_judge(cell: Path, judge: str) -> Optional[Dict[str, Dict[str, Any]]]:
    import pyarrow.parquet as pq
    p = cell / "freeform_judge" / f"{judge}.parquet"
    if not p.exists():
        return None
    rows = pq.read_table(p, columns=["id", "score", "rationale", "parse_ok"]).to_pylist()
    return {r["id"]: r for r in rows}


def scores(verdicts: Optional[Dict[str, Dict[str, Any]]]) -> Optional[Dict[str, float]]:
    if verdicts is None:
        return None
    return {i: float(v["score"]) for i, v in verdicts.items() if v.get("score") is not None}


# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------

def fmt(x: Optional[float], nd: int = 3, sign: bool = True) -> str:
    if x is None:
        return "—"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−")


def fmt_ci(d: Dict[str, Any], key: str = "delta_mean") -> str:
    if d.get(key) is None:
        return "—"
    return f"{fmt(d[key])} [{fmt(d['ci_low'])}, {fmt(d['ci_high'])}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(EXP / "_paper_finish/linebreak"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    held = [json.loads(line) for line in HELDOUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    ref = {h["id"]: h for h in held}
    all_ids = sorted(ref)
    nl = {i: ref[i]["reference"].count("\n") for i in all_ids}
    subsets = {"ge1": [i for i in all_ids if nl[i] >= 1], "ge2": [i for i in all_ids if nl[i] >= 2]}
    comp = {k: [i for i in all_ids if i not in set(v)] for k, v in subsets.items()}
    ref_par = {i: paragraphs(ref[i]["reference"]) for i in all_ids}

    result: Dict[str, Any] = {
        "script": "scripts/linebreak_differential.py", "heldout_file": str(HELDOUT.relative_to(REPO)),
        "n_prompts": len(all_ids), "n_boot": N_BOOT, "seed": SEED, "primary_judge": PRIMARY,
        "subsets": {k: {"n": len(v), "n_complement": len(comp[k]),
                        "by_stratum": {s: sum(ref[i]["stratum"] == s for i in v) for s in ("short", "medium", "long")}}
                    for k, v in subsets.items()},
        "reference_layout": {k: {"ge2_paragraphs": sum(ref_par[i] >= 2 for i in v),
                                 "list_marker": sum(has_list(ref[i]["reference"]) for i in v),
                                 "heading": sum(has_heading(ref[i]["reference"]) for i in v)}
                             for k, v in subsets.items()},
        "models": {},
    }
    rows: List[Dict[str, Any]] = []

    for model, m in MODELS.items():
        cells = cells_of(model)
        gens = {a: load_generations(c) for a, c in cells.items()}
        verd = {a: {j: load_judge(c, j) for j in JUDGES + (LAYOUT_JUDGE,)} for a, c in cells.items()}
        sc = {a: {j: scores(v) for j, v in verd[a].items()} for a in cells}
        mres: Dict[str, Any] = {"label": m["label"], "cells": {a: str(c.relative_to(REPO)) for a, c in cells.items()},
                                "layout_judge_present": {a: verd[a][LAYOUT_JUDGE] is not None for a in cells}}

        # per-prompt rows
        for i in all_ids:
            for a in cells:
                g = gens[a].get(i, "")
                rp = paragraphs(g)
                row = {"model": model, "arm": a, "id": i, "stratum": ref[i]["stratum"], "ref_line_breaks": nl[i],
                       "ref_paragraphs": ref_par[i], "answer_line_breaks": g.count("\n"), "answer_paragraphs": rp,
                       "answer_list": has_list(g), "answer_heading": has_heading(g),
                       "layout_match": (ref_par[i] >= 2) == (rp >= 2)}
                for j in JUDGES + (LAYOUT_JUDGE,):
                    v = (verd[a][j] or {}).get(i)
                    row[f"score_{j}"] = None if v is None else v.get("score")
                    row[f"format_mention_{j}"] = None if v is None else mentions_format(v.get("rationale"))
                    row[f"layout_mention_strict_{j}"] = None if v is None else mentions_layout_strict(v.get("rationale"))
                rows.append(row)

        # subset means, rule-based layout, rationale scan
        arms: Dict[str, Any] = {}
        for a in cells:
            g = gens[a]
            ent: Dict[str, Any] = {"answers_with_line_break": sum("\n" in g.get(i, "") for i in all_ids)}
            for k, ids in subsets.items():
                ent[k] = {
                    "answers_ge2_paragraphs": sum(paragraphs(g.get(i, "")) >= 2 for i in ids),
                    "answers_list": sum(has_list(g.get(i, "")) for i in ids),
                    "answers_heading": sum(has_heading(g.get(i, "")) for i in ids),
                    "layout_match_share": round(sum((ref_par[i] >= 2) == (paragraphs(g.get(i, "")) >= 2)
                                                    for i in ids) / len(ids), 4),
                    "layout_match_share_among_ge2_par_refs": (
                        round(sum(paragraphs(g.get(i, "")) >= 2 for i in ids if ref_par[i] >= 2)
                              / max(1, sum(ref_par[i] >= 2 for i in ids)), 4)),
                }
                for j in JUDGES + (LAYOUT_JUDGE,):
                    s = sc[a][j]
                    if s is None:
                        continue
                    lay = [s[i] for i in ids if i in s]
                    com = [s[i] for i in comp[k] if i in s]
                    ent[k][f"mean_{j}"] = {"layout": round(sum(lay) / len(lay), 4) if lay else None,
                                           "complement": round(sum(com) / len(com), 4) if com else None,
                                           "all": round(sum(s.values()) / len(s), 4) if s else None}
                    vv = verd[a][j]
                    ent[k][f"format_mention_share_{j}"] = {
                        "layout": round(sum(mentions_format(vv[i].get("rationale")) for i in ids if i in vv) / len(ids), 4),
                        "complement": round(sum(mentions_format(vv[i].get("rationale")) for i in comp[k] if i in vv)
                                            / len(comp[k]), 4)}
                    ent[k][f"layout_mention_strict_share_{j}"] = {
                        "layout": round(sum(mentions_layout_strict(vv[i].get("rationale")) for i in ids if i in vv)
                                        / len(ids), 4),
                        "complement": round(sum(mentions_layout_strict(vv[i].get("rationale")) for i in comp[k] if i in vv)
                                            / len(comp[k]), 4)}
            arms[a] = ent
        mres["arms"] = arms

        # pairs
        pres: Dict[str, Any] = {}
        for name, other, base, kind in PAIRS:
            pe: Dict[str, Any] = {"kind": kind, "other": other, "base": base}
            for j in JUDGES + (LAYOUT_JUDGE,):
                so, sb = sc[other][j], sc[base][j]
                if so is None or sb is None:
                    continue
                je: Dict[str, Any] = {"all": delta(sb, so, all_ids)}
                for k, ids in subsets.items():
                    je[k] = {"layout": delta(sb, so, ids), "complement": delta(sb, so, comp[k]),
                             "differential": differential(sb, so, ids, comp[k])}
                pe[j] = je
            pres[name] = pe
        mres["pairs"] = pres
        result["models"][model] = mres

    # the reading rule (§3.20), primary judge, subset ge1
    verdict: Dict[str, Any] = {"rule": "material for a shipped-versus-ours pair if, under the primary judge on the "
                                       "≥ 1-line-break subset, D > 0 with its 95 % interval above zero on ≥ 2 of 3 "
                                       "models; the control pair must not show a D interval excluding zero on ≥ 2 models",
                               "pairs": {}}
    for name, other, base, kind in PAIRS:
        per = {mo: result["models"][mo]["pairs"][name][PRIMARY]["ge1"]["differential"] for mo in MODELS}
        if kind == "control":
            n = sum(excludes_zero(d) for d in per.values())
            verdict["pairs"][name] = {"kind": kind, "models_excluding_zero": n, "control_clean": n < 2}
        else:
            n = sum(excludes_zero_positive(d) for d in per.values())
            verdict["pairs"][name] = {"kind": kind, "models_D_above_zero": n, "fires": n >= 2}
    control_clean = verdict["pairs"]["araroopat-bpe16k"]["control_clean"]
    fired = [n for n, v in verdict["pairs"].items() if v.get("fires")]
    verdict["control_clean"] = control_clean
    verdict["outcome"] = ("material" if fired and control_clean else
                          "not attributable to layout (control shows a differential)" if fired else
                          "not detectable at this sample size")
    verdict["fired_pairs"] = fired
    result["reading_rule"] = verdict

    import pandas as pd
    pd.DataFrame(rows).to_parquet(out_dir / "per_prompt.parquet", index=False)
    (out_dir / "linebreak_differential.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
    (out_dir / "linebreak_differential.md").write_text(render_md(result), encoding="utf-8")
    print(f"outcome: {verdict['outcome']} (fired: {fired or 'none'}; control clean: {control_clean}) → {out_dir}")
    return 0


def render_md(r: Dict[str, Any]) -> str:
    L: List[str] = ["# The line-break differential", "",
                    f"Generated by `{r['script']}` from `{r['heldout_file']}`, the free-form dumps and the judge files "
                    f"(paired bootstrap {r['n_boot']} resamples, seed {r['seed']}; D = Δ(layout) − Δ(complement), the "
                    "two subsets resampled independently). Δ = first arm − second arm; positive D means the first arm's "
                    "advantage is larger on the layout-demanding prompts.", ""]
    s = r["subsets"]
    L += [f"Subsets: ≥ 1 line break in the reference **{s['ge1']['n']}** prompts "
          f"({s['ge1']['by_stratum']}; complement {s['ge1']['n_complement']}); ≥ 2 line breaks **{s['ge2']['n']}** "
          f"(complement {s['ge2']['n_complement']}). References with ≥ 2 paragraphs / a list marker / a heading on the "
          f"≥ 1 subset: {r['reference_layout']['ge1']}.", ""]
    v = r["reading_rule"]
    parts = []
    for n, p in v["pairs"].items():
        if p["kind"] == "control":
            parts.append(f"{PAIR_LABEL[n]}: D interval excludes zero on {p['models_excluding_zero']} of 3 models")
        else:
            parts.append(f"{PAIR_LABEL[n]}: D interval above zero on {p['models_D_above_zero']} of 3 models")
    L += ["## Reading rule (primary judge, ≥ 1 subset)", "",
          f"**Outcome: {v['outcome']}.** Control clean: {v['control_clean']}. " + "; ".join(parts) + ".", ""]
    for j in JUDGES + (LAYOUT_JUDGE,):
        for k in ("ge1", "ge2"):
            hdr = f"## Pairs — {j}, subset {'≥ 1' if k == 'ge1' else '≥ 2'} line break(s)"
            body = ["| model | pair | Δ all 250 | Δ layout | Δ complement | D [95 % CI] |", "|---|---|---|---|---|---|"]
            any_row = False
            for mo, mres in r["models"].items():
                for name, pe in mres["pairs"].items():
                    if j not in pe:
                        continue
                    any_row = True
                    e = pe[j]
                    d = e[k]["differential"]
                    body.append(f"| {mres['label']} | {PAIR_LABEL[name]} | {fmt_ci(e['all'])} | {fmt_ci(e[k]['layout'])} "
                                f"| {fmt_ci(e[k]['complement'])} | {fmt(d['D'])} [{fmt(d['ci_low'])}, {fmt(d['ci_high'])}] |")
            if any_row:
                L += [hdr, ""] + body + [""]
    k = "ge1"
    n = r["subsets"][k]["n"]
    L += ["## Arms — subset means and layout from the text (≥ 1 subset)", "",
          "| model | arm | answers with a line break (of 250) | ≥ 2 paragraphs (layout subset) | layout match (layout subset) "
          "| list / heading (layout subset) | gpt-oss layout / complement | Falcon layout / complement "
          "| layout judge layout / complement | format mentioned, gpt-oss (layout subset; broad / strict) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for mo, mres in r["models"].items():
        for a in ARM_ORDER:
            if a not in mres["arms"]:
                continue
            e = mres["arms"][a]
            kk = e[k]

            def mm(j: str) -> str:
                x = kk.get(f"mean_{j}")
                return "—" if not x else f"{fmt(x['layout'], 3, False)} / {fmt(x['complement'], 3, False)}"
            fm = kk.get(f"format_mention_share_{PRIMARY}")
            fs = kk.get(f"layout_mention_strict_share_{PRIMARY}")
            fm_s = "—" if not fm else f"{fm['layout']:.3f} / {fs['layout']:.3f}"
            L.append(f"| {mres['label']} | {ARM_LABEL[a]} | {e['answers_with_line_break']} | "
                     f"{kk['answers_ge2_paragraphs']} / {n} | {kk['layout_match_share']:.3f} | "
                     f"{kk['answers_list']} / {kk['answers_heading']} | {mm(PRIMARY)} | {mm('falcon_h1_34b')} | "
                     f"{mm(LAYOUT_JUDGE)} | {fm_s} |")
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
