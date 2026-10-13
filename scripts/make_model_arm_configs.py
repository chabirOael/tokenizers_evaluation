#!/usr/bin/env python
"""Generate the five arm configs of a new base model from the Qwen3-4B arms (brief v6 §5.3), so every training
block is the Qwen3-4B one by construction — only the model and the native tokenizer change.

    .venv/bin/python scripts/make_model_arm_configs.py --model llama3b        # → configs/experiments/models_x_tokenizers/
    .venv/bin/python scripts/make_model_arm_configs.py --model gemma3_4b --check-only
    .venv/bin/python scripts/make_model_arm_configs.py --model llama3b --mcq          # the five *_mcq4096 cells
    .venv/bin/python scripts/make_model_arm_configs.py --model llama3b --rotations    # P4: 25 letter-slot cells

Per model, five cells under ``outputs/experiments/<experiment>/``:
  native_base_m2_c2400        the untrained model, free-form eval only   ← qwen_native_no_presteps_after_LOOP_fix.yaml
  native_sft_v5_distill       Phase 3 only (the v5 block)                ← qwen_native_sft_v5_distill.yaml
  araroopat_3phase_v5_distill Phases 1 + 2 of qwen_araroopat_3phase_v3.yaml + Phase 3 of qwen_araroopat_3phase_v5_distill.yaml
  bpe_16k_3phase_v5_distill   Phases 1 + 2 of qwen_bpe16k_3phase_v3.yaml   + Phase 3 of qwen_bpe16k_3phase_v5_distill.yaml
  bpe_128k_3phase_v5_distill  qwen_bpe128k_3phase_v5_distill.yaml (one run, three phases)
in one run each, the three phases enabled for the from-scratch arms. Tokenizers by ``load_path`` (never retrained);
pools by ``total_words`` (the same fingerprints, so the same packed text as on Qwen3-4B); ``embedding_init`` with
``base_tokenizer: null`` = the model's own; learning rates and batch sizes the Qwen3-4B ones.

After writing, every config is loaded through ``load_config`` and its resolved ``training.phases`` and
``pretraining_mix`` / ``corpus_params`` are compared with the Qwen3-4B source's — the run stops on any difference.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

EXP = REPO / "configs" / "experiments"
OUT = EXP / "models_x_tokenizers"

MODELS = {
    "llama3b": {"label": "Llama-3.2-3B", "experiment": "llama3b_tokenizers",
                "model": {"type": "llama", "name_or_path": "meta-llama/Llama-3.2-3B"},
                "native": {"type": "native_llama", "params": {"model_name_or_path": "meta-llama/Llama-3.2-3B"},
                           "save_path": "outputs/tokenizers/native_llama_3b"}},
    "gemma3_4b": {"label": "Gemma-3-4B", "experiment": "gemma3_4b_tokenizers",
                  "model": {"type": "gemma3", "name_or_path": "google/gemma-3-4b-pt"},
                  "native": {"type": "native_gemma", "params": {"model_name_or_path": "google/gemma-3-4b-pt"},
                             "save_path": "outputs/tokenizers/native_gemma3_4b"}},
}

# cell → (Phase 1/2 source, Phase 3 source)
ARMS = {
    "native_base_m2_c2400": ("qwen_native_no_presteps_after_LOOP_fix.yaml", None),
    "native_sft_v5_distill": ("qwen_native_sft_v5_distill.yaml", None),
    "araroopat_3phase_v5_distill": ("qwen_araroopat_3phase_v3.yaml", "qwen_araroopat_3phase_v5_distill.yaml"),
    "bpe_16k_3phase_v5_distill": ("qwen_bpe16k_3phase_v3.yaml", "qwen_bpe16k_3phase_v5_distill.yaml"),
    "bpe_128k_3phase_v5_distill": ("qwen_bpe128k_3phase_v5_distill.yaml", None),
}


def load(name: str) -> Dict[str, Any]:
    return yaml.safe_load((EXP / name).read_text(encoding="utf-8"))


def build(model_key: str, cell: str) -> Dict[str, Any]:
    m = MODELS[model_key]
    p12_src, p3_src = ARMS[cell]
    if p3_src is None:
        cfg = copy.deepcopy(load(p12_src))
    else:
        cfg = copy.deepcopy(load(p3_src))                        # tokenizer (load_path), corpus_params, Phase 3, eval, sweep
        v3 = load(p12_src)
        for ph in ("embedding_alignment", "warmup"):
            cfg["training"]["phases"][ph] = copy.deepcopy(v3["training"]["phases"][ph])
            cfg["training"]["phases"][ph]["enabled"] = True
        cfg["training"]["pretraining_mix"] = copy.deepcopy(v3["training"]["pretraining_mix"])
        cfg["model"] = copy.deepcopy(v3["model"])                # embedding_init: surface_avg / char_len
    native = cell.startswith("native_")
    model = cfg.setdefault("model", {})
    model["type"] = m["model"]["type"]
    model["name_or_path"] = m["model"]["name_or_path"]
    if "embedding_init" in model:
        model["embedding_init"]["base_tokenizer"] = None         # = model.name_or_path: the model's own tokenizer
    if native:
        tok = cfg["tokenizer"]
        tok["type"] = m["native"]["type"]
        tok["params"] = dict(m["native"]["params"])
        tok["save_path"] = m["native"]["save_path"]
        tok["load_path"] = None
        tok.pop("vocab_size", None)
        for t in (cfg.get("sweep") or {}).get("tokenizers") or []:
            t["type"] = m["native"]["type"]
            t["params"] = dict(m["native"]["params"])
    exp = cfg.setdefault("experiment", {})
    exp["name"] = f"{model_key}_{cell}"
    exp["output_dir"] = f"outputs/experiments/{m['experiment']}/{cell}"
    exp["description"] = (f"{m['label']} — {cell}: generated by scripts/make_model_arm_configs.py from "
                          f"{p12_src}{' + ' + p3_src if p3_src else ''} (brief v6 §5.3); only the model"
                          f"{' and the native tokenizer' if native else ''} differ from the Qwen3-4B arm")
    exp.pop("created_at", None)
    exp.pop("runs", None)
    return cfg


# MCQ eval-only cells (brief v6 §5.3: "the five *_mcq4096 cells"), from the Qwen3-4B templates of mcq_v5/
MCQ_TEMPLATES = {
    "native_base_m2_c2400": "mcq_v5/mcq_native_base.yaml",
    "native_sft_v5_distill": "mcq_v5/mcq_native_v5.yaml",
    "araroopat_3phase_v5_distill": "mcq_v5/mcq_araroopat_v5.yaml",
    "bpe_16k_3phase_v5_distill": "mcq_v5/mcq_bpe16k_v5.yaml",
    "bpe_128k_3phase_v5_distill": "mcq_v5/mcq_bpe128k_v5.yaml",
}


def build_mcq(model_key: str, cell: str) -> Dict[str, Any]:
    m = MODELS[model_key]
    cfg = copy.deepcopy(load(MCQ_TEMPLATES[cell]))
    native = cell.startswith("native_")
    cfg["model"]["type"] = m["model"]["type"]
    cfg["model"]["name_or_path"] = (m["model"]["name_or_path"] if cell == "native_base_m2_c2400"
                                    else f"outputs/experiments/{m['experiment']}/{cell}/training/sft")
    if native:
        tok = cfg["tokenizer"]
        tok["type"] = m["native"]["type"]
        tok["params"] = dict(m["native"]["params"])
        tok["save_path"] = m["native"]["save_path"]
        tok["load_path"] = None
        for t in (cfg.get("sweep") or {}).get("tokenizers") or []:
            t["type"] = m["native"]["type"]
            t["params"] = dict(m["native"]["params"])
    name = "native_base_mcq4096" if cell == "native_base_m2_c2400" else f"{cell}_mcq4096"
    exp = cfg["experiment"]
    exp["name"] = f"{model_key}_{name}"
    exp["output_dir"] = f"outputs/experiments/{m['experiment']}/{name}"
    exp["description"] = (f"{m['label']} — MCQ benchmarks of {cell} (eval only, the {MCQ_TEMPLATES[cell]} protocol: "
                          f"max_length 4096, 3-shot, ACVA 0-shot, char+pmi); generated by scripts/make_model_arm_configs.py")
    exp.pop("created_at", None)
    exp.pop("runs", None)
    return cfg


# letter-slot rotation cells (brief v6 P4 / M16: "the rotations for the four trained arms + native base"), from the
# Qwen3-4B templates of mcq_letter_slot/ (docs/report.md §3.11): rotations 0–3 at 3-shot + a 0-shot pass, on the
# same 1 000-row subsets (rows_<task>.json), max_length 4096, char+pmi
LETTER_ARMS = {
    "native_base_m2_c2400": "native_base", "native_sft_v5_distill": "native_v5",
    "araroopat_3phase_v5_distill": "araroopat_v5", "bpe_16k_3phase_v5_distill": "bpe16k_v5",
    "bpe_128k_3phase_v5_distill": "bpe128k_v5",
}
LETTER_CONDITIONS = ("rot0", "rot1", "rot2", "rot3", "0shot")
LETTER_OUT = OUT / "letter_slot"


def build_rotation(model_key: str, cell: str, cond: str) -> Dict[str, Any]:
    m = MODELS[model_key]
    arm = LETTER_ARMS[cell]
    tmpl = f"mcq_letter_slot/{arm}_{cond}.yaml"
    cfg = copy.deepcopy(load(tmpl))
    cfg["model"]["type"] = m["model"]["type"]
    cfg["model"]["name_or_path"] = (m["model"]["name_or_path"] if cell == "native_base_m2_c2400"
                                    else f"outputs/experiments/{m['experiment']}/{cell}/training/sft")
    if cell.startswith("native_"):
        tok = cfg["tokenizer"]
        tok["type"] = m["native"]["type"]
        tok["params"] = dict(m["native"]["params"])
        tok["save_path"] = m["native"]["save_path"]
        tok["load_path"] = None
        for t in (cfg.get("sweep") or {}).get("tokenizers") or []:
            t["type"] = m["native"]["type"]
            t["params"] = dict(m["native"]["params"])
    exp = cfg["experiment"]
    exp["name"] = f"{model_key}_{arm}_{cond}"
    exp["output_dir"] = f"outputs/experiments/{m['experiment']}/_letter_slot/{arm}_{cond}"
    exp["description"] = (f"{m['label']} — letter-slot diagnostic of {cell} under {cond} (the {tmpl} protocol); "
                          f"generated by scripts/make_model_arm_configs.py --rotations")
    exp.pop("created_at", None)
    exp.pop("runs", None)
    return cfg


def compare_rotation(cell: str, cond: str, path: Path) -> None:
    tmpl = EXP / f"mcq_letter_slot/{LETTER_ARMS[cell]}_{cond}.yaml"
    new, ref = resolved(path), resolved(tmpl)
    for sec in ("evaluation", "data"):
        assert getattr(new, sec).model_dump() == getattr(ref, sec).model_dump(), f"{cell} {cond}: {sec} differs"
    assert [t.model_dump() for t in new.sweep.tasks] == [t.model_dump() for t in ref.sweep.tasks], f"{cell} {cond}: tasks differ"
    for ph in ("embedding_alignment", "warmup", "sft"):
        assert not getattr(new.training.phases, ph).enabled, f"{cell} {cond}: {ph} enabled in an eval-only config"


def compare_mcq(cell: str, path: Path) -> None:
    new, ref = resolved(path), resolved(EXP / MCQ_TEMPLATES[cell])
    for sec in ("evaluation", "data"):
        assert getattr(new, sec).model_dump() == getattr(ref, sec).model_dump(), f"{cell}: {sec} differs"
    assert [t.model_dump() for t in new.sweep.tasks] == [t.model_dump() for t in ref.sweep.tasks], f"{cell}: tasks differ"
    for ph in ("embedding_alignment", "warmup", "sft"):
        assert not getattr(new.training.phases, ph).enabled, f"{cell}: {ph} enabled in an eval-only config"


def resolved(path: Path) -> Any:
    from arabic_eval.config import load_config
    return load_config(str(path), base_path=str(REPO / "configs" / "base.yaml"))


def compare(model_key: str, cell: str, path: Path) -> None:
    """The resolved training blocks must equal the Qwen3-4B source's."""
    p12_src, p3_src = ARMS[cell]
    new = resolved(path).training
    a = resolved(EXP / p12_src).training
    for ph in ("embedding_alignment", "warmup"):
        x, y = getattr(new.phases, ph).model_dump(), getattr(a.phases, ph).model_dump()
        if p3_src is not None:
            y["enabled"] = True
        assert x == y, f"{cell}: phase {ph} differs from {p12_src}"
    b = resolved(EXP / (p3_src or p12_src)).training
    assert new.phases.sft.model_dump() == b.phases.sft.model_dump(), f"{cell}: Phase 3 differs"
    assert new.corpus_params == b.corpus_params, f"{cell}: corpus_params differ"
    assert new.pretraining_mix.model_dump() == a.pretraining_mix.model_dump(), f"{cell}: pretraining_mix differs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--mcq", action="store_true", help="generate the five eval-only MCQ configs instead")
    ap.add_argument("--rotations", action="store_true",
                    help="generate the 25 letter-slot rotation configs (5 cells × rot0–3 + 0shot) instead")
    args = ap.parse_args()
    if args.rotations:
        LETTER_OUT.mkdir(parents=True, exist_ok=True)
        for cell, arm in LETTER_ARMS.items():
            for cond in LETTER_CONDITIONS:
                path = LETTER_OUT / f"{args.model}_{arm}_{cond}.yaml"
                if not args.check_only:
                    head = (f"# GENERATED by scripts/make_model_arm_configs.py --model {args.model} --rotations — do not edit by hand.\n"
                            f"# Eval only: the mcq_letter_slot/{arm}_{cond}.yaml protocol with this model's checkpoint and tokenizer.\n")
                    path.write_text(head + yaml.safe_dump(build_rotation(args.model, cell, cond), allow_unicode=True,
                                                          sort_keys=False, width=120), encoding="utf-8")
                compare_rotation(cell, cond, path)
                r = resolved(path)
                print(f"{path.relative_to(REPO)}: valid; {r.model.type} {r.model.name_or_path}; {r.tokenizer.type}; "
                      f"rot {[t.params.get('label_rotation') for t in r.sweep.tasks]} shots {r.evaluation.num_fewshot} → {r.output_dir}")
        return 0
    if args.mcq:
        OUT.mkdir(parents=True, exist_ok=True)
        for cell, tmpl in MCQ_TEMPLATES.items():
            name = "native_base_mcq4096" if cell == "native_base_m2_c2400" else f"{cell}_mcq4096"
            path = OUT / f"{args.model}_{name}.yaml"
            if not args.check_only:
                head = (f"# GENERATED by scripts/make_model_arm_configs.py --model {args.model} --mcq — do not edit by hand. Eval only:\n"
                        f"# the {tmpl} protocol with this model's checkpoint and tokenizer (brief v6 §5.3).\n")
                path.write_text(head + yaml.safe_dump(build_mcq(args.model, cell), allow_unicode=True, sort_keys=False,
                                                      width=120), encoding="utf-8")
            compare_mcq(cell, path)
            r = resolved(path)
            print(f"{path.relative_to(REPO)}: valid; {r.model.type} {r.model.name_or_path}; {r.tokenizer.type} "
                  f"{r.tokenizer.load_path or ''}; tasks {[t.type for t in r.sweep.tasks]} → {r.output_dir}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for cell in ARMS:
        path = OUT / f"{args.model}_{cell}.yaml"
        if not args.check_only:
            cfg = build(args.model, cell)
            p12, p3 = ARMS[cell]
            head = (f"# GENERATED by scripts/make_model_arm_configs.py --model {args.model} — do not edit by hand; edit the\n"
                    f"# Qwen3-4B source(s) and regenerate. Sources: {p12}{' (Phases 1 + 2) + ' + p3 + ' (Phase 3, corpus_params, eval)' if p3 else ''}.\n"
                    f"# Brief v6 §5.3: every training block equals the Qwen3-4B arm's (checked on generation); only the model\n"
                    f"# ({MODELS[args.model]['model']['name_or_path']}){' and the native tokenizer' if cell.startswith('native_') else ''} differ.\n")
            path.write_text(head + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
        compare(args.model, cell, path)
        r = resolved(path)
        print(f"{path.relative_to(REPO)}: valid; model {r.model.type} {r.model.name_or_path}; tokenizer {r.tokenizer.type} "
              f"{r.tokenizer.load_path or ''}; phases " +
              "/".join(str(getattr(r.training.phases, p).enabled)[0] for p in ("embedding_alignment", "warmup", "sft")) +
              f"; → {r.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
