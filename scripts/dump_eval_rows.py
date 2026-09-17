#!/usr/bin/env python
"""Re-run the evaluation pass of a finished experiment to produce its eval-row dump.

Experiments that ran before ``evaluation.eval_row_dump`` existed have accuracy
numbers but no per-row record, so the console's Eval-rows tab has nothing to
show for them. This script rebuilds exactly the model that produced those
numbers, re-runs the scoring, and writes
``<cell>/eval_rows/<task>.parquet``. Training is not touched.

    # one cell, every benchmark in its config
    .venv/bin/python scripts/dump_eval_rows.py \
        --cell outputs/experiments/all_tokenizers_sweep_pretrain_mix/native_llama

    # a quick look: 200 rows of one benchmark (minutes, not an hour)
    .venv/bin/python scripts/dump_eval_rows.py --cell <dir> --tasks acva --max-rows 200

    # every cell of a sweep
    .venv/bin/python scripts/dump_eval_rows.py --sweep outputs/experiments/all_tokenizers_sweep

Because scoring is deterministic, the accuracy recomputed here should match
the archived ``all_metrics.json``. The script compares them and says so
either way: a match is a reproducibility check on the archived number, a
mismatch means the cell's inputs have moved (a re-trained tokenizer, a
changed prompt, a benchmark updated on the Hub) and the dump describes
today's code rather than the run you remember.

**Checkpoint loading deliberately avoids** ``LlamaAdapter.load_checkpoint``:
that calls ``from_pretrained`` on the checkpoint directory, which rebuilds a
*vanilla* architecture from its ``config.json``. For CharacterBERT, char-JABER
and Charformer cells the custom embedding and output-head weights would then
be unexpected keys and the replaced modules would stay randomly initialised —
a silently wrong model. Here the adapter is built, adapted to the tokenizer
(which installs those modules), and only then loaded, strictly.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arabic_eval.config import (  # noqa: E402
    EvaluationConfig,
    ModelConfig,
    TaskConfig,
    TokenizerConfig,
)
from arabic_eval.registry import model_registry, task_registry, tokenizer_registry  # noqa: E402
from arabic_eval.utils.io import ensure_dir, load_json  # noqa: E402
from arabic_eval.utils.reproducibility import set_seed  # noqa: E402

import arabic_eval.models  # noqa: E402,F401 — populate registries
import arabic_eval.tasks  # noqa: E402,F401
import arabic_eval.tokenizers  # noqa: E402,F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dump_eval_rows")

PHASE_ORDER = ("sft", "warmup", "embedding_alignment")


def _rel(path: Path) -> str:
    """Repo-relative when possible — the console addresses dumps that way."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class CellSpec:
    """The parts of an archived ``config.json`` this script actually uses.

    Deliberately *not* a full ``ExperimentConfig``. An archived config was
    written by the code of its day and may no longer validate against a
    schema that has since gained a required field — the pretraining-mix
    phases are the live example. Re-validating a training block we will never
    run would refuse to dump perfectly good experiments, so each section is
    validated on its own and the training block is read as plain data.
    """

    def __init__(self, cell_dir: Path) -> None:
        path = cell_dir / "config.json"
        if not path.exists():
            raise SystemExit(f"no config.json in {cell_dir} — is this an experiment output dir?")
        self.raw: Dict[str, Any] = load_json(path)
        self.cell_dir = cell_dir
        try:
            self.tokenizer = TokenizerConfig(**(self.raw.get("tokenizer") or {}))
            self.model = ModelConfig(**(self.raw.get("model") or {}))
            self.evaluation = EvaluationConfig(**(self.raw.get("evaluation") or {}))
            self.tasks = [TaskConfig(**t) for t in ((self.raw.get("sweep") or {}).get("tasks") or [])]
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"{path}: cannot read the eval half of this config — {e}") from e
        self.seed = int(self.raw.get("seed") or 42)
        if not self.tasks:
            raise SystemExit(f"{path} declares no sweep.tasks — nothing to evaluate")

    def phase_enabled(self, phase: str) -> bool:
        phases = (self.raw.get("training") or {}).get("phases") or {}
        return bool((phases.get(phase) or {}).get("enabled"))


def pick_checkpoint(cell_dir: Path, spec: "CellSpec", want: Optional[str]) -> Optional[Path]:
    """The checkpoint the eval pass would have run against: the last enabled
    phase that actually saved one."""
    training = cell_dir / "training"
    if want:
        p = training / want
        if not (p / "model.safetensors").exists():
            raise SystemExit(f"no checkpoint at {p}")
        return p
    for phase in PHASE_ORDER:
        p = training / phase
        if spec.phase_enabled(phase) and (p / "model.safetensors").exists():
            return p
    return None


def load_state_into(adapter, checkpoint: Path) -> None:
    """Load a phase checkpoint into an already-adapted model.

    Tied-embedding models (Llama-3.2-1B, Qwen3-4B-Base) do not store
    ``lm_head.weight`` — it is the same tensor as ``embed_tokens.weight`` —
    so that one key is allowed to be missing and the weights are re-tied
    afterwards. Anything else missing, and any unexpected key at all, is
    fatal: a key that does not land is a layer left at its random init.
    """
    from safetensors.torch import load_file

    state = load_file(str(checkpoint / "model.safetensors"))
    missing, unexpected = adapter.model.load_state_dict(state, strict=False)
    tied = bool(getattr(adapter.model.config, "tie_word_embeddings", False))
    allowed_missing = {"lm_head.weight"} if tied else set()
    real_missing = [k for k in missing if k not in allowed_missing]
    if unexpected:
        raise SystemExit(
            f"checkpoint {checkpoint} has {len(unexpected)} key(s) the adapted model "
            f"does not expect, e.g. {unexpected[:3]} — the tokenizer/model pair does "
            f"not match this checkpoint."
        )
    if real_missing:
        raise SystemExit(
            f"checkpoint {checkpoint} is missing {len(real_missing)} key(s) the model "
            f"needs, e.g. {real_missing[:3]} — refusing to evaluate a partly random model."
        )
    # Re-tie only when the model really is the plain tied pair: the key was
    # missing *because* it is tied. The CharacterBERT / char-JABER /
    # Charformer branches replace both modules with ones that have no
    # ``weight`` to tie (and store both explicitly in the checkpoint), so
    # ``tie_weights()`` there would raise on an embedding it cannot find.
    retied = tied and "lm_head.weight" in missing
    if retied:
        adapter.model.tie_weights()
        head = adapter.model.lm_head.weight
        embed = adapter.model.model.embed_tokens.weight
        if head.data_ptr() != embed.data_ptr():
            raise SystemExit(
                f"{checkpoint}: lm_head did not re-tie to embed_tokens — the output "
                f"head would be running on weights the checkpoint never trained."
            )
    log.info("loaded %s (%d tensors%s)", checkpoint, len(state),
             ", lm_head re-tied" if retied else "")


def build_tokenizer(cfg: "CellSpec"):
    tok_cls = tokenizer_registry.get(cfg.tokenizer.type)
    tokenizer = tok_cls(**cfg.tokenizer.params)
    path = cfg.tokenizer.load_path or cfg.tokenizer.save_path
    full = REPO_ROOT / path
    if not full.exists():
        raise SystemExit(
            f"tokenizer directory {path} is gone — it is needed to reproduce the eval "
            f"(re-train it with scripts/train_tokenizer.py or point load_path elsewhere)."
        )
    tokenizer.load(str(full))
    log.info("tokenizer %s loaded from %s (vocab=%s)",
             cfg.tokenizer.type, path, tokenizer.vocab_size)
    return tokenizer


def dump_cell(
    cell_dir: Path,
    only_tasks: Optional[List[str]],
    max_rows: Optional[int],
    phase: Optional[str],
    device: Optional[str],
    overwrite: bool,
) -> Dict[str, Any]:
    cell_dir = Path(cell_dir).resolve()
    cfg = CellSpec(cell_dir)
    if device:
        cfg.model.device = device
    set_seed(cfg.seed)

    tasks = [t for t in cfg.tasks if not only_tasks or t.type in only_tasks]
    if not tasks:
        raise SystemExit(
            f"no matching tasks in {cell_dir} "
            f"(config has: {', '.join(t.type for t in cfg.tasks)})"
        )
    out_dir = ensure_dir(cell_dir / "eval_rows")
    pending = [
        t for t in tasks
        if overwrite or not (out_dir / f"{t.type}.parquet").exists()
    ]
    if not pending:
        log.info("%s: every requested dump already exists (use --overwrite to redo)", cell_dir)
        return {"cell": str(cell_dir), "skipped": True}

    tokenizer = build_tokenizer(cfg)
    adapter = model_registry.get(cfg.model.type)(
        model_name_or_path=cfg.model.name_or_path,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
        **cfg.model.params,
    )
    adapter.adapt_to_tokenizer(tokenizer)
    checkpoint = pick_checkpoint(cell_dir, cfg, phase)
    if checkpoint is None:
        log.warning("%s: no phase checkpoint found — evaluating the *untrained* "
                    "adapted model; the dump will not match the archived numbers",
                    cell_dir)
    else:
        load_state_into(adapter, checkpoint)

    archived = {}
    metrics_path = cell_dir / "all_metrics.json"
    if metrics_path.exists():
        archived = (load_json(metrics_path).get("downstream") or {})

    report: Dict[str, Any] = {"cell": str(cell_dir), "checkpoint": str(checkpoint or ""),
                              "tasks": {}}
    for task_cfg in pending:
        params = dict(task_cfg.params)
        params.setdefault("num_fewshot", cfg.evaluation.num_fewshot)
        task = task_registry.get(task_cfg.type)(params)
        cap = max_rows if max_rows is not None else cfg.evaluation.num_eval_samples
        log.info("=== %s / %s (max_rows=%s) ===", cell_dir.name, task_cfg.type, cap)
        t0 = time.perf_counter()
        metrics = task.evaluate(
            adapter, tokenizer,
            max_samples=cap,
            score_normalization=cfg.evaluation.score_normalization,
            row_dump_dir=out_dir,
        )
        elapsed = time.perf_counter() - t0
        entry: Dict[str, Any] = {
            "num_samples": metrics.get("num_samples"),
            "accuracy": metrics.get("accuracy"),
            "seconds": round(elapsed, 1),
            "path": _rel(out_dir / f"{task_cfg.type}.parquet"),
        }
        before = archived.get(task_cfg.type) or {}
        if before.get("accuracy") is not None and metrics.get("accuracy") is not None:
            delta = float(metrics["accuracy"]) - float(before["accuracy"])
            entry["archived_accuracy"] = before["accuracy"]
            entry["delta"] = round(delta, 6)
            partial = cap is not None and before.get("num_samples") not in (None, metrics.get("num_samples"))
            if partial:
                log.info("  %s: %.4f on %s rows (archived %.4f on %s rows — partial dump, "
                         "not comparable)", task_cfg.type, metrics["accuracy"],
                         metrics.get("num_samples"), before["accuracy"], before.get("num_samples"))
            elif abs(delta) < 1e-4:
                log.info("  %s: %.4f — reproduces the archived number", task_cfg.type, metrics["accuracy"])
            else:
                log.warning("  %s: %.4f vs archived %.4f (delta %+.4f) — the inputs have "
                            "moved since that run; this dump describes today's code",
                            task_cfg.type, metrics["accuracy"], before["accuracy"], delta)
        report["tasks"][task_cfg.type] = entry
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cell", help="one experiment output dir (holding config.json)")
    src.add_argument("--sweep", help="a sweep dir; every cell under it is dumped")
    ap.add_argument("--tasks", help="comma-separated benchmark keys (default: all in the config)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap rows per benchmark — a fast preview of the tab")
    ap.add_argument("--phase", choices=PHASE_ORDER, default=None,
                    help="checkpoint to evaluate (default: last enabled phase that saved one)")
    ap.add_argument("--device", default=None, help='override model device, e.g. "cuda:0"')
    ap.add_argument("--overwrite", action="store_true", help="redo dumps that already exist")
    args = ap.parse_args()

    only = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    if args.cell:
        cells = [Path(args.cell)]
    else:
        root = Path(args.sweep)
        cells = sorted(p.parent for p in root.glob("*/config.json"))
        if not cells:
            raise SystemExit(f"no cells with a config.json under {root}")
        log.info("sweep %s: %d cell(s)", root, len(cells))

    reports = []
    for cell in cells:
        try:
            reports.append(dump_cell(cell, only, args.max_rows, args.phase,
                                     args.device, args.overwrite))
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - one bad cell must not sink the sweep
            log.error("%s failed: %s", cell, e, exc_info=True)
            reports.append({"cell": str(cell), "error": str(e)})
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
