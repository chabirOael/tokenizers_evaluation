"""End-to-end experiment orchestrator (3-phase training pipeline).

Flow per (tokenizer, vocab_size):
  1) Load main Arabic corpus → tokenizer training + intrinsic eval texts.
  2) Train (or load) tokenizer.
  3) Intrinsic evaluation (size/coverage + Arabic morphological metrics).
  4) Load model and adapt embedding/lm_head to the tokenizer.
  5) Phase 1 (embedding_alignment), Phase 2 (warmup), Phase 3 (sft) —
     each runs only if its ``enabled`` flag is true. Phase 3 enables
     periodic eval on TyDiQA-val + ARCD-val and stops on stagnation.
  6) Evaluate the trained model on every task in ``config.sweep.tasks``,
     forcing ``eval_full=True`` so the entire benchmark is the eval set.
  7) Compute MEI per LightEval MCQ task; save combined metrics.

Key invariant: training is task-AGNOSTIC under this pipeline (Phase 3 SFT
uses TyDiQA-Arabic + ARCD, NOT the benchmark). One trained model is
shared across all eval tasks in the sweep.
"""
from __future__ import annotations

import inspect
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from arabic_eval.config import ExperimentConfig, PhaseConfig
from arabic_eval.data.finetune_corpora import build_qa_dataloader, load_corpora
from arabic_eval.data.loader import extract_texts, load_arabic_dataset
from arabic_eval.evaluation.evaluator import Evaluator
from arabic_eval.evaluation.metrics import compute_mei
from arabic_eval.evaluation.reporter import generate_report
from arabic_eval.registry import model_registry, task_registry, tokenizer_registry
from arabic_eval.tasks.lighteval import LightEvalBenchmarkTask
from arabic_eval.training.phases import PhaseResult, run_phase
from arabic_eval.utils.io import ensure_dir, load_json, save_json
from arabic_eval.utils.reproducibility import set_seed

logger = logging.getLogger("arabic_eval.pipeline")

# Ensure registries are populated.
import arabic_eval.tokenizers  # noqa: F401, E402
import arabic_eval.models      # noqa: F401, E402
import arabic_eval.tasks        # noqa: F401, E402


# --------------------------------------------------------------------------
# Phase orchestration
# --------------------------------------------------------------------------

_PHASE_NAMES: List[str] = ["embedding_alignment", "warmup", "sft"]


def _phase_eval_loader(
    phase_cfg: PhaseConfig,
    tokenizer,
):
    """Build the eval loader for SFT (TyDiQA-val + ARCD-val by default).

    Other phases pass ``eval_loader=None`` to ``run_phase``. Only invoked
    when ``phase_cfg.early_stopping`` is set and enabled.
    """
    es = phase_cfg.early_stopping
    if es is None or not es.enabled:
        return None
    eval_records = load_corpora(list(es.eval_splits.keys()), es.eval_splits)
    if not eval_records:
        raise RuntimeError(
            f"Eval split is empty for phase with early_stopping enabled "
            f"(splits={es.eval_splits}); cannot run early-stop logic."
        )
    return build_qa_dataloader(
        eval_records, tokenizer,
        batch_size=phase_cfg.batch_size,
        max_length=phase_cfg.max_length,
        loss_target="answer_only",
        shuffle=False,
    )


def _run_all_phases(
    adapter,
    tokenizer,
    training_cfg,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run the three phases in sequence. Skipped phases produce a
    ``{"status": "skipped"}`` record."""
    history: Dict[str, Any] = {}
    for phase_name in _PHASE_NAMES:
        phase_cfg: PhaseConfig = getattr(training_cfg.phases, phase_name)
        if not phase_cfg.enabled:
            logger.info("[%s] skipped (enabled=false)", phase_name)
            history[phase_name] = {"status": "skipped"}
            continue

        # Build the train loader from the phase's own corpus list.
        train_records = load_corpora(phase_cfg.datasets, splits="train")
        train_loader = build_qa_dataloader(
            train_records, tokenizer,
            batch_size=phase_cfg.batch_size,
            max_length=phase_cfg.max_length,
            loss_target=phase_cfg.loss_target,
            shuffle=True,
        )

        eval_loader = _phase_eval_loader(phase_cfg, tokenizer)

        result: PhaseResult = run_phase(
            phase_name=phase_name,
            adapter=adapter,
            phase_cfg=phase_cfg,
            train_loader=train_loader,
            eval_loader=eval_loader,
            output_dir=output_dir,
            bf16=training_cfg.bf16,
            fp16=training_cfg.fp16,
            logging_steps=training_cfg.logging_steps,
        )
        history[phase_name] = {
            "status": "ok",
            "steps_completed": result.steps_completed,
            "final_train_loss": result.final_train_loss,
            "best_eval_loss": result.best_eval_loss,
            "best_eval_step": result.best_eval_step,
            "early_stopped": result.early_stopped,
            "checkpoint_path": result.checkpoint_path,
            "wall_time_sec": round(result.wall_time_sec, 2),
            # Truncate per-step train losses to the last 200 entries so
            # the JSON stays human-readable.
            "train_losses_tail": result.train_losses[-200:],
            "eval_losses": result.eval_losses,
        }
    return history


# --------------------------------------------------------------------------
# Single experiment (one tokenizer, multiple tasks)
# --------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run one experiment: train tokenizer → intrinsic → 3 phases → eval all tasks.

    ``config.sweep.tasks`` is the list of LightEval benchmarks to evaluate
    after training. Each task is forced into ``eval_full=True`` mode (no
    SFT split, all rows used for eval).
    """
    set_seed(config.seed, config.deterministic)
    output_dir = Path(config.output_dir)
    ensure_dir(output_dir)

    if config.sweep is None or not config.sweep.tasks:
        raise ValueError(
            "ExperimentConfig must declare sweep.tasks (the eval task list)"
        )

    logger.info("=" * 60)
    logger.info("Experiment: %s", config.name)
    logger.info("Tokenizer: %s (vocab=%s)", config.tokenizer.type, config.tokenizer.vocab_size)
    logger.info("Model: %s", config.model.name_or_path)
    logger.info("Eval tasks: %s", [t.type for t in config.sweep.tasks])
    logger.info("=" * 60)

    save_json(config.model_dump(), output_dir / "config.json")

    # 1) Load main Arabic corpus
    logger.info("Step 1/7: Loading Arabic corpus...")
    dataset = load_arabic_dataset(
        dataset_name=config.data.dataset_name,
        dataset_config=config.data.dataset_config,
        cache_dir=config.data.cache_dir,
        max_train_samples=config.data.max_train_samples,
        max_eval_samples=config.data.max_eval_samples,
        preprocessing_config=config.data.preprocessing,
        seed=config.seed,
    )
    train_texts = extract_texts(dataset["train"])
    eval_texts = extract_texts(dataset.get("eval", dataset["train"]))

    # 2) Tokenizer
    logger.info("Step 2/7: Preparing tokenizer '%s'...", config.tokenizer.type)
    tokenizer_cls = tokenizer_registry.get(config.tokenizer.type)
    tokenizer = tokenizer_cls(**config.tokenizer.params)
    if config.tokenizer.load_path:
        logger.info("  loading from %s", config.tokenizer.load_path)
        tokenizer.load(config.tokenizer.load_path)
    else:
        logger.info("  training on %d texts", len(train_texts))
        tokenizer.train(
            train_texts,
            vocab_size=config.tokenizer.vocab_size or 32_000,
            **config.tokenizer.params,
        )
        tokenizer.save(config.tokenizer.save_path)

    # 3) Intrinsic eval
    results: Dict[str, Any] = {
        "config": {
            "tokenizer": config.tokenizer.type,
            "vocab_size": config.tokenizer.vocab_size,
            "model": config.model.name_or_path,
            "tasks": [t.type for t in config.sweep.tasks],
        }
    }
    if config.evaluation.intrinsic_metrics:
        logger.info("Step 3/7: Running intrinsic evaluation...")
        evaluator = Evaluator(
            tokenizer=tokenizer,
            eval_texts=eval_texts,
            output_dir=str(output_dir),
        )
        results["intrinsic"] = evaluator.run_intrinsic(
            num_samples=config.evaluation.num_eval_samples,
            morphological_metrics=config.evaluation.morphological_metrics,
            morph_sample_size=config.evaluation.morph_sample_size,
        )

    # 4) Load model + adapt
    logger.info("Step 4/7: Loading and adapting model...")
    model_cls = model_registry.get(config.model.type)
    adapter = model_cls(
        model_name_or_path=config.model.name_or_path,
        device=config.model.device,
        dtype=config.model.dtype,
        **config.model.params,
    )
    adapter.adapt_to_tokenizer(tokenizer)

    # 5) Run the 3 phases (each may be skipped via enabled=false)
    logger.info("Step 5/7: Running training phases...")
    training_dir = output_dir / "training"
    ensure_dir(training_dir)
    results["training"] = _run_all_phases(
        adapter, tokenizer, config.training, training_dir
    )

    # 6+7) Evaluate on each benchmark + compute MEI
    if config.evaluation.downstream_metrics:
        logger.info("Step 6-7/7: Evaluating on %d benchmark task(s)...",
                    len(config.sweep.tasks))
        downstream: Dict[str, Any] = {}
        mei_per_task: Dict[str, Any] = {}
        for task_cfg in config.sweep.tasks:
            task_type = task_cfg.type
            task_cls = task_registry.get(task_type)
            task = task_cls(dict(task_cfg.params))

            eval_kwargs: Dict[str, Any] = {
                "split": "test",
                "max_samples": config.evaluation.num_eval_samples,
            }
            eval_params = inspect.signature(task.evaluate).parameters
            if config.evaluation.failure_reports and "failure_report_dir" in eval_params:
                fdir = output_dir / "failure_reports"
                ensure_dir(fdir)
                eval_kwargs["failure_report_dir"] = fdir
            if config.evaluation.score_normalization != "char" and "score_normalization" in eval_params:
                eval_kwargs["score_normalization"] = config.evaluation.score_normalization

            # Tokenizer warmup (avoids cold-start charging the timer)
            try:
                tokenizer.encode("نص قصير للإحماء")
            except Exception as e:  # noqa: BLE001
                logger.warning("Tokenizer warmup failed (non-fatal): %s", e)

            t0 = time.perf_counter()
            metrics = task.evaluate(adapter, tokenizer, **eval_kwargs)
            inference_time_sec = time.perf_counter() - t0
            metrics = dict(metrics)
            metrics["inference_time_sec"] = round(inference_time_sec, 4)
            downstream[task_type] = metrics
            logger.info("  [%s] %s (eval=%.1fs)",
                        task_type, _summarize_metrics(metrics), inference_time_sec)

            # MEI per task (LightEval MCQ only)
            if "accuracy_pmi" in metrics:
                mei_acc = metrics["accuracy_pmi"]
                mei_acc_src = "accuracy_pmi"
            else:
                mei_acc = metrics.get("accuracy")
                mei_acc_src = "accuracy"
            intrinsic_block = results.get("intrinsic", {}) or {}
            mei_record = compute_mei(
                accuracy=mei_acc,
                rps=intrinsic_block.get("root_conservation_rate"),
                compression=intrinsic_block.get("compression_ratio"),
                inference_time_sec=inference_time_sec,
                num_eval_rows=metrics.get("num_samples"),
                is_lighteval_mcq=isinstance(task, LightEvalBenchmarkTask),
                accuracy_source=mei_acc_src,
            )
            mei_per_task[task_type] = mei_record

        results["downstream"] = downstream
        results["mei"] = mei_per_task

    save_json(results, output_dir / "all_metrics.json")
    logger.info("Experiment '%s' done -> %s", config.name, output_dir)
    return results


def _summarize_metrics(metrics: Dict[str, Any]) -> str:
    """One-line metrics summary for log readability."""
    keys = ("accuracy", "accuracy_char_norm", "accuracy_pmi", "f1", "exact_match", "perplexity")
    parts = []
    for k in keys:
        if k in metrics and isinstance(metrics[k], (int, float)):
            parts.append(f"{k}={metrics[k]:.4f}")
    if "num_samples" in metrics:
        parts.append(f"n={metrics['num_samples']}")
    return ", ".join(parts) if parts else str(metrics)


# --------------------------------------------------------------------------
# Sweep mode (multiple tokenizers / vocab sizes, shared task list)
# --------------------------------------------------------------------------

def run_sweep(config: ExperimentConfig) -> Dict[str, Dict[str, Any]]:
    """Run ``run_experiment`` for each (tokenizer, vocab_size) cell.

    The task list ``config.sweep.tasks`` is shared across cells (training
    happens once per cell; all tasks evaluate the same trained model).
    """
    if config.sweep is None:
        raise ValueError("run_sweep requires a sweep configuration")

    all_results: Dict[str, Dict[str, Any]] = {}
    sweep_dir = Path(config.output_dir)
    ensure_dir(sweep_dir)

    for tok_cfg in config.sweep.tokenizers:
        for vocab_size in tok_cfg.vocab_sizes:
            vs_str = f"_{vocab_size // 1000}k" if vocab_size else ""
            cell_name = f"{tok_cfg.type}{vs_str}"

            cell_config = config.model_copy(deep=True)
            cell_config.name = cell_name
            cell_config.output_dir = str(sweep_dir / cell_name)
            cell_config.tokenizer.type = tok_cfg.type
            cell_config.tokenizer.vocab_size = vocab_size
            cell_config.tokenizer.params = tok_cfg.params
            cell_config.tokenizer.save_path = str(
                Path("outputs/tokenizers") / cell_name
            )
            # Keep cell_config.sweep intact — run_experiment iterates
            # over sweep.tasks for evaluation.

            existing = Path(cell_config.output_dir) / "all_metrics.json"
            if existing.exists():
                logger.info("SWEEP: skipping %s (results exist)", cell_name)
                all_results[cell_name] = load_json(existing)
                continue

            logger.info("=" * 60)
            logger.info("SWEEP cell: %s", cell_name)
            logger.info("=" * 60)
            try:
                all_results[cell_name] = run_experiment(cell_config)
            except Exception as e:
                logger.error("Cell %s failed: %s", cell_name, e, exc_info=True)
                all_results[cell_name] = {"error": str(e)}

    report_path = sweep_dir / "comparison_report.txt"
    try:
        report = generate_report(all_results, report_path)
        logger.info("\n%s", report)
    except Exception as e:  # noqa: BLE001
        logger.warning("Comparison report generation failed: %s", e)

    return all_results
