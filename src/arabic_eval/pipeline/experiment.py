"""End-to-end experiment orchestrator."""
from __future__ import annotations

import inspect
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from arabic_eval.config import ExperimentConfig
from arabic_eval.data.loader import load_arabic_dataset, extract_texts
from arabic_eval.evaluation.evaluator import Evaluator
from arabic_eval.evaluation.metrics import compute_mei
from arabic_eval.evaluation.reporter import generate_report
from arabic_eval.registry import model_registry, task_registry, tokenizer_registry
from arabic_eval.tasks.lighteval import LightEvalBenchmarkTask
from arabic_eval.training.trainer import Trainer
from arabic_eval.utils.io import ensure_dir, load_json, save_json
from arabic_eval.utils.reproducibility import set_seed

logger = logging.getLogger("arabic_eval.pipeline")

# Ensure all registries are populated
import arabic_eval.tokenizers  # noqa: F401
import arabic_eval.models      # noqa: F401
import arabic_eval.tasks        # noqa: F401


def run_single_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a single experiment: train tokenizer -> adapt model -> train -> evaluate.

    Args:
        config: Fully resolved experiment configuration.

    Returns:
        Combined results dict with intrinsic and downstream metrics.
    """
    set_seed(config.seed, config.deterministic)
    output_dir = Path(config.output_dir)
    ensure_dir(output_dir)

    logger.info("=" * 60)
    logger.info("Experiment: %s", config.name)
    logger.info("Tokenizer: %s (vocab=%s)", config.tokenizer.type, config.tokenizer.vocab_size)
    logger.info("Model: %s", config.model.name_or_path)
    logger.info("Task: %s", config.task.type)
    logger.info("=" * 60)

    # Save config
    save_json(config.model_dump(), output_dir / "config.json")

    # ---------------------------------------------------------------
    # Step 1: Load data
    # ---------------------------------------------------------------
    logger.info("Step 1: Loading dataset...")
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

    # ---------------------------------------------------------------
    # Step 2: Train or load tokenizer
    # ---------------------------------------------------------------
    logger.info("Step 2: Preparing tokenizer '%s'...", config.tokenizer.type)
    tokenizer_cls = tokenizer_registry.get(config.tokenizer.type)
    tokenizer = tokenizer_cls(**config.tokenizer.params)

    if config.tokenizer.load_path:
        logger.info("Loading tokenizer from %s", config.tokenizer.load_path)
        tokenizer.load(config.tokenizer.load_path)
    else:
        logger.info("Training tokenizer on %d texts...", len(train_texts))
        tokenizer.train(
            train_texts,
            vocab_size=config.tokenizer.vocab_size or 32_000,
            **config.tokenizer.params,
        )
        tokenizer.save(config.tokenizer.save_path)
        logger.info("Tokenizer saved to %s", config.tokenizer.save_path)

    # ---------------------------------------------------------------
    # Step 3: Intrinsic evaluation
    # ---------------------------------------------------------------
    results: Dict[str, Any] = {"config": {"tokenizer": config.tokenizer.type,
                                           "vocab_size": config.tokenizer.vocab_size,
                                           "model": config.model.name_or_path,
                                           "task": config.task.type}}
    if config.evaluation.intrinsic_metrics:
        logger.info("Step 3: Running intrinsic tokenizer evaluation...")
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

    # ---------------------------------------------------------------
    # Step 4: Load model and adapt to tokenizer
    # ---------------------------------------------------------------
    logger.info("Step 4: Loading and adapting model...")
    model_cls = model_registry.get(config.model.type)
    model = model_cls(
        model_name_or_path=config.model.name_or_path,
        device=config.model.device,
        dtype=config.model.dtype,
        **config.model.params,
    )
    model.adapt_to_tokenizer(tokenizer)

    # ---------------------------------------------------------------
    # Step 5: Get task and prepare data
    # ---------------------------------------------------------------
    logger.info("Step 5: Preparing task '%s'...", config.task.type)
    task_cls = task_registry.get(config.task.type)
    task = task_cls(config.task.params)

    if config.training.ft.enabled:
        # Pass completion_only_loss only to tasks whose get_dataloader() accepts it
        # (currently only the LightEval MCQ benchmarks). Same signature-gating
        # pattern used for failure_report_dir below.
        sft_kwargs: Dict[str, Any] = {}
        if config.training.completion_only_loss:
            dl_params = inspect.signature(task.get_dataloader).parameters
            if "completion_only_loss" in dl_params:
                sft_kwargs["completion_only_loss"] = True
            else:
                logger.warning(
                    "training.completion_only_loss=true ignored: task '%s' get_dataloader() "
                    "does not accept the kwarg.",
                    config.task.type,
                )

        train_dataloader = task.get_dataloader(
            tokenizer, split="train",
            batch_size=config.training.batch_size,
            max_samples=config.data.max_train_samples,
            shuffle=True,
            **sft_kwargs,
        )
        eval_dataloader = task.get_dataloader(
            tokenizer, split="test",
            batch_size=config.training.batch_size,
            max_samples=config.data.max_eval_samples,
            **sft_kwargs,
        )

        # ---------------------------------------------------------------
        # Step 6: Fine-tune
        # ---------------------------------------------------------------
        logger.info("Step 6: Fine-tuning model...")
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            config=config.training.model_dump(),
            output_dir=str(output_dir / "training"),
        )
        train_results = trainer.train()
        results["training"] = train_results
    else:
        logger.info(
            "Steps 5-6: Fine-tuning skipped (training.ft.enabled=false). "
            "Evaluating pretrained model directly."
        )
        results["training"] = {
            "status": "skipped",
            "reason": "training.ft.enabled=false",
        }

    # ---------------------------------------------------------------
    # Step 7: Downstream evaluation
    # ---------------------------------------------------------------
    if config.evaluation.downstream_metrics:
        logger.info("Step 7: Running downstream evaluation...")
        eval_kwargs: Dict[str, Any] = {
            "split": "test",
            "max_samples": config.evaluation.num_eval_samples,
        }
        eval_params = inspect.signature(task.evaluate).parameters
        # Pass failure_report_dir only to tasks whose evaluate() accepts it
        # (currently only the LightEval MCQ benchmarks).
        if config.evaluation.failure_reports:
            if "failure_report_dir" in eval_params:
                failure_dir = output_dir / "failure_reports"
                ensure_dir(failure_dir)
                eval_kwargs["failure_report_dir"] = failure_dir
            else:
                logger.info(
                    "failure_reports=true but task '%s' does not support "
                    "failure CSV reporting; skipping.", config.task.type,
                )
        # Pass score_normalization only to tasks whose evaluate() accepts it
        # (LightEval MCQ benchmarks). Same signature-gating pattern as
        # failure_report_dir / completion_only_loss.
        if config.evaluation.score_normalization != "char":
            if "score_normalization" in eval_params:
                eval_kwargs["score_normalization"] = (
                    config.evaluation.score_normalization
                )
            else:
                logger.warning(
                    "evaluation.score_normalization=%r ignored: task '%s' "
                    "evaluate() does not accept the kwarg.",
                    config.evaluation.score_normalization, config.task.type,
                )

        # Warm any lazy tokenizer backends (e.g. araroopat's CAMeL bridge,
        # Farasa subprocess) before the timer starts. Otherwise subprocess
        # spawn / DB load gets billed to inference_time on the first call,
        # unfairly penalizing the affected tokenizers in MEI.
        try:
            tokenizer.encode("نص قصير للإحماء")
        except Exception as e:  # noqa: BLE001
            logger.warning("Tokenizer warmup failed (non-fatal): %s", e)

        t0 = time.perf_counter()
        downstream_metrics = task.evaluate(model, tokenizer, **eval_kwargs)
        inference_time_sec = time.perf_counter() - t0
        downstream_metrics = dict(downstream_metrics)
        downstream_metrics["inference_time_sec"] = round(inference_time_sec, 4)
        results["downstream"] = {config.task.type: downstream_metrics}
        logger.info(
            "Inference time for task '%s': %.2fs", config.task.type, inference_time_sec
        )

        # ---------------------------------------------------------------
        # Step 8: Composite metric — MEI (LightEval MCQ tasks only)
        # ---------------------------------------------------------------
        intrinsic_block = results.get("intrinsic", {}) or {}
        # Prefer PMI accuracy when available — it removes the per-letter /
        # per-text prior bias and is the corrected scoring. Fall back to
        # ``accuracy`` (which aliases char-norm under "char+pmi" and is the
        # only accuracy under "char"-only mode).
        if "accuracy_pmi" in downstream_metrics:
            mei_accuracy = downstream_metrics["accuracy_pmi"]
            mei_accuracy_source = "accuracy_pmi"
        else:
            mei_accuracy = downstream_metrics.get("accuracy")
            mei_accuracy_source = "accuracy"
        mei_record = compute_mei(
            accuracy=mei_accuracy,
            rps=intrinsic_block.get("root_conservation_rate"),
            compression=intrinsic_block.get("compression_ratio"),
            inference_time_sec=inference_time_sec,
            num_eval_rows=downstream_metrics.get("num_samples"),
            is_lighteval_mcq=isinstance(task, LightEvalBenchmarkTask),
            accuracy_source=mei_accuracy_source,
        )
        results["mei"] = mei_record
        logger.info(
            "MEI: %s (status=%s)", mei_record["mei"], mei_record["status"]
        )

    # Save combined results
    save_json(results, output_dir / "all_metrics.json")
    logger.info("Experiment '%s' completed. Results: %s", config.name, output_dir)
    return results


def run_sweep(config: ExperimentConfig) -> Dict[str, Dict[str, Any]]:
    """Run a sweep over multiple tokenizer/task/vocab_size combinations.

    Args:
        config: Experiment config with a ``sweep`` section.

    Returns:
        {experiment_name: results_dict} for all combinations.
    """
    if config.sweep is None:
        raise ValueError("No sweep config provided. Use run_single_experiment instead.")

    all_results: Dict[str, Dict[str, Any]] = {}

    for tok_config in config.sweep.tokenizers:
        for vocab_size in tok_config.vocab_sizes:
            for task_config in config.sweep.tasks:
                # Build experiment name
                vs_str = f"_{vocab_size // 1000}k" if vocab_size else ""
                exp_name = f"{tok_config.type}{vs_str}_{task_config.type}"

                # Create per-experiment config
                exp_config = config.model_copy(deep=True)
                exp_config.name = exp_name
                exp_config.output_dir = str(Path(config.output_dir) / exp_name)
                exp_config.tokenizer.type = tok_config.type
                exp_config.tokenizer.vocab_size = vocab_size
                exp_config.tokenizer.params = tok_config.params
                exp_config.tokenizer.save_path = str(
                    Path("outputs/tokenizers") / f"{tok_config.type}{vs_str}"
                )
                exp_config.task = task_config
                exp_config.sweep = None  # Clear sweep for single run

                # Resume: skip if results already exist
                existing_results = Path(exp_config.output_dir) / "all_metrics.json"
                if existing_results.exists():
                    logger.info("SWEEP: Skipping %s (results already exist)", exp_name)
                    all_results[exp_name] = load_json(existing_results)
                    continue

                logger.info("=" * 60)
                logger.info("SWEEP: Running %s", exp_name)
                logger.info("=" * 60)

                try:
                    results = run_single_experiment(exp_config)
                    all_results[exp_name] = results
                except Exception as e:
                    logger.error("Experiment %s failed: %s", exp_name, e, exc_info=True)
                    all_results[exp_name] = {"error": str(e)}

    # Generate comparison report
    report_path = Path(config.output_dir) / "comparison_report.txt"
    report = generate_report(all_results, report_path)
    logger.info("\n%s", report)

    return all_results
