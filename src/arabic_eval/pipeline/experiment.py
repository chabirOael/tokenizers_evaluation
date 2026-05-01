"""End-to-end experiment orchestrator."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from arabic_eval.config import ExperimentConfig
from arabic_eval.data.loader import load_arabic_dataset, extract_texts
from arabic_eval.evaluation.evaluator import Evaluator
from arabic_eval.evaluation.reporter import generate_report
from arabic_eval.registry import model_registry, task_registry, tokenizer_registry
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

    train_dataloader = task.get_dataloader(
        tokenizer, split="train",
        batch_size=config.training.batch_size,
        max_samples=config.data.max_train_samples,
        shuffle=True,
    )
    eval_dataloader = task.get_dataloader(
        tokenizer, split="test",
        batch_size=config.training.batch_size,
        max_samples=config.data.max_eval_samples,
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

    # ---------------------------------------------------------------
    # Step 7: Downstream evaluation
    # ---------------------------------------------------------------
    if config.evaluation.downstream_metrics:
        logger.info("Step 7: Running downstream evaluation...")
        downstream_metrics = task.evaluate(
            model, tokenizer,
            split="test",
            max_samples=config.evaluation.num_eval_samples,
        )
        results["downstream"] = {config.task.type: downstream_metrics}

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
