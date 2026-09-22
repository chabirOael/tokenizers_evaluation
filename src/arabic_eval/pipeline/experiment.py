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

from torch.utils.data import DataLoader

from arabic_eval.config import ExperimentConfig, PhaseConfig
from arabic_eval.data.collation import get_collator
from arabic_eval.data.finetune_corpora import build_qa_dataloader, filter_latin_records, load_corpora
from arabic_eval.data.loader import extract_texts, load_arabic_dataset
from arabic_eval.evaluation.evaluator import Evaluator
from arabic_eval.evaluation.metrics import compute_mei
from arabic_eval.evaluation.reporter import generate_report
from arabic_eval.params_spec import specs_by_name, validate_params
from arabic_eval.registry import model_registry, task_registry, tokenizer_registry
from arabic_eval.tasks.lighteval import LightEvalBenchmarkTask
from arabic_eval.tokenizers.provenance import write_training_provenance
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
    corpus_params=None,
    exclusions=None,
):
    """Build the eval loader for SFT (TyDiQA-val + ARCD-val by default).

    Other phases pass ``eval_loader=None`` to ``run_phase``. Only invoked
    when ``phase_cfg.early_stopping`` is set and enabled.
    """
    es = phase_cfg.early_stopping
    if es is None or not es.enabled:
        return None
    eval_records = load_corpora(list(es.eval_splits.keys()), es.eval_splits, corpus_params=corpus_params,
                                exclusions=exclusions)
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


def _phase_eval_mixture_loaders(
    phase_cfg: PhaseConfig,
    tokenizer,
    corpus_params=None,
    exclusions=None,
    manifest_path=None,
):
    """One eval loader per category, composed from the phase corpora's ``dev``
    slices at the phase mixture's ratio (``early_stopping.eval_mixture``).

    Returns ``(loaders, manifest)`` or ``(None, None)`` when the phase does not
    use it — in which case the caller falls back to ``_phase_eval_loader`` and
    nothing about the old path changes.
    """
    es = phase_cfg.early_stopping
    if es is None or not es.enabled or getattr(es, "eval_mixture", None) is None:
        return None, None
    from arabic_eval.config import MixtureConfig
    from arabic_eval.data.sft_mixture import compose_mixture, load_mixture_pools

    em = es.eval_mixture
    train_mix = phase_cfg.mixture
    eval_mix = MixtureConfig(
        total_examples=em.total_examples,
        shares=dict(train_mix.shares),
        within_category=train_mix.within_category,
        weights=dict(train_mix.weights) if train_mix.weights else None,
        upsample=False,                       # a short dev pool is an error, never a repeat
        drop_truncated_answers=train_mix.drop_truncated_answers,
        seed=em.seed,
    )
    datasets = list(phase_cfg.datasets)
    pools, before = load_mixture_pools(datasets, corpus_params, phase_cfg.clean_latin_rows,
                                       exclusions, split="dev")
    encodings, manifest = compose_mixture(
        eval_mix, datasets, pools, tokenizer, phase_cfg.max_length, "answer_only",
        batch_size=phase_cfg.batch_size, pool_sizes_before_filter=before,
        clean_latin_rows=phase_cfg.clean_latin_rows, attach_category=True,
    )
    manifest["split"] = "dev"
    manifest["role"] = "early_stop_eval_mixture"

    from torch.utils.data import DataLoader
    from arabic_eval.data.collation import get_collator
    from arabic_eval.data.sft_mixture import _QATokenizedDataset
    collator = get_collator(tokenizer.embedding_type,
                            pad_token_id=getattr(tokenizer, "pad_token_id", 0),
                            max_length=phase_cfg.max_length)
    loaders = {}
    by_category: Dict[str, list] = {}
    for entry in encodings:
        by_category.setdefault(entry["_category"], []).append(entry)
    for category, entries in sorted(by_category.items()):
        loaders[category] = DataLoader(
            _QATokenizedDataset(entries), batch_size=phase_cfg.batch_size,
            shuffle=False, collate_fn=collator,
        )
    logger.info(
        "early-stop eval mixture: %d dev records — %s",
        len(encodings),
        "; ".join(f"{c} {len(e)}" for c, e in sorted(by_category.items())),
    )
    if manifest_path is not None:
        save_json(manifest, manifest_path)
    return loaders, manifest


def _packed_mix_loader(
    phase_name: str,
    phase_cfg: PhaseConfig,
    packed,
    tokenizer,
    block_size: int,
    qa_packed=None,
) -> "tuple[DataLoader, Dict[str, Any]]":  # noqa: UP037
    """Train loader over the phase's slice of the packed pretraining mix.

    The phase trains on ``mix_tokens / block_size`` blocks. Without a
    ``qa_blend`` they are the next unread range of the raw-text corpus
    (consecutive mix phases never overlap; blocks were shuffled at pack
    time, so the loader runs ``shuffle=False``). With a ``qa_blend``,
    ``round(share × blocks)`` of them are the next unread blocks of the
    packed QA corpus instead, and the two slices are interleaved through a
    seeded permutation (``BlendedBlockDataset``) so every batch is a random
    mixture. The config validator guarantees ``steps × batch_size`` equals
    the block count: every block is read exactly once. The collator is
    dispatched on ``embedding_type`` exactly like ``build_qa_dataloader``;
    a packed block carries ``input_ids`` (+ ``char_ids`` for character_cnn)
    and the collator derives full-sequence causal-LM labels from it.
    """
    from arabic_eval.data.pretraining_mix.packing import BlendedBlockDataset

    total_blocks = phase_cfg.mix_tokens // block_size
    qa_blocks = int(round(phase_cfg.qa_blend.share * total_blocks)) if phase_cfg.qa_blend is not None else 0
    dataset, info = packed.take(phase_name, total_blocks - qa_blocks)
    if qa_blocks:
        assert qa_packed is not None
        qa_dataset, qa_info = qa_packed.take(phase_name, qa_blocks)
        dataset = BlendedBlockDataset([dataset, qa_dataset], seed=phase_cfg.qa_blend.seed)
        info = {
            **info,
            "qa_blend": {
                **qa_info,
                "datasets": list(phase_cfg.qa_blend.datasets),
                "split": phase_cfg.qa_blend.split,
                "share_target": phase_cfg.qa_blend.share,
                "achieved_share": round(qa_blocks / total_blocks, 4),
                "records_covered_approx": int(round(qa_blocks * block_size / max(qa_packed.manifest.get("tokens_per_record", 1.0), 1e-9))),
            },
            "phase_blocks": total_blocks,
            "phase_tokens": total_blocks * block_size,
        }
    collator = get_collator(
        tokenizer.embedding_type,
        pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        max_length=phase_cfg.max_length,
    )
    loader = DataLoader(dataset, batch_size=phase_cfg.batch_size, shuffle=False, collate_fn=collator)
    logger.info(
        "[%s] pretraining_mix: blocks [%d, %d) of %d (%d tokens, %.1f%% of the packed corpus)",
        phase_name, info["block_start"], info["block_end"], info["corpus_blocks"],
        info["n_tokens"], 100 * info["corpus_share"],
    )
    if qa_blocks:
        qb = info["qa_blend"]
        logger.info(
            "[%s] qa_blend: %d of %d blocks (%.1f%%, %d tokens ≈ %d records of %s) — QA blocks [%d, %d) of %d, interleaved",
            phase_name, qa_blocks, total_blocks, 100 * qb["achieved_share"], qb["n_tokens"],
            qb["records_covered_approx"], "+".join(qb["datasets"]), qb["block_start"], qb["block_end"], qb["corpus_blocks"],
        )
    return loader, info


def _run_all_phases(
    adapter,
    tokenizer,
    training_cfg,
    output_dir: Path,
    tokenizer_type: str = "",
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the three phases in sequence. Skipped phases produce a
    ``{"status": "skipped"}`` record.

    A phase whose ``datasets`` is ``["pretraining_mix"]`` trains on the
    packed raw-text corpus (packed once per tokenizer under the pool cache;
    ``data_dir``, default ``{output_dir}/../data/pretraining_mix``, gets a
    manifest pointing at it) instead of QA records — optionally blended
    with packed SFT-format QA text (``qa_blend``, its own cache beside the
    pool); the other phases keep the QA path. ``history[phase]["data"]``
    records which blocks each mix phase consumed.
    """
    history: Dict[str, Any] = {}
    packed = None
    qa_packed = None
    data_dir = Path(data_dir) if data_dir is not None else Path(output_dir).parent / "data" / "pretraining_mix"
    # Committed contamination list: training record ids carrying a held-out
    # evaluation passage, dropped from every train / dev split loaded below
    # (None when training.contamination_exclusions is null).
    from arabic_eval.data.contamination import load_exclusions
    exclusions = load_exclusions(getattr(training_cfg, "contamination_exclusions", None))
    if exclusions is not None:
        logger.info("contamination exclusions: %s (%d record ids over %d corpora)",
                    exclusions.path, exclusions.total(), len(exclusions.ids))
    for phase_name in _PHASE_NAMES:
        phase_cfg: PhaseConfig = getattr(training_cfg.phases, phase_name)
        if not phase_cfg.enabled:
            logger.info("[%s] skipped (enabled=false)", phase_name)
            history[phase_name] = {"status": "skipped"}
            continue

        data_info: Dict[str, Any]
        uses_mix = phase_cfg.datasets == ["pretraining_mix"]
        if uses_mix:
            if packed is None:
                from arabic_eval.data.pretraining_mix.packing import build_packed_corpus
                packed = build_packed_corpus(
                    training_cfg.pretraining_mix, tokenizer, tokenizer_type, cell_data_dir=data_dir,
                )
            if phase_cfg.clean_latin_rows:
                logger.info(
                    "[%s] clean_latin_rows applies to the qa_blend records only (the pool's "
                    "Latin-letter-ratio rule already applied to the raw text)", phase_name,
                )
            if phase_cfg.qa_blend is not None and qa_packed is None:
                from arabic_eval.data.pretraining_mix.packing import pack_qa_blend
                qa_packed = pack_qa_blend(
                    phase_cfg.qa_blend, training_cfg.pretraining_mix.block_size,
                    Path(training_cfg.pretraining_mix.cache_dir), tokenizer, tokenizer_type,
                    clean_latin_rows=phase_cfg.clean_latin_rows, cell_data_dir=data_dir,
                    corpus_params=training_cfg.corpus_params, exclusions=exclusions,
                )
            train_loader, data_info = _packed_mix_loader(
                phase_name, phase_cfg, packed, tokenizer, training_cfg.pretraining_mix.block_size,
                qa_packed=qa_packed,
            )
        elif phase_cfg.mixture is not None:
            # Ratio-controlled composition: exact per-category counts drawn
            # from the phase's corpora (see data/sft_mixture.py). The full
            # manifest (with the record ids drawn) lands beside the cell.
            from arabic_eval.data.sft_mixture import build_mixture_dataloader, manifest_summary
            train_loader, manifest = build_mixture_dataloader(
                phase_cfg.mixture, phase_cfg.datasets, tokenizer,
                batch_size=phase_cfg.batch_size,
                max_length=phase_cfg.max_length,
                loss_target=phase_cfg.loss_target,
                corpus_params=training_cfg.corpus_params,
                clean_latin_rows=phase_cfg.clean_latin_rows,
                exclusions=exclusions,
            )
            manifest_path = data_dir.parent / f"{phase_name}_mixture_manifest.json"
            save_json({"phase": phase_name, **manifest}, manifest_path)
            logger.info("[%s] mixture manifest → %s", phase_name, manifest_path)
            data_info = {
                "datasets": list(phase_cfg.datasets),
                "n_records": manifest["total_examples"],
                "mixture": manifest_summary(manifest),
                "mixture_manifest_path": str(manifest_path),
            }
        else:
            # Build the train loader from the phase's own corpus list.
            train_records = load_corpora(phase_cfg.datasets, splits="train", corpus_params=training_cfg.corpus_params,
                                         exclusions=exclusions)
            if phase_cfg.clean_latin_rows:
                n_before = len(train_records)
                train_records = filter_latin_records(train_records)
                n_after = len(train_records)
                if n_after == 0:
                    raise ValueError(
                        f"[{phase_name}] clean_latin_rows dropped every record (was {n_before})"
                    )
                logger.info(
                    "[%s] clean_latin_rows: dropped %d/%d records (%.1f%% removed)",
                    phase_name, n_before - n_after, n_before,
                    100.0 * (n_before - n_after) / n_before,
                )
            train_loader = build_qa_dataloader(
                train_records, tokenizer,
                batch_size=phase_cfg.batch_size,
                max_length=phase_cfg.max_length,
                loss_target=phase_cfg.loss_target,
                shuffle=True,
            )
            data_info = {"datasets": list(phase_cfg.datasets), "n_records": len(train_records)}
        if exclusions is not None:
            data_info["contamination"] = {"exclusions": str(exclusions.path), "records_excluded": dict(exclusions.dropped)}
        if not uses_mix or phase_cfg.qa_blend is not None:
            # Provenance of the rendered prompt text (QA records, or the QA
            # blend of a mix phase): which template version they were built with.
            from arabic_eval.data.finetune_corpora import TEMPLATE_VERSION
            data_info["template_version"] = TEMPLATE_VERSION

        eval_mix_path = output_dir / "data" / "sft_eval_mixture_manifest.json"
        eval_loaders, eval_mix_manifest = _phase_eval_mixture_loaders(
            phase_cfg, tokenizer, corpus_params=training_cfg.corpus_params,
            exclusions=exclusions, manifest_path=eval_mix_path,
        )
        if eval_loaders is not None:
            eval_loader = None
            data_info["eval_mixture"] = {
                **manifest_summary(eval_mix_manifest),
                "split": "dev",
                "manifest_path": str(eval_mix_path),
            }
        else:
            eval_loader = _phase_eval_loader(phase_cfg, tokenizer, corpus_params=training_cfg.corpus_params,
                                             exclusions=exclusions)

        result: PhaseResult = run_phase(
            phase_name=phase_name,
            adapter=adapter,
            phase_cfg=phase_cfg,
            train_loader=train_loader,
            eval_loader=eval_loader,
            eval_loaders=eval_loaders,
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
            "eval_history": result.eval_history,
            "eval_loss_definition": result.eval_loss_definition,
            "data": data_info,
        }
    if packed is not None:
        history["pretraining_mix"] = {
            "packed_manifest": packed.manifest,
            "consumption": packed.consumption,
        }
        if qa_packed is not None:
            history["pretraining_mix"]["qa_blend"] = {
                "packed_manifest": qa_packed.manifest,
                "consumption": qa_packed.consumption,
            }
    return history


# --------------------------------------------------------------------------
# Single experiment (one tokenizer, multiple tasks)
# --------------------------------------------------------------------------

def _task_params(task_cls: type, params: Dict[str, Any], config: ExperimentConfig) -> Dict[str, Any]:
    """The params a task is constructed with: the YAML's ``sweep.tasks[].params``
    plus the global ``evaluation.num_fewshot`` — injected **only when the task's
    ``param_spec`` declares ``num_fewshot``** (the LightEval MCQ tasks) and the
    YAML did not set it. It used to go into every task; the free-form task
    ignored it and the console flagged it as an undeclared key."""
    task_params = dict(params)
    declared = specs_by_name(task_cls.param_spec())
    if "num_fewshot" in declared:
        task_params.setdefault("num_fewshot", config.evaluation.num_fewshot)
    return task_params


def _warn_task_params(config: ExperimentConfig) -> List[str]:
    """Advisory check of every ``params`` dict against its owner's ``param_spec``
    at run start — before hours of training, not at Step 6: ``sweep.tasks[].params``
    against the task, ``tokenizer.params`` against the tokenizer that runs. Each
    finding is one WARNING line; nothing is fatal (an old YAML with a key a task
    no longer reads keeps running exactly as before, the key is ignored).
    Returns the findings (for tests)."""
    findings: List[str] = []
    for i, task_cfg in enumerate(config.sweep.tasks if config.sweep else []):
        try:
            task_cls = task_registry.get(task_cfg.type)
        except KeyError as e:
            findings.append(f"sweep.tasks[{i}]: {e}")
            continue
        for msg in validate_params(task_cls.param_spec(), task_cfg.params, owner=task_cfg.type):
            findings.append(f"sweep.tasks[{i}].params: {msg}")
    try:
        tok_cls = tokenizer_registry.get(config.tokenizer.type)
    except KeyError as e:
        findings.append(f"tokenizer: {e}")
    else:
        for msg in validate_params(tok_cls.param_spec(), config.tokenizer.params, owner=config.tokenizer.type):
            findings.append(f"tokenizer.params: {msg}")
    for f in findings:
        logger.warning("declared params: %s", f)
    return findings


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
    _warn_task_params(config)

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
        write_training_provenance(
            config.tokenizer.save_path,
            dataset_name=config.data.dataset_name,
            preprocessing=config.data.preprocessing,
            num_texts=len(train_texts),
            entry_point="pipeline.run_experiment",
            tokenizer_type=config.tokenizer.type,
            tokenizer_params=config.tokenizer.params,
        )

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
        intrinsic_unk_csv: Optional[str] = None
        if config.evaluation.intrinsic_unk_report:
            intrinsic_unk_csv = str(output_dir / "intrinsic_unks.parquet")
        results["intrinsic"] = evaluator.run_intrinsic(
            num_samples=config.evaluation.num_eval_samples,
            morphological_metrics=config.evaluation.morphological_metrics,
            morph_sample_size=config.evaluation.morph_sample_size,
            unk_report_path=intrinsic_unk_csv,
        )

    # 4) Load model + adapt
    logger.info("Step 4/7: Loading and adapting model...")
    model_cls = model_registry.get(config.model.type)
    adapter = model_cls(
        model_name_or_path=config.model.name_or_path,
        device=config.model.device,
        dtype=config.model.dtype,
        embedding_init=config.model.embedding_init,
        **config.model.params,
    )
    adapter.adapt_to_tokenizer(tokenizer)

    # 5) Run the 3 phases (each may be skipped via enabled=false)
    logger.info("Step 5/7: Running training phases...")
    training_dir = output_dir / "training"
    ensure_dir(training_dir)
    results["training"] = _run_all_phases(
        adapter, tokenizer, config.training, training_dir,
        tokenizer_type=config.tokenizer.type,
        data_dir=output_dir / "data" / "pretraining_mix",
    )
    init_report = getattr(adapter, "embedding_init_report", None)
    if init_report is not None:
        results["training"]["embedding_init"] = init_report.to_json()

    # 6+7) Evaluate on each benchmark + compute MEI
    if config.evaluation.downstream_metrics:
        logger.info("Step 6-7/7: Evaluating on %d benchmark task(s)...",
                    len(config.sweep.tasks))
        downstream: Dict[str, Any] = {}
        mei_per_task: Dict[str, Any] = {}
        for task_cfg in config.sweep.tasks:
            task_type = task_cfg.type
            task_cls = task_registry.get(task_type)
            task = task_cls(_task_params(task_cls, task_cfg.params, config))

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
            if config.evaluation.downstream_unk_report and "unk_report_dir" in eval_params:
                udir = output_dir / "unk_reports"
                ensure_dir(udir)
                eval_kwargs["unk_report_dir"] = udir
            if config.evaluation.eval_row_dump and "row_dump_dir" in eval_params:
                rdir = output_dir / "eval_rows"
                ensure_dir(rdir)
                eval_kwargs["row_dump_dir"] = rdir

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
    keys = ("accuracy", "accuracy_char_norm", "accuracy_pmi", "f1", "exact_match", "perplexity",
            "chrf", "bertscore_f1", "degenerate_rate")
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
