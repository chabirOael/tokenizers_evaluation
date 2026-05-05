"""Step-driven training loop for the 3-phase pipeline.

One ``run_phase`` core handles all three phases. Differences across phases
(freezing strategy, loss target, learning rate, dataset, early stop) are
captured in the ``PhaseConfig`` instance — the runner doesn't branch on
phase identity.

For Phase 1 (``embedding_alignment``): freeze the body, train only
``embed_tokens`` + ``lm_head``. Phases 2 and 3 unfreeze everything. Phase
3 additionally enables periodic eval + early-stop-on-stagnation; phases
1 and 2 run for a fixed step budget without eval.
"""
from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..config import PhaseConfig
from ..models.base import BaseModelAdapter
from .freezing import apply_trainable_filter

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Result records
# --------------------------------------------------------------------------

@dataclass
class PhaseResult:
    """Outcome of one phase run."""
    phase_name: str
    steps_completed: int
    final_train_loss: float
    train_losses: List[Tuple[int, float]] = field(default_factory=list)  # (step, loss)
    eval_losses: List[Tuple[int, float]] = field(default_factory=list)
    best_eval_loss: Optional[float] = None
    best_eval_step: Optional[int] = None
    early_stopped: bool = False
    checkpoint_path: Optional[str] = None
    wall_time_sec: float = 0.0


# --------------------------------------------------------------------------
# Schedulers
# --------------------------------------------------------------------------

def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Linear warmup over ``warmup_steps``, then schedule for the remainder.

    cosine: cosine decay from 1.0 to 0.0 over the post-warmup portion.
    linear: linear decay from 1.0 to 0.0 over the post-warmup portion.
    constant: no decay (lr stays at peak after warmup).
    """
    decay_steps = max(total_steps - warmup_steps, 1)

    if scheduler_name == "constant":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(warmup_steps, 1)
            return 1.0
    elif scheduler_name == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / decay_steps
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    elif scheduler_name == "linear":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / decay_steps
            return max(0.0, 1.0 - min(progress, 1.0))
    else:
        raise ValueError(f"unknown lr_scheduler {scheduler_name!r}")

    return LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------
# Eval helper
# --------------------------------------------------------------------------

@torch.no_grad()
def _eval_loss(
    adapter: BaseModelAdapter,
    eval_loader: DataLoader,
    bf16: bool,
    fp16: bool,
    max_batches: Optional[int] = None,
) -> float:
    """Compute mean per-batch loss on ``eval_loader``."""
    adapter.model.eval()
    total = 0.0
    count = 0
    autocast_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else None)
    for i, batch in enumerate(eval_loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = _to_device(batch, adapter.device)
        if autocast_dtype is not None:
            with autocast(device_type=adapter.device.type, dtype=autocast_dtype):
                out = adapter.forward(batch)
        else:
            out = adapter.forward(batch)
        total += out["loss"].item()
        count += 1
    adapter.model.train()
    if count == 0:
        return float("nan")
    return total / count


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------
# Phase runner
# --------------------------------------------------------------------------

def run_phase(
    *,
    phase_name: str,
    adapter: BaseModelAdapter,
    phase_cfg: PhaseConfig,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader] = None,
    output_dir: Path,
    bf16: bool = True,
    fp16: bool = False,
    logging_steps: int = 50,
) -> PhaseResult:
    """Run one training phase. See module docstring for the per-phase semantics.

    ``eval_loader`` is required only when ``phase_cfg.early_stopping`` is set
    and enabled. Phase 1 / Phase 2 pass ``eval_loader=None``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    early_stop = phase_cfg.early_stopping
    if early_stop is not None and early_stop.enabled:
        if eval_loader is None:
            raise ValueError(
                f"phase {phase_name!r} has early_stopping enabled but no eval_loader was provided"
            )

    # 1) Freeze / unfreeze
    trainable_names = apply_trainable_filter(adapter.model, phase_cfg.trainable_parameters)
    trainable_params = [p for p in adapter.model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in adapter.model.parameters())
    logger.info(
        "[%s] trainable params: %d / %d (%.2f%%)",
        phase_name, n_trainable, n_total, 100 * n_trainable / max(n_total, 1),
    )

    # 2) Optimizer
    optimizer = AdamW(
        trainable_params,
        lr=phase_cfg.learning_rate,
        weight_decay=phase_cfg.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # 3) Scheduler
    scheduler = _build_lr_scheduler(
        optimizer,
        phase_cfg.lr_scheduler,
        phase_cfg.warmup_steps,
        phase_cfg.steps,
    )

    autocast_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else None)
    device = adapter.device

    # 4) Step-driven loop with grad accum + dataloader looping
    adapter.model.train()
    optimizer.zero_grad()

    train_iter = iter(train_loader)
    accumulated = 0
    last_logged_loss = float("nan")
    best_eval_loss = float("inf")
    best_eval_step = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_counter = 0
    early_stopped = False

    result = PhaseResult(
        phase_name=phase_name,
        steps_completed=0,
        final_train_loss=float("nan"),
    )

    t_start = time.perf_counter()
    for step in range(1, phase_cfg.steps + 1):
        # Cycle the dataloader if exhausted (fixed-step regime)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = _to_device(batch, device)

        if autocast_dtype is not None:
            with autocast(device_type=device.type, dtype=autocast_dtype):
                out = adapter.forward(batch)
                loss = out["loss"] / phase_cfg.gradient_accumulation_steps
        else:
            out = adapter.forward(batch)
            loss = out["loss"] / phase_cfg.gradient_accumulation_steps

        loss.backward()
        accumulated += 1

        if accumulated >= phase_cfg.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(trainable_params, phase_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accumulated = 0

        unscaled = loss.item() * phase_cfg.gradient_accumulation_steps
        last_logged_loss = unscaled
        result.train_losses.append((step, unscaled))

        if step % logging_steps == 0 or step == phase_cfg.steps:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                "[%s] step %d/%d loss=%.4f lr=%.2e",
                phase_name, step, phase_cfg.steps, unscaled, lr_now,
            )

        # Periodic eval + early stop
        if (early_stop is not None and early_stop.enabled
                and step >= early_stop.min_steps_before_stop
                and step % early_stop.eval_every_n_steps == 0):
            eval_loss_value = _eval_loss(adapter, eval_loader, bf16=bf16, fp16=fp16)
            result.eval_losses.append((step, eval_loss_value))
            logger.info(
                "[%s] step %d eval_loss=%.4f (best=%.4f@step %d, patience=%d/%d)",
                phase_name, step, eval_loss_value, best_eval_loss, best_eval_step,
                patience_counter, early_stop.patience,
            )
            if eval_loss_value + early_stop.min_delta < best_eval_loss:
                best_eval_loss = eval_loss_value
                best_eval_step = step
                patience_counter = 0
                if early_stop.restore_best_at_end:
                    best_state = copy.deepcopy({
                        k: v.detach().cpu() for k, v in adapter.model.state_dict().items()
                    })
            else:
                patience_counter += 1
                if patience_counter >= early_stop.patience:
                    logger.info(
                        "[%s] early-stop at step %d (best eval_loss=%.4f at step %d)",
                        phase_name, step, best_eval_loss, best_eval_step,
                    )
                    early_stopped = True
                    result.steps_completed = step
                    break

        result.steps_completed = step

    result.final_train_loss = last_logged_loss
    result.early_stopped = early_stopped
    if best_eval_step > 0:
        result.best_eval_loss = best_eval_loss
        result.best_eval_step = best_eval_step
    result.wall_time_sec = time.perf_counter() - t_start

    # 5) Restore best (only when SFT early-stop is on)
    if best_state is not None:
        adapter.model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info(
            "[%s] restored best checkpoint from step %d (eval_loss=%.4f)",
            phase_name, best_eval_step, best_eval_loss,
        )

    # 6) Save checkpoint
    if phase_cfg.save_checkpoint:
        ckpt_path = output_dir / phase_name
        ckpt_path.mkdir(parents=True, exist_ok=True)
        adapter.save_checkpoint(ckpt_path)
        result.checkpoint_path = str(ckpt_path)
        logger.info("[%s] saved checkpoint to %s", phase_name, ckpt_path)

    logger.info(
        "[%s] complete: steps=%d, final_loss=%.4f, wall=%.1fs%s",
        phase_name,
        result.steps_completed,
        result.final_train_loss,
        result.wall_time_sec,
        f", best_eval={result.best_eval_loss:.4f}@step{result.best_eval_step}"
        if result.best_eval_loss is not None else "",
    )
    return result
