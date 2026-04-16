"""Training loop with gradient accumulation, mixed precision, and callbacks."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.utils.io import ensure_dir, save_json

logger = logging.getLogger("arabic_eval.training")


@dataclass
class TrainState:
    """Tracks training progress."""
    global_step: int = 0
    epoch: int = 0
    best_metric: float = float("inf")
    patience_counter: int = 0
    train_losses: List[float] = field(default_factory=list)
    eval_metrics: List[Dict[str, float]] = field(default_factory=list)


class Trainer:
    """Training loop orchestrator.

    Args:
        model: The adapted model adapter.
        train_dataloader: Training data.
        eval_dataloader: Evaluation data (optional).
        config: Training configuration dict.
        output_dir: Where to save checkpoints and logs.
    """

    def __init__(
        self,
        model: BaseModelAdapter,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "outputs/training",
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

        cfg = config or {}
        self.num_epochs = cfg.get("num_epochs", 3)
        self.learning_rate = cfg.get("learning_rate", 2e-5)
        self.weight_decay = cfg.get("weight_decay", 0.01)
        self.warmup_ratio = cfg.get("warmup_ratio", 0.1)
        self.max_grad_norm = cfg.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 4)
        self.logging_steps = cfg.get("logging_steps", 50)
        self.eval_steps = cfg.get("eval_steps", 500)
        self.save_steps = cfg.get("save_steps", 500)
        self.save_total_limit = cfg.get("save_total_limit", 2)
        self.bf16 = cfg.get("bf16", True)
        self.fp16 = cfg.get("fp16", False)
        self.early_stopping_patience = cfg.get("early_stopping_patience", None)
        self.early_stopping_metric = cfg.get("early_stopping_metric", "eval_loss")

        # Optimizer
        self.optimizer = AdamW(
            model.get_trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # LR scheduler
        total_steps = len(train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps - warmup_steps, 1)
        )
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # Mixed precision — only on CUDA; fall back gracefully on CPU
        self.device_type = model.device.type
        self.use_amp = (self.fp16 or self.bf16) and self.device_type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.bf16 else torch.float16
        self.scaler = GradScaler(self.device_type, enabled=self.fp16 and self.device_type == "cuda")

        self.state = TrainState()

    def train(self) -> Dict[str, Any]:
        """Run the training loop. Returns final metrics."""
        logger.info(
            "Starting training: %d epochs, %d steps/epoch, grad_accum=%d",
            self.num_epochs, len(self.train_dataloader), self.gradient_accumulation_steps,
        )

        self.model.model.train()
        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.state.epoch = epoch
            epoch_loss = self._train_epoch()

            # End-of-epoch evaluation
            if self.eval_dataloader is not None:
                eval_metrics = self._evaluate()
                self.state.eval_metrics.append(eval_metrics)
                logger.info("Epoch %d — train_loss: %.4f, eval_loss: %.4f",
                           epoch, epoch_loss, eval_metrics.get("eval_loss", 0))

                # Early stopping
                if self.early_stopping_patience:
                    metric_val = eval_metrics.get(self.early_stopping_metric, epoch_loss)
                    if metric_val < self.state.best_metric:
                        self.state.best_metric = metric_val
                        self.state.patience_counter = 0
                        self.model.save_checkpoint(self.output_dir / "best")
                    else:
                        self.state.patience_counter += 1
                        if self.state.patience_counter >= self.early_stopping_patience:
                            logger.info("Early stopping at epoch %d", epoch)
                            break
            else:
                logger.info("Epoch %d — train_loss: %.4f", epoch, epoch_loss)

        elapsed = time.time() - start_time
        # Save final checkpoint
        self.model.save_checkpoint(self.output_dir / "final")

        result = {
            "train_loss": self.state.train_losses[-1] if self.state.train_losses else 0,
            "total_steps": self.state.global_step,
            "training_time_seconds": round(elapsed, 2),
            "epochs_completed": self.state.epoch + 1,
        }
        save_json(result, self.output_dir / "train_results.json")
        return result

    def _train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.model.train()
        total_loss = 0.0
        num_steps = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_dataloader):
            # Move to device
            batch = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with autocast(self.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model.forward(batch)
                loss = outputs["loss"]
                if loss is None:
                    continue
                loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_trainable_parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.state.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_steps += 1

            # Logging
            if self.state.global_step % self.logging_steps == 0 and self.state.global_step > 0:
                avg = total_loss / num_steps
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    "Step %d — loss: %.4f, lr: %.2e", self.state.global_step, avg, lr
                )

            # Periodic save
            if (self.save_steps and self.state.global_step % self.save_steps == 0
                    and self.state.global_step > 0):
                ckpt_path = self.output_dir / f"checkpoint-{self.state.global_step}"
                self.model.save_checkpoint(ckpt_path)
                self._cleanup_checkpoints()

        avg_loss = total_loss / max(num_steps, 1)
        self.state.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation on eval_dataloader."""
        self.model.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in self.eval_dataloader:
            batch = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with autocast(self.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model.forward(batch)

            if outputs["loss"] is not None:
                labels = batch.get("labels", batch.get("input_ids"))
                n_tokens = (labels != -100).sum().item()
                total_loss += outputs["loss"].item() * n_tokens
                total_tokens += n_tokens

        self.model.model.train()
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 100))

        return {
            "eval_loss": round(avg_loss, 6),
            "eval_perplexity": round(perplexity, 4),
        }

    def _cleanup_checkpoints(self) -> None:
        """Keep only the most recent checkpoints."""
        if not self.save_total_limit:
            return
        ckpts = sorted(self.output_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        while len(ckpts) > self.save_total_limit:
            old = ckpts.pop(0)
            import shutil
            shutil.rmtree(old, ignore_errors=True)
