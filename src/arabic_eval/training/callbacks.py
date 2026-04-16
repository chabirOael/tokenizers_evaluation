"""Training callbacks for logging and tracking."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("arabic_eval.training.callbacks")


class WandbCallback:
    """Optional Weights & Biases logging callback."""

    def __init__(self, project: str, entity: Optional[str] = None, config: Optional[dict] = None):
        try:
            import wandb
            self._wandb = wandb
            self._run = wandb.init(project=project, entity=entity, config=config)
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
            self._wandb = None
            self._run = None

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
