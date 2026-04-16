"""Abstract base class for LLM model adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from arabic_eval.tokenizers.base import BaseTokenizer


class BaseModelAdapter(ABC):
    """Wraps an LLM so that any tokenizer can be plugged in.

    Responsibilities:
      1. Load the pretrained model
      2. Replace/resize embedding and output layers for the new tokenizer
      3. Provide a uniform forward() and generate() interface
      4. Handle checkpointing
    """

    @abstractmethod
    def __init__(self, model_name_or_path: str, device: str = "auto", **kwargs: Any) -> None:
        ...

    @abstractmethod
    def adapt_to_tokenizer(self, tokenizer: BaseTokenizer) -> None:
        """Swap the model's embedding/output layers to match the tokenizer."""
        ...

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass. Returns dict with at least 'loss' and 'logits'."""
        ...

    @abstractmethod
    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate tokens autoregressively."""
        ...

    @abstractmethod
    def get_trainable_parameters(self) -> list:
        """Return parameters that should be optimized."""
        ...

    @abstractmethod
    def save_checkpoint(self, path: Path | str) -> None:
        ...

    @abstractmethod
    def load_checkpoint(self, path: Path | str) -> None:
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    @property
    @abstractmethod
    def model(self):
        """Access the underlying model."""
        ...
