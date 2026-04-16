"""Abstract base class for downstream evaluation tasks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader

from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.tokenizers.base import BaseTokenizer


class BaseTask(ABC):
    """Defines a downstream task: its data, training loop, and evaluation."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def get_dataloader(
        self,
        tokenizer: BaseTokenizer,
        split: str = "train",
        batch_size: int = 8,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        """Load and tokenize the task's dataset. Returns a DataLoader."""
        ...

    @abstractmethod
    def evaluate(
        self,
        model: BaseModelAdapter,
        tokenizer: BaseTokenizer,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate the model. Returns metric_name -> value dict."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def metric_names(self) -> List[str]:
        ...
