"""Abstract base class for downstream evaluation tasks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.tokenizers.base import BaseTokenizer


class BaseTask(ABC):
    """Defines a downstream evaluation task: how to load it and how to score it.

    Under the 3-phase training pipeline tasks no longer own a dataloader —
    training data comes from ``arabic_eval.data.finetune_corpora`` (Arabic-
    SQuAD for Phase 1+2; TyDiQA-Arabic + ARCD for Phase 3). Tasks are eval-
    only.
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None:
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
