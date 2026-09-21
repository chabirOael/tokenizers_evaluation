"""Abstract base class for downstream evaluation tasks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from arabic_eval.models.base import BaseModelAdapter
from arabic_eval.params_spec import ParamSpec
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

    @classmethod
    def param_spec(cls) -> List[ParamSpec]:
        """The parameters this task reads from ``sweep.tasks[].params``, with
        their type, default and one-line help (see ``arabic_eval.params_spec``).

        The single source of truth for the console's typed task card, the
        ``?`` tooltips, the assistant's field reference, the generated
        ``configs/tasks/<type>.yaml`` and the advisory warnings at validation
        and run start (an unknown key is *warned about*, never fatal — the
        Pydantic field stays ``Dict[str, Any]``). Default: no declared
        parameters, which turns the checks off for this task.
        """
        return []

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
