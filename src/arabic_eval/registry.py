"""Generic registry pattern for extensibility."""
from __future__ import annotations

from typing import Callable, Dict, Type, TypeVar

T = TypeVar("T")


class Registry:
    """Maps string keys to classes. Used for tokenizers, models, and tasks.

    Usage::

        tokenizer_registry = Registry("tokenizer")

        @tokenizer_registry.register("bpe")
        class BPETokenizer(BaseTokenizer):
            ...

        cls = tokenizer_registry.get("bpe")
        instance = cls(config)
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, Type] = {}

    def register(self, key: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a class under *key*."""

        def decorator(cls: Type[T]) -> Type[T]:
            if key in self._registry:
                raise ValueError(f"{self._name} '{key}' is already registered.")
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> Type:
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"Unknown {self._name} '{key}'. Available: {available}")
        return self._registry[key]

    def list_available(self) -> list[str]:
        return sorted(self._registry.keys())


tokenizer_registry = Registry("tokenizer")
model_registry = Registry("model")
task_registry = Registry("task")
