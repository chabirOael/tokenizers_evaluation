"""Qwen3 model adapter — a thin registration over :class:`LlamaAdapter`.

``Qwen3ForCausalLM`` exposes exactly the attribute surface ``LlamaAdapter``
relies on — ``model.model.embed_tokens``, ``model.model.layers``,
``model.lm_head``, ``config.hidden_size`` and an ``inputs_embeds`` forward —
so all four embedding branches (standard / character_cnn / char_jaber /
charformer) work unchanged. The subclass exists so experiment YAMLs can say
``model.type: "qwen3"`` and get the right default checkpoint, and so the
registry documents which architectures have been verified rather than
silently accepting anything under the ``llama`` key.

Facts about ``Qwen/Qwen3-4B-Base`` that matter for the pipeline:

* ``tie_word_embeddings: true`` (also 0.6B / 1.7B; 8B+ are untied) — same
  as Llama-3.2-1B, so ``lm_head`` is absent from ``named_parameters()`` and
  the freezing helper's warn-and-continue path applies in Phase 1.
* ``vocab_size`` (embedding rows) is 151936 while ``len(tokenizer)`` is
  151669 — the matrix is padded. ``NativeQwen3Tokenizer`` reports the padded
  size so ``resize_token_embeddings`` stays a no-op for the native tokenizer.
* ``hidden_size`` 2560, 36 layers, ~4.0B params — expect ~3× Llama-3.2-1B
  per-step cost under full fine-tuning.
"""
from __future__ import annotations

from typing import Any

from arabic_eval.models.llama_adapter import LlamaAdapter
from arabic_eval.registry import model_registry

DEFAULT_MODEL = "Qwen/Qwen3-4B-Base"


@model_registry.register("qwen3")
class Qwen3Adapter(LlamaAdapter):
    """Adapter for Qwen3 causal LMs (``Qwen3ForCausalLM``).

    Inherits every method from ``LlamaAdapter``; only the default checkpoint
    differs. See the module docstring for why no override is needed.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: str = "bfloat16",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            dtype=dtype,
            **kwargs,
        )
