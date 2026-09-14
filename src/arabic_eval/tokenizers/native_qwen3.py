"""Native Qwen3 tokenizer wrapper — pretrained, no from-scratch training.

Subclass of :class:`NativeLlamaTokenizer` that wraps ``Qwen/Qwen3-4B-Base``'s
pretrained tokenizer (or any Qwen3 checkpoint via ``model_name_or_path``).
``encode`` / ``decode`` / ``save`` / ``load`` are inherited unchanged; only
the special-token map and the vocab-size contract differ from Llama.

Qwen-specific facts this wrapper encodes (verified against the Hub
``config.json`` + ``tokenizer_config.json`` of ``Qwen/Qwen3-4B-Base``):

* **No BOS.** ``Qwen2Tokenizer`` adds no special tokens on encode
  (``add_bos_token=False``), so sequences start with the first content
  token. The pipeline is BOS-agnostic — answer-only masking uses an LCP
  over prompt/full encodings and LightEval scoring uses
  ``full_len - ctx_len`` — so nothing needs a BOS. We report
  ``bos_token = 151643`` (``<|endoftext|>``) purely to mirror the model's
  ``config.bos_token_id``; it is never inserted.
* **No UNK.** Qwen is byte-level BPE with byte fallback, so no UNK can ever
  be produced. The ``unk_token`` key is therefore *omitted* from
  ``special_tokens``; both UNK consumers (``intrinsic_metrics`` and
  ``unk_reports``) use ``.get("unk_token")`` and document the "no key →
  UNK rate 0 / header-only CSV" path.
* **pad = eos = 151643** (``<|endoftext|>``), the model's own convention.
  Safe for the same reason as Llama's pad=eos: collators build
  ``attention_mask`` from real-token positions, not by comparing to pad_id.
* **Padded embedding matrix.** ``len(tokenizer)`` is 151669 but the model
  ships 151936 embedding rows. ``vocab_size`` reports the *model's* row
  count (read from ``AutoConfig``) so ``resize_token_embeddings`` is a true
  no-op and the pretrained embeddings stay byte-identical — the same
  invariant ``native_llama`` gets for free because Llama's ``len(tok)``
  equals its matrix size.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from transformers import AutoConfig

from arabic_eval.registry import tokenizer_registry
from arabic_eval.tokenizers.native_llama import NativeLlamaTokenizer

logger = logging.getLogger("arabic_eval.tokenizers.native_qwen3")

DEFAULT_MODEL = "Qwen/Qwen3-4B-Base"

# <|endoftext|> — Qwen3-Base's eos, pad and (nominal) bos.
ENDOFTEXT_TOKEN_ID = 151643


@tokenizer_registry.register("native_qwen3")
class NativeQwen3Tokenizer(NativeLlamaTokenizer):
    """Wrapper around a pretrained Qwen3 tokenizer."""

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL, **kwargs: Any) -> None:
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        self._model_vocab_size: Optional[int] = None

    def load(self, path) -> None:
        super().load(path)
        # The marker may have pointed us at a different checkpoint — re-resolve.
        self._model_vocab_size = None

    def _resolve_model_vocab_size(self) -> int:
        """Embedding-row count of the checkpoint this tokenizer ships with.

        Read once from ``AutoConfig`` (a ~1 KB JSON, cached by the Hub client).
        Falls back to ``len(tokenizer)`` with a warning if the config cannot
        be read — the resize then truncates the unused padding rows, which is
        harmless for correctness but breaks the no-op invariant.
        """
        if self._model_vocab_size is not None:
            return self._model_vocab_size
        n_tok = len(self._hf_tokenizer)
        try:
            n_model = int(AutoConfig.from_pretrained(self._model_name_or_path).vocab_size)
        except Exception as e:  # noqa: BLE001 — any Hub/IO failure lands here
            logger.warning(
                "Could not read vocab_size from %s config (%s); falling back to "
                "len(tokenizer)=%d. resize_token_embeddings will NOT be a no-op.",
                self._model_name_or_path, e, n_tok,
            )
            n_model = n_tok
        if n_model < n_tok:
            logger.warning(
                "Model vocab_size %d < len(tokenizer) %d for %s — using the larger "
                "so every tokenizer id has an embedding row.",
                n_model, n_tok, self._model_name_or_path,
            )
        self._model_vocab_size = max(n_model, n_tok)
        return self._model_vocab_size

    @property
    def vocab_size(self) -> int:
        return self._resolve_model_vocab_size()

    @property
    def special_tokens(self) -> Dict[str, int]:
        eos = self._hf_tokenizer.eos_token_id
        if eos is None:
            eos = ENDOFTEXT_TOKEN_ID
        pad = self._hf_tokenizer.pad_token_id
        if pad is None:
            pad = eos
        # No unk_token key on purpose — see module docstring.
        return {
            "pad_token": pad,
            "bos_token": eos,
            "eos_token": eos,
        }
