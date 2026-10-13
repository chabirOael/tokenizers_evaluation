"""Native Gemma-3 tokenizer wrapper — pretrained, no from-scratch training.

Subclass of :class:`NativeQwen3Tokenizer` (itself a :class:`NativeLlamaTokenizer`) wrapping ``google/gemma-3-4b-pt``'s
pretrained tokenizer (or any Gemma-3 checkpoint via ``model_name_or_path``). ``encode`` / ``decode`` / ``save`` /
``load`` are inherited unchanged; the special-token map and the vocab-size source differ.

Gemma-3 facts this wrapper encodes (read from the tokenizer, not hard-coded — verify on the real checkpoint):

* **BOS, no EOS on encode.** The tokenizer prepends ``<bos>`` and appends nothing, so ``tokenize_record`` appends
  the EOS to every QA record as it does for Qwen, and the MCQ scorer's longest-common-prefix window treats the BOS
  as a shared prefix (docs/report.md §3.9).
* **Special tokens** (expected ``<pad>`` 0, ``<eos>`` 1, ``<bos>`` 2, ``<unk>`` 3) are taken from the tokenizer's
  own ids. The pretrained (``-pt``) model ends a document with ``<eos>``; ``<end_of_turn>`` belongs to the
  instruction-tuned chat format and is not used here.
* **Padded embedding matrix.** The 4B text config has 262 208 rows, the tokenizer 262 144 entries. ``vocab_size``
  reports the text config's row count (``AutoConfig(...).get_text_config().vocab_size`` — the multimodal top-level
  config has no ``vocab_size``) so ``resize_token_embeddings`` stays a no-op and the pretrained rows stay
  byte-identical, the invariant ``native_qwen3`` keeps the same way.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from transformers import AutoConfig

from arabic_eval.registry import tokenizer_registry
from arabic_eval.tokenizers.native_qwen3 import NativeQwen3Tokenizer

logger = logging.getLogger("arabic_eval.tokenizers.native_gemma")

DEFAULT_MODEL = "google/gemma-3-4b-pt"


@tokenizer_registry.register("native_gemma")
class NativeGemmaTokenizer(NativeQwen3Tokenizer):
    """Wrapper around a pretrained Gemma-3 tokenizer."""

    DEFAULT_MODEL_FOR_SPEC = DEFAULT_MODEL

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL, **kwargs: Any) -> None:
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)

    def _resolve_model_vocab_size(self) -> int:
        """Embedding-row count of the checkpoint's *text* model (padded beyond ``len(tokenizer)``)."""
        if self._model_vocab_size is not None:
            return self._model_vocab_size
        n_tok = len(self._hf_tokenizer)
        try:
            cfg = AutoConfig.from_pretrained(self._model_name_or_path)
            n_model = int(cfg.get_text_config().vocab_size)
        except Exception as e:  # noqa: BLE001 — any Hub/IO failure lands here
            logger.warning("Could not read the text vocab_size of %s (%s); falling back to len(tokenizer)=%d — "
                           "resize_token_embeddings will NOT be a no-op.", self._model_name_or_path, e, n_tok)
            n_model = n_tok
        self._model_vocab_size = max(n_model, n_tok)
        return self._model_vocab_size

    @property
    def special_tokens(self) -> Dict[str, int]:
        t = self._hf_tokenizer
        out = {"pad_token": t.pad_token_id, "bos_token": t.bos_token_id, "eos_token": t.eos_token_id,
               "unk_token": t.unk_token_id}
        missing = [k for k, v in out.items() if v is None and k != "unk_token"]
        if missing:
            raise ValueError(f"{self._model_name_or_path}: the tokenizer has no {missing} — not a Gemma-3 tokenizer?")
        return {k: int(v) for k, v in out.items() if v is not None}
