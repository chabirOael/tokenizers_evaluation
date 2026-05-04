"""Native Llama tokenizer wrapper — pretrained, no from-scratch training.

Wraps `meta-llama/Llama-3.2-1B`'s pretrained tokenizer (or any
``AutoTokenizer.from_pretrained`` target) so it slots into the pipeline like
any other tokenizer. Unlike the eight from-scratch tokenizers, ``train()`` is
a no-op — the tokenizer ships with the model.

Used as the basis for two baselines:
  (a) native tokenizer + native pretrained weights, no SFT — pretrained ceiling
  (b) native tokenizer + native pretrained weights + same SFT pipeline —
      isolates the cost of using a from-scratch tokenizer in the existing sweep

Special-token IDs follow Llama-3.2's actual layout:
  bos = 128000  <|begin_of_text|>
  eos = 128001  <|end_of_text|>
  pad = 128001  (= eos, HF standard pattern for causal LMs without dedicated pad)
  unk = 128002  (= <|reserved_special_token_0|>, never emitted; UNK rate stays 0)

The pad=eos collision is safe because the collators build attention_mask from
"real token positions" (not from comparing input_ids to pad_id), so trailing
pads are masked to -100 in labels regardless, and real EOS in mid-sequence
stays attention_mask=1 and contributes to loss.

Vocab size uses ``len(tokenizer)`` (=128256) rather than ``tokenizer.vocab_size``
(=128000, BPE merges only). 128256 matches the Llama-3.2-1B embedding matrix,
so ``model.resize_token_embeddings(128256)`` is a pass-through (no resize, no
reinit) and pretrained embeddings stay byte-identical.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from arabic_eval.registry import tokenizer_registry
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput

logger = logging.getLogger("arabic_eval.tokenizers.native_llama")

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"

# Llama-3.2 reserved special tokens used for our pad/unk slots:
#   128001 = <|end_of_text|>            (real EOS — re-used as pad, HF standard)
#   128002 = <|reserved_special_token_0|>  (never emitted in practice — used as UNK slot)
PAD_TOKEN_ID = 128001
BOS_TOKEN_ID = 128000
EOS_TOKEN_ID = 128001
UNK_TOKEN_ID = 128002


@tokenizer_registry.register("native_llama")
class NativeLlamaTokenizer(BaseTokenizer):
    """Wrapper around a pretrained Llama tokenizer."""

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL, **kwargs: Any) -> None:
        self._model_name_or_path = model_name_or_path
        self._hf_tokenizer = self._load_hf(model_name_or_path)
        logger.info(
            "Loaded pretrained tokenizer from %s (vocab_size=%d, len=%d)",
            model_name_or_path, self._hf_tokenizer.vocab_size, len(self._hf_tokenizer),
        )

    @staticmethod
    def _load_hf(name_or_path: str):
        tok = AutoTokenizer.from_pretrained(
            name_or_path,
            clean_up_tokenization_spaces=False,
        )
        # Llama tokenizers ship without a pad_token; set it to eos so HF's
        # padding path works. Our PAD_TOKEN_ID == EOS_TOKEN_ID == 128001 mirrors
        # this — the collator-level mask is built from "real token positions",
        # not from comparing against pad_id, so the pad/eos collision is safe.
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def train(self, texts: List[str], vocab_size: int, **kwargs: Any) -> None:
        logger.info(
            "native_llama.train() is a no-op — tokenizer is pretrained "
            "(ignoring vocab_size=%s, %d input texts)",
            vocab_size, len(texts),
        )

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> TokenizerOutput:
        encoded = self._hf_tokenizer(
            text,
            max_length=max_length if (truncation or padding) else None,
            truncation=truncation,
            padding="max_length" if (padding and max_length) else False,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        tokens = self._hf_tokenizer.convert_ids_to_tokens(input_ids)
        return TokenizerOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokens=tokens,
        )

    def decode(self, ids: List[int]) -> str:
        return self._hf_tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._hf_tokenizer.save_pretrained(str(path))
        # Marker so load() knows where this came from even if save_pretrained's
        # files end up shared with another snapshot.
        with open(path / "native_llama_marker.json", "w", encoding="utf-8") as f:
            json.dump({"model_name_or_path": self._model_name_or_path}, f)

    def load(self, path: Path | str) -> None:
        path = Path(path)
        marker = path / "native_llama_marker.json"
        if marker.exists():
            with open(marker, "r", encoding="utf-8") as f:
                self._model_name_or_path = json.load(f).get(
                    "model_name_or_path", self._model_name_or_path
                )
        self._hf_tokenizer = self._load_hf(str(path))

    @property
    def vocab_size(self) -> int:
        # len(tokenizer) includes Llama's 256 reserved special tokens at IDs
        # 128000-128255; matches model.config.vocab_size (128256 for Llama-3.2-1B)
        # so resize_token_embeddings is a no-op.
        return len(self._hf_tokenizer)

    @property
    def embedding_type(self) -> str:
        return EmbeddingType.STANDARD

    @property
    def special_tokens(self) -> Dict[str, int]:
        return {
            "pad_token": PAD_TOKEN_ID,
            "bos_token": BOS_TOKEN_ID,
            "eos_token": EOS_TOKEN_ID,
            "unk_token": UNK_TOKEN_ID,
        }
