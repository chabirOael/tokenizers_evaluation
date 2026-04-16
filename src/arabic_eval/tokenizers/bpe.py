"""BPE tokenizer using HuggingFace tokenizers library."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

from arabic_eval.registry import tokenizer_registry
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput

logger = logging.getLogger("arabic_eval.tokenizers.bpe")

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]


@tokenizer_registry.register("bpe")
class BPETokenizer(BaseTokenizer):
    """Byte Pair Encoding tokenizer trained from scratch."""

    def __init__(self, **kwargs: Any) -> None:
        self._tokenizer: Optional[Tokenizer] = None
        self._special_token_map: Dict[str, int] = {}

    def train(self, texts: List[str], vocab_size: int, **kwargs: Any) -> None:
        min_frequency = kwargs.get("min_frequency", 2)

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )

        logger.info("Training BPE tokenizer (vocab_size=%d) on %d texts", vocab_size, len(texts))
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Post-processor: add BOS/EOS
        bos_id = tokenizer.token_to_id("<s>")
        eos_id = tokenizer.token_to_id("</s>")
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"<s>:0 $A:0 </s>:0",
            pair=f"<s>:0 $A:0 </s>:0 <s>:1 $B:1 </s>:1",
            special_tokens=[("<s>", bos_id), ("</s>", eos_id)],
        )

        self._tokenizer = tokenizer
        self._build_special_token_map()
        logger.info("BPE tokenizer trained — vocab size: %d", self.vocab_size)

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> TokenizerOutput:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded.")

        if truncation and max_length:
            self._tokenizer.enable_truncation(max_length=max_length)
        else:
            self._tokenizer.no_truncation()

        if padding and max_length:
            self._tokenizer.enable_padding(length=max_length, pad_id=self.pad_token_id)
        else:
            self._tokenizer.no_padding()

        encoded = self._tokenizer.encode(text)
        return TokenizerOutput(
            input_ids=encoded.ids,
            attention_mask=encoded.attention_mask,
            tokens=encoded.tokens,
        )

    def decode(self, ids: List[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded.")
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path / "tokenizer.json"))

    def load(self, path: Path | str) -> None:
        path = Path(path)
        self._tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self._build_special_token_map()

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size() if self._tokenizer else 0

    @property
    def embedding_type(self) -> str:
        return EmbeddingType.STANDARD

    @property
    def special_tokens(self) -> Dict[str, int]:
        return self._special_token_map

    def _build_special_token_map(self) -> None:
        self._special_token_map = {
            "pad_token": self._tokenizer.token_to_id("<pad>"),
            "bos_token": self._tokenizer.token_to_id("<s>"),
            "eos_token": self._tokenizer.token_to_id("</s>"),
            "unk_token": self._tokenizer.token_to_id("<unk>"),
            "mask_token": self._tokenizer.token_to_id("<mask>"),
        }
