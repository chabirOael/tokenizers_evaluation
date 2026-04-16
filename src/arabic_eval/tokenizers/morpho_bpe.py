"""Morphological BPE tokenizer: Farasa segmentation + BPE."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

from arabic_eval.registry import tokenizer_registry
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput

logger = logging.getLogger("arabic_eval.tokenizers.morpho_bpe")

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]

# Farasa morphological segmentation marker
_FARASA_SEP = "+"


def _get_farasa_segmenter():
    """Lazy-load Farasa segmenter (requires Java)."""
    from farasa.segmenter import FarasaSegmenter
    return FarasaSegmenter(interactive=True)


def segment_with_farasa(texts: List[str], segmenter=None) -> List[str]:
    """Apply Farasa morphological segmentation to a list of texts.

    Farasa splits Arabic words into morphemes separated by '+'.
    We replace '+' with spaces so BPE treats morphemes as separate units.
    """
    if segmenter is None:
        segmenter = _get_farasa_segmenter()

    segmented = []
    for text in texts:
        seg = segmenter.segment(text)
        # Replace Farasa's '+' delimiter with space for BPE
        seg = seg.replace(_FARASA_SEP, " ")
        segmented.append(seg)
    return segmented


@tokenizer_registry.register("morpho_bpe")
class MorphoBPETokenizer(BaseTokenizer):
    """Morphological BPE: first segment with Farasa, then apply BPE."""

    def __init__(self, **kwargs: Any) -> None:
        self._tokenizer: Optional[Tokenizer] = None
        self._segmenter = None
        self._special_token_map: Dict[str, int] = {}

    def _ensure_segmenter(self):
        if self._segmenter is None:
            self._segmenter = _get_farasa_segmenter()

    def train(self, texts: List[str], vocab_size: int, **kwargs: Any) -> None:
        min_frequency = kwargs.get("min_frequency", 2)

        # Step 1: Morphological segmentation with Farasa
        logger.info("Segmenting %d texts with Farasa...", len(texts))
        self._ensure_segmenter()
        segmented_texts = segment_with_farasa(texts, self._segmenter)

        # Step 2: Train BPE on segmented text
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )

        logger.info("Training BPE on Farasa-segmented text (vocab_size=%d)", vocab_size)
        tokenizer.train_from_iterator(segmented_texts, trainer=trainer)

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
        logger.info("MorphoBPE tokenizer trained — vocab size: %d", self.vocab_size)

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> TokenizerOutput:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded.")

        # First apply Farasa segmentation
        self._ensure_segmenter()
        segmented = segment_with_farasa([text], self._segmenter)[0]

        if truncation and max_length:
            self._tokenizer.enable_truncation(max_length=max_length)
        else:
            self._tokenizer.no_truncation()

        if padding and max_length:
            self._tokenizer.enable_padding(length=max_length, pad_id=self.pad_token_id)
        else:
            self._tokenizer.no_padding()

        encoded = self._tokenizer.encode(segmented)
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
