"""Farasa + CharacterBERT tokenizer.

Combines MorphoBPE's Farasa morphological pre-segmentation with
CharacterBERT's char-CNN encoding. After Farasa splits each Arabic word
into morphemes, every morpheme (instead of every whitespace word) becomes
the unit that is encoded as a fixed-length character ID vector and fed
through the CharCNN embedding.

Embedding family stays ``character_cnn``, so the existing
``LlamaAdapter`` branch and ``CharacterCNNCollator`` are reused as-is.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from arabic_eval.registry import tokenizer_registry
from arabic_eval.tokenizers.base import TokenizerOutput
from arabic_eval.tokenizers.character_bert import (
    CharacterBERTTokenizer,
    DEFAULT_MAX_CHAR_LEN,
)
from arabic_eval.tokenizers.morpho_bpe import segment_with_farasa, _get_farasa_segmenter

logger = logging.getLogger("arabic_eval.tokenizers.farasa_character_bert")

# Morphemes are shorter than whole words; a smaller char window is plenty
# and keeps the 3D char tensor lighter.
DEFAULT_MORPHEME_MAX_CHAR_LEN = 25


@tokenizer_registry.register("farasa_character_bert")
class FarasaCharacterBERTTokenizer(CharacterBERTTokenizer):
    """Farasa-segmented CharacterBERT.

    Pipeline:
      1. Farasa segments each word into morphemes (e.g. ``والكتاب`` → ``و ال كتاب``).
      2. Each morpheme is converted to a fixed-length char ID vector.
      3. CharCNN embedding consumes the 3D ``[batch, n_morphemes, max_char_len]`` tensor.

    The ``input_ids`` (used by the output head) index a *morpheme* vocabulary
    rather than a word vocabulary — fewer unique units, lower OOV rate.
    """

    def __init__(
        self,
        max_char_len: int = DEFAULT_MORPHEME_MAX_CHAR_LEN,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_char_len=max_char_len, **kwargs)
        self._segmenter = None

    def _ensure_segmenter(self) -> None:
        if self._segmenter is None:
            self._segmenter = _get_farasa_segmenter()

    def train(self, texts: List[str], vocab_size: int = 0, **kwargs: Any) -> None:
        logger.info("Segmenting %d texts with Farasa...", len(texts))
        self._ensure_segmenter()
        segmented_texts = segment_with_farasa(texts, self._segmenter)
        logger.info("Building char + morpheme vocabularies from segmented texts")
        super().train(segmented_texts, vocab_size=vocab_size, **kwargs)
        logger.info(
            "FarasaCharacterBERT trained — char vocab: %d, morpheme vocab: %d",
            self.char_vocab_size, self.vocab_size,
        )

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> TokenizerOutput:
        self._ensure_segmenter()
        segmented = segment_with_farasa([text], self._segmenter)[0]
        return super().encode(
            segmented,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
        )

    def save(self, path: Path | str) -> None:
        super().save(path)

    def load(self, path: Path | str) -> None:
        super().load(path)
