"""char-JABER tokenizer: character-level tokenization for Arabic.

char-JABER treats each character (including spaces and punctuation) as an
individual token. The vocabulary is small and fixed — all observed Unicode
codepoints plus special tokens.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from arabic_eval.registry import tokenizer_registry
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput

logger = logging.getLogger("arabic_eval.tokenizers.char_jaber")

# Reserved IDs
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SPACE_ID = 4


@tokenizer_registry.register("char_jaber")
class CharJaberTokenizer(BaseTokenizer):
    """Character-level tokenizer for Arabic (char-JABER style).

    Every character is a token. The vocabulary is small (~300-500 entries).
    Sequences are much longer than subword tokenizers.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._char_to_id: Dict[str, int] = {}
        self._id_to_char: Dict[int, str] = {}

    def train(self, texts: List[str], vocab_size: int = 0, **kwargs: Any) -> None:
        """Build the character vocabulary from training texts.

        ``vocab_size`` is ignored — the vocab is all observed characters.
        """
        logger.info("Building char-JABER character vocabulary from %d texts", len(texts))

        char_set: set = set()
        for text in texts:
            char_set.update(text)

        # Build vocab: reserved tokens first, then sorted characters
        self._char_to_id = {
            "<pad>": PAD_ID,
            "<s>": BOS_ID,
            "</s>": EOS_ID,
            "<unk>": UNK_ID,
            " ": SPACE_ID,
        }
        next_id = len(self._char_to_id)
        for ch in sorted(char_set):
            if ch not in self._char_to_id:
                self._char_to_id[ch] = next_id
                next_id += 1

        self._id_to_char = {v: k for k, v in self._char_to_id.items()}
        logger.info("char-JABER tokenizer built — vocab size: %d", self.vocab_size)

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> TokenizerOutput:
        ids = [BOS_ID]
        tokens = ["<s>"]

        for ch in text:
            ids.append(self._char_to_id.get(ch, UNK_ID))
            tokens.append(ch)

        ids.append(EOS_ID)
        tokens.append("</s>")

        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]
            tokens = tokens[:max_length]

        attention_mask = [1] * len(ids)

        if padding and max_length and len(ids) < max_length:
            pad_count = max_length - len(ids)
            ids.extend([PAD_ID] * pad_count)
            tokens.extend(["<pad>"] * pad_count)
            attention_mask.extend([0] * pad_count)

        return TokenizerOutput(
            input_ids=ids,
            attention_mask=attention_mask,
            tokens=tokens,
        )

    def decode(self, ids: List[int]) -> str:
        chars = []
        for cid in ids:
            if cid in (PAD_ID, BOS_ID, EOS_ID):
                continue
            ch = self._id_to_char.get(cid, "")
            chars.append(ch)
        return "".join(chars)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump({"char_to_id": self._char_to_id}, f, ensure_ascii=False, indent=2)

    def load(self, path: Path | str) -> None:
        path = Path(path)
        with open(path / "tokenizer.json", encoding="utf-8") as f:
            data = json.load(f)
        self._char_to_id = data["char_to_id"]
        self._id_to_char = {int(v): k for k, v in self._char_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._char_to_id)

    @property
    def embedding_type(self) -> str:
        return EmbeddingType.CHAR_JABER

    @property
    def special_tokens(self) -> Dict[str, int]:
        return {
            "pad_token": PAD_ID,
            "bos_token": BOS_ID,
            "eos_token": EOS_ID,
            "unk_token": UNK_ID,
        }

    def get_embedding_config(self) -> Dict[str, Any]:
        return {
            "char_vocab_size": self.vocab_size,
        }
