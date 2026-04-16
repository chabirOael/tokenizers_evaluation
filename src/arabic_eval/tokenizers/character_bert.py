"""CharacterBERT tokenizer: word-level segmentation with per-word character IDs.

CharacterBERT does not use a subword vocabulary. Instead, text is split into
words, and each word is represented by a fixed-length sequence of character
IDs that are later fed through a CharCNN embedding module.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from arabic_eval.registry import tokenizer_registry
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput

logger = logging.getLogger("arabic_eval.tokenizers.character_bert")

# Reserved character IDs
PAD_CHAR = 0
BOW_CHAR = 1  # Beginning of word
EOW_CHAR = 2  # End of word
UNK_CHAR = 3
# Dedicated sentinel character IDs for special word tokens so they are never
# confused with ordinary characters or with each other.
BOS_CHAR = 4   # sentinel for <s> word token
EOS_CHAR = 5   # sentinel for </s> word token
UNK_WORD_CHAR = 6  # sentinel for <unk> word token
_NUM_RESERVED_CHARS = 7  # start regular chars after this

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

DEFAULT_MAX_CHAR_LEN = 50


@tokenizer_registry.register("character_bert")
class CharacterBERTTokenizer(BaseTokenizer):
    """Word-level tokenizer with character IDs for CharCNN embeddings.

    Each word is represented as a fixed-length vector of character IDs.
    The embedding is handled by a CharacterCNN module in the model adapter.
    """

    def __init__(self, max_char_len: int = DEFAULT_MAX_CHAR_LEN, **kwargs: Any) -> None:
        self.max_char_len = max_char_len
        self._char_to_id: Dict[str, int] = {}
        self._id_to_char: Dict[int, str] = {}
        self._word_to_id: Dict[str, int] = {}
        self._id_to_word: Dict[int, str] = {}
        # Special word tokens
        self._special_word_tokens = {
            PAD_TOKEN: 0,
            BOS_TOKEN: 1,
            EOS_TOKEN: 2,
            UNK_TOKEN: 3,
        }
        self._next_word_id = len(self._special_word_tokens)

    def train(self, texts: List[str], vocab_size: int = 0, **kwargs: Any) -> None:
        """Build the character vocabulary from the training texts.

        ``vocab_size`` is ignored for CharacterBERT as the character vocabulary
        is fixed (all observed Unicode codepoints). The word vocabulary is
        built for the output head (lm_head) — capped at ``max_word_vocab``
        most-frequent words.
        """
        max_word_vocab = kwargs.get("max_word_vocab", 50_000)
        logger.info("Building character vocabulary from %d texts", len(texts))

        # Build character vocab
        char_set: set = set()
        word_freq: Dict[str, int] = {}
        for text in texts:
            for ch in text:
                char_set.add(ch)
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1

        # Character vocab: reserved IDs + sorted chars
        self._char_to_id = {
            "<pad>": PAD_CHAR,
            "<bow>": BOW_CHAR,
            "<eow>": EOW_CHAR,
            "<unk>": UNK_CHAR,
            "<bos_char>": BOS_CHAR,
            "<eos_char>": EOS_CHAR,
            "<unk_word_char>": UNK_WORD_CHAR,
        }
        for i, ch in enumerate(sorted(char_set), start=_NUM_RESERVED_CHARS):
            self._char_to_id[ch] = i
        self._id_to_char = {v: k for k, v in self._char_to_id.items()}

        # Word vocab (for output head): most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        self._word_to_id = dict(self._special_word_tokens)
        self._next_word_id = len(self._special_word_tokens)
        for word, _ in sorted_words[:max_word_vocab]:
            if word not in self._word_to_id:
                self._word_to_id[word] = self._next_word_id
                self._next_word_id += 1
        self._id_to_word = {v: k for k, v in self._word_to_id.items()}

        logger.info(
            "CharacterBERT tokenizer built — char vocab: %d, word vocab: %d",
            len(self._char_to_id), len(self._word_to_id),
        )

    def _word_to_char_ids(self, word: str) -> List[int]:
        """Convert a word to a fixed-length character ID sequence.

        Special word tokens get dedicated sentinel character IDs so they are
        always distinguishable from each other and from regular characters.
        Regular words are represented as [BOW, char..., EOW, PAD...], with
        characters truncated (not EOW) when the word is too long.
        """
        # Special tokens: use dedicated sentinel chars, never go through the
        # char vocab (their ASCII chars are almost certainly absent from Arabic text).
        if word == PAD_TOKEN:
            return [PAD_CHAR] * self.max_char_len
        if word == BOS_TOKEN:
            ids = [BOW_CHAR, BOS_CHAR, EOW_CHAR]
            ids.extend([PAD_CHAR] * (self.max_char_len - len(ids)))
            return ids
        if word == EOS_TOKEN:
            ids = [BOW_CHAR, EOS_CHAR, EOW_CHAR]
            ids.extend([PAD_CHAR] * (self.max_char_len - len(ids)))
            return ids
        if word == UNK_TOKEN:
            ids = [BOW_CHAR, UNK_WORD_CHAR, EOW_CHAR]
            ids.extend([PAD_CHAR] * (self.max_char_len - len(ids)))
            return ids

        # Regular word: BOW + chars + EOW, truncating chars (not EOW) to fit.
        max_chars = self.max_char_len - 2  # reserve slots for BOW and EOW
        ids = [BOW_CHAR]
        for ch in word[:max_chars]:
            ids.append(self._char_to_id.get(ch, UNK_CHAR))
        ids.append(EOW_CHAR)
        ids.extend([PAD_CHAR] * (self.max_char_len - len(ids)))
        return ids

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> TokenizerOutput:
        words = text.split()

        # Add BOS/EOS
        word_ids = [self._special_word_tokens[BOS_TOKEN]]
        char_ids = [self._word_to_char_ids(BOS_TOKEN)]

        for w in words:
            wid = self._word_to_id.get(w, self._special_word_tokens[UNK_TOKEN])
            word_ids.append(wid)
            char_ids.append(self._word_to_char_ids(w))

        word_ids.append(self._special_word_tokens[EOS_TOKEN])
        char_ids.append(self._word_to_char_ids(EOS_TOKEN))

        if truncation and max_length and len(word_ids) > max_length:
            word_ids = word_ids[:max_length]
            char_ids = char_ids[:max_length]

        attention_mask = [1] * len(word_ids)

        if padding and max_length and len(word_ids) < max_length:
            pad_count = max_length - len(word_ids)
            word_ids.extend([self._special_word_tokens[PAD_TOKEN]] * pad_count)
            char_ids.extend([[PAD_CHAR] * self.max_char_len] * pad_count)
            attention_mask.extend([0] * pad_count)

        return TokenizerOutput(
            input_ids=word_ids,
            attention_mask=attention_mask,
            tokens=[BOS_TOKEN] + words + [EOS_TOKEN],
            char_ids=char_ids,
        )

    def decode(self, ids: List[int]) -> str:
        words = []
        for wid in ids:
            if wid in self._id_to_word:
                word = self._id_to_word[wid]
                if word not in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                    words.append(word)
        return " ".join(words)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "char_to_id": self._char_to_id,
            "word_to_id": self._word_to_id,
            "max_char_len": self.max_char_len,
        }
        with open(path / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: Path | str) -> None:
        path = Path(path)
        with open(path / "tokenizer.json", encoding="utf-8") as f:
            data = json.load(f)
        self._char_to_id = data["char_to_id"]
        self._id_to_char = {v: k for k, v in self._char_to_id.items()}
        self._word_to_id = {k: int(v) for k, v in data["word_to_id"].items()}
        self._id_to_word = {v: k for k, v in self._word_to_id.items()}
        self._next_word_id = max(self._word_to_id.values()) + 1 if self._word_to_id else len(self._special_word_tokens)
        self.max_char_len = data["max_char_len"]

    @property
    def vocab_size(self) -> int:
        """Word vocabulary size (used for output head)."""
        return len(self._word_to_id)

    @property
    def char_vocab_size(self) -> int:
        return len(self._char_to_id)

    @property
    def embedding_type(self) -> str:
        return EmbeddingType.CHARACTER_CNN

    @property
    def special_tokens(self) -> Dict[str, int]:
        return {
            "pad_token": self._special_word_tokens[PAD_TOKEN],
            "bos_token": self._special_word_tokens[BOS_TOKEN],
            "eos_token": self._special_word_tokens[EOS_TOKEN],
            "unk_token": self._special_word_tokens[UNK_TOKEN],
        }

    def get_embedding_config(self) -> Dict[str, Any]:
        return {
            "char_vocab_size": self.char_vocab_size,
            "char_embed_dim": 16,
            "max_char_len": self.max_char_len,
            "cnn_filters": [
                [1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024],
            ],
            "num_highway_layers": 2,
            "output_vocab_size": self.vocab_size,
        }
