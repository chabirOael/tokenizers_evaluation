"""Abstract base class for all tokenizers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from arabic_eval.params_spec import ParamSpec


@dataclass
class TokenizerOutput:
    """Standardized output from any tokenizer."""
    input_ids: List[int]
    attention_mask: List[int]
    tokens: List[str] = field(default_factory=list)
    # For character-level tokenizers that need extra data:
    char_ids: Optional[List[List[int]]] = None  # Per-word character ID sequences


class EmbeddingType:
    STANDARD = "standard"             # nn.Embedding (BPE, WordPiece, MorphoBPE)
    CHARACTER_CNN = "character_cnn"   # CharCNN (CharacterBERT)
    CHAR_JABER = "char_jaber"         # char-JABER character embedding
    CHARFORMER = "charformer"         # Charformer GBST: byte embed -> block enum/score/mix -> downsample


class BaseTokenizer(ABC):
    """Abstract interface that all tokenizers must implement."""

    @classmethod
    def param_spec(cls) -> List[ParamSpec]:
        """The keys this tokenizer reads from ``tokenizer.params`` (the pipeline
        passes the dict to **both** the constructor and ``train()``), with type,
        default and one-line help — see ``arabic_eval.params_spec``. Defaults are
        the code's own constants, never retyped: a changed default would change
        every future vocabulary. Advisory only (the Pydantic field stays
        ``Dict[str, Any]``); an unknown key is warned about at validation and at
        run start, and the console renders a typed row per entry. Default: none
        declared, which turns the checks off for this tokenizer."""
        return []

    @abstractmethod
    def train(self, texts: List[str], vocab_size: int, **kwargs) -> None:
        """Train the tokenizer from a list of text strings."""
        ...

    @abstractmethod
    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> TokenizerOutput:
        """Encode a single text string."""
        ...

    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[TokenizerOutput]:
        """Encode a batch of text strings. Default: sequential encode."""
        return [self.encode(t, max_length, padding, truncation) for t in texts]

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        ...

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Save trained tokenizer to disk."""
        ...

    @abstractmethod
    def load(self, path: Path | str) -> None:
        """Load a trained tokenizer from disk."""
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        """Return the EmbeddingType this tokenizer requires."""
        ...

    @property
    @abstractmethod
    def special_tokens(self) -> Dict[str, int]:
        """Return mapping of special token names to IDs.

        Must include: pad_token, bos_token, eos_token, unk_token.
        """
        ...

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["pad_token"]

    def get_embedding_config(self) -> Dict[str, Any]:
        """Extra config for the embedding layer. Override for char-level tokenizers."""
        return {}

    def token_surfaces(self) -> Dict[int, List[Tuple[str, float]]]:
        """The surface strings each token id stands for, with weights — the input of
        the ``surface_avg`` embedding initialiser (``model.embedding_init``).

        Default for a standard tokenizer: every id maps to its own decoded string
        with weight 1 (a ByteLevel BPE piece keeps its leading space, a WordPiece
        piece loses its ``##``, both through ``decode``); the special tokens, any
        id that decodes to whitespace and a ByteLevel piece that is a partial
        UTF-8 sequence (decodes to U+FFFD — a lone byte of an Arabic letter, no
        text of its own) map to an empty list, which the initialiser fills with
        the base matrix's global mean. Tokenizers whose ids are not surface
        pieces (AraRooPat's ``[ROOT_*]`` / ``[PAT_*]``) override this with the
        words the token participates in.
        """
        specials = set((self.special_tokens or {}).values())
        out: Dict[int, List[Tuple[str, float]]] = {}
        for tid in range(self.vocab_size):
            if tid in specials:
                out[tid] = []
                continue
            try:
                text = self.decode([tid])
            except Exception:  # noqa: BLE001 — an undecodable id has no surface
                text = ""
            out[tid] = [(text, 1.0)] if text and text.strip() and "\ufffd" not in text else []
        return out
