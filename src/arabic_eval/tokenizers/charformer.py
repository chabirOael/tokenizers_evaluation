"""Charformer tokenizer: byte-level UTF-8 tokenization.

The Charformer paper (Tay et al., 2021) keeps the input vocabulary trivially
small — 256 possible byte values plus a handful of special tokens — and
delegates *all* subword-formation work to the GBST module sitting inside the
model (see ``arabic_eval.models.embeddings.charformer_embed``). So this
"tokenizer" is essentially a UTF-8 byte encoder; ``train()`` is a no-op
because there's nothing to learn at this layer.

Special tokens occupy ids 0..3, then byte values 0..255 are mapped to ids
4..259 (vocab_size = 260). We chose this layout (rather than the paper's
"reuse 100 byte ids as sentinels") for consistency with the rest of the
codebase, where pad=0, bos=1, eos=2, unk=3 across every tokenizer.

Arabic note: each Arabic character is 2 bytes in UTF-8, so a typical word
expands to ~5-10 byte ids (vs ~1-3 BPE tokens), and full sentences are
roughly 4-6x longer than under subword tokenization. The GBST module's
downsampling (default d_s=2) absorbs much of that cost on the transformer
side, but the *tokenizer* output still needs ~2048 byte ids of headroom
per example — see ``configs/tokenizers/charformer.yaml``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from arabic_eval.registry import tokenizer_registry
from arabic_eval.tokenizers.base import BaseTokenizer, EmbeddingType, TokenizerOutput

logger = logging.getLogger("arabic_eval.tokenizers.charformer")

# Reserved special-token ids — match the convention used by every other
# tokenizer in this codebase so the trainer / collator / pipeline don't need
# tokenizer-specific branches.
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
N_SPECIAL = 4

# Byte values 0..255 occupy ids 4..259.
BYTE_OFFSET = N_SPECIAL
VOCAB_SIZE = N_SPECIAL + 256  # 260


@tokenizer_registry.register("charformer")
class CharformerTokenizer(BaseTokenizer):
    """Byte-level UTF-8 tokenizer for Charformer.

    Vocabulary is fixed at 260 ids:
      0..3   - pad / bos / eos / unk
      4..259 - byte values 0..255

    There is nothing to train: ``train()`` is intentionally a no-op (it just
    logs that the byte vocabulary is fixed). The "subword learning" happens
    inside the GBST module of the model, not here.
    """

    def __init__(self, **kwargs: Any) -> None:
        # No state to learn — vocabulary is the same for every Charformer
        # instance — but expose ``params`` for symmetry with sibling tokenizers.
        self._params = dict(kwargs)

    def train(self, texts: List[str], vocab_size: int = 0, **kwargs: Any) -> None:
        """No-op: Charformer's byte vocabulary is fixed at 256 + 4 specials."""
        logger.info(
            "Charformer tokenizer: byte vocabulary is fixed (%d ids). "
            "No training performed; ignoring vocab_size=%s and %d training texts.",
            VOCAB_SIZE, vocab_size, len(texts),
        )

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> TokenizerOutput:
        # UTF-8 bytes -> ids = byte + BYTE_OFFSET. Surround with bos/eos.
        ids: List[int] = [BOS_ID]
        tokens: List[str] = ["<s>"]

        for byte in text.encode("utf-8"):
            ids.append(byte + BYTE_OFFSET)
            # Token strings are reported as 2-digit hex (e.g. "0x61") so they
            # are intelligible in metric reports without trying to decode each
            # individual byte (most Arabic bytes are continuation bytes that
            # don't represent a character on their own).
            tokens.append(f"0x{byte:02x}")

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
        # Collect raw bytes for non-special ids, then UTF-8 decode at the
        # very end — decoding byte-by-byte breaks Arabic multi-byte chars.
        skip = {PAD_ID, BOS_ID, EOS_ID, UNK_ID}
        raw_bytes = bytearray()
        for cid in ids:
            if cid in skip:
                continue
            byte_val = cid - BYTE_OFFSET
            if 0 <= byte_val <= 255:
                raw_bytes.append(byte_val)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # The vocabulary is a constant, but we still write a manifest so
        # ``load()`` can sanity-check that the saved tokenizer matches.
        with open(path / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "charformer",
                    "vocab_size": VOCAB_SIZE,
                    "byte_offset": BYTE_OFFSET,
                    "special_tokens": {
                        "<pad>": PAD_ID,
                        "<s>": BOS_ID,
                        "</s>": EOS_ID,
                        "<unk>": UNK_ID,
                    },
                    "params": self._params,
                },
                f,
                indent=2,
            )

    def load(self, path: Path | str) -> None:
        path = Path(path)
        with open(path / "tokenizer.json", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("vocab_size") != VOCAB_SIZE:
            raise ValueError(
                f"Saved Charformer tokenizer has vocab_size={data.get('vocab_size')}, "
                f"expected {VOCAB_SIZE}"
            )
        self._params = data.get("params", {})

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def embedding_type(self) -> str:
        return EmbeddingType.CHARFORMER

    @property
    def special_tokens(self) -> Dict[str, int]:
        return {
            "pad_token": PAD_ID,
            "bos_token": BOS_ID,
            "eos_token": EOS_ID,
            "unk_token": UNK_ID,
        }

    def get_embedding_config(self) -> Dict[str, Any]:
        # Forward GBST hyperparameters declared in the tokenizer config (e.g.
        # ``params: {max_block_size: 4, downsample_rate: 2, ...}``) to the
        # adapter, which constructs GBSTEmbedding from this dict.
        return {
            "vocab_size": VOCAB_SIZE,
            "max_block_size": self._params.get("max_block_size", 4),
            "downsample_rate": self._params.get("downsample_rate", 2),
            "conv_kernel_size": self._params.get("conv_kernel_size", 5),
            "block_attention": self._params.get("block_attention", False),
        }
