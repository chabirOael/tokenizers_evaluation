"""Data collation factories dispatched by tokenizer embedding type."""
from __future__ import annotations

from typing import Any, Dict, List

import torch


class StandardCollator:
    """Collate batches for subword tokenizers (BPE, WordPiece, MorphoBPE).

    Pads 1-D ``input_ids`` and builds ``attention_mask`` / ``labels``.
    """

    def __init__(self, pad_token_id: int = 0, max_length: int = 512) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(ex["input_ids"][: self.max_length], dtype=torch.long)
                     for ex in batch]
        max_len = max(t.size(0) for t in input_ids)

        padded_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

        for i, ids in enumerate(input_ids):
            padded_ids[i, : ids.size(0)] = ids
            attention_mask[i, : ids.size(0)] = 1

        result = {"input_ids": padded_ids, "attention_mask": attention_mask}

        # Labels for LM training (shift handled by the model)
        if "labels" in batch[0]:
            labels = [torch.tensor(ex["labels"][: self.max_length], dtype=torch.long)
                      for ex in batch]
            padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
            for i, lab in enumerate(labels):
                padded_labels[i, : lab.size(0)] = lab
            result["labels"] = padded_labels
        else:
            # Default: causal LM labels = input_ids (ignore pad)
            labels = padded_ids.clone()
            labels[attention_mask == 0] = -100
            result["labels"] = labels

        return result


class CharacterCNNCollator:
    """Collate batches for CharacterBERT (3-D ``char_ids`` tensors).

    Each example has ``char_ids`` of shape ``[num_words, max_char_len]``.
    """

    def __init__(
        self, pad_token_id: int = 0, char_pad_id: int = 0,
        max_words: int = 512, max_char_len: int = 50,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.char_pad_id = char_pad_id
        self.max_words = max_words
        self.max_char_len = max_char_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # char_ids: list of [num_words, max_char_len]
        char_ids_list = [
            torch.tensor(ex["char_ids"][: self.max_words], dtype=torch.long)
            for ex in batch
        ]
        max_words = max(t.size(0) for t in char_ids_list)

        padded = torch.full(
            (len(batch), max_words, self.max_char_len), self.char_pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros(len(batch), max_words, dtype=torch.long)

        for i, cids in enumerate(char_ids_list):
            n = cids.size(0)
            padded[i, :n, : cids.size(1)] = cids
            attention_mask[i, :n] = 1

        result = {"char_ids": padded, "attention_mask": attention_mask}

        # Word-level labels: prefer explicit "labels", fall back to word-level
        # input_ids (present when called from text_generation / QA tasks).
        if "labels" in batch[0]:
            labels = [torch.tensor(ex["labels"][: self.max_words], dtype=torch.long)
                      for ex in batch]
            padded_labels = torch.full((len(batch), max_words), -100, dtype=torch.long)
            for i, lab in enumerate(labels):
                padded_labels[i, : lab.size(0)] = lab
            result["labels"] = padded_labels
        elif "input_ids" in batch[0]:
            # Derive causal LM labels from word IDs (same as StandardCollator fallback).
            word_ids = [
                torch.tensor(ex["input_ids"][: self.max_words], dtype=torch.long)
                for ex in batch
            ]
            padded_word_ids = torch.full((len(batch), max_words), self.pad_token_id, dtype=torch.long)
            for i, wids in enumerate(word_ids):
                padded_word_ids[i, : wids.size(0)] = wids
            word_labels = padded_word_ids.clone()
            word_labels[attention_mask == 0] = -100
            result["labels"] = word_labels

        return result


class CharJaberCollator:
    """Collate batches for char-JABER (1-D character ID sequences, longer)."""

    def __init__(self, pad_token_id: int = 0, max_length: int = 2048) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(ex["input_ids"][: self.max_length], dtype=torch.long)
                     for ex in batch]
        max_len = max(t.size(0) for t in input_ids)

        padded_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

        for i, ids in enumerate(input_ids):
            padded_ids[i, : ids.size(0)] = ids
            attention_mask[i, : ids.size(0)] = 1

        labels = padded_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": padded_ids, "attention_mask": attention_mask, "labels": labels}


class CharformerCollator:
    """Collate batches for Charformer (1-D byte ID sequences).

    Shape-wise identical to ``CharJaberCollator``: input_ids is a single byte
    stream per example. The downsampling that Charformer applies happens
    inside the model (GBST), not here, so the collator just builds a flat
    byte-level batch and lets the adapter shrink the attention mask.

    Default ``max_length`` is 2048 because Arabic UTF-8 inflates ~2-3x over
    char counts (each Arabic char is 2 bytes), so a typical paragraph that
    fits in 512 BPE tokens needs ~2-3k bytes.
    """

    def __init__(self, pad_token_id: int = 0, max_length: int = 2048) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(ex["input_ids"][: self.max_length], dtype=torch.long)
                     for ex in batch]
        max_len = max(t.size(0) for t in input_ids)

        padded_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

        for i, ids in enumerate(input_ids):
            padded_ids[i, : ids.size(0)] = ids
            attention_mask[i, : ids.size(0)] = 1

        labels = padded_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": padded_ids, "attention_mask": attention_mask, "labels": labels}


def get_collator(embedding_type: str, pad_token_id: int = 0, **kwargs):
    """Factory: return the right collator for a tokenizer's embedding type."""
    if embedding_type == "standard":
        return StandardCollator(pad_token_id=pad_token_id, **kwargs)
    elif embedding_type == "character_cnn":
        # CharacterCNNCollator uses max_words (not max_length) for its sequence
        # dimension.  Callers uniformly pass max_length, so map it here.
        cnn_kwargs = dict(kwargs)
        if "max_length" in cnn_kwargs:
            cnn_kwargs.setdefault("max_words", cnn_kwargs.pop("max_length"))
        return CharacterCNNCollator(pad_token_id=pad_token_id, **cnn_kwargs)
    elif embedding_type == "char_jaber":
        return CharJaberCollator(pad_token_id=pad_token_id, **kwargs)
    elif embedding_type == "charformer":
        return CharformerCollator(pad_token_id=pad_token_id, **kwargs)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
