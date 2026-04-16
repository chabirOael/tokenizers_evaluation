"""Intrinsic tokenizer evaluation metrics."""
from __future__ import annotations

import logging
from typing import Dict, List

from tqdm import tqdm

from arabic_eval.tokenizers.base import BaseTokenizer

logger = logging.getLogger("arabic_eval.tokenizers.intrinsic")


def compute_intrinsic_metrics(
    tokenizer: BaseTokenizer,
    texts: List[str],
) -> Dict[str, float]:
    """Compute intrinsic tokenizer quality metrics on a corpus.

    Returns:
        fertility: average number of tokens per whitespace word
        compression_ratio: average characters per token
        unk_rate: fraction of tokens that are <unk>
        vocab_coverage: fraction of unique words that have no <unk> tokens
        avg_token_count: average number of tokens per text
    """
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_unk = 0
    words_with_unk = 0
    unique_words: set = set()
    unique_words_with_unk: set = set()
    token_counts: List[int] = []

    unk_id = tokenizer.special_tokens.get("unk_token")

    for text in tqdm(texts, desc="Intrinsic metrics", unit="text"):
        words = text.split()
        total_words += len(words)
        total_chars += len(text)

        encoded = tokenizer.encode(text)
        n_tokens = len(encoded.input_ids)
        total_tokens += n_tokens
        token_counts.append(n_tokens)

        # Count UNK tokens
        if unk_id is not None:
            unk_count = encoded.input_ids.count(unk_id)
            total_unk += unk_count

        # Per-word UNK analysis
        for word in words:
            unique_words.add(word)
            word_enc = tokenizer.encode(word)
            if unk_id is not None and unk_id in word_enc.input_ids:
                unique_words_with_unk.add(word)

    n_texts = len(texts)
    fertility = total_tokens / max(total_words, 1)
    compression_ratio = total_chars / max(total_tokens, 1)
    unk_rate = total_unk / max(total_tokens, 1)
    vocab_coverage = 1.0 - len(unique_words_with_unk) / max(len(unique_words), 1)
    avg_token_count = total_tokens / max(n_texts, 1)

    metrics = {
        "fertility": round(fertility, 4),
        "compression_ratio": round(compression_ratio, 4),
        "unk_rate": round(unk_rate, 6),
        "vocab_coverage": round(vocab_coverage, 4),
        "avg_token_count": round(avg_token_count, 2),
        "vocab_size": tokenizer.vocab_size,
    }

    logger.info("Intrinsic metrics: %s", metrics)
    return metrics
