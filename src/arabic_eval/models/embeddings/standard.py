"""Standard embedding resizing for subword tokenizers."""
from __future__ import annotations

import logging
import torch
import torch.nn as nn

logger = logging.getLogger("arabic_eval.models.embeddings.standard")


def resize_token_embeddings(model, new_vocab_size: int) -> None:
    """Resize model's input embeddings and lm_head to match new vocab size.

    New embedding rows are initialized with small random values.
    """
    old_embeddings = model.get_input_embeddings()
    old_vocab_size = old_embeddings.weight.shape[0]
    embedding_dim = old_embeddings.weight.shape[1]

    if old_vocab_size == new_vocab_size:
        logger.info("Vocab size unchanged (%d), skipping resize.", new_vocab_size)
        return

    logger.info("Resizing embeddings: %d -> %d", old_vocab_size, new_vocab_size)

    # Resize via the HF method (handles both embed_tokens and lm_head)
    model.resize_token_embeddings(new_vocab_size)

    # Reinitialize new rows with small random values
    new_embeddings = model.get_input_embeddings()
    if new_vocab_size > old_vocab_size:
        with torch.no_grad():
            nn.init.normal_(
                new_embeddings.weight[old_vocab_size:],
                mean=0.0,
                std=0.02,
            )

    # Also reinitialize lm_head if it exists and was resized
    if hasattr(model, "lm_head") and model.lm_head.out_features == new_vocab_size:
        if new_vocab_size > old_vocab_size:
            with torch.no_grad():
                nn.init.normal_(
                    model.lm_head.weight[old_vocab_size:],
                    mean=0.0,
                    std=0.02,
                )
