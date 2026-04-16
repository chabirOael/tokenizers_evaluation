"""char-JABER embedding module.

For char-JABER, each character is a token, so a standard nn.Embedding suffices.
However, character-level sequences are 4-6x longer than subword sequences, so
an optional convolutional downsampler can reduce sequence length before the
transformer, and an upsampler restores it for the output head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharJaberEmbedding(nn.Module):
    """Character embedding for char-JABER with optional downsampling.

    Args:
        char_vocab_size: Size of the character vocabulary.
        output_dim: Hidden dimension of the model.
        downsample_factor: If > 1, apply strided convolution to reduce
            sequence length by this factor. Set to 1 for no downsampling.
    """

    def __init__(
        self,
        char_vocab_size: int,
        output_dim: int = 2048,
        downsample_factor: int = 1,
    ) -> None:
        super().__init__()
        self.char_vocab_size = char_vocab_size
        self.output_dim = output_dim
        self.downsample_factor = downsample_factor

        self.embedding = nn.Embedding(char_vocab_size, output_dim, padding_idx=0)

        if downsample_factor > 1:
            self.downsampler = nn.Conv1d(
                output_dim, output_dim,
                kernel_size=downsample_factor,
                stride=downsample_factor,
                padding=0,
            )
        else:
            self.downsampler = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len] character IDs.

        Returns:
            Embeddings of shape [batch_size, seq_len // factor, output_dim].
        """
        emb = self.embedding(input_ids)  # [B, S, D]

        if self.downsampler is not None:
            # Conv1d expects [B, D, S]
            emb = emb.transpose(1, 2)
            emb = self.downsampler(emb)
            emb = emb.transpose(1, 2)

        return emb


class CharJaberOutputHead(nn.Module):
    """Output head for char-JABER that optionally upsamples back to character level.

    Args:
        hidden_dim: Model hidden dimension.
        char_vocab_size: Character vocabulary size.
        upsample_factor: Must match the downsample_factor used in embedding.
    """

    def __init__(
        self,
        hidden_dim: int,
        char_vocab_size: int,
        upsample_factor: int = 1,
    ) -> None:
        super().__init__()
        self.upsample_factor = upsample_factor

        if upsample_factor > 1:
            self.upsampler = nn.ConvTranspose1d(
                hidden_dim, hidden_dim,
                kernel_size=upsample_factor,
                stride=upsample_factor,
            )
        else:
            self.upsampler = None

        self.output_projection = nn.Linear(hidden_dim, char_vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]

        Returns:
            Logits of shape [batch_size, original_seq_len, char_vocab_size].
        """
        if self.upsampler is not None:
            h = hidden_states.transpose(1, 2)  # [B, D, S]
            h = self.upsampler(h)               # [B, D, S*factor]
            hidden_states = h.transpose(1, 2)   # [B, S*factor, D]

        return self.output_projection(hidden_states)
