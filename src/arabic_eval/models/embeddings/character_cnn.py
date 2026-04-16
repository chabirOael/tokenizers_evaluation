"""Character CNN embedding module for CharacterBERT integration.

Replaces the standard nn.Embedding lookup with a CharCNN that produces
word-level embeddings from character ID sequences.

Architecture:
  char IDs [batch, seq_len, max_char_len]
  -> char embedding [batch, seq_len, max_char_len, char_embed_dim]
  -> multi-width 1D CNNs (per word)
  -> max-pool over char dimension
  -> highway layers
  -> linear projection to hidden_dim
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    """Highway layer: y = gate * transform(x) + (1 - gate) * x."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.transform = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = F.relu(self.transform(x))
        g = torch.sigmoid(self.gate(x))
        return g * t + (1 - g) * x


class CharacterCNNEmbedding(nn.Module):
    """CharCNN embedding: maps character ID tensors to word-level embeddings.

    Args:
        char_vocab_size: Number of character types (including special chars).
        char_embed_dim: Dimension of character embeddings.
        cnn_filters: List of [kernel_width, num_filters] pairs.
        num_highway_layers: Number of highway layers after CNN.
        output_dim: Final output dimension (must match model hidden_size).
        max_char_len: Maximum characters per word.
    """

    def __init__(
        self,
        char_vocab_size: int,
        char_embed_dim: int = 16,
        cnn_filters: List[List[int]] | None = None,
        num_highway_layers: int = 2,
        output_dim: int = 2048,
        max_char_len: int = 50,
    ) -> None:
        super().__init__()
        if cnn_filters is None:
            cnn_filters = [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]]

        self.char_embed_dim = char_embed_dim
        self.max_char_len = max_char_len

        self.char_embedding = nn.Embedding(
            char_vocab_size, char_embed_dim, padding_idx=0
        )

        # CNN filters: each operates on char_embed_dim input channels
        self.convolutions = nn.ModuleList()
        total_filters = 0
        for width, num_f in cnn_filters:
            conv = nn.Conv1d(
                in_channels=char_embed_dim,
                out_channels=num_f,
                kernel_size=width,
                padding=0,
            )
            self.convolutions.append(conv)
            total_filters += num_f

        # Highway layers
        self.highways = nn.ModuleList(
            [Highway(total_filters) for _ in range(num_highway_layers)]
        )

        # Final projection to match model hidden size
        self.projection = nn.Linear(total_filters, output_dim)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_ids: [batch_size, seq_len, max_char_len] integer tensor.

        Returns:
            Embeddings of shape [batch_size, seq_len, output_dim].
        """
        batch_size, seq_len, max_chars = char_ids.shape

        # Flatten batch and seq dimensions
        flat_char_ids = char_ids.view(-1, max_chars)  # [B*S, C]

        # Character embedding
        char_emb = self.char_embedding(flat_char_ids)  # [B*S, C, D]
        char_emb = char_emb.transpose(1, 2)  # [B*S, D, C] — channels first for Conv1d

        # Apply CNN filters and max-pool
        conv_outputs = []
        for conv in self.convolutions:
            c = F.relu(conv(char_emb))  # [B*S, num_f, C-k+1]
            c = c.max(dim=2).values     # [B*S, num_f]
            conv_outputs.append(c)

        cnn_out = torch.cat(conv_outputs, dim=1)  # [B*S, total_filters]

        # Highway layers
        for highway in self.highways:
            cnn_out = highway(cnn_out)

        # Project to hidden size
        out = self.projection(cnn_out)  # [B*S, output_dim]

        return out.view(batch_size, seq_len, -1)  # [B, S, output_dim]
