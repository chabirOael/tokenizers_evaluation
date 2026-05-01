"""Charformer GBST (Gradient-Based Subword Tokenization) embedding.

Implements the GBST module from Tay et al., 2021 (ICLR 2022),
https://arxiv.org/abs/2106.12672. The module replaces the standard
``nn.Embedding`` lookup with a learned byte->latent-subword pipeline:

  byte ids [B, L]
  -> byte embedding              [B, L, D]
  -> (optional) pre-GBST 1D conv [B, L, D]
  -> for each block size b in 1..M:
       pad to multiple of b, mean-pool stride b -> [B, L/b, D]
       score each block via Linear(D, 1)        -> [B, L/b, 1]
       repeat-interleave both back to length L  -> [B, L, D] / [B, L, 1]
  -> stack across block sizes, softmax over the b-dim
  -> (optional) position-wise score calibration: P_hat = softmax(P P^T) P
  -> weighted sum to form latent subwords        [B, L, D]
  -> mean-pool stride d_s                        [B, L/d_s, D]

The trailing mean-pool downsampling makes the transformer that consumes
this module's output operate on ``L/d_s`` positions instead of ``L``,
which is the central efficiency contribution of Charformer.

Causality: GBST is non-causal *within* a block window of size M (a block
of size 4 starting at position i pools X[i:i+4], so position i sees up
to position i+M-1). This is harmless for teacher-forced LM training and
log-likelihood scoring of fixed sequences (which is what our pipeline
does), but means autoregressive generation byte-by-byte is not
well-defined. We don't expose generate() for this embedding type.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GBSTEmbedding(nn.Module):
    """Gradient-Based Subword Tokenization embedding (Charformer).

    Args:
        vocab_size: Number of byte/character ids (incl. specials).
        output_dim: Hidden dimension of the model (matches the transformer).
        max_block_size: ``M`` in the paper. Block sizes b in {1, ..., M} are
            enumerated. Default 4 (paper's main config).
        downsample_rate: ``d_s``. Final mean-pool stride applied to the
            latent-subword sequence. Default 2.
        conv_kernel_size: Pre-GBST 1D conv kernel. The paper uses 5 in main
            experiments to "smooth" over context as a cheaper alternative
            to enumerating offset blocks. Set to 0 to disable.
        block_attention: Whether to apply position-wise score calibration
            P_hat = softmax(P P^T) P. The paper shows this helps in English
            and has little effect multilingually.
        padding_idx: Embedding padding index (typically 0).
    """

    def __init__(
        self,
        vocab_size: int,
        output_dim: int = 2048,
        max_block_size: int = 4,
        downsample_rate: int = 2,
        conv_kernel_size: int = 5,
        block_attention: bool = False,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if max_block_size < 1:
            raise ValueError(f"max_block_size must be >= 1, got {max_block_size}")
        if downsample_rate < 1:
            raise ValueError(f"downsample_rate must be >= 1, got {downsample_rate}")

        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.max_block_size = max_block_size
        self.downsample_rate = downsample_rate
        self.conv_kernel_size = conv_kernel_size
        self.block_attention = block_attention

        self.embedding = nn.Embedding(vocab_size, output_dim, padding_idx=padding_idx)

        if conv_kernel_size and conv_kernel_size > 0:
            # 'same' padding so length is preserved after the pre-GBST conv.
            pad = (conv_kernel_size - 1) // 2
            self.pre_conv = nn.Conv1d(
                output_dim, output_dim,
                kernel_size=conv_kernel_size,
                padding=pad,
            )
        else:
            self.pre_conv = None

        # Block-scoring network: linear(D -> 1), no bias (matches paper code).
        self.block_scoring = nn.Linear(output_dim, 1, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] byte/char ids.

        Returns:
            Latent subwords of shape [batch, seq_len // downsample_rate, D].
        """
        x = self.embedding(input_ids)  # [B, L, D]

        if self.pre_conv is not None:
            x = self.pre_conv(x.transpose(1, 2)).transpose(1, 2)  # [B, L, D]

        L = x.size(1)
        all_candidates = []
        all_scores = []

        for b in range(1, self.max_block_size + 1):
            # Pad on the right so length is divisible by b.
            pad_amt = (b - L % b) % b
            if pad_amt:
                # F.pad on a [B, L, D] tensor pads the *last* dim by default,
                # so explicitly pad on the seq (second-to-last) dim.
                padded = F.pad(x, (0, 0, 0, pad_amt))
            else:
                padded = x

            # Mean-pool with stride b: [B, L_padded/b, D].
            pooled = F.avg_pool1d(
                padded.transpose(1, 2),
                kernel_size=b,
                stride=b,
            ).transpose(1, 2)

            # Score each block: [B, L_padded/b, 1].
            scores = self.block_scoring(pooled)

            # Upsample by replicating each block b times: [B, L_padded, D] / [B, L_padded, 1].
            upsampled_seq = pooled.repeat_interleave(b, dim=1)
            upsampled_scores = scores.repeat_interleave(b, dim=1)

            # Trim back to L (the right-pad padding we added is removed).
            upsampled_seq = upsampled_seq[:, :L, :]
            upsampled_scores = upsampled_scores[:, :L, :]

            all_candidates.append(upsampled_seq.unsqueeze(-1))   # [B, L, D, 1]
            all_scores.append(upsampled_scores.unsqueeze(-1))    # [B, L, 1, 1]

        # [B, L, D, M] and [B, L, 1, M].
        candidates = torch.cat(all_candidates, dim=-1)
        block_scores = torch.cat(all_scores, dim=-1)

        # Softmax over block sizes per position.
        block_scores = F.softmax(block_scores, dim=-1)  # [B, L, 1, M]

        if self.block_attention:
            # P_hat = softmax(P P^T) P, no projections (paper §2.1.4).
            P = block_scores.squeeze(2)                       # [B, L, M]
            attn = torch.matmul(P, P.transpose(1, 2))         # [B, L, L]
            attn = F.softmax(attn, dim=-1)
            P = torch.matmul(attn, P)                         # [B, L, M]
            block_scores = P.unsqueeze(2)                     # [B, L, 1, M]

        # Weighted sum over block sizes: [B, L, D].
        latent = (candidates * block_scores).sum(dim=-1)

        # Final mean-pool downsampling.
        if self.downsample_rate > 1:
            # Right-pad if needed so length is divisible by d_s.
            L_lat = latent.size(1)
            pad_amt = (self.downsample_rate - L_lat % self.downsample_rate) % self.downsample_rate
            if pad_amt:
                latent = F.pad(latent, (0, 0, 0, pad_amt))
            latent = F.avg_pool1d(
                latent.transpose(1, 2),
                kernel_size=self.downsample_rate,
                stride=self.downsample_rate,
            ).transpose(1, 2)

        return latent  # [B, ceil(L/d_s), D]


class CharformerOutputHead(nn.Module):
    """Output head for Charformer: upsample hidden states then project to vocab.

    The transformer operates on the downsampled sequence (length ~L/d_s).
    To compute byte-level logits aligned with the original byte labels, we
    upsample by ``downsample_rate`` via a transposed 1D convolution and
    project to the byte vocabulary.

    Args:
        hidden_dim: Model hidden dimension.
        vocab_size: Byte vocabulary size (matches GBSTEmbedding.vocab_size).
        upsample_factor: Must equal the GBSTEmbedding.downsample_rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        upsample_factor: int = 2,
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

        self.output_projection = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, downsampled_len, hidden_dim].

        Returns:
            Logits of shape [batch, byte_len, vocab_size].
        """
        if self.upsampler is not None:
            h = hidden_states.transpose(1, 2)  # [B, D, S]
            h = self.upsampler(h)              # [B, D, S * factor]
            hidden_states = h.transpose(1, 2)  # [B, S * factor, D]

        return self.output_projection(hidden_states)
