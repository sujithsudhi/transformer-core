"""Embedding modules used by transformer models."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """Lookup embedding layer with optional padding index."""

    def __init__(self,
                 vocab_size  : int = 256,
                 embed_dim   : int = 256,
                 padding_idx : int | None = None) -> None:
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings = vocab_size,
                                      embedding_dim  = embed_dim,
                                      padding_idx    = padding_idx)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens)


class PositionalEncoding(nn.Module):
    """Sinusoidal or trainable positional encoding."""

    def __init__(self,
                 max_len   : int,
                 embed_dim : int,
                 dropout   : float,
                 method    : str = "normal") -> None:
        super().__init__()

        if max_len <= 0:
            raise ValueError("max_len must be positive.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if method not in {"normal", "trainable"}:
            raise ValueError("method must be either 'normal' or 'trainable'.")

        self.max_len = max_len
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if method == "trainable":
            positional_table      = torch.zeros(1, max_len, embed_dim)
            self.positional_table = nn.Parameter(positional_table)
            nn.init.trunc_normal_(self.positional_table, std=0.02)
        else:
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
            )
            pe       = torch.zeros(max_len, embed_dim, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
            self.register_buffer("positional_table", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        length = x.size(1)
        if offset + length > self.max_len:
            raise ValueError(
                f"offset({offset}) + length({length}) > max_len({self.max_len}). Increase max_len."
            )
        x = x + self.positional_table[:, offset : offset + length]
        return self.dropout(x)
