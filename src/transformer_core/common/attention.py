import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional KV caching."""

    def __init__(self,
                 embed_dim       : int,
                 num_heads       : int,
                 dropout         : float = 0.0,
                 flash_attention : bool = False) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("Embedded dimension should be divisible by number of heads.")

        self.num_heads       = num_heads
        self.embed_dim       = embed_dim
        self.head_dim        = embed_dim // num_heads
        self.dropout_p       = dropout
        self.dropout         = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.flash_attention = flash_attention
        self._flash_warned   = False

        self.w_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=True)

    def scaled_dot_product(self,
                           q    : Tensor,
                           k    : Tensor,
                           v    : Tensor,
                           mask : Optional[Tensor] = None) -> Tensor:
        if self.flash_attention:
            if not self._flash_warned and not torch.cuda.is_available():
                warnings.warn(
                    "flash_attention=True but CUDA is unavailable; using the math SDP kernel instead.",
                    stacklevel=2,
                )
                self._flash_warned = True
            attn_mask = None
            if mask is not None:
                mask = mask.to(dtype=torch.bool, device=q.device)
                attn_mask = ~mask
            dropout_p = self.dropout_p if self.training else 0.0
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

        dk = q.size(-1)
        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=attention.device)
            attention = attention.masked_fill(~mask, float("-inf"))

        attention_prob = torch.softmax(attention, dim=-1)
        attention_prob = self.dropout(attention_prob)
        return torch.matmul(attention_prob, v)

    def split_heads(self, x: Tensor) -> Tensor:
        """Reshape embeddings into per-head projections."""
        batch_size, seq_len, _ = x.shape
        return (
            x.view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, self.num_heads, seq_len, self.head_dim)
        )

    def combine_heads(self, x: Tensor) -> Tensor:
        """Merge per-head projections back into model embeddings."""
        batch_size, _, seq_len, _ = x.shape
        return (
            x.view(batch_size, self.num_heads, seq_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

    def forward(self,
                x         : Tensor,
                mask      : Optional[Tensor] = None,
                past_kv   : Optional[tuple[Tensor, Tensor]] = None,
                use_cache : bool = False) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        attn_mask = None
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask > 0
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]
            elif mask.dim() != 4:
                raise ValueError("Unsupported attention mask rank.")
            attn_mask = mask.to(q.device)

        attention = self.scaled_dot_product(q=q, k=k, v=v, mask=attn_mask)
        attention = self.combine_heads(attention)
        attention = self.dropout(attention)
        attention = self.w_o(attention)

        if use_cache:
            return attention, (k, v)
        return attention
