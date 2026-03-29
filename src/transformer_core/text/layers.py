from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from transformer_core.common import FeedForward, MultiHeadSelfAttention, ResidualBlock


class TransformerEncoderLayer(nn.Module):
    """Text transformer encoder block built from shared common blocks."""

    def __init__(self,
                 embed_dim         : int,
                 num_heads         : int,
                 mlp_ratio         : int = 4,
                 activation        : Optional[nn.Module] = None,
                 attention_dropout : float = 0.0,
                 dropout           : float = 0.0,
                 norm_first        : bool = True,
                 flash_attention   : bool = False) -> None:
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.attention = MultiHeadSelfAttention(embed_dim       = embed_dim,
                                                num_heads       = num_heads,
                                                dropout         = attention_dropout,
                                                flash_attention = flash_attention)
        
        self.residual_attention = ResidualBlock(embed_dim  = embed_dim,
                                                module     = self.attention,
                                                dropout    = dropout,
                                                norm_first = norm_first)
        
        self.ff = FeedForward(input_dim   = embed_dim,
                              hidden_dim  = hidden_dim,
                              output_dim  = embed_dim,
                              activation  = activation or nn.GELU(),
                              dropout     = dropout)
        
        self.residual_mlp = ResidualBlock(embed_dim  = embed_dim,
                                          module     = self.ff,
                                          dropout    = dropout,
                                          norm_first = norm_first)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.residual_attention(x, mask=mask)
        x = self.residual_mlp(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Text transformer decoder block with causal masking and optional KV caching."""

    def __init__(self,
                 embed_dim         : int,
                 num_heads         : int,
                 mlp_ratio         : int = 4,
                 activation        : Optional[nn.Module] = None,
                 attention_dropout : float = 0.0,
                 dropout           : float = 0.0,
                 norm_first        : bool = True,
                 flash_attention   : bool = False,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.attention = MultiHeadSelfAttention(embed_dim       = embed_dim,
                                                num_heads       = num_heads,
                                                dropout         = attention_dropout,
                                                flash_attention = flash_attention)
        
        self.residual_attention = ResidualBlock(embed_dim  = embed_dim,
                                                module     = self.attention,
                                                dropout    = dropout,
                                                norm_first = norm_first)
        
        self.ff = FeedForward(input_dim   = embed_dim,
                              hidden_dim  = hidden_dim,
                              output_dim  = embed_dim,
                              activation  = activation or nn.GELU(),
                              dropout     = dropout)
        
        self.residual_mlp = ResidualBlock(embed_dim  = embed_dim,
                                          module     = self.ff,
                                          dropout    = dropout,
                                          norm_first = norm_first)

    def _build_causal_mask(self, x: Tensor, mask: Optional[Tensor], past_len: int = 0) -> Tensor:
        batch_size, seq_len, _ = x.shape
        total_len = past_len + seq_len
        causal = torch.tril(torch.ones(total_len, total_len, device=x.device, dtype=torch.bool))
        if seq_len != total_len:
            causal = causal[total_len - seq_len : total_len, :]
        causal = causal.unsqueeze(0).expand(batch_size, seq_len, total_len)

        if mask is None:
            return causal
        if mask.dtype != torch.bool:
            mask = mask > 0
        if mask.dim() == 2:
            pad = mask[:, None, :].expand(batch_size, seq_len, mask.size(1))
            if mask.size(1) != total_len:
                raise ValueError("Padding mask length does not match total sequence length.")
            return causal & pad
        if mask.dim() == 3:
            return causal & mask
        raise ValueError("Unsupported attention mask rank.")

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        past_kv: Optional[tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        past_len = 0 if past_kv is None else past_kv[0].size(2)
        attn_mask = None
        if not (use_cache and past_kv is not None and x.size(1) == 1):
            attn_mask = self._build_causal_mask(x, mask, past_len=past_len)

        out = self.residual_attention(x, mask=attn_mask, past_kv=past_kv, use_cache=use_cache)
        if isinstance(out, tuple):
            x, present = out
        else:
            x = out
            present = None
        x = self.residual_mlp(x)
        if use_cache:
            return x, present
        return x
