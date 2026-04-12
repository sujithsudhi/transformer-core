import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import RotaryEmbedding


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional KV caching, RoPE, and tracing."""

    def __init__(self,
                 embed_dim       : int,
                 num_heads       : int,
                 dropout         : float = 0.0,
                 flash_attention : bool = False,
                 qkv_bias        : bool = True,
                 use_rope        : bool = False,
                 rope_base       : int = 10000,
             ) -> None:
        """
        Initialize a multi-head self-attention block.
        Args:
            embed_dim       : Total embedding dimension of the attention inputs and outputs.
            num_heads       : Number of attention heads.
            dropout         : Dropout probability applied to attention probabilities and outputs.
            flash_attention : Whether to prefer torch scaled-dot-product attention kernels.
            qkv_bias        : Whether the Q/K/V projection layers include bias terms.
            use_rope        : Whether to apply rotary embeddings to the query and key tensors.
            rope_base       : Base used to construct rotary embedding frequencies.
        Returns:
            None.
        Raises:
            ValueError: If embed_dim is not divisible by num_heads.
        """
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

        self.w_q  = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.w_k  = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.w_v  = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.w_o  = nn.Linear(embed_dim, embed_dim, bias=True)
        self.rope = RotaryEmbedding(head_dim=self.head_dim, base=rope_base) if use_rope else None

        self.capture_attention   = False
        self.capture_qkv         = False
        self.last_attention_weights: Optional[Tensor] = None
        self.last_q              : Optional[Tensor] = None
        self.last_k              : Optional[Tensor] = None
        self.last_v              : Optional[Tensor] = None

    def set_trace(self,
                  enabled     : bool = True,
                  capture_qkv : bool = False,
              ) -> None:
        """
        Enable or disable tracing of attention weights and optional Q/K/V tensors.
        Args:
            enabled     : Whether tracing should be enabled.
            capture_qkv : Whether traced query, key, and value tensors should also be stored.
        Returns:
            None.
        """
        self.capture_attention = enabled
        self.capture_qkv       = enabled and capture_qkv
        if not enabled:
            self.clear_trace()

    def clear_trace(self) -> None:
        """
        Clear any cached traced tensors from the last forward pass.
        Args:
            None.
        Returns:
            None.
        """
        self.last_attention_weights = None
        self.last_q                 = None
        self.last_k                 = None
        self.last_v                 = None

    def scaled_dot_product(self,
                           q            : Tensor,
                           k            : Tensor,
                           v            : Tensor,
                           mask         : Optional[Tensor] = None,
                           is_causal    : bool = False,
                           need_weights : bool = False,
                       ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Compute scaled dot-product attention.
        Args:
            q            : Query tensor of shape (batch_size, num_heads, q_len, head_dim).
            k            : Key tensor of shape (batch_size, num_heads, k_len, head_dim).
            v            : Value tensor of shape (batch_size, num_heads, k_len, head_dim).
            mask         : Optional boolean mask broadcastable to the attention scores.
            is_causal    : Whether a causal lower-triangular mask should be enforced.
            need_weights : Whether attention weights should be returned alongside the outputs.
        Returns:
            Attention output tensor, or a tuple of output tensor and attention weights.
        """
        if self.flash_attention and not need_weights:
            if not self._flash_warned and not torch.cuda.is_available():
                warnings.warn("flash_attention=True but CUDA is unavailable; using the math SDP kernel instead.",
                              stacklevel=2)
                self._flash_warned = True

            attn_mask = None
            if mask is not None:
                mask      = mask.to(dtype=torch.bool, device=q.device)
                attn_mask = ~mask

            dropout_p = self.dropout_p if self.training else 0.0
            return F.scaled_dot_product_attention(q,
                                                  k,
                                                  v,
                                                  attn_mask = attn_mask,
                                                  dropout_p = dropout_p,
                                                  is_causal = is_causal)

        dk               = q.size(-1)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)

        if is_causal:
            q_len            = attention_scores.size(-2)
            k_len            = attention_scores.size(-1)
            causal           = torch.tril(torch.ones(q_len, k_len, device=attention_scores.device, dtype=torch.bool))
            attention_scores = attention_scores.masked_fill(~causal, float("-inf"))

        if mask is not None:
            mask             = mask.to(dtype=torch.bool, device=attention_scores.device)
            attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        attention_prob   = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(self.dropout(attention_prob), v)
        if need_weights:
            return attention_output, attention_prob
        return attention_output

    def split_heads(self, x: Tensor) -> Tensor:
        """
        Reshape the last embedding dimension into attention heads.
        Args:
            x : Tensor of shape (batch_size, seq_len, embed_dim).
        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len, _ = x.shape
        return (x.view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(batch_size, self.num_heads, seq_len, self.head_dim))

    def combine_heads(self, x: Tensor) -> Tensor:
        """
        Merge the attention-head dimension back into the embedding dimension.
        Args:
            x : Tensor of shape (batch_size, num_heads, seq_len, head_dim).
        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, _, seq_len, _ = x.shape
        return (x.view(batch_size, self.num_heads, seq_len, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, self.embed_dim))

    def forward(self,
                x            : Tensor,
                mask         : Optional[Tensor] = None,
                past_kv      : Optional[tuple[Tensor, Tensor]] = None,
                use_cache    : bool = False,
                is_causal    : bool = False,
                need_weights : bool = False,
            ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Run the self-attention block.
        Args:
            x            : Tensor of shape (batch_size, seq_len, embed_dim).
            mask         : Optional padding mask or explicit attention mask.
            past_kv      : Optional cached key/value tensors for autoregressive decoding.
            use_cache    : Whether to return the updated key/value cache.
            is_causal    : Whether causal masking should be applied.
            need_weights : Whether attention weights should be returned for tracing.
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim), or a tuple with cache.
        Raises:
            ValueError: If the provided mask rank is unsupported.
        """
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))

        position_offset = 0 if past_kv is None else past_kv[0].size(2)

        if self.rope is not None:
            q, k = self.rope(q, k, position_offset=position_offset)

        if past_kv is not None:
            past_k, past_v = past_kv
            k              = torch.cat([past_k, k], dim=2)
            v              = torch.cat([past_v, v], dim=2)

        attn_mask = None
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask > 0

            if mask.dim() == 2:
                if mask.shape[0] == mask.shape[1]:
                    mask = mask[None, None, :, :]
                else:
                    mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]
            elif mask.dim() != 4:
                raise ValueError("Unsupported attention mask rank.")

            attn_mask = mask.to(q.device)

        capture_attention = need_weights or self.capture_attention
        attention_out     = self.scaled_dot_product(q            = q,
                                                    k            = k,
                                                    v            = v,
                                                    mask         = attn_mask,
                                                    is_causal    = is_causal,
                                                    need_weights = capture_attention)

        if capture_attention:
            attention, attention_weights = attention_out
            self.last_attention_weights  = attention_weights.detach()
        else:
            attention                   = attention_out
            self.last_attention_weights = None

        if self.capture_qkv:
            self.last_q = q.detach()
            self.last_k = k.detach()
            self.last_v = v.detach()
        else:
            self.last_q = None
            self.last_k = None
            self.last_v = None

        attention = self.combine_heads(attention)
        attention = self.dropout(attention)
        attention = self.w_o(attention)

        if use_cache:
            return attention, (k, v)
        return attention
