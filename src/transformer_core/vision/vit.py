from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from transformer_core.common import FeedForward, MultiHeadSelfAttention, ResidualBlock


class ViTEncoderLayer(nn.Module):
    """Minimal Vision Transformer encoder block built from shared common blocks."""

    def __init__(self,
                 embed_dim         : int,
                 num_heads         : int,
                 mlp_ratio         : float = 4.0,
                 activation        : Optional[nn.Module] = None,
                 attention_dropout : float = 0.0,
                 dropout           : float = 0.0,
                 norm_first        : bool = True,
                 flash_attention   : bool = False,
             ) -> None:
        """
        Initialize a Vision Transformer encoder layer.
        Args:
            embed_dim         : Embedding dimension for the token sequence.
            num_heads         : Number of attention heads.
            mlp_ratio         : Hidden-size multiplier used for the MLP expansion layer.
            activation        : Optional activation module for the feed-forward block.
            attention_dropout : Dropout probability used inside self-attention.
            dropout           : Residual and MLP dropout probability.
            norm_first        : Whether layer normalization is applied before each sublayer.
            flash_attention   : Whether to prefer torch scaled-dot-product attention kernels.
        Returns:
            None.
        """
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

        self.feed_forward = FeedForward(input_dim   = embed_dim,
                                        hidden_dim  = hidden_dim,
                                        output_dim  = embed_dim,
                                        activation  = activation or nn.GELU(),
                                        dropout     = dropout)

        self.residual_mlp = ResidualBlock(embed_dim  = embed_dim,
                                          module     = self.feed_forward,
                                          dropout    = dropout,
                                          norm_first = norm_first)

    def forward(self,
                x    : Tensor,
                mask : Optional[Tensor] = None,
            ) -> Tensor:
        """
        Run the ViT encoder layer.
        Args:
            x    : Tensor of shape (batch_size, seq_len, embed_dim).
            mask : Optional attention mask broadcastable to self-attention.
        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim).
        """
        x = self.residual_attention(x, mask=mask)
        x = self.residual_mlp(x)
        return x
