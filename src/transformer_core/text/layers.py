from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from transformer_core.common import FeedForward, MultiHeadSelfAttention, ResidualBlock


def _build_activation(activation: Optional[nn.Module | str]) -> nn.Module:
    """
    Resolve an activation specifier into a module instance.
    Args:
        activation : Optional activation module or supported activation name.
    Returns:
        Activation module instance.
    Raises:
        ValueError: If the activation string is unsupported.
    """
    if activation is None:
        return nn.GELU()
    if isinstance(activation, nn.Module):
        return activation
    if activation == "gelu":
        return nn.GELU()
    if activation == "relu":
        return nn.ReLU()
    if activation == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {activation}")


def _resolve_block_config(config: object | None = None,
                          **kwargs: object,
                      ) -> dict[str, object]:
    """
    Normalize explicit keyword arguments or a shared config object into layer kwargs.
    Args:
        config : Optional config object exposing attention and MLP sub-configs.
        **kwargs : Explicit constructor arguments used when config is not provided.
    Returns:
        Dictionary of resolved block configuration values.
    """
    if config is not None:
        attention_cfg = config.attention
        mlp_cfg       = config.mlp

        attention_kwargs = {"embed_dim"        : attention_cfg.embedding_dim,
                            "num_heads"        : attention_cfg.num_heads,
                            "attention_dropout": attention_cfg.dropout,
                            "qkv_bias"         : attention_cfg.qkv_bias,
                            "use_rope"         : False,
                            "rope_base"        : 10000,
                            "flash_attention"  : getattr(config, "flash_attention", False)}
        if hasattr(config, "resolve_attention_kwargs"):
            attention_kwargs.update(config.resolve_attention_kwargs())

        mlp_kwargs = {"mlp_hidden_dim": mlp_cfg.hidden_dim,
                      "activation"    : mlp_cfg.activation,
                      "dropout"       : mlp_cfg.dropout}
        if hasattr(config, "resolve_mlp_kwargs"):
            mlp_kwargs.update(config.resolve_mlp_kwargs())

        resolved = {**attention_kwargs,
                    **mlp_kwargs,
                    "norm_first"    : getattr(config, "pre_norm", True),
                    "layer_norm_eps": getattr(config, "layer_norm_eps", 1e-5),
                    "drop_path"     : getattr(config, "drop_path", 0.0),
                    "attention_type": getattr(config, "attention_type", None),
                    "window_size"   : getattr(config, "window_size", None)}
        return resolved

    embed_dim  = kwargs["embed_dim"]
    num_heads  = kwargs["num_heads"]
    mlp_ratio  = kwargs.get("mlp_ratio", 4.0)
    hidden_dim = kwargs.get("mlp_hidden_dim")

    if hidden_dim is None:
        hidden_dim = int(embed_dim * mlp_ratio)

    return {"embed_dim"        : embed_dim,
            "num_heads"        : num_heads,
            "mlp_hidden_dim"   : hidden_dim,
            "activation"       : kwargs.get("activation"),
            "attention_dropout": kwargs.get("attention_dropout", 0.0),
            "dropout"          : kwargs.get("dropout", 0.0),
            "norm_first"       : kwargs.get("norm_first", True),
            "flash_attention"  : kwargs.get("flash_attention", False),
            "qkv_bias"         : kwargs.get("qkv_bias", True),
            "use_rope"         : kwargs.get("use_rope", False),
            "rope_base"        : kwargs.get("rope_base", 10000),
            "layer_norm_eps"   : kwargs.get("layer_norm_eps", 1e-5),
            "drop_path"        : kwargs.get("drop_path", 0.0),
            "attention_type"   : kwargs.get("attention_type"),
            "window_size"      : kwargs.get("window_size")}


class _TransformerLayerBase(nn.Module):
    """Shared encoder/decoder transformer block utilities."""

    def __init__(self, resolved: dict[str, object]) -> None:
        """
        Build the shared residual attention and MLP sublayers.
        Args:
            resolved : Dictionary of normalized block configuration values.
        Returns:
            None.
        """
        super().__init__()
        self.attention_type = resolved["attention_type"]
        self.window_size    = resolved["window_size"]

        self.residual_attention = ResidualBlock(embed_dim      = resolved["embed_dim"],
                                                module         = MultiHeadSelfAttention(embed_dim       = resolved["embed_dim"],
                                                                                         num_heads       = resolved["num_heads"],
                                                                                         dropout         = resolved["attention_dropout"],
                                                                                         flash_attention = resolved["flash_attention"],
                                                                                         qkv_bias        = resolved["qkv_bias"],
                                                                                         use_rope        = resolved["use_rope"],
                                                                                         rope_base       = resolved["rope_base"]),
                                                dropout        = resolved["dropout"],
                                                norm_first     = resolved["norm_first"],
                                                layer_norm_eps = resolved["layer_norm_eps"],
                                                drop_path      = resolved["drop_path"])

        self.residual_mlp = ResidualBlock(embed_dim      = resolved["embed_dim"],
                                          module         = FeedForward(input_dim   = resolved["embed_dim"],
                                                                       hidden_dim  = resolved["mlp_hidden_dim"],
                                                                       output_dim  = resolved["embed_dim"],
                                                                       activation  = _build_activation(resolved["activation"]),
                                                                       dropout     = resolved["dropout"]),
                                          dropout        = resolved["dropout"],
                                          norm_first     = resolved["norm_first"],
                                          layer_norm_eps = resolved["layer_norm_eps"],
                                          drop_path      = resolved["drop_path"])

    def _forward_block(self,
                       x         : Tensor,
                       mask      : Optional[Tensor] = None,
                       past_kv   : Optional[tuple[Tensor, Tensor]] = None,
                       use_cache : bool = False,
                       is_causal : bool = False,
                   ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Run the shared attention block followed by the residual MLP block.
        Args:
            x         : Tensor of shape (batch_size, seq_len, embed_dim).
            mask      : Optional mask tensor for attention broadcasting rules.
            past_kv   : Optional cached key/value tensors for autoregressive decoding.
            use_cache : Whether to return the newly computed key/value cache.
            is_causal : Whether causal masking should be applied inside attention.
        Returns:
            Output tensor, or a tuple of output tensor and present key/value cache.
        """
        out = self.residual_attention(x,
                                      mask      = mask,
                                      past_kv   = past_kv,
                                      use_cache = use_cache,
                                      is_causal = is_causal)

        if isinstance(out, tuple):
            x, present = out
        else:
            x       = out
            present = None

        x = self.residual_mlp(x)

        if use_cache:
            return x, present
        return x


class TransformerEncoderLayer(_TransformerLayerBase):
    """Transformer encoder block configurable via explicit args or a shared config object."""

    def __init__(self,
                 embed_dim         : Optional[int] = None,
                 num_heads         : Optional[int] = None,
                 mlp_ratio         : float = 4.0,
                 activation        : Optional[nn.Module | str] = None,
                 attention_dropout : float = 0.0,
                 dropout           : float = 0.0,
                 norm_first        : bool = True,
                 flash_attention   : bool = False,
                 qkv_bias          : bool = True,
                 use_rope          : bool = False,
                 rope_base         : int = 10000,
                 mlp_hidden_dim    : Optional[int] = None,
                 layer_norm_eps    : float = 1e-5,
                 drop_path         : float = 0.0,
                 config            : object | None = None,
             ) -> None:
        """
        Initialize a transformer encoder layer.
        Args:
            embed_dim         : Embedding dimension for the input and output sequence.
            num_heads         : Number of attention heads.
            mlp_ratio         : Hidden-size multiplier used when mlp_hidden_dim is omitted.
            activation        : Optional activation module or activation name for the MLP.
            attention_dropout : Dropout probability used inside attention.
            dropout           : Residual and MLP dropout probability.
            norm_first        : Whether layer normalization is applied before each sublayer.
            flash_attention   : Whether to use torch scaled-dot-product attention kernels.
            qkv_bias          : Whether the Q/K/V projections use bias parameters.
            use_rope          : Whether rotary position embeddings are enabled.
            rope_base         : Base used when constructing rotary embeddings.
            mlp_hidden_dim    : Optional explicit hidden dimension for the feed-forward block.
            layer_norm_eps    : Epsilon used by layer normalization.
            drop_path         : Stochastic depth probability for residual branches.
            config            : Optional shared config object used instead of explicit kwargs.
        Returns:
            None.
        """
        resolved = _resolve_block_config(config            = config,
                                         embed_dim         = embed_dim,
                                         num_heads         = num_heads,
                                         mlp_ratio         = mlp_ratio,
                                         activation        = activation,
                                         attention_dropout = attention_dropout,
                                         dropout           = dropout,
                                         norm_first        = norm_first,
                                         flash_attention   = flash_attention,
                                         qkv_bias          = qkv_bias,
                                         use_rope          = use_rope,
                                         rope_base         = rope_base,
                                         mlp_hidden_dim    = mlp_hidden_dim,
                                         layer_norm_eps    = layer_norm_eps,
                                         drop_path         = drop_path)
        super().__init__(resolved)

    def forward(self,
                x    : Tensor,
                mask : Optional[Tensor] = None,
            ) -> Tensor:
        """
        Run the encoder layer.
        Args:
            x    : Tensor of shape (batch_size, seq_len, embed_dim).
            mask : Optional attention mask broadcastable to self-attention.
        Returns:
            Encoded tensor of shape (batch_size, seq_len, embed_dim).
        """
        return self._forward_block(x, mask=mask)


class TransformerDecoderLayer(_TransformerLayerBase):
    """Transformer decoder block configurable via explicit args or a shared config object."""

    def __init__(self,
                 embed_dim         : Optional[int] = None,
                 num_heads         : Optional[int] = None,
                 mlp_ratio         : float = 4.0,
                 activation        : Optional[nn.Module | str] = None,
                 attention_dropout : float = 0.0,
                 dropout           : float = 0.0,
                 norm_first        : bool = True,
                 flash_attention   : bool = False,
                 qkv_bias          : bool = True,
                 use_rope          : bool = False,
                 rope_base         : int = 10000,
                 mlp_hidden_dim    : Optional[int] = None,
                 layer_norm_eps    : float = 1e-5,
                 drop_path         : float = 0.0,
                 config            : object | None = None,
                 *args,
                 **kwargs,
             ) -> None:
        """
        Initialize a transformer decoder layer.
        Args:
            embed_dim         : Embedding dimension for the input and output sequence.
            num_heads         : Number of attention heads.
            mlp_ratio         : Hidden-size multiplier used when mlp_hidden_dim is omitted.
            activation        : Optional activation module or activation name for the MLP.
            attention_dropout : Dropout probability used inside attention.
            dropout           : Residual and MLP dropout probability.
            norm_first        : Whether layer normalization is applied before each sublayer.
            flash_attention   : Whether to use torch scaled-dot-product attention kernels.
            qkv_bias          : Whether the Q/K/V projections use bias parameters.
            use_rope          : Whether rotary position embeddings are enabled.
            rope_base         : Base used when constructing rotary embeddings.
            mlp_hidden_dim    : Optional explicit hidden dimension for the feed-forward block.
            layer_norm_eps    : Epsilon used by layer normalization.
            drop_path         : Stochastic depth probability for residual branches.
            config            : Optional shared config object used instead of explicit kwargs.
            *args             : Unused positional arguments preserved for compatibility.
            **kwargs          : Unused keyword arguments preserved for compatibility.
        Returns:
            None.
        """
        del args, kwargs

        resolved = _resolve_block_config(config            = config,
                                         embed_dim         = embed_dim,
                                         num_heads         = num_heads,
                                         mlp_ratio         = mlp_ratio,
                                         activation        = activation,
                                         attention_dropout = attention_dropout,
                                         dropout           = dropout,
                                         norm_first        = norm_first,
                                         flash_attention   = flash_attention,
                                         qkv_bias          = qkv_bias,
                                         use_rope          = use_rope,
                                         rope_base         = rope_base,
                                         mlp_hidden_dim    = mlp_hidden_dim,
                                         layer_norm_eps    = layer_norm_eps,
                                         drop_path         = drop_path)
        super().__init__(resolved)

    def _build_causal_mask(self,
                           x        : Tensor,
                           mask     : Tensor,
                           past_len : int = 0,
                       ) -> Tensor:
        """
        Combine a lower-triangular causal mask with an optional padding or 3D mask.
        Args:
            x        : Tensor of shape (batch_size, seq_len, embed_dim).
            mask     : Tensor of shape (batch_size, total_len) or (batch_size, seq_len, total_len).
            past_len : Number of cached sequence positions already present in past_kv.
        Returns:
            Boolean attention mask of shape (batch_size, seq_len, total_len).
        Raises:
            ValueError: If the provided mask rank or length is unsupported.
        """
        batch_size, seq_len, _ = x.shape
        total_len              = past_len + seq_len

        causal = torch.tril(torch.ones(total_len, total_len, device=x.device, dtype=torch.bool))
        if seq_len != total_len:
            causal = causal[total_len - seq_len : total_len, :]
        causal = causal.unsqueeze(0).expand(batch_size, seq_len, total_len)

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

    def forward(self,
                x         : Tensor,
                mask      : Optional[Tensor] = None,
                past_kv   : Optional[tuple[Tensor, Tensor]] = None,
                use_cache : bool = False,
            ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Run the decoder layer with optional cached autoregressive state.
        Args:
            x         : Tensor of shape (batch_size, seq_len, embed_dim).
            mask      : Optional padding mask or explicit 3D attention mask.
            past_kv   : Optional cached key/value tensors from previous decode steps.
            use_cache : Whether to return the updated key/value cache.
        Returns:
            Output tensor, or a tuple of output tensor and present key/value cache.
        """
        past_len  = 0 if past_kv is None else past_kv[0].size(2)
        attn_mask = None
        is_causal = False

        if not (use_cache and past_kv is not None and x.size(1) == 1):
            if mask is None:
                is_causal = True
            else:
                attn_mask = self._build_causal_mask(x, mask, past_len=past_len)

        return self._forward_block(x,
                                   mask      = attn_mask,
                                   past_kv   = past_kv,
                                   use_cache = use_cache,
                                   is_causal = is_causal)
