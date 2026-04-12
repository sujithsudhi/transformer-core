"""
Explainable AI tools for transformer attention mechanisms.

This module provides functions to extract and analyze attention patterns
from transformer models, including attention visualization and rollout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import torch
from torch import Tensor

from transformer_core.common.attention import MultiHeadSelfAttention


def _to_attention_tensor(attention_matrix: Tensor | Sequence[Sequence[float]]) -> Tensor:
    """
    Convert attention-like inputs into a float tensor.
    Args:
        attention_matrix : Tensor or nested Python sequence containing attention values.
    Returns:
        Tensor containing detached float32 attention values.
    """
    if isinstance(attention_matrix, Tensor):
        return attention_matrix.detach().to(dtype=torch.float32)
    return torch.as_tensor(attention_matrix, dtype=torch.float32)


def extract_attention_weights(
    model: torch.nn.Module,
    inputs: Tensor,
    layer_idx: Optional[int] = None,
    head_idx: Optional[int] = None,
    mask: Optional[Tensor] = None,
    is_causal: bool = False,
    capture_qkv: bool = False,
) -> dict[str, Any]:
    """
    Extract attention weights from all or part of a transformer model.
    Args:
        model       : Transformer model containing ``MultiHeadSelfAttention`` modules.
        inputs      : Tensor of shape (batch_size, seq_len, embed_dim).
        layer_idx   : Optional attention-module index to extract instead of all layers.
        head_idx    : Optional head index to extract instead of all heads.
        mask        : Optional attention mask passed to the model forward call.
        is_causal   : Reserved for API compatibility with future decoder-specific flows.
        capture_qkv : Whether traced Q/K/V tensors should also be returned.
    Returns:
        Dictionary containing traced attention maps, module names, and optional Q/K/V tensors.
    Raises:
        IndexError: If layer_idx or head_idx is outside the traced range.
    """
    del is_causal

    attention_modules = [(name, module)
                         for name, module in model.named_modules()
                         if isinstance(module, MultiHeadSelfAttention)]

    if layer_idx is not None:
        if layer_idx < 0 or layer_idx >= len(attention_modules):
            raise IndexError(f"layer_idx={layer_idx} is out of range for {len(attention_modules)} layers")
        attention_modules = [attention_modules[layer_idx]]

    previous_states = [(module.capture_attention, module.capture_qkv)
                       for _, module in attention_modules]
    for _, module in attention_modules:
        module.set_trace(True, capture_qkv=capture_qkv)

    try:
        with torch.no_grad():
            try:
                _ = model(inputs, mask=mask)
            except TypeError:
                _ = model(inputs)

        attention_maps: list[Tensor] = []
        q_tensors     : list[Tensor] = []
        k_tensors     : list[Tensor] = []
        v_tensors     : list[Tensor] = []

        for _, module in attention_modules:
            attention_map = module.last_attention_weights
            if attention_map is None:
                continue

            if head_idx is not None:
                if head_idx < 0 or head_idx >= attention_map.shape[1]:
                    raise IndexError(
                        f"head_idx={head_idx} is out of range for {attention_map.shape[1]} heads"
                    )
                attention_map = attention_map[:, head_idx : head_idx + 1]

            attention_maps.append(attention_map.detach().cpu())

            if capture_qkv:
                q_tensor = module.last_q
                k_tensor = module.last_k
                v_tensor = module.last_v
                if q_tensor is None or k_tensor is None or v_tensor is None:
                    continue
                if head_idx is not None:
                    q_tensor = q_tensor[:, head_idx : head_idx + 1]
                    k_tensor = k_tensor[:, head_idx : head_idx + 1]
                    v_tensor = v_tensor[:, head_idx : head_idx + 1]
                q_tensors.append(q_tensor.detach().cpu())
                k_tensors.append(k_tensor.detach().cpu())
                v_tensors.append(v_tensor.detach().cpu())
    finally:
        for (_, module), (capture_attention, capture_saved_qkv) in zip(attention_modules, previous_states):
            if capture_attention:
                module.capture_attention = True
                module.capture_qkv       = capture_saved_qkv
            else:
                module.set_trace(False)

    result = {"attention_maps": attention_maps,
              "layer_names"   : [name for name, _ in attention_modules],
              "num_layers"    : len(attention_maps),
              "batch_size"    : inputs.shape[0],
              "seq_len"       : inputs.shape[1]}

    if attention_maps:
        result["num_heads"] = attention_maps[0].shape[1]

    if capture_qkv:
        result["q"] = q_tensors
        result["k"] = k_tensors
        result["v"] = v_tensors

    return result


def rollout_attention(attention_maps: list[Tensor],
                      residual_connections: bool = True,
                  ) -> Tensor:
    """
    Compute attention rollout across multiple layers.
    Args:
        attention_maps        : List of attention tensors, one per layer.
        residual_connections  : Whether residual self-connections are added before rollout.
    Returns:
        Rollout attention tensor of shape (seq_len, seq_len).
    Raises:
        ValueError: If no attention maps are provided.
    """
    if not attention_maps:
        raise ValueError("No attention maps provided")

    rollout = torch.eye(attention_maps[0].shape[-1], device=attention_maps[0].device)

    for attn in attention_maps:
        if attn.dim() == 4:
            attn = attn.mean(dim=1)
        if attn.dim() == 3:
            attn = attn.mean(dim=0)

        if residual_connections:
            attn = attn + torch.eye(attn.shape[-1], device=attn.device)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        rollout = torch.matmul(attn, rollout)

    return rollout


def explain_attention(
    model: torch.nn.Module,
    inputs: Tensor,
    target_tokens: Optional[list[int]] = None,
    layer_idx: Optional[int] = None,
    head_idx: Optional[int] = None,
    mask: Optional[Tensor] = None,
    is_causal: bool = False,
    use_rollout: bool = True,
) -> dict[str, Any]:
    """
    Explain transformer behavior using attention tracing and optional rollout.
    Args:
        model         : Transformer model containing attention modules.
        inputs        : Tensor of shape (batch_size, seq_len, embed_dim).
        target_tokens : Optional token indices used to pool rollout importance.
        layer_idx     : Optional attention-module index for single-layer analysis.
        head_idx      : Optional attention-head index for single-head analysis.
        mask          : Optional attention mask passed to the model.
        is_causal     : Whether the underlying model should be treated as causal.
        use_rollout   : Whether to compute rollout attention across layers.
    Returns:
        Dictionary containing traced attention data and derived explanation tensors.
    """
    attn_data = extract_attention_weights(model      = model,
                                          inputs     = inputs,
                                          layer_idx  = layer_idx,
                                          head_idx   = head_idx,
                                          mask       = mask,
                                          is_causal  = is_causal,
                                          capture_qkv = False)

    result = {"attention_data": attn_data,
              "target_tokens" : target_tokens,
              "layer_idx"     : layer_idx,
              "head_idx"      : head_idx}

    if use_rollout and attn_data["attention_maps"]:
        rollout = rollout_attention(attn_data["attention_maps"])
        result["rollout_attention"] = rollout.cpu()

        if target_tokens:
            valid_tokens = [token_idx for token_idx in target_tokens if 0 <= token_idx < rollout.shape[0]]
            if valid_tokens:
                token_index = torch.as_tensor(valid_tokens, dtype=torch.long, device=rollout.device)
                result["token_importance"] = rollout.index_select(0, token_index).mean(dim=0).cpu()

    return result


def visualize_attention(attention_matrix: Tensor | Sequence[Sequence[float]],
                        tokens: Optional[list[str]] = None,
                        title: str = "Attention Map",
                        save_path: Optional[str] = None,
                    ) -> None:
    """
    Visualize an attention matrix as a heatmap.
    Args:
        attention_matrix : Attention weights of shape (seq_len, seq_len) or compatible nested sequence.
        tokens           : Optional token strings used as axis labels.
        title            : Plot title.
        save_path        : Optional path used to save the figure instead of displaying it.
    Returns:
        None.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required for visualization")
        return

    matrix = _to_attention_tensor(attention_matrix).cpu().tolist()
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix,
                xticklabels = tokens,
                yticklabels = tokens,
                cmap        = "viridis",
                square      = True)
    plt.title(title)
    plt.xlabel("To Token")
    plt.ylabel("From Token")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def attention_entropy(attention_matrix: Tensor | Sequence[Sequence[float]]) -> float:
    """
    Compute the mean entropy of an attention distribution.
    Args:
        attention_matrix : Attention weights of shape (seq_len, seq_len) or a higher-rank attention tensor.
    Returns:
        Average entropy across all positions.
    """
    matrix = _to_attention_tensor(attention_matrix)
    if matrix.dim() > 2:
        matrix = matrix.mean(dim=tuple(range(matrix.dim() - 2)))

    row_sums   = matrix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    normalized = matrix / row_sums
    entropy    = -(normalized * normalized.clamp_min(1e-10).log()).sum(dim=-1).mean()
    return float(entropy.item())


def attention_sparsity(attention_matrix: Tensor | Sequence[Sequence[float]],
                       threshold: float = 0.1,
                   ) -> float:
    """
    Compute the fraction of attention weights below a threshold.
    Args:
        attention_matrix : Attention weights of any shape.
        threshold        : Sparsity threshold.
    Returns:
        Fraction of weights smaller than the threshold.
    """
    matrix = _to_attention_tensor(attention_matrix)
    return float((matrix < threshold).to(dtype=torch.float32).mean().item())
