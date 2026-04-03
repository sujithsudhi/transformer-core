"""
Explainable AI tools for transformer attention mechanisms.

This module provides functions to extract and analyze attention patterns
from transformer models, including attention visualization and rollout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from transformer_core.common.attention import MultiHeadSelfAttention


def _to_attention_tensor(attention_matrix: Tensor | Sequence[Sequence[float]]) -> Tensor:
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
) -> Dict[str, Any]:
    """
    Extract attention weights from a transformer model.

    Args:
        model: The transformer model (should have layers with attention)
        inputs: Input tensor of shape (batch_size, seq_len, embed_dim)
        layer_idx: Specific layer to extract from (None for all layers)
        head_idx: Specific head to extract from (None for all heads)
        mask: Attention mask
        is_causal: Reserved for API compatibility with future decoder-specific flows
        capture_qkv: Whether to also return traced Q/K/V tensors

    Returns:
        Dictionary containing attention weights and metadata
    """
    del is_causal

    attention_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, MultiHeadSelfAttention)
    ]

    if layer_idx is not None:
        if layer_idx < 0 or layer_idx >= len(attention_modules):
            raise IndexError(f"layer_idx={layer_idx} is out of range for {len(attention_modules)} layers")
        attention_modules = [attention_modules[layer_idx]]

    previous_states = [(module.capture_attention, module.capture_qkv) for _, module in attention_modules]
    for _, module in attention_modules:
        module.set_trace(True, capture_qkv=capture_qkv)

    try:
        with torch.no_grad():
            try:
                _ = model(inputs, mask=mask)
            except TypeError:
                _ = model(inputs)

        attention_maps: List[Tensor] = []
        q_tensors: List[Tensor] = []
        k_tensors: List[Tensor] = []
        v_tensors: List[Tensor] = []

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
                module.capture_qkv = capture_saved_qkv
            else:
                module.set_trace(False)

    result = {
        "attention_maps": attention_maps,
        "layer_names": [name for name, _ in attention_modules],
        "num_layers": len(attention_maps),
        "batch_size": inputs.shape[0],
        "seq_len": inputs.shape[1],
    }

    if attention_maps:
        result["num_heads"] = attention_maps[0].shape[1]

    if capture_qkv:
        result["q"] = q_tensors
        result["k"] = k_tensors
        result["v"] = v_tensors

    return result


def rollout_attention(
    attention_maps: List[Tensor],
    residual_connections: bool = True
) -> Tensor:
    """
    Compute attention rollout across layers.

    Args:
        attention_maps: List of attention tensors, one per layer
        residual_connections: Whether to include residual connections

    Returns:
        Rollout attention matrix of shape (seq_len, seq_len)
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
    target_tokens: Optional[List[int]] = None,
    layer_idx: Optional[int] = None,
    head_idx: Optional[int] = None,
    mask: Optional[Tensor] = None,
    is_causal: bool = False,
    use_rollout: bool = True
) -> Dict[str, Any]:
    """
    Main function to explain attention patterns in transformer models.

    Args:
        model: The transformer model
        inputs: Input tensor
        target_tokens: Specific tokens to focus on (None for all)
        layer_idx: Layer to analyze (None for rollout across all)
        head_idx: Head to analyze (None for all heads)
        mask: Attention mask
        is_causal: Whether causal masking is used
        use_rollout: Whether to compute attention rollout

    Returns:
        Dictionary with attention explanations
    """
    attn_data = extract_attention_weights(
        model=model,
        inputs=inputs,
        layer_idx=layer_idx,
        head_idx=head_idx,
        mask=mask,
        is_causal=is_causal,
    )

    result = {
        "attention_data": attn_data,
        "target_tokens": target_tokens,
        "layer_idx": layer_idx,
        "head_idx": head_idx,
    }

    if use_rollout and attn_data["attention_maps"]:
        rollout = rollout_attention(attn_data["attention_maps"])
        result["rollout_attention"] = rollout.cpu()

        if target_tokens:
            valid_tokens = [token_idx for token_idx in target_tokens if 0 <= token_idx < rollout.shape[0]]
            if valid_tokens:
                token_index = torch.as_tensor(valid_tokens, dtype=torch.long, device=rollout.device)
                result["token_importance"] = rollout.index_select(0, token_index).mean(dim=0).cpu()

    return result


def visualize_attention(
    attention_matrix: Tensor | Sequence[Sequence[float]],
    tokens: Optional[List[str]] = None,
    title: str = "Attention Map",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize attention matrix as a heatmap.

    Args:
        attention_matrix: Attention weights of shape (seq_len, seq_len)
        tokens: Token strings for axis labels
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required for visualization")
        return

    matrix = _to_attention_tensor(attention_matrix).cpu().tolist()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        square=True,
    )
    plt.title(title)
    plt.xlabel("To Token")
    plt.ylabel("From Token")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def attention_entropy(attention_matrix: Tensor | Sequence[Sequence[float]]) -> float:
    """
    Compute entropy of attention distribution.

    Args:
        attention_matrix: Attention weights of shape (seq_len, seq_len)

    Returns:
        Average entropy across all positions
    """
    matrix = _to_attention_tensor(attention_matrix)
    if matrix.dim() > 2:
        matrix = matrix.mean(dim=tuple(range(matrix.dim() - 2)))

    row_sums = matrix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    normalized = matrix / row_sums
    entropy = -(normalized * normalized.clamp_min(1e-10).log()).sum(dim=-1).mean()
    return float(entropy.item())


def attention_sparsity(
    attention_matrix: Tensor | Sequence[Sequence[float]],
    threshold: float = 0.1,
) -> float:
    """
    Compute sparsity of attention (fraction of weights below threshold).

    Args:
        attention_matrix: Attention weights
        threshold: Sparsity threshold

    Returns:
        Sparsity ratio
    """
    matrix = _to_attention_tensor(attention_matrix)
    return float((matrix < threshold).to(dtype=torch.float32).mean().item())
