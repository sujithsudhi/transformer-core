"""
Structured result containers for explainability utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from torch import Tensor


@dataclass
class AttributionResult:
    """Reusable container for attribution tensors and derived scores."""

    attribution: Tensor
    method: str
    token_importance: Optional[Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "attribution": self.attribution,
            "method": self.method,
            "metadata": dict(self.metadata),
        }
        if self.token_importance is not None:
            payload["token_importance"] = self.token_importance
        return payload


@dataclass
class AttentionTraceResult:
    """Structured attention trace for one model forward pass."""

    attention_maps: list[Tensor]
    layer_names: list[str]
    batch_size: int
    seq_len: int
    num_heads: Optional[int] = None
    q: Optional[list[Tensor]] = None
    k: Optional[list[Tensor]] = None
    v: Optional[list[Tensor]] = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "attention_maps": self.attention_maps,
            "layer_names": self.layer_names,
            "num_layers": len(self.attention_maps),
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
        }
        if self.num_heads is not None:
            payload["num_heads"] = self.num_heads
        if self.q is not None:
            payload["q"] = self.q
        if self.k is not None:
            payload["k"] = self.k
        if self.v is not None:
            payload["v"] = self.v
        return payload


@dataclass
class VLMExplanationResult:
    """Joint result container for text and image-side VLM explanations."""

    layout: Any
    text_token_scores: Optional[Tensor] = None
    image_patch_scores: Optional[Tensor] = None
    image_patch_grid_scores: Optional[Tensor] = None
    attention_rollout: Optional[Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "layout": self.layout,
            "metadata": dict(self.metadata),
        }
        if self.text_token_scores is not None:
            payload["text_token_scores"] = self.text_token_scores
        if self.image_patch_scores is not None:
            payload["image_patch_scores"] = self.image_patch_scores
        if self.image_patch_grid_scores is not None:
            payload["image_patch_grid_scores"] = self.image_patch_grid_scores
        if self.attention_rollout is not None:
            payload["attention_rollout"] = self.attention_rollout
        return payload
