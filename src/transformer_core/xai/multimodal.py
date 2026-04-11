"""
VLM-oriented metadata helpers for patch/text sequence layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from transformer_core.xai.attention import explain_attention
from transformer_core.xai.gradients import explain_with_gradients
from transformer_core.xai.results import VLMExplanationResult
from transformer_core.vision.patch_embedding import PatchEmbedding


@dataclass(frozen=True)
class PatchGrid:
    """Image patch grid metadata for ViT-style patch embeddings."""

    image_height: int
    image_width: int
    patch_height: int
    patch_width: int
    grid_height: int
    grid_width: int

    @property
    def num_patches(self) -> int:
        return self.grid_height * self.grid_width

    def patch_index_to_coord(self, patch_index: int) -> tuple[int, int]:
        if patch_index < 0 or patch_index >= self.num_patches:
            raise IndexError(f"patch_index={patch_index} is out of range for {self.num_patches} patches")
        return divmod(patch_index, self.grid_width)

    def coord_to_patch_index(self, row: int, col: int) -> int:
        if row < 0 or row >= self.grid_height:
            raise IndexError(f"row={row} is out of range for grid height {self.grid_height}")
        if col < 0 or col >= self.grid_width:
            raise IndexError(f"col={col} is out of range for grid width {self.grid_width}")
        return row * self.grid_width + col


def infer_patch_grid(patch_embedding: PatchEmbedding) -> PatchGrid:
    """Build patch-grid metadata from a PatchEmbedding module."""

    return PatchGrid(
        image_height=patch_embedding.image_size,
        image_width=patch_embedding.image_size,
        patch_height=patch_embedding.patch_size,
        patch_width=patch_embedding.patch_size,
        grid_height=patch_embedding.grid_size,
        grid_width=patch_embedding.grid_size,
    )


@dataclass(frozen=True)
class ModalitySpan:
    """Contiguous region in a multimodal sequence."""

    modality: str
    start: int
    end: int
    label: Optional[str] = None

    @property
    def length(self) -> int:
        return self.end - self.start

    def contains(self, position: int) -> bool:
        return self.start <= position < self.end

    def positions(self) -> list[int]:
        return list(range(self.start, self.end))


@dataclass(frozen=True)
class MultimodalLayout:
    """Sequence layout describing where each modality lives in a fused VLM input."""

    spans: tuple[ModalitySpan, ...]

    @property
    def total_length(self) -> int:
        return max((span.end for span in self.spans), default=0)

    def positions(self, modality: str) -> list[int]:
        result: list[int] = []
        for span in self.spans:
            if span.modality == modality:
                result.extend(span.positions())
        return result

    def span_for_position(self, position: int) -> ModalitySpan:
        for span in self.spans:
            if span.contains(position):
                return span
        raise IndexError(f"position={position} is outside the multimodal layout")

    def relative_position(self, position: int) -> tuple[ModalitySpan, int]:
        span = self.span_for_position(position)
        return span, position - span.start


def build_vlm_layout(
    *,
    patch_grid: PatchGrid,
    num_text_tokens: int,
    image_first: bool = True,
    cls_tokens: int = 0,
    separator_tokens: int = 0,
    suffix_special_tokens: int = 0,
) -> MultimodalLayout:
    """
    Build a simple fused text/image layout for VLM inputs.

    The layout supports common forms like:
    `[CLS] [IMG_PATCHES] [SEP] [TEXT]`
    or
    `[CLS] [TEXT] [SEP] [IMG_PATCHES]`
    """
    spans: list[ModalitySpan] = []
    cursor = 0

    if cls_tokens > 0:
        spans.append(ModalitySpan("special", cursor, cursor + cls_tokens, label="cls"))
        cursor += cls_tokens

    primary_modality = "image_patch" if image_first else "text_token"
    secondary_modality = "text_token" if image_first else "image_patch"
    primary_length = patch_grid.num_patches if image_first else num_text_tokens
    secondary_length = num_text_tokens if image_first else patch_grid.num_patches

    spans.append(ModalitySpan(primary_modality, cursor, cursor + primary_length))
    cursor += primary_length

    if separator_tokens > 0:
        spans.append(ModalitySpan("special", cursor, cursor + separator_tokens, label="separator"))
        cursor += separator_tokens

    spans.append(ModalitySpan(secondary_modality, cursor, cursor + secondary_length))
    cursor += secondary_length

    if suffix_special_tokens > 0:
        spans.append(ModalitySpan("special", cursor, cursor + suffix_special_tokens, label="suffix"))

    return MultimodalLayout(spans=tuple(spans))


def select_modality_scores(scores: Tensor, layout: MultimodalLayout, modality: str) -> Tensor:
    """Select the scores that belong to a specific modality."""

    positions = layout.positions(modality)
    if not positions:
        shape = scores.shape[:-1] + (0,) if scores.dim() > 0 else (0,)
        return scores.new_empty(shape)

    indices = torch.as_tensor(positions, dtype=torch.long, device=scores.device)
    if scores.dim() == 1:
        return scores.index_select(0, indices)
    return scores.index_select(-1, indices)


def reshape_patch_scores(patch_scores: Tensor, patch_grid: PatchGrid) -> Tensor:
    """Project flat patch scores back onto the 2D patch grid."""

    if patch_scores.shape[-1] != patch_grid.num_patches:
        raise ValueError(
            f"Expected {patch_grid.num_patches} patch scores, got {patch_scores.shape[-1]}"
        )

    if patch_scores.dim() == 1:
        return patch_scores.view(patch_grid.grid_height, patch_grid.grid_width)
    if patch_scores.dim() == 2:
        return patch_scores.view(patch_scores.shape[0], patch_grid.grid_height, patch_grid.grid_width)
    raise ValueError("patch_scores must be a 1D or 2D tensor.")


def explain_vlm_with_gradients(
    model: torch.nn.Module,
    inputs: Tensor,
    *,
    layout: MultimodalLayout,
    patch_grid: PatchGrid,
    method: str = "integrated_gradients",
    target: Optional[Tensor] = None,
    **kwargs,
) -> VLMExplanationResult:
    """
    Run an existing gradient explainer and split the resulting scores by modality.
    """
    gradient_result = explain_with_gradients(
        model=model,
        inputs=inputs,
        method=method,
        target=target,
        **kwargs,
    )
    token_scores = gradient_result["token_importance"]
    text_scores = select_modality_scores(token_scores, layout, "text_token")
    image_scores = select_modality_scores(token_scores, layout, "image_patch")
    image_grid_scores = reshape_patch_scores(image_scores, patch_grid) if image_scores.numel() else None

    metadata = {
        "method": method,
        "raw_result": gradient_result,
    }
    return VLMExplanationResult(
        layout=layout,
        text_token_scores=text_scores,
        image_patch_scores=image_scores,
        image_patch_grid_scores=image_grid_scores,
        metadata=metadata,
    )


def explain_vlm_attention(
    model: torch.nn.Module,
    inputs: Tensor,
    *,
    layout: MultimodalLayout,
    patch_grid: PatchGrid,
    target_positions: Optional[list[int]] = None,
    layer_idx: Optional[int] = None,
    head_idx: Optional[int] = None,
    mask: Optional[Tensor] = None,
    use_rollout: bool = True,
) -> VLMExplanationResult:
    """
    Run attention-based explanation and split the resulting rollout scores by modality.
    """
    if target_positions is None:
        target_positions = layout.positions("text_token")

    attention_result = explain_attention(
        model=model,
        inputs=inputs,
        target_tokens=target_positions,
        layer_idx=layer_idx,
        head_idx=head_idx,
        mask=mask,
        use_rollout=use_rollout,
    )

    token_scores = attention_result.get("token_importance")
    text_scores = None if token_scores is None else select_modality_scores(token_scores, layout, "text_token")
    image_scores = None if token_scores is None else select_modality_scores(token_scores, layout, "image_patch")
    image_grid_scores = None
    if image_scores is not None and image_scores.numel():
        image_grid_scores = reshape_patch_scores(image_scores, patch_grid)

    return VLMExplanationResult(
        layout=layout,
        text_token_scores=text_scores,
        image_patch_scores=image_scores,
        image_patch_grid_scores=image_grid_scores,
        attention_rollout=attention_result.get("rollout_attention"),
        metadata={
            "attention_result": attention_result,
            "target_positions": target_positions,
        },
    )
