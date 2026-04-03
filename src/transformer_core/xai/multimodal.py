"""
VLM-oriented metadata helpers for patch/text sequence layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
