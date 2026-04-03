"""Tests for multimodal XAI metadata helpers."""

import torch

from transformer_core.vision.patch_embedding import PatchEmbedding
from transformer_core.xai.multimodal import build_vlm_layout, infer_patch_grid
from transformer_core.xai.results import AttributionResult, AttentionTraceResult, VLMExplanationResult


def test_infer_patch_grid():
    patch_embedding = PatchEmbedding(image_size=32, patch_size=8, embed_dim=16)
    patch_grid = infer_patch_grid(patch_embedding)

    assert patch_grid.grid_height == 4
    assert patch_grid.grid_width == 4
    assert patch_grid.num_patches == 16


def test_patch_grid_index_round_trip():
    patch_embedding = PatchEmbedding(image_size=32, patch_size=8, embed_dim=16)
    patch_grid = infer_patch_grid(patch_embedding)

    row, col = patch_grid.patch_index_to_coord(6)
    assert (row, col) == (1, 2)
    assert patch_grid.coord_to_patch_index(row, col) == 6


def test_build_vlm_layout_image_first():
    patch_grid = infer_patch_grid(PatchEmbedding(image_size=32, patch_size=8, embed_dim=16))
    layout = build_vlm_layout(
        patch_grid=patch_grid,
        num_text_tokens=5,
        image_first=True,
        cls_tokens=1,
        separator_tokens=1,
    )

    assert layout.total_length == 23
    assert layout.positions("image_patch")[0] == 1
    assert layout.positions("image_patch")[-1] == 16
    assert layout.positions("text_token") == [18, 19, 20, 21, 22]
    span, relative = layout.relative_position(4)
    assert span.modality == "image_patch"
    assert relative == 3


def test_build_vlm_layout_text_first():
    patch_grid = infer_patch_grid(PatchEmbedding(image_size=32, patch_size=8, embed_dim=16))
    layout = build_vlm_layout(
        patch_grid=patch_grid,
        num_text_tokens=4,
        image_first=False,
        cls_tokens=1,
        separator_tokens=2,
        suffix_special_tokens=1,
    )

    assert layout.total_length == 24
    assert layout.positions("text_token") == [1, 2, 3, 4]
    assert layout.positions("image_patch")[0] == 7
    assert layout.positions("image_patch")[-1] == 22
    assert layout.span_for_position(23).modality == "special"


def test_result_containers_to_dict():
    attribution = AttributionResult(
        attribution=torch.ones(1, 2, 3),
        method="saliency",
        token_importance=torch.ones(1, 2),
        metadata={"target": "cls"},
    )
    trace = AttentionTraceResult(
        attention_maps=[torch.ones(1, 2, 3, 3)],
        layer_names=["encoder.layers.0.attention"],
        batch_size=1,
        seq_len=3,
        num_heads=2,
    )
    layout = build_vlm_layout(
        patch_grid=infer_patch_grid(PatchEmbedding(image_size=16, patch_size=8, embed_dim=16)),
        num_text_tokens=2,
        cls_tokens=1,
    )
    vlm = VLMExplanationResult(
        layout=layout,
        text_token_scores=torch.ones(2),
        image_patch_scores=torch.ones(4),
        metadata={"mode": "joint"},
    )

    attribution_dict = attribution.to_dict()
    trace_dict = trace.to_dict()
    vlm_dict = vlm.to_dict()

    assert attribution_dict["method"] == "saliency"
    assert trace_dict["num_layers"] == 1
    assert vlm_dict["metadata"]["mode"] == "joint"
