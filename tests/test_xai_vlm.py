"""Tests for VLM-oriented XAI helpers."""

import torch

from transformer_core.text.layers import TransformerEncoderLayer
from transformer_core.vision.patch_embedding import PatchEmbedding
from transformer_core.xai.multimodal import (
    build_vlm_layout,
    explain_vlm_attention,
    explain_vlm_with_gradients,
    infer_patch_grid,
    reshape_patch_scores,
    select_modality_scores,
)


class MockVLMClassifier(torch.nn.Module):
    """Simple fused-sequence model for VLM-style explainability tests."""

    def __init__(self, embed_dim=32, num_heads=4, num_layers=2, num_classes=3):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.classifier = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


def test_select_modality_scores_and_patch_reshape():
    patch_grid = infer_patch_grid(PatchEmbedding(image_size=16, patch_size=8, embed_dim=16))
    layout = build_vlm_layout(
        patch_grid=patch_grid,
        num_text_tokens=3,
        cls_tokens=1,
        separator_tokens=1,
    )
    token_scores = torch.arange(layout.total_length, dtype=torch.float32)

    text_scores = select_modality_scores(token_scores, layout, "text_token")
    image_scores = select_modality_scores(token_scores, layout, "image_patch")
    image_grid = reshape_patch_scores(image_scores, patch_grid)

    assert text_scores.shape == (3,)
    assert image_scores.shape == (4,)
    assert image_grid.shape == (2, 2)


def test_explain_vlm_with_gradients():
    patch_grid = infer_patch_grid(PatchEmbedding(image_size=16, patch_size=8, embed_dim=16))
    layout = build_vlm_layout(
        patch_grid=patch_grid,
        num_text_tokens=3,
        cls_tokens=1,
        separator_tokens=1,
    )
    model = MockVLMClassifier(embed_dim=32, num_layers=1)
    inputs = torch.randn(1, layout.total_length, 32)

    result = explain_vlm_with_gradients(
        model,
        inputs,
        layout=layout,
        patch_grid=patch_grid,
        method="saliency",
    )

    assert result.text_token_scores is not None
    assert result.image_patch_scores is not None
    assert result.image_patch_grid_scores is not None
    assert result.text_token_scores.shape == (1, 3)
    assert result.image_patch_scores.shape == (1, 4)
    assert result.image_patch_grid_scores.shape == (1, 2, 2)


def test_explain_vlm_attention():
    patch_grid = infer_patch_grid(PatchEmbedding(image_size=16, patch_size=8, embed_dim=16))
    layout = build_vlm_layout(
        patch_grid=patch_grid,
        num_text_tokens=3,
        cls_tokens=1,
        separator_tokens=1,
    )
    model = MockVLMClassifier(embed_dim=32, num_layers=2)
    inputs = torch.randn(1, layout.total_length, 32)

    result = explain_vlm_attention(
        model,
        inputs,
        layout=layout,
        patch_grid=patch_grid,
    )

    assert result.attention_rollout is not None
    assert result.text_token_scores is not None
    assert result.image_patch_scores is not None
    assert result.image_patch_grid_scores is not None
    assert result.attention_rollout.shape == (layout.total_length, layout.total_length)
    assert result.text_token_scores.shape == (3,)
    assert result.image_patch_scores.shape == (4,)
    assert result.image_patch_grid_scores.shape == (2, 2)
