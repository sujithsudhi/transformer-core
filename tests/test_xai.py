"""Tests for XAI (explainability) tools."""

import pytest
import torch

from transformer_core.text.layers import TransformerEncoderLayer
from transformer_core.xai.attention import (
    attention_entropy,
    attention_sparsity,
    explain_attention,
    extract_attention_weights,
    rollout_attention,
)
from transformer_core.xai.gradients import (
    explain_with_gradients,
    gradient_x_input,
    integrated_gradients,
    occlusion_sensitivity,
    saliency_map,
    smooth_gradients,
)


class MockAttentionModel(torch.nn.Module):
    """Mock model for testing attention extraction."""

    def __init__(self, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class MockClassifier(torch.nn.Module):
    """Simple transformer-backed classifier for gradient-based XAI tests."""

    def __init__(self, embed_dim=32, num_heads=4, num_layers=1, num_classes=3):
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


def test_extract_attention_weights():
    """Test attention weight extraction."""
    model = MockAttentionModel()
    batch_size, seq_len, embed_dim = 2, 10, 64
    inputs = torch.randn(batch_size, seq_len, embed_dim)

    result = extract_attention_weights(model, inputs, capture_qkv=True)

    assert "attention_maps" in result
    assert "layer_names" in result
    assert result["num_layers"] == 2
    assert result["batch_size"] == batch_size
    assert result["seq_len"] == seq_len
    assert result["attention_maps"][0].shape == (batch_size, 4, seq_len, seq_len)
    assert len(result["q"]) == 2
    assert result["q"][0].shape == (batch_size, 4, seq_len, embed_dim // 4)


def test_rollout_attention():
    """Test attention rollout computation."""
    seq_len = 5
    attention_maps = [
        torch.softmax(torch.randn(1, 4, seq_len, seq_len), dim=-1),
        torch.softmax(torch.randn(1, 4, seq_len, seq_len), dim=-1),
    ]

    rollout = rollout_attention(attention_maps)

    assert rollout.shape == (seq_len, seq_len)
    assert torch.all(rollout >= 0)
    assert torch.allclose(rollout.sum(dim=-1), torch.ones(seq_len), atol=1e-5)


def test_attention_entropy():
    """Test attention entropy calculation."""
    seq_len = 4
    uniform_attn = torch.ones(seq_len, seq_len) / seq_len
    entropy_uniform = attention_entropy(uniform_attn)

    concentrated_attn = torch.zeros(seq_len, seq_len)
    concentrated_attn[:, 0] = 1.0
    entropy_concentrated = attention_entropy(concentrated_attn)

    assert entropy_concentrated < entropy_uniform


def test_attention_sparsity():
    """Test attention sparsity calculation."""
    seq_len = 4
    sparse_attn = torch.zeros(seq_len, seq_len)
    sparse_attn[0, 0] = 1.0
    sparsity = attention_sparsity(sparse_attn, threshold=0.1)

    assert 0.0 <= sparsity <= 1.0
    assert sparsity > 0.5


def test_integrated_gradients():
    """Test integrated gradients computation."""
    model = MockClassifier(embed_dim=32, num_layers=1)
    batch_size, seq_len, embed_dim = 1, 5, 32
    inputs = torch.randn(batch_size, seq_len, embed_dim)

    attributions, delta = integrated_gradients(model, inputs, steps=10)

    assert attributions.shape == inputs.shape
    assert delta is None

    attributions, delta = integrated_gradients(
        model,
        inputs,
        steps=10,
        return_convergence_delta=True,
    )

    assert attributions.shape == inputs.shape
    assert delta is not None
    assert delta.shape == (batch_size,)


def test_saliency_map():
    """Test saliency map computation."""
    model = MockClassifier(embed_dim=32, num_layers=1)
    batch_size, seq_len, embed_dim = 1, 5, 32
    inputs = torch.randn(batch_size, seq_len, embed_dim)

    saliency = saliency_map(model, inputs)

    assert saliency.shape == inputs.shape
    assert torch.all(saliency >= 0)


def test_gradient_x_input():
    """Test gradients x input attribution."""
    model = MockClassifier(embed_dim=32, num_layers=1)
    batch_size, seq_len, embed_dim = 1, 5, 32
    inputs = torch.randn(batch_size, seq_len, embed_dim)

    attribution = gradient_x_input(model, inputs)

    assert attribution.shape == inputs.shape


def test_smooth_gradients():
    """Test SmoothGrad computation."""
    model = MockClassifier(embed_dim=32, num_layers=1)
    batch_size, seq_len, embed_dim = 1, 5, 32
    inputs = torch.randn(batch_size, seq_len, embed_dim)

    smooth_attr = smooth_gradients(model, inputs, num_samples=5)

    assert smooth_attr.shape == inputs.shape
    assert torch.all(smooth_attr >= 0)


def test_occlusion_sensitivity():
    """Test occlusion sensitivity."""
    model = MockClassifier(embed_dim=32, num_layers=1)
    batch_size, seq_len, embed_dim = 1, 5, 32
    inputs = torch.randn(batch_size, seq_len, embed_dim)

    occlusion_attr = occlusion_sensitivity(model, inputs, occlusion_size=1)

    assert occlusion_attr.shape == (batch_size, seq_len)


def test_explain_with_gradients():
    """Test unified gradient explanation interface."""
    model = MockClassifier(embed_dim=32, num_layers=1)
    batch_size, seq_len, embed_dim = 1, 5, 32
    inputs = torch.randn(batch_size, seq_len, embed_dim)

    result = explain_with_gradients(model, inputs, method="saliency")

    assert "attribution" in result
    assert "method" in result
    assert result["method"] == "saliency"
    assert "token_importance" in result


def test_explain_attention():
    """Test attention explanation."""
    model = MockAttentionModel()
    batch_size, seq_len, embed_dim = 1, 5, 64
    inputs = torch.randn(batch_size, seq_len, embed_dim)

    result = explain_attention(model, inputs, use_rollout=True)

    assert "attention_data" in result
    assert "rollout_attention" in result
    assert result["rollout_attention"].shape == (seq_len, seq_len)


if __name__ == "__main__":
    pytest.main([__file__])
