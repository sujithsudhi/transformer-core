import torch

from transformer_core import MultiHeadSelfAttention, TransformerDecoderLayer, TransformerEncoderLayer


def test_multi_head_attention_accepts_padding_mask() -> None:
    attention = MultiHeadSelfAttention(embed_dim=8, num_heads=2)
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

    output = attention(x, mask=mask)

    assert output.shape == (2, 4, 8)


def test_encoder_layer_forward_shape() -> None:
    layer = TransformerEncoderLayer(embed_dim=8, num_heads=2)
    x = torch.randn(2, 4, 8)

    output = layer(x)

    assert output.shape == (2, 4, 8)


def test_decoder_layer_cache_growth() -> None:
    layer = TransformerDecoderLayer(embed_dim=8, num_heads=2)
    x = torch.randn(2, 3, 8)

    output, cache = layer(x, use_cache=True)
    next_output, next_cache = layer(torch.randn(2, 1, 8), past_kv=cache, use_cache=True)

    assert output.shape == (2, 3, 8)
    assert next_output.shape == (2, 1, 8)
    assert cache[0].shape == (2, 2, 3, 4)
    assert next_cache[0].shape == (2, 2, 4, 4)
