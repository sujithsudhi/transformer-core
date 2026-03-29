import torch

from transformer_core import PatchEmbedding, ViTEncoderLayer


def test_patch_embedding_flattens_into_tokens() -> None:
    embedding = PatchEmbedding(image_size=32, patch_size=8, in_channels=3, embed_dim=16)
    x = torch.randn(2, 3, 32, 32)

    output = embedding(x)

    assert output.shape == (2, 16, 16)


def test_vit_encoder_layer_forward_shape() -> None:
    layer = ViTEncoderLayer(embed_dim=16, num_heads=4)
    x = torch.randn(2, 17, 16)

    output = layer(x)

    assert output.shape == (2, 17, 16)
