import pytest
import torch

from transformer_core import PositionalEncoding, TokenEmbedding


def test_token_embedding_respects_padding_idx() -> None:
    embedding = TokenEmbedding(vocab_size=8, embed_dim=4, padding_idx=0)
    tokens = torch.tensor([[0, 1, 2]])

    output = embedding(tokens)

    assert output.shape == (1, 3, 4)
    assert torch.equal(output[0, 0], torch.zeros(4, dtype=output.dtype))


@pytest.mark.parametrize("embed_dim", [6, 7])
def test_positional_encoding_supports_even_and_odd_dims(embed_dim: int) -> None:
    encoding = PositionalEncoding(max_len=8, embed_dim=embed_dim, dropout=0.0)
    x = torch.zeros(2, 3, embed_dim)

    output = encoding(x)

    assert output.shape == (2, 3, embed_dim)
    assert torch.any(output != 0)


def test_positional_encoding_offset_bounds() -> None:
    encoding = PositionalEncoding(max_len=4, embed_dim=6, dropout=0.0)

    with pytest.raises(ValueError, match="Increase max_len"):
        encoding(torch.zeros(1, 3, 6), offset=2)
