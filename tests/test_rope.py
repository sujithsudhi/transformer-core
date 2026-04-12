import torch

from transformer_core.common.rope import RotaryEmbedding


def test_rotary_embedding_reuses_precomputed_cache() -> None:
    rope = RotaryEmbedding(head_dim=4, max_seq_len=8)
    q = torch.randn(2, 3, 4, 4)
    k = torch.randn(2, 3, 4, 4)

    cos_ptr_before = rope.cos_cached.data_ptr()
    sin_ptr_before = rope.sin_cached.data_ptr()

    rotated_q, rotated_k = rope(q, k, position_offset=2)

    expected_cos = rope.cos_cached[:, :, 2:6, :].to(dtype=q.dtype)
    expected_sin = rope.sin_cached[:, :, 2:6, :].to(dtype=q.dtype)

    assert rope.cos_cached.data_ptr() == cos_ptr_before
    assert rope.sin_cached.data_ptr() == sin_ptr_before
    assert torch.allclose(rotated_q, (q * expected_cos) + (rope._rotate_half(q) * expected_sin))
    assert torch.allclose(rotated_k, (k * expected_cos) + (rope._rotate_half(k) * expected_sin))


def test_rotary_embedding_grows_cache_when_needed() -> None:
    rope = RotaryEmbedding(head_dim=4, max_seq_len=2)
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)

    rope(q, k, position_offset=2)

    assert rope.max_seq_len >= 5
    assert rope.cos_cached.shape == (1, 1, rope.max_seq_len, 4)
    assert rope.sin_cached.shape == (1, 1, rope.max_seq_len, 4)
