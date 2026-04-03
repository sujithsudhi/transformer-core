import torch
from torch import Tensor, nn

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")

        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _build_cos_sin(self,
                       seq_len: int,
                       device: torch.device,
                       dtype: torch.dtype,
                       position_offset: int = 0) -> tuple[Tensor, Tensor]:
        positions = torch.arange(position_offset,
                                 position_offset + seq_len,
                                 device=device,
                                 dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        cos = torch.cat((torch.cos(freqs), torch.cos(freqs)), dim=-1)
        sin = torch.cat((torch.sin(freqs), torch.sin(freqs)), dim=-1)
        cos = cos.unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        sin = sin.unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        return cos, sin

    def _apply_rotary(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(self,
                q: Tensor,
                k: Tensor,
                position_offset: int = 0) -> tuple[Tensor, Tensor]:
        seq_len = q.size(2)
        if k.size(2) != seq_len:
            raise ValueError("RoPE expects q and k to have the same current sequence length.")

        cos, sin = self._build_cos_sin(seq_len=seq_len,
                                       device=q.device,
                                       dtype=q.dtype,
                                       position_offset=position_offset)
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)
