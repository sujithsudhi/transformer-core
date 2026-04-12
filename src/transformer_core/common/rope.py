import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding helper for attention Q/K tensors."""

    def __init__(self,
                 head_dim : int,
                 base     : int = 10000,
             ) -> None:
        """
        Initialize rotary embedding frequencies for a given head dimension.
        Args:
            head_dim : Attention head dimension, which must be even.
            base     : Frequency base used to build the inverse-frequency table.
        Returns:
            None.
        Raises:
            ValueError: If head_dim is odd.
        """
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")

        self.head_dim = head_dim
        inv_freq      = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x: Tensor) -> Tensor:
        """
        Swap the two feature halves used by the rotary transform.
        Args:
            x : Tensor whose last dimension is head_dim.
        Returns:
            Tensor with the two feature halves rotated.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _build_cos_sin(self,
                       seq_len         : int,
                       device          : torch.device,
                       dtype           : torch.dtype,
                       position_offset : int = 0,
                   ) -> tuple[Tensor, Tensor]:
        """
        Build cosine and sine rotation tables for the requested sequence window.
        Args:
            seq_len         : Number of current sequence positions.
            device          : Device on which the lookup tables should be materialized.
            dtype           : Output dtype for the rotation tables.
            position_offset : Starting position used for cached decoding.
        Returns:
            Tuple of cosine and sine tensors shaped for attention broadcasting.
        """
        positions = torch.arange(position_offset,
                                 position_offset + seq_len,
                                 device=device,
                                 dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        cos   = torch.cat((torch.cos(freqs), torch.cos(freqs)), dim=-1)
        sin   = torch.cat((torch.sin(freqs), torch.sin(freqs)), dim=-1)
        cos   = cos.unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        sin   = sin.unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        return cos, sin

    def _apply_rotary(self,
                      x   : Tensor,
                      cos : Tensor,
                      sin : Tensor,
                  ) -> Tensor:
        """
        Apply precomputed rotary tables to an attention tensor.
        Args:
            x   : Tensor of shape (batch_size, num_heads, seq_len, head_dim).
            cos : Cosine table broadcastable to x.
            sin : Sine table broadcastable to x.
        Returns:
            Tensor with rotary embeddings applied.
        """
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(self,
                q               : Tensor,
                k               : Tensor,
                position_offset : int = 0,
            ) -> tuple[Tensor, Tensor]:
        """
        Apply rotary embeddings to query and key tensors.
        Args:
            q               : Query tensor of shape (batch_size, num_heads, seq_len, head_dim).
            k               : Key tensor of shape (batch_size, num_heads, seq_len, head_dim).
            position_offset : Starting position used for cached decoding.
        Returns:
            Tuple of rotated query and key tensors.
        Raises:
            ValueError: If q and k do not share the same current sequence length.
        """
        seq_len = q.size(2)
        if k.size(2) != seq_len:
            raise ValueError("RoPE expects q and k to have the same current sequence length.")

        cos, sin = self._build_cos_sin(seq_len         = seq_len,
                                       device          = q.device,
                                       dtype           = q.dtype,
                                       position_offset = position_offset)
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)
