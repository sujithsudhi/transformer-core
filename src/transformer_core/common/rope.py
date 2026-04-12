import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding helper for attention Q/K tensors."""

    def __init__(self,
                 head_dim    : int,
                 base        : int = 10000,
                 max_seq_len : int = 2048,
             ) -> None:
        """
        Initialize rotary embedding frequencies and an initial cosine/sine cache.
        Args:
            head_dim    : Attention head dimension, which must be even.
            base        : Frequency base used to build the inverse-frequency table.
            max_seq_len : Initial number of positions to precompute for reuse.
        Returns:
            None.
        Raises:
            ValueError: If head_dim is odd or max_seq_len is not positive.
        """
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        cos_cached, sin_cached = self._build_cache(max_seq_len=max_seq_len)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

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

    def _build_cache(self, max_seq_len: int) -> tuple[Tensor, Tensor]:
        """
        Precompute cosine and sine tables up to a maximum sequence length.
        Args:
            max_seq_len : Number of positions to precompute.
        Returns:
            Tuple of cached cosine and sine tensors shaped for attention broadcasting.
        """
        positions = torch.arange(max_seq_len,
                                 device=self.inv_freq.device,
                                 dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        cos   = torch.cat((torch.cos(freqs), torch.cos(freqs)), dim=-1)
        sin   = torch.cat((torch.sin(freqs), torch.sin(freqs)), dim=-1)
        cos   = cos.unsqueeze(0).unsqueeze(0)
        sin   = sin.unsqueeze(0).unsqueeze(0)
        return cos, sin

    def _ensure_cache_capacity(self, required_len: int) -> None:
        """
        Grow the cached cosine/sine tables when a longer sequence is requested.
        Args:
            required_len : Total number of positions needed by the current call.
        Returns:
            None.
        """
        if required_len <= self.max_seq_len:
            return

        new_max_seq_len = max(required_len, self.max_seq_len * 2)
        cos_cached, sin_cached = self._build_cache(max_seq_len=new_max_seq_len)
        self.max_seq_len = new_max_seq_len
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def _build_cos_sin(self,
                       seq_len         : int,
                       dtype           : torch.dtype,
                       position_offset : int = 0,
                   ) -> tuple[Tensor, Tensor]:
        """
        Slice cached cosine and sine tables for the requested sequence window.
        Args:
            seq_len         : Number of current sequence positions.
            dtype           : Output dtype for the sliced rotation tables.
            position_offset : Starting position used for cached decoding.
        Returns:
            Tuple of cosine and sine tensors shaped for attention broadcasting.
        Raises:
            ValueError: If seq_len or position_offset is negative.
        """
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative.")
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative.")

        required_len = position_offset + seq_len
        self._ensure_cache_capacity(required_len)

        cos = self.cos_cached[:, :, position_offset : required_len, :].to(dtype=dtype)
        sin = self.sin_cached[:, :, position_offset : required_len, :].to(dtype=dtype)
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
                                       dtype           = q.dtype,
                                       position_offset = position_offset)
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)
