from torch import Tensor, nn
import torch


class DropPath(nn.Module):
    """Stochastic depth applied to residual branches."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x * random_tensor / keep_prob


class ResidualBlock(nn.Module):
    """Residual wrapper with configurable norm, dropout, and stochastic depth."""

    def __init__(self,
                 embed_dim      : int,
                 module         : nn.Module,
                 dropout        : float = 0.0,
                 norm_first     : bool = True,
                 layer_norm_eps : float = 1e-5,
                 drop_path      : float = 0.0) -> None:
        super().__init__()

        self.norm_first = norm_first
        self.module     = module
        self.dropout    = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path  = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm       = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor | tuple[Tensor, object]:
        out = self.module(self.norm(x), *args, **kwargs) if self.norm_first else self.module(
            x, *args, **kwargs
        )
        extra = None
        if isinstance(out, tuple):
            out, extra = out
        residual = self.drop_path(self.dropout(out))
        residue = x + residual if self.norm_first else self.norm(x + residual)
        if extra is not None:
            return residue, extra
        return residue
