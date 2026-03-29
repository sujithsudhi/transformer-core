from torch import Tensor, nn


class ResidualBlock(nn.Module):
    """Residual wrapper with configurable pre-norm or post-norm behavior."""

    def __init__(self,
                 embed_dim  : int,
                 module     : nn.Module,
                 dropout    : float = 0.0,
                 norm_first : bool = True) -> None:
        super().__init__()

        self.norm_first = norm_first
        self.module     = module
        self.dropout    = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm       = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor | tuple[Tensor, object]:
        out = self.module(self.norm(x), *args, **kwargs) if self.norm_first else self.module(
            x, *args, **kwargs
        )
        extra = None
        if isinstance(out, tuple):
            out, extra = out
        residue = x + self.dropout(out) if self.norm_first else self.norm(x + self.dropout(out))
        if extra is not None:
            return residue, extra
        return residue
