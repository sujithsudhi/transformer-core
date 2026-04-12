from torch import Tensor, nn


class FeedForward(nn.Module):
    """Two-layer feed-forward network used inside transformer blocks."""

    def __init__(self,
                 input_dim  : int,
                 hidden_dim : int,
                 output_dim : int,
                 activation : nn.Module | None = None,
                 dropout    : float = 0.0,
                 bias       : bool = True,
             ) -> None:
        """
        Initialize a transformer MLP block.
        Args:
            input_dim  : Input feature dimension.
            hidden_dim : Hidden feature dimension used by the expansion layer.
            output_dim : Output feature dimension, typically equal to input_dim.
            activation : Optional activation module inserted between the two linear layers.
            dropout    : Dropout probability applied after the activation.
            bias       : Whether the linear layers include bias terms.
        Returns:
            None.
        Raises:
            ValueError: If input_dim and output_dim differ.
        """
        if input_dim != output_dim:
            raise ValueError("Input and output dimension should be the same")
        super().__init__()

        self.dropout    = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc1        = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2        = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.activation = activation or nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the feed-forward network to the input tensor.
        Args:
            x : Tensor of shape (..., input_dim).
        Returns:
            Tensor of shape (..., output_dim).
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
