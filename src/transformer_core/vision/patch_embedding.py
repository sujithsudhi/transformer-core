from __future__ import annotations

from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """Convert images into patch tokens using a strided convolution."""

    def __init__(self,
                 image_size  : int = 224,
                 patch_size  : int = 16,
                 in_channels : int = 3,
                 embed_dim   : int = 768,
                 flatten     : bool = True,
             ) -> None:
        """
        Initialize the image-to-patch projection module.
        Args:
            image_size  : Input image height and width in pixels.
            patch_size  : Patch height and width in pixels.
            in_channels : Number of image input channels.
            embed_dim   : Embedding dimension produced for each patch.
            flatten     : Whether to return flattened patch tokens instead of a feature map.
        Returns:
            None.
        Raises:
            ValueError: If image_size is not divisible by patch_size.
        """
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size  = image_size
        self.patch_size  = patch_size
        self.grid_size   = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.flatten     = flatten

        self.proj = nn.Conv2d(in_channels,
                              embed_dim,
                              kernel_size = patch_size,
                              stride      = patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Project an image batch into patch embeddings.
        Args:
            x : Tensor of shape (batch_size, in_channels, image_size, image_size).
        Returns:
            Tensor of shape (batch_size, num_patches, embed_dim) when flatten is True,
            otherwise tensor of shape (batch_size, embed_dim, grid_size, grid_size).
        """
        x = self.proj(x)

        if self.flatten:
            return x.flatten(2).transpose(1, 2)
        return x
