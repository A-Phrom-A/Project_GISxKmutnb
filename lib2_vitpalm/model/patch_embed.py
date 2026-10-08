"""
lib2_vitpalm.model.patch_embed
================================
PatchEmbed: แปลง (B, C, H, W) → (B, N, D)
โดย Conv2d ขนาด patch_size x patch_size

รองรับ in_channels=12 (S1+S2+Gsat fused)
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Split image into non-overlapping patches and project to embedding dim.

    Args:
        in_channels : int  — number of input channels (default 12)
        patch_size  : int  — size of each patch in pixels (default 16)
        img_size    : int  — spatial size of input image (default 256)
        embed_dim   : int  — embedding dimension (default 512)
    """

    def __init__(
        self,
        in_channels: int = 12,
        patch_size:  int = 16,
        img_size:    int = 256,
        embed_dim:   int = 512,
    ):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim   = embed_dim

        # Single conv projects each patch to embed_dim
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)
        Returns:
            tokens : (B, N, D)  where N = num_patches
        """
        x = self.proj(x)                    # (B, D, h, w)
        x = x.flatten(2).transpose(1, 2)   # (B, N, D)
        return x
