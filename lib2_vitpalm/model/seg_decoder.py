"""
lib2_vitpalm.model.seg_decoder
================================
Segmentation Decoder: แปลง ViT token sequence (B, N, D)
กลับมาเป็น probability map (B, H, W)

Pipeline:
  (B, N, D) → reshape (B, D, h, w)  h=w=img_size/patch_size
            → ConvTranspose2d upsample ×16 → (B, 1, H, W)
            → sigmoid → (B, H, W)  probability map [0, 1]
"""

import torch
import torch.nn as nn
import math


class SegDecoder(nn.Module):
    """
    Convolutional upsampling decoder for palm segmentation.

    Reshapes patch token sequence back to spatial feature map,
    then progressively upsamples to original image resolution.

    Args:
        embed_dim  : int — ViT embedding dimension (must match encoder)
        patch_size : int — patch size used in ViTEncoder (default 16)
        img_size   : int — target output size (default 256)
    """

    def __init__(
        self,
        embed_dim:  int = 512,
        patch_size: int = 16,
        img_size:   int = 256,
    ):
        super().__init__()
        self.h = img_size // patch_size      # spatial grid H (e.g. 16)
        self.w = img_size // patch_size      # spatial grid W (e.g. 16)
        self.embed_dim = embed_dim

        # n_upsamples: log2(patch_size) — each ConvTranspose2d doubles resolution
        n_up = int(math.log2(patch_size))   # 16 → 4 upsamples

        # Build progressive upsample: embed_dim → 256 → 128 → 64 → 32 → 1
        dims = [embed_dim] + [max(32, embed_dim // (2 ** i)) for i in range(1, n_up + 1)]
        dims[-1] = 32   # ensure final feature dim before palm head

        layers = []
        for i in range(n_up):
            layers += [
                nn.ConvTranspose2d(dims[i], dims[i + 1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(dims[i + 1]),
                nn.GELU(),
            ]

        # Palm head: 1×1 conv → single channel
        layers.append(nn.Conv2d(dims[-1], 1, kernel_size=1))

        self.up = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens : (B, N, D) — patch tokens from ViTEncoder (CLS excluded)
        Returns:
            prob_map : (B, H, W) — palm probability map [0, 1]
        """
        B, N, D = tokens.shape
        # Reshape to spatial grid
        x = tokens.transpose(1, 2).reshape(B, D, self.h, self.w)  # (B, D, h, w)
        x = self.up(x)            # (B, 1, H, W)
        x = torch.sigmoid(x)
        return x.squeeze(1)       # (B, H, W)
