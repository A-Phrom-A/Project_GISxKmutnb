"""
lib2_vitpalm.model.mae_encoder
================================
MAE Encoder — extension ของ ViTEncoder สำหรับ
Masked Autoencoder self-supervised pretraining.

เพิ่ม random_masking() utility ที่ส่ง visible indices ให้ ViTEncoder
"""

import torch
import torch.nn as nn
from lib2_vitpalm.model.vit_encoder import ViTEncoder


def random_masking(
    B: int,
    N: int,
    mask_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random masking indices for MAE.

    Args:
        B          : batch size
        N          : total number of patches
        mask_ratio : fraction of patches to mask (e.g. 0.75)
        device     : torch device

    Returns:
        ids_keep   : (B, N_keep)   — indices of visible patches
        ids_masked : (B, N_masked) — indices of masked patches
    """
    N_keep   = int(N * (1 - mask_ratio))
    noise    = torch.rand(B, N, device=device)
    ids_all  = torch.argsort(noise, dim=1)
    ids_keep   = ids_all[:, :N_keep]
    ids_masked = ids_all[:, N_keep:]
    return ids_keep, ids_masked


class MAEDecoder(nn.Module):
    """
    Simple linear decoder for MAE reconstruction.
    Predicts pixel values of masked patches.

    Args:
        embed_dim  : int — ViT embedding dimension
        patch_size : int — patch size in pixels
        num_bands  : int — number of input channels (default 12)
    """

    def __init__(
        self,
        embed_dim:  int = 512,
        patch_size: int = 16,
        num_bands:  int = 12,
    ):
        super().__init__()
        self.pred_dim = patch_size * patch_size * num_bands
        self.pred = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.pred_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens : (B, N_visible, D) — visible token outputs from ViTEncoder
        Returns:
            pred : (B, N_visible, patch_size^2 * num_bands) — predicted pixel values
        """
        return self.pred(tokens)


class MAEModel(nn.Module):
    """
    Full MAE model: ViTEncoder + MAEDecoder.
    ใช้สำหรับ self-supervised pretraining บน satellite tiles.

    Args:
        in_channels  : int   — number of input channels (default 12)
        img_size     : int   — tile spatial size (default 256)
        patch_size   : int   — patch size (default 16)
        embed_dim    : int   — embedding dim (default 512)
        depth        : int   — transformer depth (default 6)
        num_heads    : int   — attention heads (default 8)
        mlp_ratio    : float — MLP ratio (default 4.0)
        mask_ratio   : float — fraction of patches to mask (default 0.75)
    """

    def __init__(
        self,
        in_channels: int   = 12,
        img_size:    int   = 256,
        patch_size:  int   = 16,
        embed_dim:   int   = 512,
        depth:       int   = 6,
        num_heads:   int   = 8,
        mlp_ratio:   float = 4.0,
        mask_ratio:  float = 0.75,
    ):
        super().__init__()
        self.mask_ratio  = mask_ratio
        self.patch_size  = patch_size
        self.num_bands   = in_channels

        self.encoder = ViTEncoder(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=0.0,       # No dropout during MAE pretraining
        )

        self.decoder = MAEDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_bands=in_channels,
        )

        self.num_patches = self.encoder.num_patches

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for MAE pretraining.

        Args:
            x : (B, C, H, W) — satellite tile

        Returns:
            pred       : (B, N_keep, patch_px) — predicted pixel values for visible patches
            ids_keep   : (B, N_keep)            — visible patch indices
            ids_masked : (B, N_masked)          — masked patch indices
        """
        B = x.size(0)
        device = x.device

        # Random masking
        ids_keep, ids_masked = random_masking(
            B, self.num_patches, self.mask_ratio, device
        )

        # Encode only visible patches
        tokens = self.encoder(x, mask_indices=ids_keep)  # (B, N_keep+1, D)

        # Decode visible patches (skip CLS token at index 0)
        pred = self.decoder(tokens[:, 1:, :])             # (B, N_keep, patch_px)

        return pred, ids_keep, ids_masked

    def get_patch_targets(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """
        Extract ground truth pixel values for a set of patch indices.

        Args:
            x   : (B, C, H, W) — original tile
            ids : (B, K)        — patch indices to extract

        Returns:
            targets : (B, K, patch_size^2 * C)
        """
        B = x.size(0)
        # For pixel reconstruction, patchify manually
        p = self.patch_size
        C = self.num_bands
        # Unfold into patches: (B, C, h*p, w*p) → (B, N, p*p*C)
        x_unf = x.unfold(2, p, p).unfold(3, p, p)              # (B, C, h, w, p, p)
        x_unf = x_unf.contiguous().view(B, C, -1, p * p)       # (B, C, N, p*p)
        x_unf = x_unf.permute(0, 2, 1, 3).reshape(B, -1, C * p * p)  # (B, N, C*p*p)

        targets = torch.stack([x_unf[b][ids[b]] for b in range(B)])   # (B, K, C*p*p)
        return targets
