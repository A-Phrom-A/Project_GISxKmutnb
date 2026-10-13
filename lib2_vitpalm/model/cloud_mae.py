"""
lib2_vitpalm.model.cloud_mae
=============================
Cloud-MAE: Radar-Guided Masked Autoencoder for Optical Cloud Removal and Representation Learning.

Key Characteristics:
- S1 (SAR: 2 bands VV, VH): 100% visible (never masked, penetrating clouds).
- S2 (Optical: 7 bands B2, B3, B4, B8A, B8, B11, B12): Simulated cloud mask (50% - 75%)
  via Block Masking or Random Patch Masking.
- Target: Reconstruct masked S2 pixels guided by unmasked SAR and surviving Optical patches.
- Loss: MSE computed strictly over the masked (cloud-covered) pixel regions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from pathlib import Path


class CloudMaskGenerator:
    """
    Generates cloud masks for S2 optical imagery.
    Supports:
      1. 'block': Rectangular simulated cloud patches (mimicking realistic cloud blobs).
      2. 'random': Random patch-level masking.
    """

    @staticmethod
    def generate_random_patch_mask(
        B: int,
        H: int,
        W: int,
        patch_size: int = 16,
        mask_ratio: float = 0.70,
        device: torch.device = torch.device('cpu'),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mask_pixel: (B, 1, H, W) binary mask (1 = masked/cloud, 0 = visible)
            mask_patch: (B, N) binary patch mask where N = (H//P)*(W//P)
        """
        num_h = H // patch_size
        num_w = W // patch_size
        N = num_h * num_w
        num_masked = int(N * mask_ratio)

        rand_noise = torch.rand(B, N, device=device)
        sorted_indices = torch.argsort(rand_noise, dim=1)
        
        mask_patch = torch.zeros(B, N, device=device)
        for b in range(B):
            mask_patch[b, sorted_indices[b, :num_masked]] = 1.0

        # Upsample patch mask to pixel mask
        mask_grid = mask_patch.view(B, 1, num_h, num_w)
        mask_pixel = F.interpolate(mask_grid, size=(H, W), mode='nearest')
        return mask_pixel, mask_patch

    @staticmethod
    def generate_block_cloud_mask(
        B: int,
        H: int,
        W: int,
        patch_size: int = 16,
        target_ratio: float = 0.65,
        min_block: int = 64,
        max_block: int = 128,
        device: torch.device = torch.device('cpu'),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates simulated realistic rectangular cloud blobs covering ~target_ratio of pixels.
        """
        mask_pixel = torch.zeros(B, 1, H, W, device=device)
        for b in range(B):
            curr_coverage = 0.0
            attempts = 0
            while curr_coverage < target_ratio and attempts < 15:
                attempts += 1
                bw = int(torch.randint(min_block, max_block + 1, (1,)).item())
                bh = int(torch.randint(min_block, max_block + 1, (1,)).item())
                x = int(torch.randint(0, max(1, W - bw + 1), (1,)).item())
                y = int(torch.randint(0, max(1, H - bh + 1), (1,)).item())
                mask_pixel[b, 0, y:y + bh, x:x + bw] = 1.0
                curr_coverage = mask_pixel[b, 0].mean().item()

        # Downsample to patch level
        num_h = H // patch_size
        num_w = W // patch_size
        mask_patch = F.adaptive_avg_pool2d(mask_pixel, (num_h, num_w))
        mask_patch = (mask_patch.view(B, -1) > 0.4).float()
        return mask_pixel, mask_patch


class CloudMAEEncoder(nn.Module):
    """
    Vision Transformer Encoder for Cloud-MAE.
    Embeds S1 (SAR) and S2 (Optical), fuses them per patch position,
    and applies Transformer self-attention blocks.
    """

    def __init__(
        self,
        s1_channels: int = 2,
        s2_channels: int = 7,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # 1. Modality patch projections
        self.s1_proj = nn.Conv2d(s1_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.s2_proj = nn.Conv2d(s2_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Modality & Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.mod_embed_s1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mod_embed_s2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cloud_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. Cross-modal projection fusion
        self.fuse_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mod_embed_s1, std=0.02)
        nn.init.trunc_normal_(self.mod_embed_s2, std=0.02)
        nn.init.trunc_normal_(self.cloud_token, std=0.02)

    def forward(
        self,
        s1: torch.Tensor,
        s2_masked: torch.Tensor,
        mask_patch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            s1         : (B, 2, H, W) SAR
            s2_masked  : (B, 6, H, W) Optical with masked cloud pixels zeroed
            mask_patch : (B, N) binary patch mask (1 = cloud, 0 = visible)
        Returns:
            latent     : (B, N, embed_dim)
        """
        B = s1.shape[0]

        # Patch Embeddings: (B, embed_dim, h, w) -> (B, N, embed_dim)
        t_s1 = self.s1_proj(s1).flatten(2).transpose(1, 2)
        t_s2 = self.s2_proj(s2_masked).flatten(2).transpose(1, 2)

        # Add modality embeddings
        t_s1 = t_s1 + self.mod_embed_s1
        t_s2 = t_s2 + self.mod_embed_s2

        # Replace masked Optical patches with learnable cloud_token
        if mask_patch is not None:
            m = mask_patch.unsqueeze(-1)  # (B, N, 1)
            t_s2 = t_s2 * (1.0 - m) + self.cloud_token.expand(B, self.num_patches, -1) * m

        # Multi-modal Token Fusion per spatial patch
        fused = torch.cat([t_s1, t_s2], dim=-1)  # (B, N, embed_dim * 2)
        tokens = self.fuse_proj(fused) + self.pos_embed  # (B, N, embed_dim)

        # Transformer blocks
        latent = self.transformer(tokens)
        latent = self.norm(latent)
        return latent


class CloudMAEDecoder(nn.Module):
    """
    Lightweight ViT Decoder for reconstructing masked Optical (S2) bands.
    """

    def __init__(
        self,
        s2_channels: int = 7,
        patch_size: int = 16,
        embed_dim: int = 512,
        decoder_dim: int = 256,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.s2_channels = s2_channels
        self.pred_dim = patch_size * patch_size * s2_channels

        self.dec_proj = nn.Linear(embed_dim, decoder_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, self.pred_dim)

    def forward(self, latent: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            latent : (B, N, embed_dim)
        Returns:
            pred_s2: (B, 6, H, W) reconstructed optical image
        """
        B, N, _ = latent.shape
        x = self.dec_proj(latent)
        x = self.transformer(x)
        x = self.norm(x)
        pred_patches = self.head(x)  # (B, N, P * P * C)

        # Unpatchify into (B, C, H, W)
        num_h = H // self.patch_size
        num_w = W // self.patch_size
        P = self.patch_size
        C = self.s2_channels

        # Reshape to (B, num_h, num_w, P, P, C) -> (B, C, H, W)
        pred = pred_patches.view(B, num_h, num_w, P, P, C)
        pred = pred.permute(0, 5, 1, 3, 2, 4).contiguous()
        pred = pred.view(B, C, H, W)
        return pred


class CloudMAEModel(nn.Module):
    """
    Complete Cloud-MAE Model combining Encoder, Decoder, Cloud Masking, and MSE Loss.
    """

    def __init__(
        self,
        s1_channels: int = 2,
        s2_channels: int = 7,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        decoder_dim: int = 256,
        decoder_depth: int = 3,
        mask_ratio: float = 0.65,
        mask_type: str = "block",
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

        self.encoder = CloudMAEEncoder(
            s1_channels=s1_channels,
            s2_channels=s2_channels,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )

        self.decoder = CloudMAEDecoder(
            s2_channels=s2_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=num_heads,
        )

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        mask_pixel: Optional[torch.Tensor] = None,
        mask_patch: Optional[torch.Tensor] = None,
        target_s2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        """
        B, _, H, W = s2.shape

        # 1. Generate cloud mask if not explicitly supplied
        if mask_pixel is None or mask_patch is None:
            if self.mask_type == "block":
                mask_pixel, mask_patch = CloudMaskGenerator.generate_block_cloud_mask(
                    B, H, W, self.patch_size, self.mask_ratio, device=s2.device
                )
            else:
                mask_pixel, mask_patch = CloudMaskGenerator.generate_random_patch_mask(
                    B, H, W, self.patch_size, self.mask_ratio, device=s2.device
                )

        # 2. Simulate cloud-covered optical input (zero out masked pixels)
        s2_masked = s2 * (1.0 - mask_pixel)

        # 3. Encode S1 (full radar) + S2 (cloud masked)
        latent = self.encoder(s1, s2_masked, mask_patch)

        # 4. Decode / Reconstruct optical image
        pred_s2 = self.decoder(latent, H, W)

        # 5. Compute MSE Loss strictly over the masked (cloud-covered) pixels
        # mask_pixel is (B, 1, H, W)
        target = target_s2 if target_s2 is not None else s2
        mask_expanded = mask_pixel.expand_as(target)
        diff_sq = (pred_s2 - target) ** 2
        masked_diff = diff_sq * mask_expanded
        
        # Avoid division by zero
        num_masked_pixels = mask_expanded.sum()
        if num_masked_pixels > 0:
            loss = masked_diff.sum() / num_masked_pixels
        else:
            loss = diff_sq.mean()

        return {
            "loss": loss,
            "pred_s2": pred_s2,
            "s2_masked": s2_masked,
            "mask_pixel": mask_pixel,
            "mask_patch": mask_patch,
            "latent": latent,
        }

    def get_encoder(self) -> CloudMAEEncoder:
        """Returns encoder for downstream segmentation fine-tuning."""
        return self.encoder

    def transfer_to_late_fusion(self, late_fusion_model: nn.Module) -> None:
        """
        Transfers Cloud-MAE representations to ViTLateFusionModel.
        - S1 encoder projection & Transformer blocks
        - S2 (7 bands) encoder projection & Transformer blocks
        """
        with torch.no_grad():
            # 1. Transfer S1 encoder
            if hasattr(late_fusion_model, "encoder_s1"):
                late_fusion_model.encoder_s1.patch_embed.proj.weight.copy_(self.encoder.s1_proj.weight)
                late_fusion_model.encoder_s1.patch_embed.proj.bias.copy_(self.encoder.s1_proj.bias)
                late_fusion_model.encoder_s1.transformer.load_state_dict(self.encoder.transformer.state_dict())

            # 2. Transfer S2 encoder (7 bands)
            if hasattr(late_fusion_model, "encoder_s2"):
                late_fusion_model.encoder_s2.patch_embed.proj.weight.copy_(self.encoder.s2_proj.weight)
                late_fusion_model.encoder_s2.patch_embed.proj.bias.copy_(self.encoder.s2_proj.bias)
                late_fusion_model.encoder_s2.transformer.load_state_dict(self.encoder.transformer.state_dict())
        print("  [Cloud-MAE] Successfully transferred weights to ViTLateFusionModel (S1 + S2 encoders)!")

    def transfer_to_cross_attn(self, cross_attn_model: nn.Module) -> None:
        """
        Transfers Cloud-MAE representations to ViTCrossAttnModel.
        - S1 encoder projection & Transformer blocks
        - S2 (7 bands) encoder projection & Transformer blocks
        """
        with torch.no_grad():
            if hasattr(cross_attn_model, "encoder_s1"):
                cross_attn_model.encoder_s1.patch_embed.proj.weight.copy_(self.encoder.s1_proj.weight)
                cross_attn_model.encoder_s1.patch_embed.proj.bias.copy_(self.encoder.s1_proj.bias)
                cross_attn_model.encoder_s1.transformer.load_state_dict(self.encoder.transformer.state_dict())

            if hasattr(cross_attn_model, "encoder_s2"):
                cross_attn_model.encoder_s2.patch_embed.proj.weight.copy_(self.encoder.s2_proj.weight)
                cross_attn_model.encoder_s2.patch_embed.proj.bias.copy_(self.encoder.s2_proj.bias)
                cross_attn_model.encoder_s2.transformer.load_state_dict(self.encoder.transformer.state_dict())
        print("  [Cloud-MAE] Successfully transferred weights to ViTCrossAttnModel (S1 + S2 encoders)!")

    def transfer_to_early_fusion(self, vitpalm_model: nn.Module) -> None:
        """
        Transfers Cloud-MAE representations to ViTPalmModel (12 channels Early Fusion).
        - Shared Transformer backbone layers
        - Patch projection weights for S1 (ch 0..1), S2 (ch 2..8), and GSat RGB initialized from S2 RGB
        """
        with torch.no_grad():
            if hasattr(vitpalm_model, "encoder"):
                # Transfer transformer blocks
                vitpalm_model.encoder.transformer.load_state_dict(self.encoder.transformer.state_dict())
                
                # Transfer patch projections into 12-channel Conv2d
                # S1 (0:2)
                vitpalm_model.encoder.patch_embed.proj.weight[:, 0:2, :, :].copy_(self.encoder.s1_proj.weight)
                # S2 (2:9)
                vitpalm_model.encoder.patch_embed.proj.weight[:, 2:9, :, :].copy_(self.encoder.s2_proj.weight)
                # GSat (9:12) initialized from S2 RGB (B4=idx 2, B3=idx 1, B2=idx 0)
                vitpalm_model.encoder.patch_embed.proj.weight[:, 9:12, :, :].copy_(self.encoder.s2_proj.weight[:, [2, 1, 0], :, :])
        print("  [Cloud-MAE] Successfully transferred weights to ViTPalmModel (Early Fusion 12ch)!")

