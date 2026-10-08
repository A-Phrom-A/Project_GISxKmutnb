"""
lib2_vitpalm.model.vit_encoder
================================
ViT Encoder สำหรับ 12-band satellite imagery.

ใช้ PatchEmbed → CLS token + Positional Embed → TransformerEncoder → LayerNorm

Output: (B, N, D) token sequence พร้อมส่งให้ SegDecoder หรือ MAEDecoder
"""

import torch
import torch.nn as nn
from lib2_vitpalm.model.patch_embed import PatchEmbed


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder.

    Args:
        in_channels  : int   — number of input spectral bands (default 12)
        img_size     : int   — spatial tile size in pixels (default 256)
        patch_size   : int   — patch size in pixels (default 16)
        embed_dim    : int   — token embedding dimension (default 512)
        depth        : int   — number of TransformerEncoderLayer blocks (default 6)
        num_heads    : int   — number of attention heads (default 8)
        mlp_ratio    : float — MLP hidden dim = embed_dim * mlp_ratio (default 4.0)
        dropout      : float — attention + FFN dropout (default 0.1)
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
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch tokenizer
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            patch_size=patch_size,
            img_size=img_size,
            embed_dim=embed_dim,
        )
        N = self.patch_embed.num_patches

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, embed_dim))

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.num_patches = N
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ViT paper recommendations."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x            : (B, C, H, W) — input satellite tile
            mask_indices : (B, N_keep) — indices of visible patches for MAE.
                           If None, all patches are used (fine-tuning mode).
        Returns:
            tokens : (B, N_out, D)
                     N_out = N_keep+1 (MAE mode) or N+1 (full mode)
                     index 0 = CLS token
        """
        B = x.size(0)
        tokens = self.patch_embed(x)                   # (B, N, D)
        cls    = self.cls_token.expand(B, -1, -1)      # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)       # (B, N+1, D)
        tokens = tokens + self.pos_embed               # add positional embed

        if mask_indices is not None:
            # MAE mode: keep only visible patches (+ CLS)
            # mask_indices shape: (B, N_keep) — indices into patch tokens [1:]
            cls_tokens     = tokens[:, :1, :]          # (B, 1, D)
            patch_tokens   = tokens[:, 1:, :]          # (B, N, D)
            visible = torch.stack(
                [patch_tokens[b][mask_indices[b]] for b in range(B)]
            )                                          # (B, N_keep, D)
            tokens = torch.cat([cls_tokens, visible], dim=1)  # (B, N_keep+1, D)

        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return tokens   # (B, N_out, D)
