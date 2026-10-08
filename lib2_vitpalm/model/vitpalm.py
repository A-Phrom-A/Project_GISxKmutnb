"""
lib2_vitpalm.model.vitpalm
============================
ViTPalmModel: Combined ViT Encoder + Segmentation Decoder
สำหรับ fine-tuning บน palm detection task.

รองรับการ load pretrained MAE encoder weights.
"""

import torch
import torch.nn as nn
from pathlib import Path

from lib2_vitpalm.model.vit_encoder import ViTEncoder
from lib2_vitpalm.model.seg_decoder import SegDecoder


class ViTPalmModel(nn.Module):
    """
    Palm Detection model: ViT Encoder + ConvTranspose Segmentation Decoder.

    Input  : (B, 12, 256, 256) — 12-band fused satellite tile
    Output : (B, 256, 256)     — palm probability map [0, 1]

    Args:
        in_channels  : int   — number of input channels (default 12)
        img_size     : int   — tile spatial size (default 256)
        patch_size   : int   — patch size in pixels (default 16)
        embed_dim    : int   — embedding dimension (default 512)
        depth        : int   — transformer depth (default 6)
        num_heads    : int   — attention heads (default 8)
        mlp_ratio    : float — MLP hidden ratio (default 4.0)
        dropout      : float — dropout rate (default 0.1)
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
        self.encoder = ViTEncoder(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.decoder = SegDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)
        Returns:
            prob_map : (B, H, W) — palm probability [0, 1]
        """
        tokens   = self.encoder(x)           # (B, N+1, D)
        prob_map = self.decoder(tokens[:, 1:, :])  # exclude CLS token
        return prob_map

    def load_mae_weights(self, ckpt_path: str | Path, strict: bool = False) -> None:
        """
        Load pretrained MAE encoder weights into this model's encoder.

        Args:
            ckpt_path : path to checkpoint saved by MAEModel training
                        (expected keys: 'encoder' or 'model')
            strict    : if True, raise error on key mismatch (default False)
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # Try different common checkpoint key formats
        if "encoder" in ckpt:
            state_dict = ckpt["encoder"]
        elif "model" in ckpt:
            # Try to extract encoder sub-keys
            state_dict = {
                k.replace("encoder.", ""): v
                for k, v in ckpt["model"].items()
                if k.startswith("encoder.")
            }
        else:
            # Raw state dict, likely from train_mae.py model.state_dict()
            state_dict = {
                k.replace("encoder.", ""): v
                for k, v in ckpt.items()
                if k.startswith("encoder.")
            }

        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=strict)
        loaded_keys = len(state_dict) - len(missing)
        print(f"  MAE weights loaded: {loaded_keys} keys matched")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
