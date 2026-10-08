"""
lib3_geoai.config
==================
GeoAIConfig — dataclass รวมการตั้งค่าทุกอย่างในที่เดียว

ใช้ได้สองแบบ:
    1. สร้าง config object แล้วส่งให้ pipeline
       cfg = GeoAIConfig(fusion="late", epochs=100)
       palm = GeoAI.palm(config=cfg)

    2. ส่ง keyword arguments โดยตรง (lib จะสร้าง GeoAIConfig ให้)
       palm = GeoAI.palm(fusion="late", epochs=100)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class GeoAIConfig:
    """
    การตั้งค่าทั้งหมดสำหรับ GeoAI pipeline

    Attributes
    ----------
    Data:
        data_dir        : path ไปยัง split directory (มี train/ val/ test/)
        tile_size       : ขนาด tile (default 256)
        batch_size      : batch size (default 16)
        num_workers     : DataLoader workers (default 0 สำหรับ Windows)
        augment         : เปิด augmentation สำหรับ train set (default True)

    Model:
        fusion          : "early" | "late" | "cross"
        embed_dim       : ViT embedding dimension (default 256)
        depth           : จำนวน Transformer layers (default 6)
        num_heads       : จำนวน attention heads (default 8)
        patch_size      : ขนาด patch ใน pixels (default 16)
        mlp_ratio       : MLP hidden ratio (default 4.0)
        dropout         : dropout rate (default 0.1)
        in_channels     : จำนวน input bands (default 12 = S1×2 + S2×7 + GSat×3)

    Training:
        epochs          : จำนวน epoch (default 100)
        lr              : learning rate (default 1e-4)
        weight_decay    : AdamW weight decay (default 0.05)
        patience        : early stopping patience (default 10)
        use_amp         : Mixed Precision BF16 (default True)
        seed            : random seed (default 42)
        save_every      : บันทึก checkpoint ทุก N epochs (default 10)
        save_dir        : directory สำหรับ checkpoint (default "checkpoints/")

    Loss:
        pos_weight      : weight สำหรับ positive class (default 4.75)
        bce_weight      : BCE loss weight (default 0.5)
        dice_weight     : Dice loss weight (default 0.5)
        threshold       : probability threshold สำหรับ binary pred (default 0.5)

    Pretraining (MAE):
        mask_ratio      : fraction of patches to mask (default 0.75)
        mask_strategy   : "random" | "cloud" | "band" | "mixed" (default "random")
        cloud_mask_dir  : path ไปยัง cloud mask files (ใช้เมื่อ strategy="cloud")

    Output:
        output_mode     : "binary" | "probability" | "both" (default "probability")
        use_wandb       : เปิด WandB logging (default False)
        wandb_project   : ชื่อ WandB project (default "geoai")
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir:    str | Path = "data/processed/split"
    tile_size:   int = 256
    batch_size:  int = 16
    num_workers: int = 0        # Windows: ใช้ 0 เสมอ
    augment:     bool = True

    # ── Model ─────────────────────────────────────────────────────────────────
    fusion:      Literal["early", "late", "cross"] = "late"
    decoder:     Literal["seg", "linear", "unet"] = "seg"
    embed_dim:   int   = 256
    depth:       int   = 6
    num_heads:   int   = 8
    patch_size:  int   = 16
    mlp_ratio:   float = 4.0
    dropout:     float = 0.1
    in_channels: int   = 12     # S1×2 + S2×7 + GSat×3

    # ── Training ──────────────────────────────────────────────────────────────
    epochs:       int   = 100
    lr:           float = 1e-4
    weight_decay: float = 0.05
    patience:     int   = 10
    use_amp:      bool  = True
    seed:         int   = 42
    save_every:   int   = 10
    save_dir:     str | Path = "checkpoints/"

    # ── Loss ──────────────────────────────────────────────────────────────────
    pos_weight:  float = 4.75
    bce_weight:  float = 0.5
    dice_weight: float = 0.5
    threshold:   float = 0.5

    # ── Pretraining ───────────────────────────────────────────────────────────
    mask_ratio:     float = 0.75
    mask_strategy:  Literal["random", "cloud", "band", "mixed"] = "random"
    cloud_mask_dir: Optional[str | Path] = None

    # ── Output ────────────────────────────────────────────────────────────────
    output_mode:    Literal["binary", "probability", "both"] = "probability"
    use_wandb:      bool = False
    wandb_project:  str  = "geoai"

    # ── Computed Properties ───────────────────────────────────────────────────
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.save_dir = Path(self.save_dir)
        if self.cloud_mask_dir is not None:
            self.cloud_mask_dir = Path(self.cloud_mask_dir)

    @classmethod
    def palm_default(cls) -> "GeoAIConfig":
        """Config ที่แนะนำสำหรับ Palm Detection (Chonburi dataset)."""
        return cls(
            fusion      = "late",
            embed_dim   = 256,
            depth       = 6,
            num_heads   = 8,
            pos_weight  = 4.75,   # จากการวิเคราะห์ class imbalance
            lr          = 1e-4,
            epochs      = 100,
            use_amp     = True,
            seed        = 42,
        )

    @classmethod
    def mae_default(cls) -> "GeoAIConfig":
        """Config สำหรับ MAE pretraining."""
        return cls(
            embed_dim    = 512,
            depth        = 6,
            num_heads    = 8,
            lr           = 1.5e-4,
            weight_decay = 0.05,
            epochs       = 100,
            mask_ratio   = 0.75,
            mask_strategy = "random",
            use_amp      = True,
            seed         = 42,
        )

    def summary(self) -> str:
        """แสดงสรุป config ที่สำคัญ."""
        lines = [
            "GeoAIConfig:",
            f"  fusion       = {self.fusion}",
            f"  in_channels  = {self.in_channels}",
            f"  embed_dim    = {self.embed_dim}  depth={self.depth}  heads={self.num_heads}",
            f"  patch_size   = {self.patch_size}  tile_size={self.tile_size}",
            f"  lr           = {self.lr}  wd={self.weight_decay}",
            f"  epochs       = {self.epochs}  patience={self.patience}",
            f"  pos_weight   = {self.pos_weight}",
            f"  use_amp      = {self.use_amp}  seed={self.seed}",
            f"  output_mode  = {self.output_mode}",
            f"  data_dir     = {self.data_dir}",
            f"  save_dir     = {self.save_dir}",
        ]
        return "\n".join(lines)
