"""
lib3_geoai.models.factory
==========================
build_model() — เลือก model architecture ตาม fusion strategy

รองรับ:
  "early" → ViTPalmModel          (12-band Early Fusion)
  "late"  → ViTLateFusionModel   (3 encoder แยก modality)
  "cross" → ViTCrossAttnModel    (Cross-Attention S2 query S1+GSat)
"""

from __future__ import annotations

import torch.nn as nn
from typing import Literal

from lib3_geoai.config import GeoAIConfig

# ── Lazy imports จาก lib2 ──────────────────────────────────────────────────
from lib2_vitpalm.model.vitpalm          import ViTPalmModel
from lib2_vitpalm.model.late_fusion_model import ViTLateFusionModel
from lib2_vitpalm.model.cross_attn_model  import ViTCrossAttnModel


FUSION_MAP = {
    "early": ViTPalmModel,
    "late":  ViTLateFusionModel,
    "cross": ViTCrossAttnModel,
}


def build_model(
    config: GeoAIConfig,
    fusion: Literal["early", "late", "cross"] | None = None,
) -> nn.Module:
    """
    สร้าง model ตาม fusion strategy ใน config.

    Args:
        config : GeoAIConfig object
        fusion : override fusion strategy (optional)
                 ถ้าไม่ระบุ ใช้ config.fusion

    Returns:
        nn.Module ที่พร้อม train (ยังไม่ได้ .to(device))

    Raises:
        ValueError: ถ้า fusion ไม่รู้จัก
    """
    fusion = fusion or config.fusion

    if fusion not in FUSION_MAP:
        raise ValueError(
            f"Unknown fusion '{fusion}'. Choose: {list(FUSION_MAP.keys())}"
        )

    model_cls = FUSION_MAP[fusion]

    # ── early fusion: รับ in_channels เป็น argument ──────────────────────────
    if fusion == "early":
        model = model_cls(
            in_channels = config.in_channels,
            img_size    = config.tile_size,
            patch_size  = config.patch_size,
            embed_dim   = config.embed_dim,
            depth       = config.depth,
            num_heads   = config.num_heads,
            mlp_ratio   = config.mlp_ratio,
            dropout     = config.dropout,
        )

    # ── late / cross fusion: encoder แยก modality (S1, S2, GSat) ─────────────
    else:
        model = model_cls(
            img_size  = config.tile_size,
            patch_size = config.patch_size,
            embed_dim  = config.embed_dim,
            depth      = config.depth,
            num_heads  = config.num_heads,
            mlp_ratio  = config.mlp_ratio,
        )

    return model


def model_summary(model: nn.Module, fusion: str = "") -> None:
    """พิมพ์สรุป model architecture."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model      : {model.__class__.__name__}  [{fusion}]")
    print(f"  Parameters : {n_params:,}")
