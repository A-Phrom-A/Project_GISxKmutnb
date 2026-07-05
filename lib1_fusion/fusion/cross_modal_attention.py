"""
lib1_fusion.fusion.cross_modal_attention
=========================================
Cross-Modal Attention fusion strategy.

Each modality acts as both Query and Key/Value source for the others,
allowing the model to learn inter-sensor relationships (e.g., SAR
backscatter patterns correlating with optical reflectance).

This module provides the numpy-level configuration and interface.
The actual attention computation (with learnable weights) is implemented
in lib2_vitpalm using PyTorch/TensorFlow tensors.

Architecture overview:
  S1 features   (C1, H, W) ─┐
  S2 features   (C2, H, W) ─┼→ [Cross-Modal Attention] → fused (C_out, H, W)
  GEO features  (C3, H, W) ─┘

Cross-attention:
  Q from modality i, K/V from modality j (for all i ≠ j pairs)
  Output: attended features concatenated across all pairs
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CrossModalConfig:
    """
    Configuration for Cross-Modal Attention fusion.

    Attributes:
        num_heads:      Number of attention heads. Default: 8
        embed_dim:      Embedding dimension per modality. Default: 256
        dropout:        Attention dropout rate. Default: 0.1
        modality_names: Labels for each modality.
    """
    num_heads: int = 8
    embed_dim: int = 256
    dropout: float = 0.1
    modality_names: List[str] = None

    def __post_init__(self):
        if self.modality_names is None:
            self.modality_names = ["S1", "S2", "GEO-Sat"]


class CrossModalAttention:
    """
    Cross-Modal Attention fusion interface (numpy-level config & validation).

    The actual attention layers are instantiated inside lib2_vitpalm when
    the full PyTorch model is assembled. This class serves as:
      1. A configuration container passed to the model builder.
      2. A validation helper for input feature map shapes.

    Args:
        config: CrossModalConfig instance. Uses defaults if None.
    """

    def __init__(self, config: Optional[CrossModalConfig] = None):
        self.config = config or CrossModalConfig()

    def validate_inputs(self, feature_maps: List[np.ndarray]) -> None:
        """
        Validate that input feature maps are compatible with cross-attention.

        Args:
            feature_maps: List of arrays, each shape (C_i, H, W).

        Raises:
            ValueError: If spatial shapes do not match across modalities.
        """
        if len(feature_maps) < 2:
            raise ValueError(
                "Cross-Modal Attention requires at least 2 modality feature maps."
            )

        ref_hw = feature_maps[0].shape[1:]
        for i, fm in enumerate(feature_maps[1:], 1):
            if fm.shape[1:] != ref_hw:
                name = self.config.modality_names[i]
                raise ValueError(
                    f"Spatial shape mismatch at modality '{name}': "
                    f"expected {ref_hw}, got {fm.shape[1:]}"
                )

    def get_config_dict(self) -> dict:
        """Return config as a plain dict for passing to PyTorch model builder."""
        return {
            "num_heads": self.config.num_heads,
            "embed_dim": self.config.embed_dim,
            "dropout": self.config.dropout,
            "modality_names": self.config.modality_names,
        }

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"CrossModalAttention("
            f"num_heads={cfg.num_heads}, "
            f"embed_dim={cfg.embed_dim}, "
            f"modalities={cfg.modality_names})"
        )
