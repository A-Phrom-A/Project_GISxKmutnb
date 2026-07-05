"""
lib1_fusion.fusion.late_fusion
================================
Late Fusion strategy: encode each modality separately, then merge
at the feature level before the final prediction head.

In this approach each sensor's stack is processed independently
(e.g., by a separate CNN or linear projection), and the resulting
feature maps are concatenated or averaged before being fed to the
downstream model.

NOTE: The actual encoder modules live in lib2_vitpalm.
      This module provides the numpy-level feature concatenation helper
      and the configuration dataclass used during pipeline assembly.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional


@dataclass
class LateFusionConfig:
    """
    Configuration for Late Fusion pipeline.

    Attributes:
        merge_strategy: How to combine encoded features.
                        'concat' → concatenate along channel dim (default)
                        'mean'   → element-wise average
        modality_names: Labels for each modality (for logging/debugging).
    """
    merge_strategy: str = "concat"   # 'concat' | 'mean'
    modality_names: List[str] = field(default_factory=lambda: ["S1", "S2", "GEO-Sat"])


class LateFusion:
    """
    Late Fusion: merge modality feature maps AFTER individual encoding.

    Usage (numpy level — before deep model):
        lf = LateFusion(config)
        fused = lf.merge([s1_features, s2_features, gsat_features])

    Args:
        config: LateFusionConfig with merge strategy settings.
    """

    def __init__(self, config: Optional[LateFusionConfig] = None):
        self.config = config or LateFusionConfig()

    def merge(self, feature_maps: List[np.ndarray]) -> np.ndarray:
        """
        Merge a list of feature maps from different modalities.

        Args:
            feature_maps: List of np.ndarray, each shape (C_i, H, W).
                          All must share the same (H, W) spatial dims.

        Returns:
            Merged np.ndarray:
              - 'concat': shape (sum(C_i), H, W)
              - 'mean':   shape (C_0, H, W)  — requires all C_i equal

        Raises:
            ValueError: On shape mismatch or unknown merge_strategy.
        """
        if not feature_maps:
            raise ValueError("feature_maps list is empty")

        ref_hw = feature_maps[0].shape[1:]
        for i, fm in enumerate(feature_maps[1:], 1):
            if fm.shape[1:] != ref_hw:
                raise ValueError(
                    f"Spatial shape mismatch: modality 0 has {ref_hw}, "
                    f"modality {i} ({self.config.modality_names[i]}) has {fm.shape[1:]}"
                )

        strategy = self.config.merge_strategy
        if strategy == "concat":
            return np.concatenate(feature_maps, axis=0).astype(np.float32)
        elif strategy == "mean":
            return np.mean(np.stack(feature_maps, axis=0), axis=0).astype(np.float32)
        else:
            raise ValueError(
                f"Unknown merge_strategy '{strategy}'. Choose 'concat' or 'mean'."
            )

    def __repr__(self) -> str:
        return (
            f"LateFusion(strategy='{self.config.merge_strategy}', "
            f"modalities={self.config.modality_names})"
        )
