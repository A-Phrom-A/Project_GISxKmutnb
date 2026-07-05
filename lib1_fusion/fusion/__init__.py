"""
lib1_fusion.fusion
==================
Fusion strategy subpackage.

Available strategies:
  - early_fusion(...)           : stack all channels before model input
  - LateFusion(...)             : modality-specific encoders + feature merge
  - CrossModalAttention(...)    : cross-attention across S1, S2, GEO-Sat
"""

from lib1_fusion.fusion.early_fusion import early_fusion
from lib1_fusion.fusion.late_fusion import LateFusion
from lib1_fusion.fusion.cross_modal_attention import CrossModalAttention

__all__ = ["early_fusion", "LateFusion", "CrossModalAttention"]
