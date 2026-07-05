"""
lib1_fusion — Multi-Sensor Satellite Data Fusion Library
=========================================================
Converts Sentinel-1 (SAR), Sentinel-2 (Optical), and GEO-Satellite data
into a unified stacked tensor for use with lib2_vitpalm.

Output shape: (12, H, W) — float32, values normalized to [0.0, 1.0]

Bands layout:
  [0]  S1 VV
  [1]  S1 VH
  [2]  S2 B2  (Blue)
  [3]  S2 B3  (Green)
  [4]  S2 B4  (Red)
  [5]  S2 B8  (NIR)
  [6]  S2 B11 (SWIR-1)
  [7]  S2 B12 (SWIR-2)
  [8]  S2 RedEdge
  [9]  GEO-Sat R
  [10] GEO-Sat G
  [11] GEO-Sat B

Fusion Strategies:
  - Early Fusion  : stack all channels before model input
  - Late Fusion   : encode each modality separately, merge at feature level
  - Cross-Modal Attention : cross-attention between modalities
"""

from lib1_fusion.s1_processor import S1Processor
from lib1_fusion.s2_processor import S2Processor
from lib1_fusion.gsat_processor import GSatProcessor
from lib1_fusion.fusion.early_fusion import early_fusion
from lib1_fusion.fusion.late_fusion import LateFusion
from lib1_fusion.fusion.cross_modal_attention import CrossModalAttention
from lib1_fusion import utils

__version__ = "0.1.0"
__all__ = [
    "S1Processor",
    "S2Processor",
    "GSatProcessor",
    "early_fusion",
    "LateFusion",
    "CrossModalAttention",
    "utils",
]
