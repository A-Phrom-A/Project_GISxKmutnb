"""
lib1_fusion.s1_processor
========================
Sentinel-1 SAR data processor.

Input bands:
  - VV: Vertical-Vertical polarization backscatter (dB)
  - VH: Vertical-Horizontal polarization backscatter (dB)

Output:
  - s1_stack: np.ndarray shape (2, H, W), float32, normalized [0.0, 1.0]
"""

import numpy as np
from lib1_fusion.utils import normalize_sar_db, validate_spatial_shape, stack_bands


class S1Processor:
    """
    Processes Sentinel-1 SAR backscatter bands (VV, VH) into
    normalized float32 arrays ready for fusion.

    Args:
        db_min (float): Lower dB clip for normalization. Default: -25.0
        db_max (float): Upper dB clip for normalization. Default:   0.0
    """

    def __init__(self, db_min: float = -25.0, db_max: float = 0.0):
        self.db_min = db_min
        self.db_max = db_max

    def process(self, vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
        """
        Normalize and stack VV and VH bands.

        Args:
            vv: Sentinel-1 VV band array (H, W), values in dB.
            vh: Sentinel-1 VH band array (H, W), values in dB.

        Returns:
            np.ndarray of shape (2, H, W), float32, normalized [0.0, 1.0].
            Band order: [VV, VH]
        """
        validate_spatial_shape([vv, vh], ["VV", "VH"])

        vv_norm = normalize_sar_db(vv, self.db_min, self.db_max)
        vh_norm = normalize_sar_db(vh, self.db_min, self.db_max)

        return stack_bands(vv_norm, vh_norm)  # (2, H, W)
