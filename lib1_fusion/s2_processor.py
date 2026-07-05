"""
lib1_fusion.s2_processor
========================
Sentinel-2 Multispectral data processor.

Input bands (surface reflectance, DN 0–10000):
  - B2  : Blue       (490 nm)
  - B3  : Green      (560 nm)
  - B4  : Red        (665 nm)
  - B8  : NIR        (842 nm)
  - B11 : SWIR-1     (1610 nm)
  - B12 : SWIR-2     (2190 nm)
  - RedEdge: Red Edge (705 nm)

Output:
  - s2_stack: np.ndarray shape (7, H, W), float32, normalized [0.0, 1.0]
"""

import numpy as np
from lib1_fusion.utils import normalize_s2_reflectance, validate_spatial_shape, stack_bands


class S2Processor:
    """
    Processes Sentinel-2 surface reflectance bands into normalized
    float32 arrays ready for fusion.

    Band order in output stack: [B2, B3, B4, B8, B11, B12, RedEdge]
    """

    BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12", "RedEdge"]

    def process(
        self,
        b2: np.ndarray,
        b3: np.ndarray,
        b4: np.ndarray,
        b8: np.ndarray,
        b11: np.ndarray,
        b12: np.ndarray,
        red_edge: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize and stack all 7 Sentinel-2 bands.

        Args:
            b2, b3, b4, b8, b11, b12, red_edge:
                Each is a 2D np.ndarray (H, W) with DN values 0–10000.

        Returns:
            np.ndarray of shape (7, H, W), float32, normalized [0.0, 1.0].
            Band order: [B2, B3, B4, B8, B11, B12, RedEdge]
        """
        bands = [b2, b3, b4, b8, b11, b12, red_edge]
        validate_spatial_shape(bands, self.BAND_NAMES)

        normalized = [normalize_s2_reflectance(b) for b in bands]
        return stack_bands(*normalized)  # (7, H, W)
