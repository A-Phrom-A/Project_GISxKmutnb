"""
lib1_fusion.gsat_processor
===========================
GEO-Satellite (high-res optical) data processor.

Input bands (uint8 or float, 0–255 or 0.0–1.0):
  - R : Red channel
  - G : Green channel
  - B : Blue channel

Output:
  - gsat_stack: np.ndarray shape (3, H, W), float32, normalized [0.0, 1.0]
"""

import numpy as np
from lib1_fusion.utils import normalize_minmax, validate_spatial_shape, stack_bands


class GSatProcessor:
    """
    Processes GEO-Satellite RGB imagery into normalized float32 arrays
    ready for fusion.

    Handles both:
      - uint8 input  (0–255)  → normalized to [0.0, 1.0]
      - float input  (0.0–1.0) → passed through (clipped)

    Band order in output stack: [R, G, B]
    """

    def process(
        self,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize and stack R, G, B channels.

        Args:
            r: Red channel array   (H, W) — uint8 (0–255) or float (0–1).
            g: Green channel array (H, W) — uint8 (0–255) or float (0–1).
            b: Blue channel array  (H, W) — uint8 (0–255) or float (0–1).

        Returns:
            np.ndarray of shape (3, H, W), float32, normalized [0.0, 1.0].
            Band order: [R, G, B]
        """
        bands = [r, g, b]
        validate_spatial_shape(bands, ["R", "G", "B"])

        normalized = []
        for band in bands:
            if band.dtype == np.uint8:
                normalized.append(normalize_minmax(band, vmin=0.0, vmax=255.0))
            else:
                normalized.append(normalize_minmax(band, vmin=0.0, vmax=1.0))

        return stack_bands(*normalized)  # (3, H, W)
