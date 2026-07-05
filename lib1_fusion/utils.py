"""
lib1_fusion.utils
=================
Shared utility functions for satellite data preprocessing:
  - normalization per sensor type
  - spatial alignment / reprojection helpers
  - band validation
"""

import numpy as np
from typing import Tuple, Optional


# ─── Band index constants ──────────────────────────────────────────────────────

S1_BANDS = {"VV": 0, "VH": 1}
S2_BANDS = {"B2": 2, "B3": 3, "B4": 4, "B8": 5, "B11": 6, "B12": 7, "RedEdge": 8}
GSAT_BANDS = {"R": 9, "G": 10, "B": 11}
TOTAL_BANDS = 12  # fixed layout: 2 (S1) + 7 (S2) + 3 (GEO-Sat)


# ─── Normalization ─────────────────────────────────────────────────────────────

def normalize_minmax(
    arr: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    clip: bool = True,
) -> np.ndarray:
    """
    Min-Max normalization → [0.0, 1.0].

    Args:
        arr:  Input array of any shape.
        vmin: Minimum value (default: arr.min()).
        vmax: Maximum value (default: arr.max()).
        clip: If True, clip values outside [0, 1] after normalization.

    Returns:
        Normalized float32 array, same shape as input.
    """
    arr = arr.astype(np.float32)
    vmin = float(vmin) if vmin is not None else arr.min()
    vmax = float(vmax) if vmax is not None else arr.max()

    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.float32)

    out = (arr - vmin) / (vmax - vmin)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def normalize_sar_db(arr: np.ndarray, db_min: float = -25.0, db_max: float = 0.0) -> np.ndarray:
    """
    Normalize SAR backscatter in dB scale → [0.0, 1.0].
    Default range covers typical vegetation/soil SAR values.

    Args:
        arr:    SAR array (dB values, typically negative).
        db_min: Lower dB clip (default: -25 dB).
        db_max: Upper dB clip (default: 0 dB).

    Returns:
        Normalized float32 array.
    """
    return normalize_minmax(arr, vmin=db_min, vmax=db_max, clip=True)


def normalize_s2_reflectance(arr: np.ndarray) -> np.ndarray:
    """
    Normalize Sentinel-2 surface reflectance (0–10000 DN) → [0.0, 1.0].

    Args:
        arr: S2 band array with integer reflectance values (0–10000).

    Returns:
        Normalized float32 array.
    """
    return normalize_minmax(arr, vmin=0.0, vmax=10000.0, clip=True)


# ─── Shape Validation ──────────────────────────────────────────────────────────

def validate_spatial_shape(arrays: list, names: list) -> Tuple[int, int]:
    """
    Verify that all input arrays share the same (H, W) spatial dimensions.

    Args:
        arrays: List of 2D numpy arrays (H, W).
        names:  Corresponding names for error messages.

    Returns:
        (H, W) tuple if all match.

    Raises:
        ValueError: If shapes do not match.
    """
    shapes = [a.shape[-2:] for a in arrays]
    ref = shapes[0]
    for name, shape in zip(names, shapes):
        if shape != ref:
            raise ValueError(
                f"Shape mismatch: '{names[0]}' has shape {ref}, "
                f"but '{name}' has shape {shape}. "
                "All bands must be reprojected to the same spatial resolution before stacking."
            )
    return ref  # (H, W)


# ─── Stacking ──────────────────────────────────────────────────────────────────

def stack_bands(*bands: np.ndarray) -> np.ndarray:
    """
    Stack individual 2D band arrays into a single (N, H, W) tensor.

    Args:
        *bands: Variable number of 2D arrays (H, W).

    Returns:
        np.ndarray of shape (N, H, W) float32.
    """
    return np.stack(bands, axis=0).astype(np.float32)
