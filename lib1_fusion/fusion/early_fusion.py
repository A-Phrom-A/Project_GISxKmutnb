"""
lib1_fusion.fusion.early_fusion
================================
Early Fusion strategy: concatenate all sensor channels BEFORE the model.

This is the simplest fusion approach — all 12 bands are stacked into
a single (12, H, W) tensor that is passed directly to the ViT model.

Band layout (fixed):
  [0-1]  : Sentinel-1 (VV, VH)
  [2-8]  : Sentinel-2 (B2, B3, B4, B8, B11, B12, RedEdge)
  [9-11] : GEO-Sat    (R, G, B)
"""

import numpy as np
from lib1_fusion.utils import validate_spatial_shape, stack_bands, TOTAL_BANDS


def early_fusion(
    s1_stack: np.ndarray,
    s2_stack: np.ndarray,
    gsat_stack: np.ndarray,
) -> np.ndarray:
    """
    Stack S1, S2, and GEO-Sat tensors into a single (12, H, W) tensor.

    Args:
        s1_stack:   (2, H, W) float32 — from S1Processor
        s2_stack:   (7, H, W) float32 — from S2Processor
        gsat_stack: (3, H, W) float32 — from GSatProcessor

    Returns:
        np.ndarray of shape (12, H, W), float32.

    Raises:
        ValueError: If spatial dimensions (H, W) do not match across stacks.
        ValueError: If input channel counts are unexpected.
    """
    # Validate channel counts
    assert s1_stack.shape[0] == 2,   f"Expected S1 stack (2, H, W), got {s1_stack.shape}"
    assert s2_stack.shape[0] == 7,   f"Expected S2 stack (7, H, W), got {s2_stack.shape}"
    assert gsat_stack.shape[0] == 3, f"Expected GSat stack (3, H, W), got {gsat_stack.shape}"

    # Validate spatial alignment
    validate_spatial_shape(
        [s1_stack[0], s2_stack[0], gsat_stack[0]],
        ["S1", "S2", "GEO-Sat"],
    )

    # Concatenate along channel axis
    fused = np.concatenate([s1_stack, s2_stack, gsat_stack], axis=0)  # (12, H, W)

    assert fused.shape[0] == TOTAL_BANDS, \
        f"Expected {TOTAL_BANDS} bands after fusion, got {fused.shape[0]}"

    return fused.astype(np.float32)
