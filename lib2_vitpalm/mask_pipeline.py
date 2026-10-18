"""
lib2_vitpalm.mask_pipeline
==========================
Production Cloud Mask Generation Pipeline (Option 2: SCL + Multi-spectral Cloud Probability Ensemble).

Designed for Sentinel-2 L2A optical imagery without requiring heavy raw L1C downloads.
Key features:
1. Spectral Cloud Probability: Derived from 7 L2A surface reflectance bands (B2, B3, B4, B8, B11, B12, B8A).
2. SCL Integration: Seamlessly incorporates ESA Scene Classification Layer (Classes 8, 9, 10 for cloud, 3 for shadow).
3. Morphological Dilation: 3-pixel (~30m) dilation to envelope diffuse cloud fringes and cloud shadows.
4. Tri-mask Output:
   - hard_mask: Binary {0, 1} for loss masking, metric evaluation, and patch reinsertion.
   - soft_mask: Continuous [0.0, 1.0] opacity representation for model input.
   - type_mask: Categorical {0: Clear, 1: Thick Cloud, 2: Thin Cirrus, 3: Cloud Shadow}.
5. Coverage Binning: Strict filtering conforming to the <=20% (margin <=25%) repair scope.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, distance_transform_edt


def compute_spectral_cloud_prob(
    s2_7bands: Union[np.ndarray, torch.Tensor],
    refl_scale: float = 10000.0
) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes optical cloud probability from Sentinel-2 7 optical bands:
    [B02, B03, B04, B08, B11, B12, B8A]
    """
    is_torch = isinstance(s2_7bands, torch.Tensor)
    if is_torch:
        x = s2_7bands.detach().cpu().numpy()
    else:
        x = s2_7bands.copy()

    # Normalize to [0, 1]
    refl = np.clip(x / refl_scale, 0.0, 1.0)
    b2, b3, b4 = refl[0], refl[1], refl[2]
    b8, b11, b12, b8a = refl[3], refl[4], refl[5], refl[6]

    # 1. Visible Brightness (clouds reflect heavily across all visible bands)
    vis_mean = (b2 + b3 + b4) / 3.0
    brightness_prob = np.clip((vis_mean - 0.15) / 0.25, 0.0, 1.0)

    # 2. Whiteness (clouds are spectrally flat across RGB)
    vis_diff = np.abs(b2 - vis_mean) + np.abs(b3 - vis_mean) + np.abs(b4 - vis_mean)
    whiteness = 1.0 - np.clip(vis_diff / (vis_mean + 1e-6), 0.0, 1.0)

    # 3. SWIR distinction (clouds drop in SWIR relative to high-reflectance sand/urban)
    swir_factor = np.clip(vis_mean / (b11 + 0.05), 0.0, 2.0) / 2.0

    prob = brightness_prob * (0.6 * whiteness + 0.4 * swir_factor)
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)

    if is_torch:
        return torch.from_numpy(prob).to(device=s2_7bands.device, dtype=torch.float32)
    return prob


def classify_cloud_bin(coverage_pct: float) -> Tuple[str, bool]:
    """
    Categorizes cloud coverage into the standardized Cloud-MAE v2 bins:
    Returns (bin_name, is_valid_for_repair)
    """
    if coverage_pct < 2.0:
        return "0-2% (Clear)", False
    elif 2.0 <= coverage_pct < 5.0:
        return "2-5%", True
    elif 5.0 <= coverage_pct < 10.0:
        return "5-10%", True
    elif 10.0 <= coverage_pct <= 20.0:
        return "10-20%", True
    elif 20.0 < coverage_pct <= 25.0:
        return "20-25% (Margin)", True
    else:
        return ">25% (Rejected)", False


class CloudMaskPipeline:
    """
    Production Cloud Mask Generator implementing Option 2 Ensemble.
    """

    def __init__(
        self,
        prob_threshold: float = 0.40,
        dilation_radius_px: int = 3,
        scl_cloud_classes: Tuple[int, ...] = (8, 9, 10),
        scl_shadow_classes: Tuple[int, ...] = (3,),
    ):
        self.prob_threshold = prob_threshold
        self.dilation_radius_px = dilation_radius_px
        self.scl_cloud_classes = scl_cloud_classes
        self.scl_shadow_classes = scl_shadow_classes

        # Structure for 3 px radius morphological dilation (approx 7x7 diamond/circle)
        size = 2 * dilation_radius_px + 1
        y, x = np.ogrid[-dilation_radius_px:dilation_radius_px+1, -dilation_radius_px:dilation_radius_px+1]
        self.struct = (x**2 + y**2 <= dilation_radius_px**2)

    def generate(
        self,
        s2_optical: np.ndarray,
        scl: Optional[np.ndarray] = None
    ) -> Dict[str, Union[np.ndarray, float, str, bool]]:
        """
        Args:
            s2_optical: (7, H, W) optical reflectance [0-10000 or 0-1]
            scl: Optional (H, W) ESA Scene Classification Layer
        Returns:
            Dictionary containing hard_mask, soft_mask, type_mask, coverage_pct, coverage_bin, is_valid
        """
        assert s2_optical.ndim == 3 and s2_optical.shape[0] == 7, f"Expected (7, H, W), got {s2_optical.shape}"
        H, W = s2_optical.shape[1], s2_optical.shape[2]

        # 1. Compute multi-spectral cloud probability
        prob = compute_spectral_cloud_prob(s2_optical)

        # 2. Extract SCL components if available
        if scl is not None:
            assert scl.shape == (H, W), f"SCL shape {scl.shape} must match raster {(H, W)}"
            scl_cloud = np.isin(scl, self.scl_cloud_classes)
            scl_shadow = np.isin(scl, self.scl_shadow_classes)
            scl_cirrus = (scl == 10)
        else:
            scl_cloud = np.zeros((H, W), dtype=bool)
            scl_shadow = np.zeros((H, W), dtype=bool)
            scl_cirrus = np.zeros((H, W), dtype=bool)

        # 3. Ensemble Fusion
        prob_cloud = prob >= self.prob_threshold
        raw_cloud = scl_cloud | prob_cloud
        raw_cloud_and_shadow = raw_cloud | scl_shadow

        # 4. Morphological Dilation
        dilated_mask = binary_dilation(raw_cloud_and_shadow, structure=self.struct)

        # 5. Type Mask: {0: Clear, 1: Thick Cloud, 2: Thin Cirrus, 3: Shadow}
        type_mask = np.zeros((H, W), dtype=np.uint8)
        # Thick cloud: high prob or SCL 8, 9
        thick = (prob >= 0.65) | np.isin(scl if scl is not None else np.zeros((H, W)), [8, 9])
        # Thin cirrus: prob between threshold and 0.65 or SCL 10
        thin = (raw_cloud & ~thick) | scl_cirrus

        type_mask[scl_shadow] = 3
        type_mask[thin] = 2
        type_mask[thick] = 1

        # 6. Soft Mask: Continuous opacity [0.0, 1.0] with smooth boundary transition
        dist = distance_transform_edt(dilated_mask)
        boundary_fade = np.clip(dist / float(self.dilation_radius_px + 1), 0.0, 1.0)
        soft_mask = np.maximum(prob, dilated_mask.astype(np.float32) * boundary_fade)
        soft_mask = np.clip(soft_mask, 0.0, 1.0).astype(np.float32)

        # 7. Coverage and Binning
        hard_mask = dilated_mask.astype(np.uint8)
        total_pixels = H * W
        coverage_pct = float(np.sum(hard_mask) / total_pixels * 100.0)
        coverage_bin, is_valid = classify_cloud_bin(coverage_pct)

        return {
            "hard_mask": hard_mask,          # (H, W) uint8 {0, 1}
            "soft_mask": soft_mask,          # (H, W) float32 [0.0, 1.0]
            "type_mask": type_mask,          # (H, W) uint8 {0, 1, 2, 3}
            "prob_map": prob,                # (H, W) float32 [0.0, 1.0]
            "coverage_pct": round(coverage_pct, 3),
            "coverage_bin": coverage_bin,
            "is_valid_for_repair": is_valid
        }
