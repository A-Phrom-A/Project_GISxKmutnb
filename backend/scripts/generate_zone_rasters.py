"""
backend/scripts/generate_zone_rasters.py
========================================
Pre-generates visual raster overlays for the 4 Demo Zones:
1. GSat: High-Res Optical (0.5m)
2. S1_SAR: Sentinel-1 Radar (VV/VH False Color & Grayscale)
3. S2_RGB: Sentinel-2 True Color (B4-B3-B2)
4. S2_CIR: Sentinel-2 Color Infrared (B8-B4-B3)
5. S2_SWIR: Sentinel-2 Agriculture/SWIR (B11-B8-B2)
6. S2_NDVI: Sentinel-2 NDVI Colormap (RdYlGn)
"""

import sys
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
import matplotlib.cm as cm

ROOT_DIR = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT_DIR / "data" / "demo_zones"
SAM_DIR = ROOT_DIR / "sam_segment"
LAYERS_DIR = DEMO_DIR / "layers"
LAYERS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_band(arr, p_low=2, p_high=98):
    """Percentile normalization to [0, 255] uint8."""
    valid = arr[arr > 0]
    if len(valid) == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    vmin, vmax = np.percentile(valid, (p_low, p_high))
    if vmax <= vmin:
        vmax = vmin + 1.0
    clipped = np.clip(arr, vmin, vmax)
    norm = ((clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    return norm


def process_zone_layers(zone_id: int):
    print(f"[*] Generating raster composites for Zone {zone_id}...")
    
    # 1. GSat High-Res Optical
    gsat_src = SAM_DIR / f"{zone_id}.tif"
    if gsat_src.exists():
        with rasterio.open(gsat_src) as src:
            rgb = src.read([1, 2, 3])
            # uint8 RGB
            img_gsat = Image.fromarray(np.transpose(rgb, (1, 2, 0)))
            img_gsat.save(LAYERS_DIR / f"zone_{zone_id}_gsat.jpg", quality=85)
            print(f"    -> Saved zone_{zone_id}_gsat.jpg")

    # 2. Sentinel-1 SAR Radar (VV, VH)
    s1_src = DEMO_DIR / f"zone_{zone_id}_s1.tif"
    if s1_src.exists():
        with rasterio.open(s1_src) as src:
            vv = src.read(1)
            vh = src.read(2)
            # Dual-pol color composite: R=VV, G=VH, B=ratio
            r_vv = normalize_band(vv, 2, 98)
            g_vh = normalize_band(vh, 2, 98)
            ratio = np.where(vh > 0, vv / (vh + 1e-4), 0.0)
            b_ratio = normalize_band(ratio, 2, 98)
            
            s1_rgb = np.stack([r_vv, g_vh, b_ratio], axis=-1)
            Image.fromarray(s1_rgb).save(LAYERS_DIR / f"zone_{zone_id}_s1.jpg", quality=85)
            print(f"    -> Saved zone_{zone_id}_s1.jpg")

    # 3. Sentinel-2 Optical (7 bands)
    s2_src = DEMO_DIR / f"zone_{zone_id}_s2.tif"
    if s2_src.exists():
        with rasterio.open(s2_src) as src:
            # 1:B2, 2:B3, 3:B4, 4:B8A, 5:B8, 6:B11, 7:B12
            b2 = src.read(1).astype(np.float32)
            b3 = src.read(2).astype(np.float32)
            b4 = src.read(3).astype(np.float32)
            b8 = src.read(5).astype(np.float32)
            b11 = src.read(6).astype(np.float32)

            # A. True Color RGB (B4, B3, B2)
            r = normalize_band(b4, 2, 98)
            g = normalize_band(b3, 2, 98)
            b = normalize_band(b2, 2, 98)
            Image.fromarray(np.stack([r, g, b], axis=-1)).save(LAYERS_DIR / f"zone_{zone_id}_s2_rgb.jpg", quality=85)
            print(f"    -> Saved zone_{zone_id}_s2_rgb.jpg")

            # B. Color Infrared (B8, B4, B3)
            cir_r = normalize_band(b8, 2, 98)
            cir_g = normalize_band(b4, 2, 98)
            cir_b = normalize_band(b3, 2, 98)
            Image.fromarray(np.stack([cir_r, cir_g, cir_b], axis=-1)).save(LAYERS_DIR / f"zone_{zone_id}_s2_cir.jpg", quality=85)
            print(f"    -> Saved zone_{zone_id}_s2_cir.jpg")

            # C. Agriculture SWIR (B11, B8, B2)
            swir_r = normalize_band(b11, 2, 98)
            swir_g = normalize_band(b8, 2, 98)
            swir_b = normalize_band(b2, 2, 98)
            Image.fromarray(np.stack([swir_r, swir_g, swir_b], axis=-1)).save(LAYERS_DIR / f"zone_{zone_id}_s2_swir.jpg", quality=85)
            print(f"    -> Saved zone_{zone_id}_s2_swir.jpg")

            # D. NDVI Colormap (B8 - B4) / (B8 + B4)
            denom = b8 + b4
            ndvi = np.where(denom > 0, (b8 - b4) / (denom + 1e-6), 0.0)
            ndvi_clipped = np.clip(ndvi, 0.0, 1.0)
            colormap = cm.get_cmap("RdYlGn")
            rgba = (colormap(ndvi_clipped) * 255).astype(np.uint8)
            Image.fromarray(rgba[:, :, :3]).save(LAYERS_DIR / f"zone_{zone_id}_s2_ndvi.jpg", quality=85)
            print(f"    -> Saved zone_{zone_id}_s2_ndvi.jpg")


if __name__ == "__main__":
    for zid in range(1, 5):
        process_zone_layers(zid)
    print("\n[+] All zone raster overlays successfully generated in data/demo_zones/layers!")
