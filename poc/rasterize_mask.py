"""
poc/rasterize_mask.py
======================
Reads `ground_truth.geojson` (WGS84), reprojects it to match the Sentinel-2 
reference image (EPSG:32647), and rasterizes it into a binary mask (.tif).

Output: data/processed/palm_mask.tif (0 = background, 1 = palm)
"""

import json
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
import numpy as np
import os

GEOJSON_PATH = os.path.join("data", "raw", "ground_truth.geojson")
REF_IMG_PATH = os.path.join("data", "raw", "S2C_MSIL2A_20260410_7band_stacked_clip.tif")
OUTPUT_MASK_PATH = os.path.join("data", "processed", "palm_mask.tif")

def main():
    print(f"Loading GeoJSON: {GEOJSON_PATH}")
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loading reference image: {REF_IMG_PATH}")
    with rasterio.open(REF_IMG_PATH) as src:
        ref_meta = src.meta.copy()
        ref_crs = src.crs
        ref_transform = src.transform
        ref_shape = (src.height, src.width)

    print(f"  Reference CRS: {ref_crs}")
    print(f"  Reference Shape: {ref_shape}")

    # Extract geometries from GeoJSON
    geometries = []
    source_crs = "EPSG:4326"  # CRS84 / WGS84

    for feature in data["features"]:
        geom = feature["geometry"]
        if geom is None:
            continue
        
        try:
            # Reproject geometry to reference CRS
            reprojected_geom = transform_geom(
                source_crs, 
                ref_crs, 
                geom
            )
            # Store tuple of (geometry, value)
            geometries.append((reprojected_geom, 1))
        except Exception as e:
            pass # ignore invalid geometries

    print(f"  Total valid polygons: {len(geometries)}")

    # Rasterize
    print("Rasterizing geometries to binary mask...")
    mask = rasterize(
        shapes=geometries,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,       # Background = 0
        dtype='uint8' # 0 or 1 fits in uint8
    )

    # Save to file
    ref_meta.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint8",
        "nodata": None,
        "compress": "lzw"
    })

    os.makedirs(os.path.dirname(OUTPUT_MASK_PATH), exist_ok=True)
    with rasterio.open(OUTPUT_MASK_PATH, "w", **ref_meta) as dest:
        dest.write(mask, 1)

    print(f"\n[OK] Saved binary mask to: {OUTPUT_MASK_PATH}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Palm pixels: {np.sum(mask == 1):,}")
    print(f"  Background pixels: {np.sum(mask == 0):,}")

if __name__ == "__main__":
    main()
