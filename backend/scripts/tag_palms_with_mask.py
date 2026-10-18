"""
backend/scripts/tag_palms_with_mask.py
======================================
Enriches all 4 Demo Zone GeoJSONs with actual ground truth palm mask overlap:
1. Calculates palm_ratio for every polygon against data/processed/palm_mask.tif.
2. Sets is_palm: True for real palm plantations (ratio >= 0.20), False for unrelated land.
3. Sets land_type: 'Oil Palm' vs 'Non-Palm'.
4. Saves enriched GeoJSON back to data/demo_zones/zone_{id}_palms.geojson.
"""

from pathlib import Path
import numpy as np
import rasterio
import rasterio.features
import geopandas as gpd

ROOT_DIR = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT_DIR / "data" / "demo_zones"
PALM_MASK = ROOT_DIR / "data" / "processed" / "palm_mask.tif"


def tag_zone(zone_id: int):
    geojson_path = DEMO_DIR / f"zone_{zone_id}_palms.geojson"
    if not geojson_path.exists():
        print(f"[!] GeoJSON not found: {geojson_path}")
        return

    print(f"[*] Processing Zone {zone_id} against palm_mask.tif...")
    gdf = gpd.read_file(geojson_path)
    
    with rasterio.open(PALM_MASK) as src:
        gdf_utm = gdf.to_crs(src.crs)
        window = rasterio.windows.from_bounds(*gdf_utm.total_bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        mask_arr = src.read(1, window=window)
        win_transform = rasterio.windows.transform(window, src.transform)

        shapes = ((geom, idx) for idx, geom in enumerate(gdf_utm.geometry))
        burned = rasterio.features.rasterize(
            shapes,
            out_shape=mask_arr.shape,
            transform=win_transform,
            fill=-1,
            dtype=np.int32
        )

        palm_pixels = (mask_arr == 1)
        num_polys = len(gdf)
        totals = np.bincount(burned[burned >= 0], minlength=num_polys)
        palms = np.bincount(burned[(burned >= 0) & palm_pixels], minlength=num_polys)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(totals > 0, palms / totals, 0.0)

    # Enrich GeoPandas DataFrame
    is_palm_list = ratios >= 0.20
    gdf["palm_ratio"] = np.round(ratios, 3)
    gdf["is_palm"] = is_palm_list
    gdf["land_type"] = np.where(is_palm_list, "Oil Palm (สวนปาล์มน้ำมัน)", "Non-Palm (พื้นที่อื่น ๆ)")

    # Save enriched GeoJSON
    gdf.to_file(geojson_path, driver="GeoJSON")
    palm_count = sum(is_palm_list)
    print(f"    -> Zone {zone_id}: {palm_count}/{num_polys} ({palm_count/num_polys*100:.1f}%) are confirmed Oil Palms!")


if __name__ == "__main__":
    for zid in range(1, 5):
        tag_zone(zid)
    print("\n[+] All 4 zones successfully tagged with ground truth palm overlap!")
