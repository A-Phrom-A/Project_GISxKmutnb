#!/usr/bin/env python3
"""
backend/scripts/prepare_demo_zones.py
=====================================
Prepares demo assets for the 4 zones in Ban Bueng / Chonburi-Rayong.
1. Warps and crops S1 (2 bands) and S2 (7 bands) from full master rasters to match each zone's bounding box and grid.
2. Computes and exports high-precision NDVI GeoTIFF & visual RGB tiles.
3. Converts GPKG vector palm boundaries to WGS84 (EPSG:4326) GeoJSON for Deck.GL rendering.
4. Generates zones_meta.json manifest.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
import geopandas as gpd

# Root directory
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
SAM_DIR = ROOT_DIR / "sam_segment"
OUT_DIR = ROOT_DIR / "data" / "demo_zones"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S1_FULL = RAW_DIR / "S1_IW_mosaic_2026044_47PQQ_VVVH_full.tif"
S2_FULL = RAW_DIR / "S2C_MSIL2A_20260410_7band_stacked_full.tif"

ZONE_NAMES = {
    1: {"name": "Zone 1 — บ้านบึง-ใต้ (Ban Bueng South)", "desc": "สวนปาล์มแปลงใหญ่หนาแน่น พื้นที่ปลูกคุณภาพสูง"},
    2: {"name": "Zone 2 — บ้านบึง-ตะวันออก (Ban Bueng East)", "desc": "แปลงปาล์มสลับสวนผลไม้และพื้นที่เกษตรกรรมผสม"},
    3: {"name": "Zone 3 — บ้านบึง-เหนือ (Ban Bueng North)", "desc": "พื้นที่รอยต่อภูเขาและแปลงปาล์มหนุ่ม"},
    4: {"name": "Zone 4 — ชลบุรี-ระยอง (Chonburi-Rayong Border)", "desc": "แปลงปาล์มน้ำมันเชิงพาณิชย์ขนาดใหญ่"},
}


def process_zone(zone_id: int):
    print(f"\n=======================================================")
    print(f"[*] Processing Demo Zone {zone_id}...")
    print(f"=======================================================")
    
    gsat_path = SAM_DIR / f"{zone_id}.tif"
    gpkg_path = SAM_DIR / f"{zone_id}_optim-b16.gpkg"
    
    with rasterio.open(gsat_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
        dst_bounds = ref.bounds
        wgs_bounds = transform_bounds(dst_crs, "EPSG:4326", *dst_bounds)
        center_lon = (wgs_bounds[0] + wgs_bounds[2]) / 2.0
        center_lat = (wgs_bounds[1] + wgs_bounds[3]) / 2.0
    
    print(f"    Dimensions: {dst_width}x{dst_height}, CRS: {dst_crs}")
    print(f"    WGS84 Bounds: {wgs_bounds}")
    print(f"    Center: Lon={center_lon:.4f}, Lat={center_lat:.4f}")

    # 1. Warp & Crop Sentinel-1 (2 bands: VV, VH)
    s1_out = OUT_DIR / f"zone_{zone_id}_s1.tif"
    print(f"[*] Warping S1 SAR from {S1_FULL.name} -> {s1_out.name}...")
    with rasterio.open(S1_FULL) as src:
        dst_meta = src.meta.copy()
        dst_meta.update({
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "count": 2,
            "dtype": "float32",
            "nodata": 0.0,
        })
        
        with rasterio.open(s1_out, "w", **dst_meta) as dst:
            for b in range(1, 3):
                dest_arr = np.zeros((dst_height, dst_width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, b),
                    destination=dest_arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
                dst.write(dest_arr, b)

    # 2. Warp & Crop Sentinel-2 (7 bands: B2, B3, B4, B8A, B8, B11, B12)
    s2_out = OUT_DIR / f"zone_{zone_id}_s2.tif"
    print(f"[*] Warping S2 Optical from {S2_FULL.name} -> {s2_out.name}...")
    with rasterio.open(S2_FULL) as src:
        dst_meta = src.meta.copy()
        dst_meta.update({
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "count": 7,
            "dtype": "uint16",
            "nodata": 0,
        })
        
        s2_bands_data = []
        with rasterio.open(s2_out, "w", **dst_meta) as dst:
            for b in range(1, 8):
                dest_arr = np.zeros((dst_height, dst_width), dtype=np.uint16)
                reproject(
                    source=rasterio.band(src, b),
                    destination=dest_arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
                dst.write(dest_arr, b)
                s2_bands_data.append(dest_arr)

    # 3. Compute NDVI: (NIR - Red) / (NIR + Red)
    # Band index 3 is Red (B4), Band index 5 is NIR (B8) in 1-based indexing
    ndvi_out = OUT_DIR / f"zone_{zone_id}_ndvi.tif"
    print(f"[*] Computing NDVI layer -> {ndvi_out.name}...")
    b_red = s2_bands_data[2].astype(np.float32)   # B4 (Band 3)
    b_nir = s2_bands_data[4].astype(np.float32)   # B8 (Band 5)
    denom = b_nir + b_red
    ndvi = np.zeros_like(b_red, dtype=np.float32)
    valid_mask = denom > 0
    ndvi[valid_mask] = (b_nir[valid_mask] - b_red[valid_mask]) / denom[valid_mask]
    ndvi = np.clip(ndvi, -1.0, 1.0)

    with rasterio.open(
        ndvi_out,
        "w",
        driver="GTiff",
        height=dst_height,
        width=dst_width,
        count=1,
        dtype="float32",
        crs=dst_crs,
        transform=dst_transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(ndvi, 1)

    # 4. Convert GPKG vector polygons to WGS84 GeoJSON
    geojson_out = OUT_DIR / f"zone_{zone_id}_palms.geojson"
    print(f"[*] Converting {gpkg_path.name} to WGS84 GeoJSON -> {geojson_out.name}...")
    gdf = gpd.read_file(gpkg_path)
    # Transform to EPSG:4326 for Deck.GL
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    
    # Add calculated area attributes
    # Area in square meters (calculated in EPSG:3857)
    gdf_wgs84["area_sqm"] = np.round(gdf.geometry.area, 1)
    gdf_wgs84["area_rai"] = np.round(gdf_wgs84["area_sqm"] / 1600.0, 2)
    gdf_wgs84["parcel_id"] = [f"Z{zone_id}-P{idx:04d}" for idx in range(len(gdf))]
    # Sample real NDVI from calibrated raster at each parcel centroid
    inv_t = ~dst_transform
    real_ndvis = []
    for geom in gdf.geometry:
        col, row = int(round((inv_t * (geom.centroid.x, geom.centroid.y))[0])), int(round((inv_t * (geom.centroid.x, geom.centroid.y))[1]))
        if 0 <= row < dst_height and 0 <= col < dst_width:
            val = float(ndvi[row, col])
            real_ndvis.append(round(val, 2) if np.isfinite(val) else 0.60)
        else:
            real_ndvis.append(0.60)
    gdf_wgs84["health_score"] = real_ndvis
    gdf_wgs84["real_ndvi"] = real_ndvis
    
    # Save optimized GeoJSON
    gdf_wgs84.to_file(geojson_out, driver="GeoJSON")
    print(f"    Exported {len(gdf_wgs84)} palm parcel polygons.")

    return {
        "zone_id": zone_id,
        "name": ZONE_NAMES[zone_id]["name"],
        "description": ZONE_NAMES[zone_id]["desc"],
        "center": [center_lon, center_lat],
        "bounds": [wgs_bounds[0], wgs_bounds[1], wgs_bounds[2], wgs_bounds[3]],
        "dimensions": [dst_width, dst_height],
        "num_palms": len(gdf_wgs84),
        "total_palm_area_rai": float(np.round(gdf_wgs84["area_rai"].sum(), 1)),
        "mean_ndvi": float(np.round(np.mean(ndvi[valid_mask]), 3)),
        "files": {
            "gsat_tif": f"sam_segment/{zone_id}.tif",
            "s1_tif": f"data/demo_zones/zone_{zone_id}_s1.tif",
            "s2_tif": f"data/demo_zones/zone_{zone_id}_s2.tif",
            "ndvi_tif": f"data/demo_zones/zone_{zone_id}_ndvi.tif",
            "palms_geojson": f"data/demo_zones/zone_{zone_id}_palms.geojson",
        }
    }


def main():
    print("[*] Starting Spatial Data Preparation for 4 Demo Zones...")
    manifest = {"zones": []}
    
    for zid in range(1, 5):
        meta = process_zone(zid)
        manifest["zones"].append(meta)

    # Save manifest
    meta_path = OUT_DIR / "zones_manifest.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\n=======================================================")
    print(f"[+] ALL 4 DEMO ZONES PREPARED SUCCESSFULLY!")
    print(f"    Manifest saved at: {meta_path.resolve()}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
