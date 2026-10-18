"""
scripts/generate_all_test_samples.py
====================================
Generates comprehensive test suites for GeoAI Data Ingestion:
1. Valid datasets (Passing 100% strict prerequisite checks)
2. Negative / Edge cases (Intentionally failing specific prerequisite checks)

Organized into purpose-driven subfolders under data/test_samples/:
- 01_existingzone_valid/
- 01_existingzone_no_crs/
- 01_existingzone_insufficient_bands/
- 01_existingzone_corrupted_format/

- 02_cloudrepair_valid/
- 02_cloudrepair_s1_single_pol/
- 02_cloudrepair_s2_insufficient_bands/
- 02_cloudrepair_no_crs/

- 03_newzone_valid/
- 03_newzone_vector_points_only/
- 03_newzone_vector_empty/
- 03_newzone_gsat_no_crs/
"""

import os
import shutil
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import box, Point, LineString
from pyproj import Transformer
from pathlib import Path

BASE_DIR = Path("data/test_samples")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Shared parameters for Surat Thani synthetic landscape
X_MIN = 549000.0
Y_MAX = 1015000.0
PIXEL_SIZE = 10.0
WIDTH = 256
HEIGHT = 256
CRS_UTM = "EPSG:32647"
TRANSFORM = from_origin(X_MIN, Y_MAX, PIXEL_SIZE, PIXEL_SIZE)

print(f"Creating organized test suites under: {BASE_DIR.resolve()}")

# Clean up older surat_thani_zone5 single folder if desired, or keep as reference
# We will create clear purpose-named subfolders

np.random.seed(42)
yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]

# Base landscape arrays
palm_mask = (np.sin(xx / 20.0) * np.cos(yy / 20.0) > -0.2).astype(np.float32)
road_mask = (np.abs(yy - 128) < 3).astype(np.float32)

# S2 7 Bands Clean Array
s2_7bands = np.zeros((7, HEIGHT, WIDTH), dtype=np.uint16)
base_specs = [350, 650, 420, 3600, 4200, 1850, 980]
for b in range(7):
    arr = np.full((HEIGHT, WIDTH), base_specs[b], dtype=np.float32)
    arr += np.random.normal(0, base_specs[b] * 0.08, (HEIGHT, WIDTH))
    arr = np.where(road_mask > 0.5, 1200 + b * 200, arr)
    s2_7bands[b] = np.clip(arr, 50, 10000).astype(np.uint16)

# S2 Cloudy Array
s2_cloudy = s2_7bands.copy()
dist = np.sqrt((xx - 110)**2 + (yy - 120)**2)
c_mask = (dist < 60).astype(np.float32)
for b in range(7):
    band_c = s2_cloudy[b].astype(np.float32)
    band_c = band_c * (1.0 - c_mask) + (9200.0 + np.random.normal(0, 150, (HEIGHT, WIDTH))) * c_mask
    s2_cloudy[b] = np.clip(band_c, 50, 10000).astype(np.uint16)

# S1 Dual-Pol Array
s1_2bands = np.zeros((2, HEIGHT, WIDTH), dtype=np.float32)
s1_2bands[0] = -9.5 + np.random.normal(0, 1.2, (HEIGHT, WIDTH))
s1_2bands[1] = -15.8 + np.random.normal(0, 1.1, (HEIGHT, WIDTH))

# GSat RGB Array (uint8)
gsat_3bands = np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8)
gsat_3bands[0] = np.clip((s2_7bands[2].astype(np.float32) - 150) / 1200.0 * 255.0, 0, 255).astype(np.uint8)
gsat_3bands[1] = np.clip((s2_7bands[1].astype(np.float32) - 200) / 1400.0 * 255.0, 0, 255).astype(np.uint8)
gsat_3bands[2] = np.clip((s2_7bands[0].astype(np.float32) - 150) / 1000.0 * 255.0, 0, 255).astype(np.uint8)

# Helper writer functions
def write_geotiff(path, data, crs=CRS_UTM, transform=TRANSFORM, count=None, dtype=None):
    c = count or (data.shape[0] if data.ndim == 3 else 1)
    dt = dtype or str(data.dtype)
    h, w = data.shape[-2], data.shape[-1]
    kwargs = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": c,
        "dtype": dt,
        "compress": "deflate"
    }
    if crs:
        kwargs["crs"] = crs
    if transform:
        kwargs["transform"] = transform

    with rasterio.open(path, "w", **kwargs) as dst:
        if data.ndim == 3:
            for i in range(c):
                dst.write(data[i], i + 1)
        else:
            dst.write(data, 1)

# ==============================================================================
# SUITE 1: Existing Zone (Update Satellite)
# ==============================================================================
# 1.1 Valid
dir_1_val = BASE_DIR / "01_existingzone_valid"
dir_1_val.mkdir(parents=True, exist_ok=True)
write_geotiff(dir_1_val / "s2_new_cycle_clean_7bands.tif", s2_7bands)

# 1.2 Failing: No CRS
dir_1_nocrs = BASE_DIR / "01_existingzone_no_crs"
dir_1_nocrs.mkdir(parents=True, exist_ok=True)
write_geotiff(dir_1_nocrs / "s2_missing_crs.tif", s2_7bands, crs=None, transform=None)

# 1.3 Failing: Insufficient Bands (only 1 band instead of >= 3)
dir_1_bands = BASE_DIR / "01_existingzone_insufficient_bands"
dir_1_bands.mkdir(parents=True, exist_ok=True)
write_geotiff(dir_1_bands / "s2_only_1band_grayscale.tif", s2_7bands[0:1])

# 1.4 Failing: Corrupted format
dir_1_corrupt = BASE_DIR / "01_existingzone_corrupted_format"
dir_1_corrupt.mkdir(parents=True, exist_ok=True)
with open(dir_1_corrupt / "corrupted_satellite_image.tif", "wb") as f:
    f.write(b"NOT_A_REAL_GEOTIFF_HEADER_DATA_1234567890")

# ==============================================================================
# SUITE 2: Cloud Repair (S2 + S1 Pair)
# ==============================================================================
# 2.1 Valid
dir_2_val = BASE_DIR / "02_cloudrepair_valid"
dir_2_val.mkdir(parents=True, exist_ok=True)
write_geotiff(dir_2_val / "s2_optical_cloudy_7bands.tif", s2_cloudy)
write_geotiff(dir_2_val / "s1_sar_radar_dualpol.tif", s1_2bands)

# 2.2 Failing: S1 Single-Pol (only 1 band VV, missing VH)
dir_2_singlepol = BASE_DIR / "02_cloudrepair_s1_single_pol"
dir_2_singlepol.mkdir(parents=True, exist_ok=True)
write_geotiff(dir_2_singlepol / "s2_optical_cloudy_7bands.tif", s2_cloudy)
write_geotiff(dir_2_singlepol / "s1_sar_single_band_vv_only.tif", s1_2bands[0:1])

# 2.3 Failing: S2 Insufficient Bands
dir_2_s2bands = BASE_DIR / "02_cloudrepair_s2_insufficient_bands"
dir_2_s2bands.mkdir(parents=True, exist_ok=True)
write_geotiff(dir_2_s2bands / "s2_optical_only_1band.tif", s2_cloudy[0:1])
write_geotiff(dir_2_s2bands / "s1_sar_radar_dualpol.tif", s1_2bands)

# 2.4 Failing: Missing CRS on both
dir_2_nocrs = BASE_DIR / "02_cloudrepair_no_crs"
dir_2_nocrs.mkdir(parents=True, exist_ok=True)
write_geotiff(dir_2_nocrs / "s2_cloudy_no_crs.tif", s2_cloudy, crs=None, transform=None)
write_geotiff(dir_2_nocrs / "s1_sar_no_crs.tif", s1_2bands, crs=None, transform=None)

# ==============================================================================
# SUITE 3: Brand New Zone (S1 + S2 + GSat + Vector)
# ==============================================================================
transformer = Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)

# Generate 25 valid farm polygons
valid_parcels = []
for r in range(5):
    for c in range(5):
        ux1 = X_MIN + (30 + c * 40) * PIXEL_SIZE
        uy1 = Y_MAX - (30 + r * 40 + 30) * PIXEL_SIZE
        ux2 = ux1 + 30 * PIXEL_SIZE
        uy2 = uy1 + 30 * PIXEL_SIZE
        lon1, lat1 = transformer.transform(ux1, uy1)
        lon2, lat2 = transformer.transform(ux2, uy2)
        valid_parcels.append({
            "parcel_id": f"TEST-PARCEL-{r*5+c+1:03d}",
            "crop": "Oil Palm",
            "area_rai": 7.5,
            "geometry": box(min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))
        })
gdf_valid_polygons = gpd.GeoDataFrame(valid_parcels, crs="EPSG:4326")

# 3.1 Valid
dir_3_val = BASE_DIR / "03_newzone_valid"
dir_3_val.mkdir(parents=True, exist_ok=True)
write_geotiff(dir_3_val / "s1_sar_dualpol.tif", s1_2bands)
write_geotiff(dir_3_val / "s2_optical_7bands.tif", s2_7bands)
write_geotiff(dir_3_val / "gsat_highres_rgb.tif", gsat_3bands)
gdf_valid_polygons.to_file(dir_3_val / "parcels_boundary_polygons.geojson", driver="GeoJSON")

# 3.2 Failing: Vector has Points only (No Polygons)
dir_3_points = BASE_DIR / "03_newzone_vector_points_only"
dir_3_points.mkdir(parents=True, exist_ok=True)
points_data = [
    {"point_id": 1, "name": "GPS Waypoint 1", "geometry": Point(99.452, 9.125)},
    {"point_id": 2, "name": "GPS Waypoint 2", "geometry": Point(99.460, 9.130)},
    {"point_id": 3, "name": "GPS Waypoint 3", "geometry": Point(99.470, 9.140)},
]
gpd.GeoDataFrame(points_data, crs="EPSG:4326").to_file(dir_3_points / "gps_waypoints_points_only.geojson", driver="GeoJSON")

# 3.3 Failing: Vector Empty (0 features)
dir_3_empty = BASE_DIR / "03_newzone_vector_empty"
dir_3_empty.mkdir(parents=True, exist_ok=True)
empty_geojson = {"type": "FeatureCollection", "features": []}
with open(dir_3_empty / "empty_parcels_zero_features.geojson", "w") as f:
    json.dump(empty_geojson, f)

# 3.4 Failing: GSat High-Res No CRS
dir_3_gsat_nocrs = BASE_DIR / "03_newzone_gsat_no_crs"
dir_3_gsat_nocrs.mkdir(parents=True, exist_ok=True)
write_geotiff(dir_3_gsat_nocrs / "gsat_rgb_missing_crs.tif", gsat_3bands, crs=None, transform=None)

# 3.5 Failing: Vector Missing CRS
dir_3_vec_nocrs = BASE_DIR / "03_newzone_vector_no_crs"
dir_3_vec_nocrs.mkdir(parents=True, exist_ok=True)
gdf_no_crs = gpd.GeoDataFrame(valid_parcels[:3], crs=None)
gdf_no_crs.to_file(dir_3_vec_nocrs / "parcels_missing_crs.gpkg", driver="GPKG")

print("\nAll Organized Test Suites Generated Successfully!")
for p in sorted(BASE_DIR.glob("*")):
    if p.is_dir():
        files = [f.name for f in p.glob("*")]
        print(f"  [DIR] {p.name}/ ({len(files)} files)")
