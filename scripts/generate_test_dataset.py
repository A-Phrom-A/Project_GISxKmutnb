"""
scripts/generate_test_dataset.py
================================
Generates a realistic test dataset for a brand-new geographic area:
'Zone 5: Surat Thani (สุราษฎร์ธานี - แหล่งปลูกปาล์มน้ำมันหลักภาคใต้)'

Dataset covers a 5.3 km x 5.3 km extent at UTM 47N (EPSG:32647):
1. surat_thani_s2_7bands_clean.tif    : 7 Bands Sentinel-2 L2A (B2, B3, B4, B8A, B8, B11, B12)
2. surat_thani_s1_sar_dualpol.tif     : 2 Bands Sentinel-1 SAR (VV, VH)
3. surat_thani_s2_cloudy.tif          : 7 Bands Sentinel-2 with realistic cumulus clouds
4. surat_thani_gsat_rgb_highres.tif   : 3 Bands High-Res RGB (uint8)
5. surat_thani_parcels.geojson        : 42 Vector Farm Polygons (EPSG:4326)
"""

import os
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon
from pyproj import Transformer
from pathlib import Path

OUT_DIR = Path("data/test_samples/surat_thani_zone5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Georeference parameters for Surat Thani (Kanchanadit/Phunphin area)
# WGS84: Lon ~ 99.45, Lat ~ 9.14
# UTM 47N coordinates:
X_MIN = 549000.0
Y_MAX = 1015000.0
PIXEL_SIZE = 10.0   # 10 meters per pixel
WIDTH = 530         # 5,300 meters
HEIGHT = 530        # 5,300 meters
CRS = "EPSG:32647"

TRANSFORM = from_origin(X_MIN, Y_MAX, PIXEL_SIZE, PIXEL_SIZE)

print(f"Generating synthetic remote sensing dataset for Surat Thani...")
print(f"Extent: 5.3 km x 5.3 km | Grid: {WIDTH}x{HEIGHT} px @ {PIXEL_SIZE}m/px | CRS: {CRS}")

# 1. Base Landscape Synthesis
np.random.seed(42)
yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]

# Simulating rolling terrain, palm plantation grid, and roads
palm_mask = (np.sin(xx / 25.0) * np.cos(yy / 25.0) > -0.3).astype(np.float32)
river_mask = (np.abs((yy - 120) - 40 * np.sin(xx / 60.0)) < 12).astype(np.float32)
road_h = (np.abs(yy - 265) < 3).astype(np.float32)
road_v = (np.abs(xx - 350) < 3).astype(np.float32)
road_mask = np.clip(road_h + road_v, 0.0, 1.0)

# 2. Generate Sentinel-2 7 Bands (uint16 [0..10000])
# Bands: [B2, B3, B4, B8A, B8, B11, B12]
s2_clean = np.zeros((7, HEIGHT, WIDTH), dtype=np.uint16)

# Palm spectral profile:
# B2 (Blue): 350, B3 (Green): 650, B4 (Red): 420
# B8A (Narrow NIR): 3600, B8 (Broad NIR): 4200
# B11 (SWIR1): 1850, B12 (SWIR2): 980
palm_spec = [350, 650, 420, 3600, 4200, 1850, 980]

# Bare soil / road profile:
soil_spec = [750, 1100, 1450, 2100, 2300, 3200, 2800]

# Water profile:
water_spec = [500, 600, 300, 120, 100, 50, 40]

for b_idx in range(7):
    band_arr = np.full((HEIGHT, WIDTH), palm_spec[b_idx], dtype=np.float32)
    # Add noise & texture
    noise = np.random.normal(0, palm_spec[b_idx] * 0.07, (HEIGHT, WIDTH))
    band_arr += noise

    # Overlay soil/roads
    band_arr = np.where(road_mask > 0.5, soil_spec[b_idx] + np.random.normal(0, 50, (HEIGHT, WIDTH)), band_arr)
    # Overlay water
    band_arr = np.where(river_mask > 0.5, water_spec[b_idx] + np.random.normal(0, 20, (HEIGHT, WIDTH)), band_arr)

    s2_clean[b_idx] = np.clip(band_arr, 10, 10000).astype(np.uint16)

# Save Clean S2
s2_clean_path = OUT_DIR / "surat_thani_s2_7bands_clean.tif"
with rasterio.open(
    s2_clean_path,
    "w",
    driver="GTiff",
    height=HEIGHT,
    width=WIDTH,
    count=7,
    dtype="uint16",
    crs=CRS,
    transform=TRANSFORM,
    compress="deflate",
) as dst:
    for i in range(7):
        dst.write(s2_clean[i], i + 1)
        dst.set_band_description(i + 1, ["B2-Blue", "B3-Green", "B4-Red", "B8A-RedEdge", "B8-NIR", "B11-SWIR1", "B12-SWIR2"][i])
print(f"  [OK] Created: {s2_clean_path}")

# 3. Generate Sentinel-1 SAR Dual-Pol (VV, VH) in float32 dB
# Palm canopy: VV ~ -9.5 dB, VH ~ -15.8 dB
# Water: very low backscatter (VV ~ -23 dB, VH ~ -28 dB)
# Roads/Buildings: high backscatter (VV ~ -4 dB, VH ~ -10 dB)
s1_sar = np.zeros((2, HEIGHT, WIDTH), dtype=np.float32)

# VV
vv = np.full((HEIGHT, WIDTH), -9.5, dtype=np.float32) + np.random.normal(0, 1.2, (HEIGHT, WIDTH))
vv = np.where(river_mask > 0.5, -23.0 + np.random.normal(0, 1.5, (HEIGHT, WIDTH)), vv)
vv = np.where(road_mask > 0.5, -5.0 + np.random.normal(0, 1.0, (HEIGHT, WIDTH)), vv)
s1_sar[0] = vv

# VH
vh = np.full((HEIGHT, WIDTH), -15.8, dtype=np.float32) + np.random.normal(0, 1.1, (HEIGHT, WIDTH))
vh = np.where(river_mask > 0.5, -28.0 + np.random.normal(0, 1.5, (HEIGHT, WIDTH)), vh)
vh = np.where(road_mask > 0.5, -11.0 + np.random.normal(0, 1.0, (HEIGHT, WIDTH)), vh)
s1_sar[1] = vh

s1_sar_path = OUT_DIR / "surat_thani_s1_sar_dualpol.tif"
with rasterio.open(
    s1_sar_path,
    "w",
    driver="GTiff",
    height=HEIGHT,
    width=WIDTH,
    count=2,
    dtype="float32",
    crs=CRS,
    transform=TRANSFORM,
    compress="deflate",
) as dst:
    dst.write(s1_sar[0], 1)
    dst.set_band_description(1, "VV_Backscatter_dB")
    dst.write(s1_sar[1], 2)
    dst.set_band_description(2, "VH_Backscatter_dB")
print(f"  [OK] Created: {s1_sar_path}")

# 4. Generate Sentinel-2 Cloudy for Cloud-MAE Testing (Scenario 2)
s2_cloudy = s2_clean.copy()
# Create thick cumulus cloud patch in center-left
dist_center = np.sqrt((xx - 220)**2 + (yy - 260)**2)
cloud_patch = np.exp(- (dist_center**2) / (2 * (90**2)))
cloud_mask = (cloud_patch > 0.35).astype(np.float32)

# Cloud shadows to the north-west
dist_shadow = np.sqrt((xx - 180)**2 + (yy - 220)**2)
shadow_patch = np.exp(- (dist_shadow**2) / (2 * (65**2)))
shadow_mask = ((shadow_patch > 0.45) & (cloud_mask < 0.2)).astype(np.float32)

for b_idx in range(7):
    band_c = s2_cloudy[b_idx].astype(np.float32)
    # Bright white cloud reflectance (~8500-9500)
    band_c = band_c * (1.0 - cloud_mask) + (8800.0 + np.random.normal(0, 250, (HEIGHT, WIDTH))) * cloud_mask
    # Dark shadow
    band_c = band_c * (1.0 - shadow_mask * 0.75)
    s2_cloudy[b_idx] = np.clip(band_c, 50, 10000).astype(np.uint16)

s2_cloudy_path = OUT_DIR / "surat_thani_s2_cloudy.tif"
with rasterio.open(
    s2_cloudy_path,
    "w",
    driver="GTiff",
    height=HEIGHT,
    width=WIDTH,
    count=7,
    dtype="uint16",
    crs=CRS,
    transform=TRANSFORM,
    compress="deflate",
) as dst:
    for i in range(7):
        dst.write(s2_cloudy[i], i + 1)
        dst.set_band_description(i + 1, ["B2-Blue", "B3-Green", "B4-Red", "B8A-RedEdge", "B8-NIR", "B11-SWIR1", "B12-SWIR2"][i])
print(f"  [OK] Created: {s2_cloudy_path}")

# 5. Generate High-Res RGB GSat Imagery (uint8)
# Extract R (B4), G (B3), B (B2) and stretch to 0-255 with vibrant tone
gsat_rgb = np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8)
red_norm = np.clip((s2_clean[2].astype(np.float32) - 150) / 1200.0 * 255.0, 0, 255).astype(np.uint8)
green_norm = np.clip((s2_clean[1].astype(np.float32) - 200) / 1400.0 * 255.0, 0, 255).astype(np.uint8)
blue_norm = np.clip((s2_clean[0].astype(np.float32) - 150) / 1000.0 * 255.0, 0, 255).astype(np.uint8)
gsat_rgb[0] = red_norm
gsat_rgb[1] = green_norm
gsat_rgb[2] = blue_norm

gsat_path = OUT_DIR / "surat_thani_gsat_rgb_highres.tif"
with rasterio.open(
    gsat_path,
    "w",
    driver="GTiff",
    height=HEIGHT,
    width=WIDTH,
    count=3,
    dtype="uint8",
    crs=CRS,
    transform=TRANSFORM,
    compress="deflate",
) as dst:
    dst.write(gsat_rgb[0], 1)
    dst.set_band_description(1, "Red")
    dst.write(gsat_rgb[1], 2)
    dst.set_band_description(2, "Green")
    dst.write(gsat_rgb[2], 3)
    dst.set_band_description(3, "Blue")
print(f"  [OK] Created: {gsat_path}")

# 6. Generate Vector Farm Parcels (GeoJSON, EPSG:4326)
transformer = Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)

parcels = []
parcel_id = 1
# Create a grid of realistic agricultural parcels (avoiding the river)
for row in range(7):
    for col in range(6):
        px_x1 = 40 + col * 75 + np.random.randint(-5, 5)
        px_y1 = 40 + row * 65 + np.random.randint(-5, 5)
        px_w = 60 + np.random.randint(-6, 8)
        px_h = 50 + np.random.randint(-6, 8)

        # Skip if intersecting river center
        center_y = px_y1 + px_h // 2
        center_x = px_x1 + px_w // 2
        river_y_at_x = 120 + 40 * np.sin(center_x / 60.0)
        if abs(center_y - river_y_at_x) < 22:
            continue

        # Convert pixel to UTM coords
        utm_x1 = X_MIN + px_x1 * PIXEL_SIZE
        utm_y1 = Y_MAX - (px_y1 + px_h) * PIXEL_SIZE
        utm_x2 = X_MIN + (px_x1 + px_w) * PIXEL_SIZE
        utm_y2 = Y_MAX - px_y1 * PIXEL_SIZE

        # Convert to WGS84
        lon1, lat1 = transformer.transform(utm_x1, utm_y1)
        lon2, lat2 = transformer.transform(utm_x2, utm_y2)

        poly_wgs = box(min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))
        area_m2 = (utm_x2 - utm_x1) * (utm_y2 - utm_y1)
        area_rai = round(area_m2 / 1600.0, 1)

        parcels.append({
            "parcel_id": f"SRT-PALM-{parcel_id:04d}",
            "zone": "Surat Thani (Zone 5)",
            "district": "Kanchanadit",
            "crop_type": "Oil Palm (Elaeis guineensis)",
            "age_years": int(np.random.choice([4, 7, 12, 16, 20])),
            "is_palm": True,
            "area_rai": area_rai,
            "area_sqm": round(area_m2, 1),
            "geometry": poly_wgs
        })
        parcel_id += 1

gdf_parcels = gpd.GeoDataFrame(parcels, crs="EPSG:4326")
geojson_path = OUT_DIR / "surat_thani_parcels.geojson"
gdf_parcels.to_file(geojson_path, driver="GeoJSON")
print(f"  [OK] Created: {geojson_path} ({len(gdf_parcels)} parcels)")

print(f"\nAll Surat Thani Zone 5 test datasets generated successfully in: {OUT_DIR.resolve()}")
