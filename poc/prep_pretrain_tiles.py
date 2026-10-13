"""
poc/prep_pretrain_tiles.py
==========================
Data preparation script for Cloud-MAE Pretraining across 5 Eastern Thailand provinces:
- Chanthaburi, Chonburi, Rayong, SaKaeo, Trat

Process:
1. Reads paired Sentinel-1 (2 bands) and Sentinel-2 (7 bands) GeoTIFFs from data/pretrain/raw/
2. Normalizes:
   - S1 (SAR dB): clip [-30.0, 0.0] dB -> [0.0, 1.0]
   - S2 (Optical 7 bands): [0, 10000] DN -> [0.0, 1.0]
3. Cuts 256x256 tiles with 50% overlap (stride=128) or non-overlapping (stride=256)
4. Filters out tiles with > 20% NaN or NoData
5. Saves clean patches to data/pretrain/tiles/ as compressed .npz (s1, s2)
"""

import sys
import time
from pathlib import Path
import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "pretrain" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "pretrain" / "tiles"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROVINCES = ["Chanthaburi", "Chonburi", "Rayong", "SaKaeo", "Trat"]

TILE_SIZE = 256
STRIDE = 128          # 50% overlap giving ~6,000 to 7,500 clean patches
MAX_NAN_RATIO = 0.20  # Discard patches with > 20% NaN / border pixels
MIN_VARIANCE = 1e-4   # Discard completely flat / pure sea patches


def normalize_s1(arr: np.ndarray) -> np.ndarray:
    """Normalize SAR dB to [0, 1]."""
    db_min = -30.0
    db_max = 0.0
    arr = np.nan_to_num(arr, nan=db_min)
    norm = (arr - db_min) / (db_max - db_min)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def normalize_s2(arr: np.ndarray) -> np.ndarray:
    """Normalize Optical reflectance (0-10000 DN) to [0, 1]."""
    arr = np.nan_to_num(arr, nan=0.0)
    norm = arr / 10000.0
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def process_province(province: str) -> int:
    s1_path = RAW_DIR / f"Sentinel1_ClearSky_{province}.tif"
    s2_path = RAW_DIR / f"Sentinel2_ClearSky_{province}.tif"

    if not s1_path.exists() or not s2_path.exists():
        print(f"  [SKIP] Missing files for {province}")
        return 0

    print(f"\nProcessing Province: {province}")
    print(f"  S1: {s1_path.name}")
    print(f"  S2: {s2_path.name}")

    t0 = time.time()
    saved_count = 0

    with rasterio.open(s1_path) as src_s1, rasterio.open(s2_path) as src_s2:
        H, W = min(src_s1.height, src_s2.height), min(src_s1.width, src_s2.width)
        print(f"  Raster Grid: {H} x {W} pixels")

        # Step through in blocks to keep memory usage low (~300MB per block)
        block_rows = 1024
        for r_start in range(0, H - TILE_SIZE + 1, block_rows - TILE_SIZE):
            r_end = min(r_start + block_rows, H)
            actual_block_h = r_end - r_start

            # Read window for this block
            win = rasterio.windows.Window(0, r_start, W, actual_block_h)
            s1_block = src_s1.read(window=win)  # (2, actual_block_h, W)
            s2_block = src_s2.read(window=win)  # (7, actual_block_h, W)

            # Slide tiles within this block
            for r in range(0, actual_block_h - TILE_SIZE + 1, STRIDE):
                global_r = r_start + r
                for c in range(0, W - TILE_SIZE + 1, STRIDE):
                    s1_patch = s1_block[:, r : r + TILE_SIZE, c : c + TILE_SIZE]
                    s2_patch = s2_block[:, r : r + TILE_SIZE, c : c + TILE_SIZE]

                    # 1. Check NaN ratio
                    nan_ratio = np.isnan(s2_patch).sum() / s2_patch.size
                    if nan_ratio > MAX_NAN_RATIO:
                        continue

                    # 2. Check all-zeros / black border ratio
                    zero_ratio = (s2_patch == 0).sum() / s2_patch.size
                    if zero_ratio > MAX_NAN_RATIO:
                        continue

                    # 3. Check variance (avoid flat empty sea or constant clouds)
                    var_s2 = np.nanvar(s2_patch[2])  # check red band variance
                    if var_s2 < MIN_VARIANCE:
                        continue

                    # Normalize
                    s1_norm = normalize_s1(s1_patch)
                    s2_norm = normalize_s2(s2_patch)

                    out_file = OUT_DIR / f"tile_{province}_r{global_r:05d}_c{c:05d}.npz"
                    np.savez_compressed(out_file, s1=s1_norm, s2=s2_norm)
                    saved_count += 1

    elapsed = time.time() - t0
    print(f"  [DONE] {province}: Saved {saved_count:,} valid tiles in {elapsed:.1f}s")
    return saved_count


def run():
    print("=" * 70)
    print("  [>] PRETRAINING DATA PREPARATION: 5 EASTERN PROVINCES")
    print("=" * 70)
    print(f"Output Directory: {OUT_DIR}")
    print(f"Tile Size: {TILE_SIZE}x{TILE_SIZE} | Stride: {STRIDE} | Max NaN: {MAX_NAN_RATIO*100:.0f}%")

    total_tiles = 0
    start_all = time.time()

    for prov in PROVINCES:
        count = process_province(prov)
        total_tiles += count

    total_time = time.time() - start_all
    print("\n" + "=" * 70)
    print(f"  [SUCCESS] All 5 provinces processed!")
    print(f"  Total Pretraining Tiles: {total_tiles:,}")
    print(f"  Total Time: {total_time:.1f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    run()
