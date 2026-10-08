"""
poc/data_prep.py
================
PoC Step 1 — Data Preparation Pipeline

อ่านข้อมูลดาวเทียมจาก data/raw/, ประมวลผลด้วย lib1_fusion,
แล้วบันทึก tile (12, 256, 256) ลงใน data/processed/

ข้อมูลที่ใช้:
  GEO-Sat  : data/raw/Gsat_z17.tif                               — 3 bands (R,G,B)  EPSG:3857  ~1.2 m/px
  S1 (SAR) : data/raw/S1_IW_mosaic_2026044_47PQQ_VVVH_Clip.tif  — 2 bands (VV,VH) EPSG:32647 20 m/px
  S2 (Opt) : data/raw/S2C_MSIL2A_20260410_7band_stacked_clip.tif — 7 bands          EPSG:32647 10 m/px

Pipeline:
  1. Validate files exist
  2. Print file info (shape, CRS, resolution)
  3. Use S2 as spatial reference  (EPSG:32647, 10 m/px)
  4. Reproject S1  (20m → 10m, UTM→UTM)      via rasterio.warp
  5. Reproject Gsat (~1.2m → 10m, 3857→32647) via rasterio.warp
  6. Normalize all bands to [0, 1]
  7. Early Fusion → (12, H, W)
  8. Cut 256x256 tiles with 50% overlap
  9. Save tiles as .npy → data/processed/tiles/
"""

import os
import sys
import numpy as np
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── ชื่อไฟล์จริงที่ใช้งาน ──────────────────────────────────────────────────────
DATA_FILES = {
    "gsat" : RAW_DIR / "Gsat_z17.tif",
    "s1"   : RAW_DIR / "S1_IW_mosaic_2026044_47PQQ_VVVH_Clip.tif",
    "s2"   : RAW_DIR / "S2C_MSIL2A_20260410_7band_stacked_clip.tif",
}

# ── Tiling config ──────────────────────────────────────────────────────────────
TILE_SIZE   = 256       # tile size ที่ส่งให้ Kaggle (px × px)
TILE_STRIDE = 128       # overlap = TILE_SIZE - TILE_STRIDE = 128 px (50% overlap)
MIN_VALID   = 0.70      # % of valid (non-nodata) pixels ต่ำกว่านี้ทิ้ง tile


# ─── Check rasterio ───────────────────────────────────────────────────────────
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, calculate_default_transform
except ImportError:
    print("rasterio not installed. Run: pip install rasterio")
    sys.exit(1)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def validate_files():
    """ตรวจสอบว่าไฟล์ทั้งหมดมีอยู่จริง"""
    print("Validating input files...")
    all_ok = True
    for key, path in DATA_FILES.items():
        exists = path.exists()
        size_mb = path.stat().st_size / 1e6 if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {key.upper():5s} : {path.name}  ({size_mb:.1f} MB)")
        if not exists:
            all_ok = False
    return all_ok


def read_bands_info():
    """แสดงข้อมูล band, shape, CRS, resolution ของแต่ละไฟล์"""
    print("\nFile Info:")
    infos = {}
    for key, path in DATA_FILES.items():
        with rasterio.open(path) as src:
            info = {
                "path"   : path,
                "shape"  : (src.count, src.height, src.width),
                "dtype"  : src.dtypes[0],
                "crs"    : src.crs,
                "res"    : src.res,
                "nodata" : src.nodata,
                "bounds" : src.bounds,
                "transform": src.transform,
            }
            print(f"\n  [{key.upper()}] {path.name}")
            print(f"    Shape  : {info['shape']}  (bands, H, W)")
            print(f"    Dtype  : {info['dtype']}")
            print(f"    CRS    : {info['crs']}")
            print(f"    Res    : {info['res']} m/px")
            print(f"    NoData : {info['nodata']}")
            infos[key] = info
    return infos


def reproject_to_reference(src_path: Path, ref_meta: dict) -> np.ndarray:
    """
    Reproject ไฟล์ .tif ให้ตรงกับ reference grid (CRS, transform, shape).

    Args:
        src_path : path ของไฟล์ต้นทาง
        ref_meta : dict ที่มี keys: crs, transform, width, height

    Returns:
        np.ndarray shape (bands, H, W) float32 — aligned กับ reference
    """
    dst_crs       = ref_meta["crs"]
    dst_transform = ref_meta["transform"]
    dst_width     = ref_meta["width"]
    dst_height    = ref_meta["height"]

    with rasterio.open(src_path) as src:
        n_bands = src.count
        dst_arr = np.zeros((n_bands, dst_height, dst_width), dtype=np.float32)

        for b in range(1, n_bands + 1):
            src_band = src.read(b).astype(np.float32)

            # Replace nodata with 0 before reprojection
            if src.nodata is not None:
                src_band[src_band == src.nodata] = 0.0

            reproject(
                source        = src_band,
                destination   = dst_arr[b - 1],
                src_transform = src.transform,
                src_crs       = src.crs,
                dst_transform = dst_transform,
                dst_crs       = dst_crs,
                resampling    = Resampling.bilinear,
            )

    return dst_arr  # (bands, H, W) float32


def linear_to_db(arr: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Convert SAR linear power scale → dB scale.
    Formula: dB = 10 * log10(linear_power)

    Args:
        arr : SAR array in linear power scale (positive floats)
        eps : small value to avoid log(0)

    Returns:
        Array in dB scale (float32)
    """
    arr = np.where(arr > 0, arr, eps)  # replace 0/negative with eps
    return (10.0 * np.log10(arr)).astype(np.float32)


def normalize_bands(arr: np.ndarray, key: str) -> np.ndarray:
    """
    Normalize array to [0, 1] ตาม sensor type.

    Args:
        arr : (bands, H, W) float32
        key : 's1' | 's2' | 'gsat'

    Returns:
        np.ndarray (bands, H, W) float32 [0, 1]
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from lib1_fusion.utils import normalize_sar_db, normalize_s2_reflectance, normalize_minmax

    if key == "s1":
        # S1 stored in LINEAR power scale → convert to dB first
        # dB = 10 * log10(linear); typical range after conversion: -25 to 0 dB
        vv_db = linear_to_db(arr[0])
        vh_db = linear_to_db(arr[1])
        print(f"    S1 VV dB range: [{vv_db[vv_db > -100].min():.1f}, {vv_db.max():.1f}] dB")
        print(f"    S1 VH dB range: [{vh_db[vh_db > -100].min():.1f}, {vh_db.max():.1f}] dB")
        vv = normalize_sar_db(vv_db, db_min=-25.0, db_max=0.0)
        vh = normalize_sar_db(vh_db, db_min=-25.0, db_max=0.0)
        return np.stack([vv, vh], axis=0)   # (2, H, W)

    elif key == "s2":
        # Optical: 7 bands — DN 0-10000
        return np.stack(
            [normalize_s2_reflectance(arr[i]) for i in range(7)], axis=0
        )  # (7, H, W)

    elif key == "gsat":
        # RGB uint8 0-255
        return np.stack(
            [normalize_minmax(arr[i], vmin=0.0, vmax=255.0) for i in range(3)], axis=0
        )  # (3, H, W)

    raise ValueError(f"Unknown key: {key}")


def cut_tiles(fused: np.ndarray, tile_size: int, stride: int, mask: np.ndarray = None) -> list:
    """
    ตัด tensor (12, H, W) เป็น tiles ย่อย (12, tile_size, tile_size)
    และตัด mask (H, W) เป็น (tile_size, tile_size) ด้วยถ้ามี

    Returns:
        list of (row, col, tile, mask_tile) tuples — เฉพาะ tile ที่มี valid pixels >= MIN_VALID
    """
    _, H, W = fused.shape
    tiles = []
    for r in range(0, H - tile_size + 1, stride):
        for c in range(0, W - tile_size + 1, stride):
            tile = fused[:, r:r + tile_size, c:c + tile_size]
            valid_ratio = np.count_nonzero(tile) / tile.size
            if valid_ratio < MIN_VALID:
                continue
            
            mask_tile = None
            if mask is not None:
                mask_tile = mask[r:r + tile_size, c:c + tile_size]
                
            tiles.append((r, c, tile, mask_tile))
    return tiles


def save_tiles(tiles: list, out_dir_img: Path, out_dir_mask: Path = None, prefix: str = "tile") -> int:
    """บันทึก tiles เป็น .npy ไฟล์ ทั้งรูปและ mask"""
    out_dir_img.mkdir(parents=True, exist_ok=True)
    if out_dir_mask is not None:
        out_dir_mask.mkdir(parents=True, exist_ok=True)
        
    saved = 0
    for row, col, tile, mask_tile in tiles:
        fname_img = out_dir_img / f"{prefix}_r{row:05d}_c{col:05d}.npy"
        np.save(fname_img, tile.astype(np.float32))
        
        if mask_tile is not None and out_dir_mask is not None:
            fname_mask = out_dir_mask / f"mask_r{row:05d}_c{col:05d}.npy"
            np.save(fname_mask, mask_tile.astype(np.uint8))
            
        saved += 1
    print(f"  Saved {saved} tiles to {out_dir_img}")
    if out_dir_mask:
        print(f"  Saved {saved} masks to {out_dir_mask}")
    return saved


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  GISxKMUTNB - Data Preparation Pipeline")
    print("=" * 60)

    # Step 1: Validate files
    if not validate_files():
        print("\nMissing files. Please upload all required satellite data.")
        return

    # Step 2: File info
    read_bands_info()

    # ── Step 3: Build reference grid from S2 (10m, EPSG:32647) ─────────────────
    print("\n[Step 3] Using S2 as spatial reference (EPSG:32647, 10 m/px)...")
    with rasterio.open(DATA_FILES["s2"]) as s2_src:
        ref_meta = {
            "crs"       : s2_src.crs,
            "transform" : s2_src.transform,
            "width"     : s2_src.width,
            "height"    : s2_src.height,
        }
    print(f"  Reference grid : H={ref_meta['height']}, W={ref_meta['width']}")
    print(f"  Reference CRS  : {ref_meta['crs']}")

    # ── Step 4: Reproject S1 (20m → 10m, same UTM zone) ───────────────────────
    print("\n[Step 4] Reprojecting S1 (20m -> 10m)...")
    s1_reproj = reproject_to_reference(DATA_FILES["s1"], ref_meta)
    print(f"  S1 reprojected shape : {s1_reproj.shape}")

    # ── Step 5: Reproject Gsat (~1.2m → 10m, EPSG:3857 → 32647) ──────────────
    print("\n[Step 5] Reprojecting Gsat (~1.2m -> 10m, 3857 -> 32647) ...")
    print("  (This may take 1-2 minutes — downsampling 23k x 23k image)")
    gsat_reproj = reproject_to_reference(DATA_FILES["gsat"], ref_meta)
    print(f"  Gsat reprojected shape : {gsat_reproj.shape}")

    # ── Step 6: Load S2 directly (already the reference) ──────────────────────
    print("\n[Step 6] Loading S2 (reference - no reproject needed)...")
    with rasterio.open(DATA_FILES["s2"]) as src:
        s2_raw = src.read().astype(np.float32)
        if src.nodata is not None:
            s2_raw[s2_raw == src.nodata] = 0.0
    print(f"  S2 shape : {s2_raw.shape}")
    
    print("\n[Step 6.5] Loading Palm Mask...")
    mask_path = os.path.join("data", "processed", "palm_mask.tif")
    if os.path.exists(mask_path):
        with rasterio.open(mask_path) as src:
            palm_mask = src.read(1)
        print(f"  Mask shape : {palm_mask.shape}")
    else:
        print("  WARNING: palm_mask.tif not found! Run poc/rasterize_mask.py first.")
        palm_mask = None

    # ── Step 7: Normalize ──────────────────────────────────────────────────────
    print("\n[Step 7] Normalizing bands to [0, 1]...")
    s1_arr   = normalize_bands(s1_reproj, "s1")    # (2, H, W)
    s2_arr   = normalize_bands(s2_raw,   "s2")    # (7, H, W)
    gsat_arr = normalize_bands(gsat_reproj, "gsat") # (3, H, W)
    print(f"  S1   : {s1_arr.shape}  range [{s1_arr.min():.3f}, {s1_arr.max():.3f}]")
    print(f"  S2   : {s2_arr.shape}  range [{s2_arr.min():.3f}, {s2_arr.max():.3f}]")
    print(f"  Gsat : {gsat_arr.shape}  range [{gsat_arr.min():.3f}, {gsat_arr.max():.3f}]")

    # ── Final alignment check ──────────────────────────────────────────────────
    assert s1_arr.shape[1:] == s2_arr.shape[1:] == gsat_arr.shape[1:], \
        f"Shape mismatch after reproject! S1={s1_arr.shape}, S2={s2_arr.shape}, Gsat={gsat_arr.shape}"
    print(f"  Alignment OK : all H={s2_arr.shape[1]}, W={s2_arr.shape[2]}")

    # ── Step 8: Early Fusion → (12, H, W) ─────────────────────────────────────
    print("\n[Step 8] Early Fusion (stack channels)...")
    sys.path.insert(0, str(PROJECT_ROOT))
    from lib1_fusion.fusion.early_fusion import early_fusion
    fused = early_fusion(s1_arr, s2_arr, gsat_arr)   # (12, H, W)
    print(f"  Fused tensor shape : {fused.shape}")
    print(f"  Value range        : [{fused.min():.4f}, {fused.max():.4f}]")
    print(f"  Dtype              : {fused.dtype}")

    # ── Step 9: Tile cutting ───────────────────────────────────────────────────
    print(f"\n[Step 9] Cutting tiles ({TILE_SIZE}x{TILE_SIZE} px, stride={TILE_STRIDE})...")
    tiles = cut_tiles(fused, TILE_SIZE, TILE_STRIDE, mask=palm_mask)
    print(f"  Total valid tiles : {len(tiles)}")

    # ── Step 10: Save tiles ────────────────────────────────────────────────────
    print("\n[Step 10] Saving tiles as .npy ...")
    tiles_dir = PROCESSED_DIR / "tiles"
    masks_dir = PROCESSED_DIR / "masks" if palm_mask is not None else None
    saved = save_tiles(tiles, tiles_dir, out_dir_mask=masks_dir, prefix="tile")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Data Preparation Complete!")
    print(f"  Tiles saved : {saved}")
    print(f"  Location    : {tiles_dir}")
    if masks_dir:
        print(f"  Masks Loc   : {masks_dir}")
    print(f"  Shape/tile  : (12, {TILE_SIZE}, {TILE_SIZE})")
    print(f"  Total size  : ~{saved * 12 * TILE_SIZE * TILE_SIZE * 4 / 1e6:.1f} MB")
    print("\n  Next step: Run label pipeline -> poc/rasterize_mask.py")
    print("  Then run : poc/train_mae.py (MAE pretraining)")
    print("=" * 60)


if __name__ == "__main__":
    run()
