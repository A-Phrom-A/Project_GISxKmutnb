"""
poc/data_prep.py
================
PoC Step 1 — Data Preparation Pipeline

อ่านข้อมูลดาวเทียมจาก data/raw/, ประมวลผลด้วย lib1_fusion,
แล้วบันทึก tile (12, 256, 256) ลงใน data/processed/

ข้อมูลที่ใช้:
  GEO-Sat  : data/raw/Gsat_z17.tif                          — 3 bands (R,G,B)  [ขนาดอ้างอิง]
  S1 (SAR) : data/raw/S1_IW_mosaic_2026044_47PQQ_VVVH_Clip.tif — 2 bands (VV,VH) [clip ขนาดพอดีกับ Gsat]
  S2 (Opt) : data/raw/S2C_MSIL2A_20260410_7band_stacked_clip.tif — 7 bands        [clip ขนาดพอดีกับ Gsat]

หมายเหตุ:
  - ใช้ไฟล์ที่มี _Clip / _clip ต่อท้ายเท่านั้น (ขนาด spatial ตรงกับ Gsat แล้ว)
  - ไฟล์ _full ขนาดใหญ่กว่า Gsat ไม่ใช้ในขั้นตอนนี้
"""

import sys
import numpy as np
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
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
except ImportError:
    print("❌ rasterio not installed. Run: pip install rasterio")
    sys.exit(1)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def validate_files():
    """ตรวจสอบว่าไฟล์ทั้งหมดมีอยู่จริง"""
    print("🔍 Validating input files...")
    all_ok = True
    for key, path in DATA_FILES.items():
        exists = path.exists()
        size_mb = path.stat().st_size / 1e6 if exists else 0
        status = "✅" if exists else "❌"
        print(f"  {status} {key.upper():5s} : {path.name}  ({size_mb:.1f} MB)")
        if not exists:
            all_ok = False
    return all_ok


def read_bands_info():
    """แสดงข้อมูล band, shape, CRS, resolution ของแต่ละไฟล์"""
    print("\n📋 File Info:")
    infos = {}
    for key, path in DATA_FILES.items():
        with rasterio.open(path) as src:
            info = {
                "path"     : path,
                "shape"    : (src.count, src.height, src.width),
                "dtype"    : src.dtypes[0],
                "crs"      : src.crs,
                "res"      : src.res,
                "nodata"   : src.nodata,
                "bands"    : src.count,
            }
            print(f"\n  [{key.upper()}] {path.name}")
            print(f"    Shape  : {info['shape']}  (bands, H, W)")
            print(f"    Dtype  : {info['dtype']}")
            print(f"    CRS    : {info['crs']}")
            print(f"    Res    : {info['res']} (x_res, y_res) in CRS units")
            print(f"    NoData : {info['nodata']}")
            infos[key] = info
    return infos


def load_and_normalize(key: str) -> np.ndarray:
    """
    โหลดไฟล์ .tif และ normalize ตาม sensor type.

    Returns:
        np.ndarray shape (bands, H, W), float32, [0.0, 1.0]
    """
    path = DATA_FILES[key]
    # Add lib1_fusion to path for normalization functions
    sys.path.insert(0, str(PROJECT_ROOT))
    from lib1_fusion.utils import normalize_sar_db, normalize_s2_reflectance, normalize_minmax

    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # (bands, H, W)
        nodata = src.nodata

    # Replace nodata values with 0
    if nodata is not None:
        arr[arr == nodata] = 0.0

    if key == "s1":
        # S1 SAR: 2 bands (VV=band1, VH=band2) — dB scale, typically negative
        vv = normalize_sar_db(arr[0], db_min=-25.0, db_max=0.0)
        vh = normalize_sar_db(arr[1], db_min=-25.0, db_max=0.0)
        return np.stack([vv, vh], axis=0)            # (2, H, W)

    elif key == "s2":
        # S2 Optical: 7 bands (B2,B3,B4,B8,B11,B12,RedEdge) — DN 0-10000
        normalized = np.stack(
            [normalize_s2_reflectance(arr[i]) for i in range(7)], axis=0
        )
        return normalized                             # (7, H, W)

    elif key == "gsat":
        # GEO-Sat: 3 bands (R,G,B) — uint8 0-255
        normalized = np.stack(
            [normalize_minmax(arr[i], vmin=0.0, vmax=255.0) for i in range(3)], axis=0
        )
        return normalized                             # (3, H, W)

    raise ValueError(f"Unknown key: {key}")


def validate_spatial_alignment(s1_arr, s2_arr, gsat_arr):
    """ตรวจสอบว่า S1_clip, S2_clip และ Gsat มี H, W ตรงกัน"""
    shapes = {
        "S1 (clip)" : s1_arr.shape[1:],
        "S2 (clip)" : s2_arr.shape[1:],
        "GEO-Sat"   : gsat_arr.shape[1:],
    }
    ref = list(shapes.values())[0]
    all_match = all(s == ref for s in shapes.values())

    print("\n📐 Spatial alignment check:")
    for name, shape in shapes.items():
        status = "✅" if shape == ref else "❌ MISMATCH"
        print(f"  {status}  {name}: H={shape[0]}, W={shape[1]}")

    if not all_match:
        raise ValueError(
            "Spatial shape mismatch! S1/S2 _clip files should match Gsat size.\n"
            "Please verify the _clip files were generated from the same extent as Gsat_z17.tif"
        )
    print(f"  ✅ All match: H={ref[0]}, W={ref[1]}")
    return ref


def cut_tiles(fused: np.ndarray, tile_size: int, stride: int) -> list:
    """
    ตัด tensor (12, H, W) เป็น tiles ย่อย (12, tile_size, tile_size)
    ด้วย sliding window.

    Args:
        fused:     np.ndarray (12, H, W) — full AOI tensor
        tile_size: ขนาด tile (px)
        stride:    step size (px) — ถ้า < tile_size จะมี overlap

    Returns:
        list of (row, col, tile) tuples
    """
    _, H, W = fused.shape
    tiles = []
    for r in range(0, H - tile_size + 1, stride):
        for c in range(0, W - tile_size + 1, stride):
            tile = fused[:, r:r+tile_size, c:c+tile_size]

            # Skip tiles with too many zeros (nodata/edge)
            valid_ratio = np.count_nonzero(tile) / tile.size
            if valid_ratio < MIN_VALID:
                continue

            tiles.append((r, c, tile))
    return tiles


def save_tiles(tiles: list, out_dir: Path, prefix: str = "tile"):
    """บันทึก tiles เป็น .npy ไฟล์"""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for row, col, tile in tiles:
        fname = out_dir / f"{prefix}_r{row:05d}_c{col:05d}.npy"
        np.save(fname, tile.astype(np.float32))
        saved += 1
    print(f"  💾 Saved {saved} tiles → {out_dir}")
    return saved


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  GISxKMUTNB — Data Preparation Pipeline")
    print("=" * 60)

    # Step 1: Validate files
    if not validate_files():
        print("\n❌ Missing files. Please upload all required satellite data.")
        return

    # Step 2: File info
    read_bands_info()

    # Step 3: Load + normalize
    print("\n📥 Loading and normalizing bands...")
    s1_arr   = load_and_normalize("s1")     # (2, H, W)
    s2_arr   = load_and_normalize("s2")     # (7, H, W)
    gsat_arr = load_and_normalize("gsat")   # (3, H, W)

    print(f"  S1   shape: {s1_arr.shape}")
    print(f"  S2   shape: {s2_arr.shape}")
    print(f"  Gsat shape: {gsat_arr.shape}")

    # Step 4: Validate spatial alignment
    validate_spatial_alignment(s1_arr, s2_arr, gsat_arr)

    # Step 5: Early Fusion → (12, H, W)
    print("\n🔗 Applying Early Fusion (stack channels)...")
    sys.path.insert(0, str(PROJECT_ROOT))
    from lib1_fusion.fusion.early_fusion import early_fusion
    fused = early_fusion(s1_arr, s2_arr, gsat_arr)   # (12, H, W)
    print(f"  Fused tensor shape : {fused.shape}")
    print(f"  Value range        : [{fused.min():.4f}, {fused.max():.4f}]")
    print(f"  Dtype              : {fused.dtype}")

    # Step 6: Tile cutting
    print(f"\n✂️  Cutting tiles ({TILE_SIZE}×{TILE_SIZE} px, stride={TILE_STRIDE})...")
    tiles = cut_tiles(fused, TILE_SIZE, TILE_STRIDE)
    print(f"  Total tiles (valid): {len(tiles)}")

    # Step 7: Save tiles
    print("\n💾 Saving processed tiles...")
    tiles_dir = PROCESSED_DIR / "tiles"
    saved = save_tiles(tiles, tiles_dir, prefix="tile")

    # Step 8: Summary
    print("\n" + "=" * 60)
    print("  ✅ Data Preparation Complete!")
    print(f"  Tiles saved : {saved}")
    print(f"  Location    : {tiles_dir}")
    print(f"  Shape/tile  : (12, {TILE_SIZE}, {TILE_SIZE})")
    print(f"  Total size  : ~{saved * 12 * TILE_SIZE * TILE_SIZE * 4 / 1e6:.1f} MB")
    print("\n  Next step: Upload tiles/ folder to Kaggle as Dataset")
    print("  Then run: notebooks/kaggle/01_mae_pretrain.py")
    print("=" * 60)


if __name__ == "__main__":
    run()
