"""
scripts/download_sentinel_by_cloud_level.py
============================================
ดาวน์โหลดภาพถ่ายดาวเทียม Sentinel-2 L2A จริง (7 แบนด์) ตามระดับปริมาณเมฆ:
  - ระดับ 1: เมฆต่ำกว่า 10% (< 10%)  [ท้องฟ้าโปร่ง / จุดอ้างอิงความถูกต้อง Ground Truth]
  - ระดับ 2: เมฆต่ำกว่า 25% (10-25%) [เมฆบางส่วน / จุดที่โมเดล Cloud-MAE ซ่อมแซมได้สวยงามที่สุด]
  - ระดับ 3: เมฆต่ำกว่า 50% (25-50%) [เมฆปานกลาง / ทดสอบความทนทานในสภาพท้าทาย]

ใช้งานง่าย:
  python scripts/download_sentinel_by_cloud_level.py --level 25
  python scripts/download_sentinel_by_cloud_level.py --level 10
  python scripts/download_sentinel_by_cloud_level.py --level 50
  python scripts/download_sentinel_by_cloud_level.py --level all
"""

import argparse
import sys
import shutil
from pathlib import Path
import requests
import rasterio
from rasterio.windows import from_bounds
import numpy as np
import cv2

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ขอบเขตพื้นที่ Zone 1: อ.บ้านบึง จ.ชลบุรี (UTM Zone 47N, EPSG:32647)
TARGET_BOUNDS = (756700.0, 1448400.0, 761920.0, 1453620.0)
TARGET_CRS = "EPSG:32647"
TARGET_W, TARGET_H = 522, 522
BBOX_4326 = [101.3674, 13.0905, 101.4160, 13.1382]

BAND_NAMES = ["B02", "B03", "B04", "B8A", "B08", "B11", "B12"]

LEVEL_CONFIGS = {
    "10": {
        "title": "เมฆต่ำกว่า 10% (< 10%)",
        "date_range": "2024-01-01/2024-03-31",
        "query": {"eo:cloud_cover": {"lt": 10}},
        "out_filename": "s2_cloud_under_10pct.tif",
    },
    "25": {
        "title": "เมฆต่ำกว่า 25% (10% - 25% แนะนำที่สุด)",
        "date_range": "2024-01-01/2024-03-31",
        "query": {"eo:cloud_cover": {"lt": 25, "gte": 10}},
        "out_filename": "s2_cloud_under_25pct.tif",
    },
    "50": {
        "title": "เมฆต่ำกว่า 50% (25% - 50%)",
        "date_range": "2023-09-01/2023-11-30",
        "query": {"eo:cloud_cover": {"lt": 50, "gte": 25}},
        "out_filename": "s2_cloud_under_50pct.tif",
    },
}


def download_level(level_key: str, out_dir: Path):
    cfg = LEVEL_CONFIGS[level_key]
    print(f"\n=======================================================")
    print(f"🛰️ เริ่มค้นหาและดาวน์โหลด: {cfg['title']}")
    print(f"=======================================================")

    # 1. ค้นหาภาพผ่าน Planetary Computer STAC API
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    search_payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": BBOX_4326,
        "datetime": cfg["date_range"],
        "query": cfg["query"],
        "limit": 3,
    }

    resp = requests.post(stac_url, json=search_payload, timeout=30)
    if resp.status_code != 200:
        print(f"❌ ค้นหาล้มเหลว: {resp.text}")
        return

    features = resp.json().get("features", [])
    if not features:
        print(f"⚠️ ไม่พบภาพในช่วงเงื่อนไข: {cfg['date_range']}")
        return

    item = features[0]
    scene_id = item["id"]
    cloud_pct = item["properties"].get("eo:cloud_cover", 0.0)
    acq_date = item["properties"].get("datetime", "")
    print(f"✅ พบภาพที่ตรงเกณฑ์: {scene_id}")
    print(f"   📅 วันที่บันทึกภาพ: {acq_date}")
    print(f"   ☁️ ปริมาณเมฆทั้งซีน: {cloud_pct:.2f}%")

    BAND_NAMES_FETCH = list(BAND_NAMES)
    has_scl = "SCL" in item["assets"]
    total_bands = 8 if has_scl else 7
    stacked = np.zeros((total_bands, TARGET_H, TARGET_W), dtype=np.float32)

    for i, b_name in enumerate(BAND_NAMES):
        if b_name not in item["assets"]:
            print(f"   ⚠️ ไม่พบแบนด์ {b_name}")
            continue

        raw_href = item["assets"][b_name]["href"]
        success = False
        import time

        for attempt in range(4):
            try:
                sign_resp = requests.get(
                    f"https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={raw_href}", timeout=25
                )
                if sign_resp.status_code == 200:
                    signed_url = sign_resp.json()["href"]
                    with rasterio.open(signed_url) as src:
                        win = from_bounds(*TARGET_BOUNDS, transform=src.transform)
                        b_data = src.read(1, window=win).astype(np.float32)
                        if b_data.shape != (TARGET_H, TARGET_W):
                            b_data = cv2.resize(b_data, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
                        stacked[i] = b_data
                        success = True
                        break
            except Exception as e:
                time.sleep(1.5)

        if success:
            print(f"   ✓ โหลดสำเร็จ แบนด์ {b_name}: Min={stacked[i].min():.0f}, Max={stacked[i].max():.0f}")
        else:
            print(f"   ❌ ล้มเหลวสำหรับแบนด์ {b_name}")

    # ปรับจูนค่า Offset ESA Baseline 04.00+ (+1000) หากพบค่าเกิน 1000 ทั้งแผ่น (เฉพาะ 7 แบนด์ออปติคอล)
    if np.min(stacked[0]) >= 1000.0:
        stacked[:7] = np.clip(stacked[:7] - 1000.0, 0.0, 10000.0)
        print("   ✓ ปรับแก้ค่าชดเชยมาตรฐาน ESA Baseline 04.00+ (-1000 DN) สำเร็จ")

    # ดาวน์โหลด SCL (Scene Classification Layer) แบนด์ที่ 8
    if has_scl:
        raw_href = item["assets"]["SCL"]["href"]
        for attempt in range(4):
            try:
                sign_resp = requests.get(
                    f"https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={raw_href}", timeout=25
                )
                if sign_resp.status_code == 200:
                    signed_url = sign_resp.json()["href"]
                    with rasterio.open(signed_url) as src:
                        win = from_bounds(*TARGET_BOUNDS, transform=src.transform)
                        scl_data = src.read(1, window=win).astype(np.float32)
                        if scl_data.shape != (TARGET_H, TARGET_W):
                            scl_data = cv2.resize(scl_data, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
                        stacked[7] = scl_data
                        print(f"   🛰️ โหลดสำเร็จ แบนด์ 8: Official ESA SCL (Scene Classification Layer)")
                        break
            except Exception:
                import time
                time.sleep(1.5)

    # บันทึกไฟล์ GeoTIFF มาตรฐาน (8 แบนด์ถ้ามี SCL, 7 แบนด์ถ้าไม่มี)
    out_path = out_dir / cfg["out_filename"]
    ref_file = PROJECT_ROOT / "data" / "test_samples" / "02_cloudrepair_real_zone1" / "s2_optical_cloudy_7bands.tif"

    if ref_file.exists():
        with rasterio.open(ref_file) as ref_src:
            meta = ref_src.meta.copy()
            meta.update({"count": total_bands, "dtype": "float32"})
    else:
        from rasterio.transform import from_bounds as make_transform
        transform = make_transform(*TARGET_BOUNDS, TARGET_W, TARGET_H)
        meta = {
            "driver": "GTiff",
            "count": total_bands,
            "dtype": "float32",
            "width": TARGET_W,
            "height": TARGET_H,
            "crs": TARGET_CRS,
            "transform": transform,
        }

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(stacked)

    print(f"🎉 บันทึกภาพมัลติสเปกตรัมสำเร็จ: {out_path.name} ({total_bands} แบนด์)")
    print(f"   📁 บันทึกที่: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="ดาวน์โหลดภาพดาวเทียม Sentinel-2 ตามระดับปริมาณเมฆ")
    parser.add_argument(
        "--level",
        type=str,
        default="all",
        choices=["10", "25", "50", "all"],
        help="เลือกระดับปริมาณเมฆที่ต้องการ (10, 25, 50, all)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/test_samples/02_cloudrepair_levels",
        help="โฟลเดอร์สำหรับจัดเก็บไฟล์ผลลัพธ์",
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # คัดลอกภาพเรดาร์ S1 คู่กันมาไว้ในโฟลเดอร์เดียวกันเพื่อให้พร้อมนำเข้า Web App ทันที
    s1_source = PROJECT_ROOT / "data" / "test_samples" / "02_cloudrepair_real_zone1" / "s1_sar_radar_dualpol.tif"
    s1_dest = out_dir / "s1_sar_radar_dualpol.tif"
    if s1_source.exists() and not s1_dest.exists():
        shutil.copy2(s1_source, s1_dest)
        print(f"✓ คัดลอกเรดาร์ Sentinel-1 SAR คู่กันเตรียมไว้แล้ว: {s1_dest.name}")

    if args.level == "all":
        for lvl in ["10", "25", "50"]:
            download_level(lvl, out_dir)
    else:
        download_level(args.level, out_dir)

    print("\n=======================================================")
    print("✅ เสร็จสิ้นกระบวนการทั้งหมด!")
    print(f"ไฟล์ทั้งหมดพร้อมนำเข้าสู่หน้าต่าง GeoAI Import Studio แล้วใน:")
    print(f"👉 {out_dir}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
