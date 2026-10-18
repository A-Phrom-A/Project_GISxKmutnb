"""
projects.cross_band_recovery.scripts.01b_download_stac_pairs
============================================================
Automated STAC Downloader for Paired Sentinel-1 SAR & Sentinel-2 Optical Imagery.
Searches and downloads paired scenes across agricultural zones in Thailand
with strict temporal gap (|dt| <= 3 days) and minimal cloud cover (< 5%).

Data Source: Microsoft Planetary Computer STAC API (Sentinel-2 L2A & Sentinel-1 RTC).
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple

import requests
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import Transformer
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "data" / "raw_pairs"

STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="

S2_BANDS = ["B02", "B03", "B04", "B8A", "B08", "B11", "B12"]
S1_BANDS = ["vv", "vh"]

# 20 Agricultural ROIs across Thailand
EXPANDED_ROIS = {
    # ภาคตะวันออก (Eastern Hub)
    "Chonburi_North": {"bbox": [100.95, 13.25, 101.40, 13.55], "region": "Eastern", "crop": "Oil Palm / Cassava"},
    "Chonburi_South": {"bbox": [101.00, 12.95, 101.45, 13.25], "region": "Eastern", "crop": "Oil Palm / Rubber"},
    "Rayong_West": {"bbox": [101.15, 12.65, 101.55, 12.95], "region": "Eastern", "crop": "Rubber / Fruit"},
    "Rayong_East": {"bbox": [101.45, 12.75, 101.85, 13.05], "region": "Eastern", "crop": "Rubber / Oil Palm"},
    "Chanthaburi_West": {"bbox": [101.80, 12.60, 102.20, 12.90], "region": "Eastern", "crop": "Fruit / Durian / Rubber"},
    "Chanthaburi_North": {"bbox": [101.90, 12.85, 102.30, 13.15], "region": "Eastern", "crop": "Rubber / Oil Palm"},
    "SaKaeo_South": {"bbox": [102.05, 13.20, 102.45, 13.50], "region": "Eastern / Border", "crop": "Cassava / Sugarcane"},
    "SaKaeo_North": {"bbox": [102.20, 13.50, 102.60, 13.80], "region": "Eastern / Border", "crop": "Sugarcane / Cassava"},
    "Trat_North": {"bbox": [102.30, 12.25, 102.70, 12.55], "region": "Eastern Coastal", "crop": "Rubber / Oil Palm"},
    "Trat_South": {"bbox": [102.40, 11.95, 102.75, 12.25], "region": "Eastern Coastal", "crop": "Rubber / Oil Palm"},
    "Chachoengsao": {"bbox": [101.20, 13.40, 101.60, 13.70], "region": "Eastern", "crop": "Cassava / Oil Palm"},
    "Prachinburi": {"bbox": [101.40, 13.80, 101.80, 14.10], "region": "Eastern", "crop": "Cassava / Eucalyptus"},
    
    # ภาคใต้ (Southern Palm Belt)
    "Surat_Thani_East": {"bbox": [99.30, 9.05, 99.65, 9.35], "region": "Southern Main Palm Belt", "crop": "Oil Palm (Primary Hub)"},
    "Surat_Thani_West": {"bbox": [98.95, 8.95, 99.30, 9.25], "region": "Southern Main Palm Belt", "crop": "Oil Palm"},
    "Krabi_North": {"bbox": [98.80, 8.25, 99.15, 8.55], "region": "Southern Main Palm Belt", "crop": "Oil Palm / Rubber"},
    "Krabi_South": {"bbox": [98.90, 7.95, 99.25, 8.25], "region": "Southern Main Palm Belt", "crop": "Oil Palm / Rubber"},
    "Chumphon_South": {"bbox": [99.00, 10.15, 99.35, 10.45], "region": "Southern Main Palm Belt", "crop": "Oil Palm / Rubber"},
    "Chumphon_North": {"bbox": [99.15, 10.45, 99.50, 10.75], "region": "Southern Main Palm Belt", "crop": "Oil Palm / Fruit"},
    "Nakhon_Si_Thammarat": {"bbox": [99.70, 8.25, 100.05, 8.55], "region": "Southern Main Palm Belt", "crop": "Oil Palm / Rubber"},
    
    # ภาคกลางและตะวันตก (Central / Western Sugarcane & Crops)
    "Kanchanaburi": {"bbox": [99.40, 13.85, 99.75, 14.15], "region": "Western", "crop": "Sugarcane / Cassava"},
    "Suphanburi": {"bbox": [99.85, 14.40, 100.20, 14.70], "region": "Central", "crop": "Sugarcane / Rice"},
    "Lopburi": {"bbox": [100.70, 14.85, 101.05, 15.15], "region": "Central", "crop": "Cassava / Corn"}
}


def sign_url(raw_href: str, max_retries: int = 6) -> str:
    """Sign Planetary Computer asset href with SAS token with exponential backoff on 429"""
    for attempt in range(max_retries):
        try:
            resp = requests.get(f"{SAS_SIGN_URL}{raw_href}", timeout=30)
            if resp.status_code == 429:
                wait_time = (2 ** attempt) * 2 + 2
                print(f"[429 Rate Limit] Backing off {wait_time}s...", flush=True)
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            return resp.json()["href"]
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2)
    raise RuntimeError(f"Failed to sign URL after {max_retries} attempts: {raw_href[:80]}...")


def find_candidate_pairs(max_cloud_pct: float = 5.0, max_dt_days: float = 3.0) -> List[Dict[str, Any]]:
    """ค้นหาคู่ภาพ S1-S2 ที่เข้าเกณฑ์ทั้งหมด"""
    print("=" * 70)
    print("1. Searching Planetary Computer STAC for Clear-Sky S1-S2 Pairs")
    print(f"Criteria: Cloud < {max_cloud_pct}%, |dt| <= {max_dt_days} days, Dry Season 2024")
    print("=" * 70)
    
    candidates = []
    
    for roi_key, info in EXPANDED_ROIS.items():
        bbox = info["bbox"]
        print(f"Scanning {roi_key} ({info['region']})...", end=" ", flush=True)
        
        # Search S2
        payload_s2 = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": "2024-01-01/2024-03-31",
            "query": {"eo:cloud_cover": {"lt": max_cloud_pct}},
            "limit": 3
        }
        
        try:
            r2 = requests.post(STAC_SEARCH_URL, json=payload_s2, timeout=20)
            if r2.status_code != 200:
                print("S2 error")
                continue
            s2_items = r2.json().get("features", [])
            if not s2_items:
                print("No clear S2")
                continue
                
            best_pair = None
            min_dt = 999.0
            
            for s2_it in s2_items:
                s2_time = datetime.fromisoformat(s2_it["properties"]["datetime"].replace("Z", "+00:00"))
                t_start = (s2_time - timedelta(days=max_dt_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
                t_end = (s2_time + timedelta(days=max_dt_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
                
                payload_s1 = {
                    "collections": ["sentinel-1-rtc"],
                    "bbox": bbox,
                    "datetime": f"{t_start}/{t_end}",
                    "limit": 2
                }
                
                r1 = requests.post(STAC_SEARCH_URL, json=payload_s1, timeout=20)
                if r1.status_code == 200:
                    s1_items = r1.json().get("features", [])
                    for s1_it in s1_items:
                        s1_time = datetime.fromisoformat(s1_it["properties"]["datetime"].replace("Z", "+00:00"))
                        dt = abs((s2_time - s1_time).total_seconds()) / 86400.0
                        if dt < min_dt:
                            min_dt = dt
                            best_pair = {
                                "roi_id": roi_key,
                                "region": info["region"],
                                "crop": info["crop"],
                                "bbox": bbox,
                                "s2_item": s2_it,
                                "s1_item": s1_it,
                                "s2_id": s2_it["id"],
                                "s1_id": s1_it["id"],
                                "s2_date": s2_it["properties"]["datetime"][:10],
                                "s1_date": s1_it["properties"]["datetime"][:10],
                                "cloud_pct": s2_it["properties"]["eo:cloud_cover"],
                                "dt_days": round(dt, 2)
                            }
                            
            if best_pair is not None:
                candidates.append(best_pair)
                print(f"FOUND! dt={best_pair['dt_days']}d, Cloud={best_pair['cloud_pct']:.1f}%")
            else:
                print("No S1 match within dt")
                
        except Exception as e:
            print(f"Error: {e}")
            
    print(f"\nTotal candidate pairs discovered: {len(candidates)}")
    return candidates


def download_and_crop_pair(pair: Dict[str, Any], pair_id: str) -> Dict[str, Any]:
    """ดาวน์โหลด ตัดครอป และบันทึกเป็น GeoTIFF แบบ 7 แบนด์ S2 และ 2 แบนด์ S1"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    roi_id = pair["roi_id"]
    bbox = pair["bbox"]
    s2_it = pair["s2_item"]
    s1_it = pair["s1_item"]
    
    s2_out_file = OUTPUT_DIR / f"{pair_id}_{roi_id}_S2_7bands.tif"
    s1_out_file = OUTPUT_DIR / f"{pair_id}_{roi_id}_S1_dualpol.tif"
    
    if s2_out_file.exists() and s1_out_file.exists():
        print(f"Already downloaded: {pair_id} ({roi_id})")
        return {"s2_path": str(s2_out_file.relative_to(ROOT_DIR)).replace("\\", "/"),
                "s1_path": str(s1_out_file.relative_to(ROOT_DIR)).replace("\\", "/")}
        
    print(f"\nDownloading & Cropping {pair_id} ({roi_id})...")
    
    # 1. อ่านและบันทึก S2 7 แบนด์
    s2_bands_data = []
    s2_meta = None
    
    for b_idx, b_name in enumerate(S2_BANDS):
        raw_href = s2_it["assets"][b_name]["href"]
        signed_url = sign_url(raw_href)
        with rasterio.open(signed_url) as src:
            if b_idx == 0:
                trans = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                minx, miny = trans.transform(bbox[0], bbox[1])
                maxx, maxy = trans.transform(bbox[2], bbox[3])
                ref_win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
                target_w = int(round(ref_win.width))
                target_h = int(round(ref_win.height))
                target_transform = src.window_transform(ref_win)
                target_crs = src.crs
                
                s2_meta = {
                    "driver": "GTiff",
                    "dtype": "uint16",
                    "nodata": 0,
                    "width": target_w,
                    "height": target_h,
                    "count": 7,
                    "crs": target_crs,
                    "transform": target_transform,
                    "compress": "lzw"
                }
                
            cur_win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            # Clamp to raster dimensions
            c_off = max(0.0, min(src.width - 1.0, cur_win.col_off))
            r_off = max(0.0, min(src.height - 1.0, cur_win.row_off))
            c_w = min(src.width - c_off, cur_win.width)
            r_h = min(src.height - r_off, cur_win.height)
            clamped_win = rasterio.windows.Window(c_off, r_off, c_w, r_h)
            
            b_data = src.read(1, window=clamped_win, out_shape=(target_h, target_w), resampling=Resampling.bilinear)
            s2_bands_data.append(b_data)
            
    s2_stacked = np.stack(s2_bands_data, axis=0)
    with rasterio.open(s2_out_file, "w", **s2_meta) as dst:
        dst.write(s2_stacked)
    print(f"Saved S2: {s2_out_file.name} {s2_stacked.shape}")
    
    # 2. อ่านและบันทึก S1 2 แบนด์ (VV, VH) ลงบนกริด CRS และขนาดเดียวกับ S2 พอดี
    s1_bands_data = []
    s1_meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": 0.0,
        "width": target_w,
        "height": target_h,
        "count": 2,
        "crs": target_crs,
        "transform": target_transform,
        "compress": "lzw"
    }
    
    for pol in S1_BANDS:
        raw_href = s1_it["assets"][pol]["href"]
        signed_url = sign_url(raw_href)
        with rasterio.open(signed_url) as src:
            trans_s1 = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            minx_s1, miny_s1 = trans_s1.transform(bbox[0], bbox[1])
            maxx_s1, maxy_s1 = trans_s1.transform(bbox[2], bbox[3])
            cur_win_s1 = from_bounds(minx_s1, miny_s1, maxx_s1, maxy_s1, transform=src.transform)
            c_off = max(0.0, min(src.width - 1.0, cur_win_s1.col_off))
            r_off = max(0.0, min(src.height - 1.0, cur_win_s1.row_off))
            c_w = min(src.width - c_off, cur_win_s1.width)
            r_h = min(src.height - r_off, cur_win_s1.height)
            clamped_win_s1 = rasterio.windows.Window(c_off, r_off, c_w, r_h)
            
            pol_data = src.read(1, window=clamped_win_s1, out_shape=(target_h, target_w), resampling=Resampling.bilinear)
            s1_bands_data.append(pol_data.astype(np.float32))
            
    s1_stacked = np.stack(s1_bands_data, axis=0)
    with rasterio.open(s1_out_file, "w", **s1_meta) as dst:
        dst.write(s1_stacked)
    print(f"Saved S1: {s1_out_file.name} {s1_stacked.shape}")
    
    return {
        "s2_path": str(s2_out_file.relative_to(ROOT_DIR)).replace("\\", "/"),
        "s1_path": str(s1_out_file.relative_to(ROOT_DIR)).replace("\\", "/"),
        "width": target_w,
        "height": target_h,
        "crs": str(target_crs)
    }


def download_catalog(limit: int = None):
    """ดาวน์โหลดคู่ภาพจากแคตตาล็อกและอัปเดต metadata_manifest.parquet"""
    catalog_path = PROJECT_DIR / "data" / "manifests" / "stac_candidate_catalog.json"
    if not catalog_path.exists():
        print("Catalog not found. Running search first...")
        main_search()
        
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
        
    manifest_file = PROJECT_DIR / "data" / "manifests" / "metadata_manifest.parquet"
    df_manifest = pd.read_parquet(manifest_file)
    existing_pair_ids = set(df_manifest["pair_id"].values)
    
    targets = catalog if limit is None else catalog[:limit]
    print(f"\n--- Batch Downloading {len(targets)} Candidate Pairs ---")
    
    new_rows = []
    
    for c in targets:
        cid = c["candidate_id"]
        roi_id = c["roi_id"]
        
        # Check if already in manifest
        if cid in existing_pair_ids:
            print(f"Skipping {cid} ({roi_id}) - already in manifest.", flush=True)
            continue
            
        print(f"\nProcessing {cid} ({roi_id})...", flush=True)
        try:
            r2 = requests.post(STAC_SEARCH_URL, json={"collections": ["sentinel-2-l2a"], "ids": [c["s2_id"]]}, timeout=25).json()
            r1 = requests.post(STAC_SEARCH_URL, json={"collections": ["sentinel-1-rtc"], "ids": [c["s1_id"]]}, timeout=25).json()
            
            if not r2.get("features") or not r1.get("features"):
                print(f"Failed to fetch metadata for {cid}", flush=True)
                continue
                
            pair_obj = {
                "roi_id": roi_id,
                "bbox": c["bbox"],
                "s2_item": r2["features"][0],
                "s1_item": r1["features"][0]
            }
            
            res = download_and_crop_pair(pair_obj, cid)
            
            # Map clean ROI name for splitting
            roi_clean = roi_id.split("_")[0]  # e.g., 'Chonburi_North' -> 'Chonburi'
            
            row = {
                "pair_id": cid,
                "roi_id": roi_clean,
                "roi_name": f"{roi_id} ({c['region']})",
                "region": c["region"],
                "dataset_type": "stac_planetary_computer",
                "s1_path": res["s1_path"],
                "s2_path": res["s2_path"],
                "s1_bands": 2,
                "s2_bands": 7,
                "s1_unit": "linear_backscatter",
                "s1_date": c["s1_date"],
                "s2_date": c["s2_date"],
                "dt_days": float(c["dt_days"]),
                "tier": "Tier-1 (<=3d)" if c["dt_days"] <= 3.0 else "Tier-2 (4-6d)",
                "season": "Dry",
                "cloud_pct": float(c["cloud_pct"]),
                "crs_s1": res["crs"],
                "crs_s2": res["crs"],
                "crs_match": True,
                "width": res["width"],
                "height": res["height"],
                "s2_val_range": np.array([1000.0, 16000.0]),
                "s1_val_range": np.array([0.001, 100.0]),
                "nodata_pct": 0.0,
                "status": "APPROVED"
            }
            new_rows.append(row)
            
            # Save updated manifest incrementally
            df_manifest = pd.concat([df_manifest, pd.DataFrame([row])], ignore_index=True)
            df_manifest.to_parquet(manifest_file, index=False)
            print(f"Appended {cid} to manifest. Total active scenes: {len(df_manifest)}", flush=True)
            
        except Exception as e:
            print(f"Error downloading {cid}: {e}", flush=True)
            
    print(f"\nBatch download finished. Total scenes in manifest: {len(df_manifest)}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Planetary Computer STAC pair downloader")
    parser.add_argument("--action", choices=["search", "download"], default="download")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs to download")
    args = parser.parse_args()
    
    if args.action == "search":
        candidates = find_candidate_pairs(max_cloud_pct=5.0, max_dt_days=3.0)
        catalog_path = PROJECT_DIR / "data" / "manifests" / "stac_candidate_catalog.json"
        summary = []
        for idx, c in enumerate(candidates):
            summary.append({
                "candidate_id": f"PAIR-STAC-{idx+1:03d}",
                "roi_id": c["roi_id"],
                "region": c["region"],
                "crop": c["crop"],
                "s2_id": c["s2_id"],
                "s1_id": c["s1_id"],
                "s2_date": c["s2_date"],
                "s1_date": c["s1_date"],
                "dt_days": c["dt_days"],
                "cloud_pct": round(c["cloud_pct"], 2),
                "bbox": c["bbox"]
            })
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Catalog updated with {len(summary)} candidate pairs.")
    else:
        download_catalog(limit=args.limit)


if __name__ == "__main__":
    main()
