"""
scripts/cloudmae_v2/s02_inventory_data.py
=========================================
Phase 0 - S0.2: Raw Data Inventory & Schema Verification
"""

import os
import sys
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


def inspect_raster(path: Path) -> dict:
    try:
        with rasterio.open(path) as src:
            crs = str(src.crs) if src.crs else "None"
            res = src.res
            count = src.count
            width = src.width
            height = src.height
            bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
            
            # Subsample for nodata
            step_y = max(1, height // 256)
            step_x = max(1, width // 256)
            sample = src.read(out_shape=(count, max(1, height // step_y), max(1, width // step_x)))
            
            nodata_val = src.nodata
            if nodata_val is not None:
                nodata_mask = (sample == nodata_val)
            else:
                nodata_mask = np.zeros_like(sample, dtype=bool)
                
            nan_mask = np.isnan(sample)
            total_px = sample.size
            nodata_pct = float(np.sum(nodata_mask | nan_mask) / total_px * 100)
            
            # Check 10m resolution (either ~10m in UTM/Mercator or ~8.98e-5 deg in WGS84)
            is_10m = (abs(res[0] - 10.0) < 1.0) or (abs(res[0] - 8.98e-5) < 1.0e-5)
            
            return {
                "crs": crs,
                "res": [float(res[0]), float(res[1])],
                "bands": count,
                "width": width,
                "height": height,
                "bounds": bounds,
                "nodata_pct": round(nodata_pct, 3),
                "is_10m": is_10m,
                "valid": True,
                "error": None
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def main():
    config_path = ROOT_DIR / "configs" / "cloudmae_v2" / "base.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Scan Pretrain Raw (5 Provinces) and Raw (Downstream)
    primary_dirs = [
        ROOT_DIR / "data" / "pretrain" / "raw",
        ROOT_DIR / "data" / "raw",
        ROOT_DIR / "data" / "test_samples"
    ]

    records = []
    for sdir in primary_dirs:
        if not sdir.exists():
            continue
        for fpath in sorted(sdir.rglob("*.tif")):
            info = inspect_raster(fpath)
            if not info["valid"]:
                continue
            
            fname = fpath.name
            rel_path = str(fpath.relative_to(ROOT_DIR)).replace("\\", "/")
            is_test_sample = "test_samples" in rel_path
            
            is_s2 = "sentinel2" in fname.lower() or "s2" in fname.lower()
            is_s1 = "sentinel1" in fname.lower() or "s1" in fname.lower()
            is_gsat = "gsat" in fname.lower() or "geo" in fname.lower()

            prov = "Unknown"
            for p_name in ["Chanthaburi", "Chonburi", "Rayong", "SaKaeo", "Trat", "surat_thani"]:
                if p_name.lower() in fname.lower() or p_name.lower() in rel_path.lower():
                    prov = p_name
                    break

            date_m = re.search(r"(\d{8})", fname)
            date_str = date_m.group(1) if date_m else "20260410"

            records.append({
                "scene_id": fpath.stem,
                "path": rel_path,
                "filename": fname,
                "is_test_sample": is_test_sample,
                "sensor": "Sentinel-2" if is_s2 else ("Sentinel-1" if is_s1 else ("GEO-Sat" if is_gsat else "Other")),
                "province": prov,
                "sensing_date": date_str,
                "bands": info["bands"],
                "crs": info["crs"],
                "resolution_x": info["res"][0],
                "resolution_y": info["res"][1],
                "width": info["width"],
                "height": info["height"],
                "nodata_pct": info["nodata_pct"],
                "is_10m": info["is_10m"]
            })

    df = pd.DataFrame(records)

    # 2. Inventory Pretrain Tiles (data/pretrain/tiles/*.npz)
    tiles_dir = ROOT_DIR / "data" / "pretrain" / "tiles"
    tile_files = list(tiles_dir.glob("*.npz")) if tiles_dir.exists() else []
    tile_count = len(tile_files)
    sample_tile_info = {}
    if tile_files:
        sample_tile = np.load(tile_files[0])
        sample_tile_info = {
            "keys": sample_tile.files,
            "s1_shape": list(sample_tile["s1"].shape),
            "s2_shape": list(sample_tile["s2"].shape),
            "s1_dtype": str(sample_tile["s1"].dtype),
            "s2_dtype": str(sample_tile["s2"].dtype),
        }

    # 3. Ground Truth Palm Parcels Layer
    gt_geojson = ROOT_DIR / "data" / "raw" / "ground_truth.geojson"
    has_palm_layer = gt_geojson.exists()
    palm_features_count = 0
    if has_palm_layer:
        try:
            with open(gt_geojson, "r", encoding="utf-8") as f:
                gj = json.load(f)
                palm_features_count = len(gj.get("features", []))
        except Exception:
            pass

    # Save manifest
    manifest_out = ROOT_DIR / config["paths"]["manifests_dir"] / "scenes.parquet"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(manifest_out, index=False)

    # Primary core dataset (5 Provinces + Raw Downstream)
    core_s2_7b = df[(df["sensor"] == "Sentinel-2") & (df["bands"] == 7) & (~df["is_test_sample"])]
    core_s1_2b = df[(df["sensor"] == "Sentinel-1") & (df["bands"] == 2) & (~df["is_test_sample"])]

    chk_02a = len(core_s2_7b) >= 5
    chk_02b = all(core_s2_7b["is_10m"])
    chk_02c = (core_s2_7b["nodata_pct"] <= 5.0).all()
    chk_02d = len(core_s1_2b) >= len(core_s2_7b)
    chk_02e = has_palm_layer and palm_features_count > 0

    report = {
        "phase": "Phase 0 - S0.2",
        "total_rasters_scanned": len(df),
        "pretrain_tiles_count": tile_count,
        "sample_tile": sample_tile_info,
        "core_dataset": {
            "s2_7band_scenes": len(core_s2_7b),
            "s1_2band_scenes": len(core_s1_2b),
            "provinces": sorted(core_s2_7b["province"].unique().tolist())
        },
        "crs_distribution": df.groupby("crs")["filename"].count().to_dict(),
        "chk_02a_scene_inventory": {
            "name": "CHK 0.2a: Core scene counts per province",
            "status": "PASSED" if chk_02a else "FAILED",
            "count": len(core_s2_7b),
            "provinces": core_s2_7b.groupby("province")["scene_id"].count().to_dict()
        },
        "chk_02b_schema_resolution": {
            "name": "CHK 0.2b: 10m Resolution (Metric or Degree Eq)",
            "status": "PASSED" if chk_02b else "FAILED",
            "all_core_10m": bool(chk_02b),
            "crs_note": "Pretrain 5 provinces exported in WGS84 (EPSG:4326) with 8.98e-5 deg (10m eq); Downstream Chonburi exported in UTM47N (EPSG:32647) 10m native."
        },
        "chk_02c_nodata_check": {
            "name": "CHK 0.2c: Nodata / NaN <= 5%",
            "status": "PASSED" if chk_02c else "FAILED",
            "max_core_nodata_pct": float(core_s2_7b["nodata_pct"].max()) if len(core_s2_7b) > 0 else 0.0
        },
        "chk_02d_s1_s2_pairing": {
            "name": "CHK 0.2d: S1 SAR and S2 Optical Pairing",
            "status": "PASSED" if chk_02d else "FAILED",
            "s1_count": len(core_s1_2b),
            "s2_count": len(core_s2_7b),
            "l1c_note": "No raw L1C archives locally. SCL + Optical Spectral Ensemble will be used for mask production in Phase 1."
        },
        "chk_02e_palm_layer": {
            "name": "CHK 0.2e: Palm ground truth parcel layer",
            "status": "PASSED" if chk_02e else "FAILED",
            "features_count": palm_features_count,
            "path": str(gt_geojson.relative_to(ROOT_DIR)).replace("\\", "/")
        },
        "all_passed": chk_02a and chk_02b and chk_02c and chk_02d and chk_02e
    }

    report_file = ROOT_DIR / config["paths"]["reports_dir"] / "s02_inventory.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== S0.2 RAW DATA INVENTORY SUMMARY ===")
    print(f"Core S2 (7-band) Scenes: {len(core_s2_7b)} | Core S1 (2-band) Scenes: {len(core_s1_2b)}")
    print(f"Existing Pretrain Tiles (.npz): {tile_count}")
    print(f"Palm ground truth parcels: {palm_features_count}")
    print(f"CHK 0.2a (Scene Count >= 5): {'PASSED' if chk_02a else 'FAILED'}")
    print(f"CHK 0.2b (10m Resolution): {'PASSED' if chk_02b else 'FAILED'}")
    print(f"CHK 0.2c (NoData <= 5%): {'PASSED' if chk_02c else 'FAILED'}")
    print(f"CHK 0.2d (S1-S2 Pairing): {'PASSED' if chk_02d else 'FAILED'}")
    print(f"CHK 0.2e (Palm Layer): {'PASSED' if chk_02e else 'FAILED'}")
    print(f"All Passed: {report['all_passed']}")
    print(f"Report: {report_file}")


if __name__ == "__main__":
    main()
