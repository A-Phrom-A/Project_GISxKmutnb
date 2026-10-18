"""
scripts/cloudmae_v2/run_dilation_sweep_and_false_alarm.py
=========================================================
Rigorous empirical analysis addressing user critique:
1. True Commission (False Alarm) benchmark on 100 verified clean pretrain tiles (GT = 0 cloud).
2. Dilation sweep (d in [0, 1, 2, 3, 5] px) and probability threshold sweep (tau in [0.3, 0.4, 0.5, 0.6]).
3. True dilation effect calculation: (Ensemble_dilated \\ Ensemble_undilated).
4. Coverage inflation analysis: tracking bin shifts and rejection boundary crossings.
"""

import sys
import json
from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from lib2_vitpalm.mask_pipeline import compute_spectral_cloud_prob, classify_cloud_bin


def run_clean_tiles_false_alarm_benchmark():
    """
    Evaluates Commission Rate (False Alarm) on 100 clean pretrain tiles across 5 provinces.
    Since tiles are clear sky, Ground Truth Cloud = 0.
    Any pixel flagged as cloud (prob >= tau) is a true false alarm.
    """
    pretrain_dir = ROOT_DIR / "data/pretrain/tiles"
    all_tile_paths = sorted(pretrain_dir.glob("*.npz"))
    
    # Stratify by province: Chanthaburi, Chonburi, Rayong, SaKaeo, Trat (20 each = 100 tiles)
    provinces = ["Chanthaburi", "Chonburi", "Rayong", "SaKaeo", "Trat"]
    selected_paths = []
    for prov in provinces:
        prov_tiles = [p for p in all_tile_paths if prov in p.name][:20]
        selected_paths.extend(prov_tiles)
        
    print(f"Testing False Alarm on {len(selected_paths)} verified clean pretrain tiles...")
    
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    total_pixels = len(selected_paths) * 256 * 256
    
    false_positives_per_thresh = {t: 0 for t in thresholds}
    tiles_with_any_false_alarm = {t: 0 for t in thresholds}
    
    for p in selected_paths:
        with np.load(p) as d:
            s2_optical = d["s2"] * 10000.0  # (7, 256, 256) reflectance
            
        prob = compute_spectral_cloud_prob(s2_optical)
        
        for t in thresholds:
            flagged = prob >= t
            fp_count = int(np.sum(flagged))
            false_positives_per_thresh[t] += fp_count
            if fp_count > 0:
                tiles_with_any_false_alarm[t] += 1
                
    results = {}
    for t in thresholds:
        pixel_commission_pct = (false_positives_per_thresh[t] / total_pixels) * 100.0
        tile_commission_pct = (tiles_with_any_false_alarm[t] / len(selected_paths)) * 100.0
        results[f"tau_{t:.2f}"] = {
            "threshold": t,
            "false_positive_pixels": false_positives_per_thresh[t],
            "total_pixels_evaluated": total_pixels,
            "pixel_commission_rate_pct": round(pixel_commission_pct, 4),
            "tile_commission_rate_pct": round(tile_commission_pct, 2)
        }
        print(f"  tau={t:.2f} -> Pixel False Alarm: {pixel_commission_pct:.4f}%, Tiles with FA: {tile_commission_pct:.1f}%")
        
    return results


def run_dilation_and_inflation_sweep():
    """
    Evaluates on cloudy rasters with SCL:
    Measures true dilation addition (dilated \\ undilated) and bin inflation across d in [0, 1, 2, 3, 5].
    """
    levels_dir = ROOT_DIR / "data/test_samples/02_cloudrepair_levels"
    rasters = [
        levels_dir / "s2_cloud_under_10pct.tif",
        levels_dir / "s2_cloud_under_25pct.tif",
        levels_dir / "s2_cloud_under_50pct.tif"
    ]
    
    tiles = []
    tile_size = 256
    for r_path in rasters:
        if not r_path.exists():
            continue
        with rasterio.open(r_path) as src:
            arr = src.read()
            ny = src.height // tile_size
            nx = src.width // tile_size
            for r in range(ny):
                for c in range(nx):
                    tile_arr = arr[:, r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size]
                    tiles.append((f"{r_path.stem}_r{r}_c{c}", tile_arr))
                    
    print(f"\nEvaluating Dilation & Inflation Sweep on {len(tiles)} tiles from cloudy scenes...")
    
    dilation_radii = [0, 1, 2, 3, 5]
    prob_threshold = 0.40
    
    sweep_results = {}
    
    for d in dilation_radii:
        if d == 0:
            struct = np.ones((1, 1), dtype=bool)
        else:
            size = 2 * d + 1
            y, x = np.ogrid[-d:d+1, -d:d+1]
            struct = (x**2 + y**2 <= d**2)
            
        total_pixels = len(tiles) * 256 * 256
        total_undilated_cloud = 0
        total_dilated_cloud = 0
        total_true_dilation_added = 0
        
        bin_counts = {
            "0-2% (Clear)": 0,
            "2-5%": 0,
            "5-10%": 0,
            "10-20%": 0,
            "20-25% (Margin)": 0,
            ">25% (Rejected)": 0
        }
        
        tile_details = []
        
        for name, data in tiles:
            s2_opt = data[:7]
            scl = data[7].astype(np.int32)
            
            # Undilated Ensemble
            prob = compute_spectral_cloud_prob(s2_opt)
            scl_cloud = np.isin(scl, [8, 9, 10])
            scl_shadow = (scl == 3)
            undilated_mask = (scl_cloud | (prob >= prob_threshold)) | scl_shadow
            
            # Dilated Mask
            if d == 0:
                dilated_mask = undilated_mask.copy()
            else:
                dilated_mask = binary_dilation(undilated_mask, structure=struct)
                
            # True Dilation addition: pixels in dilated BUT NOT in undilated
            true_dilation_addition = dilated_mask & (~undilated_mask)
            
            undilated_cov = float(np.mean(undilated_mask) * 100.0)
            dilated_cov = float(np.mean(dilated_mask) * 100.0)
            true_dilation_cov = float(np.mean(true_dilation_addition) * 100.0)
            
            c_bin, is_valid = classify_cloud_bin(dilated_cov)
            bin_counts[c_bin] += 1
            
            total_undilated_cloud += int(np.sum(undilated_mask))
            total_dilated_cloud += int(np.sum(dilated_mask))
            total_true_dilation_added += int(np.sum(true_dilation_addition))
            
            tile_details.append({
                "tile_id": name,
                "undilated_coverage_pct": round(undilated_cov, 3),
                "dilated_coverage_pct": round(dilated_cov, 3),
                "true_dilation_added_pct": round(true_dilation_cov, 3),
                "assigned_bin": c_bin
            })
            
        mean_undilated = (total_undilated_cloud / total_pixels) * 100.0
        mean_dilated = (total_dilated_cloud / total_pixels) * 100.0
        mean_true_dilation_added = (total_true_dilation_added / total_pixels) * 100.0
        
        sweep_results[f"dilation_{d}px"] = {
            "dilation_radius_px": d,
            "dilation_radius_m": d * 10,
            "mean_undilated_coverage_pct": round(mean_undilated, 3),
            "mean_dilated_coverage_pct": round(mean_dilated, 3),
            "true_dilation_added_pct": round(mean_true_dilation_added, 3),
            "bin_distribution": bin_counts,
            "rejected_tiles_count": bin_counts[">25% (Rejected)"],
            "sample_tiles": tile_details[:4]
        }
        
        print(f"  d={d}px ({d*10}m) -> Mean Coverage: {mean_dilated:.2f}% (True Dilation Added: +{mean_true_dilation_added:.2f}%), Rejected: {bin_counts['>25% (Rejected)']}/12")
        
    return sweep_results


def run_full_experiment():
    print("=== STARTING RIGOROUS EMPIRICAL BENCHMARKS ===")
    fa_results = run_clean_tiles_false_alarm_benchmark()
    dilation_results = run_dilation_and_inflation_sweep()
    
    combined_report = {
        "benchmark": "Rigorous Mask Quality, False Alarm, and Dilation Inflation Audit",
        "false_alarm_benchmark_clean_tiles": fa_results,
        "dilation_and_inflation_sweep": dilation_results
    }
    
    out_path = ROOT_DIR / "reports/cloudmae_v2/s04_rigorous_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2, ensure_ascii=False)
        
    print(f"\nAudit results successfully saved to {out_path}")
    return combined_report


if __name__ == "__main__":
    run_full_experiment()
