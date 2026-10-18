"""
backend/app/api/zones.py
========================
Endpoints for retrieving 4 Demo Zones metadata, GeoJSON polygons, and NDVI statistics.
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import rasterio
import numpy as np

from backend.app.core.model_manager import AVAILABLE_MODELS
from backend.app.core.zonal_analytics import compute_zonal_palm_overlap

router = APIRouter(prefix="/api/zones", tags=["Zones"])

ROOT_DIR = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT_DIR / "data" / "demo_zones" / "zones_manifest.json"
ZONES_DIR = ROOT_DIR / "data" / "demo_zones"


def get_manifest():
    if not MANIFEST_PATH.exists():
        raise HTTPException(status_code=404, detail="Zones manifest not found. Please run prepare_demo_zones.py.")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("")
def list_zones():
    """Returns metadata for all 4 demo zones."""
    manifest = get_manifest()
    return manifest["zones"]


@router.get("/{zone_id}")
def get_zone_details(zone_id: int):
    """Returns details for a specific zone."""
    manifest = get_manifest()
    for z in manifest["zones"]:
        if z["zone_id"] == zone_id:
            return z
    raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found.")


@router.get("/{zone_id}/palms")
def get_zone_palms_geojson(zone_id: int, mode: str = "gt", model_id: str = "late_fusion_cloudmae"):
    """
    Returns WGS84 GeoJSON polygons for Deck.GL layer.
    mode:
      - 'gt': Ground Truth surveyed palm parcels (Emerald Green)
      - 'pred': AI Model Predicted palm parcels (Neon Cyan)
      - 'compare': Comparison between GT and AI (TP: Green, FP: Amber, FN: Red)
    """
    geojson_path = ZONES_DIR / f"zone_{zone_id}_palms.geojson"
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail=f"GeoJSON for Zone {zone_id} not found.")

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Real raster-to-polygon zonal statistics directly from deep learning model inference mask
    zonal_res = compute_zonal_palm_overlap(zone_id, model_id, threshold=0.40)
    ratios = zonal_res["ratios"]
    confidences = zonal_res["confidences"]
    is_predicted = zonal_res["is_predicted"]

    features = data.get("features", [])
    tp_count, fp_count, fn_count, non_palm_count = 0, 0, 0, 0
    filtered_features = []

    for idx, feat in enumerate(features):
        p = feat["properties"]
        is_actual_palm = p.get("is_palm", False)
        pred_palm = bool(is_predicted[idx]) if idx < len(is_predicted) else False
        ratio = float(ratios[idx]) if idx < len(ratios) else 0.0
        conf = float(confidences[idx]) if idx < len(confidences) else 0.50

        if is_actual_palm:
            # Actual Palm in Ground Truth
            if pred_palm:
                # AI correctly detected it (TP)
                status = "tp"
                tp_count += 1
            else:
                # AI missed it (FN)
                status = "fn"
                fn_count += 1
        else:
            # Actual Non-Palm / Unrelated Land
            non_palm_count += 1
            if pred_palm:
                # AI false alarm / over-detection (FP)
                status = "fp"
                fp_count += 1
            else:
                # Correct rejection (True Negative / Non-Palm)
                status = "non_palm"

        p["status"] = status
        p["confidence"] = round(conf, 3)
        p["palm_coverage_pct"] = round(ratio * 100.0, 1)

        if mode == "gt":
            # GT: Palm parcels are 'gt', others are 'non_palm'
            p["display_status"] = "gt" if is_actual_palm else "non_palm"
            filtered_features.append(feat)
        elif mode == "pred":
            # Pred: AI detections (TP + FP) are 'pred', others are 'non_palm'
            p["display_status"] = "pred" if status in ("tp", "fp") else "non_palm"
            filtered_features.append(feat)
        elif mode == "compare":
            # Compare: TP, FP, FN, or non_palm
            p["display_status"] = status
            filtered_features.append(feat)
        else:
            p["display_status"] = "gt" if is_actual_palm else "non_palm"
            filtered_features.append(feat)

    precision = round(tp_count / max(1, (tp_count + fp_count)), 3)
    recall = round(tp_count / max(1, (tp_count + fn_count)), 3)
    f1 = round(2 * precision * recall / max(1e-4, precision + recall), 3)

    return {
        "type": "FeatureCollection",
        "name": f"zone_{zone_id}_palms_{mode}",
        "crs": data.get("crs"),
        "meta": {
            "mode": mode,
            "model_id": model_id,
            "total_parcels": len(filtered_features),
            "actual_palms": tp_count + fn_count,
            "tp_count": tp_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
            "non_palm_count": non_palm_count,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "features": filtered_features,
    }



@router.get("/{zone_id}/layer/{layer_type}")
def get_zone_raster_layer(zone_id: int, layer_type: str):
    """
    Returns visual raster composite for a demo zone.
    layer_type:
      - 'gsat': 0.5m Optical Satellite
      - 's1': Sentinel-1 SAR Radar (Dual-Pol VV/VH)
      - 's2_rgb': Sentinel-2 True Color (B4-B3-B2)
      - 's2_cir': Sentinel-2 Color Infrared (B8-B4-B3)
      - 's2_swir': Sentinel-2 Agriculture SWIR (B11-B8-B2)
      - 's2_ndvi': Sentinel-2 NDVI Heatmap (RdYlGn)
    """
    layers_dir = ZONES_DIR / "layers"
    img_path_png = layers_dir / f"zone_{zone_id}_{layer_type}.png"
    if img_path_png.exists():
        return FileResponse(img_path_png, media_type="image/png")
    img_path_jpg = layers_dir / f"zone_{zone_id}_{layer_type}.jpg"
    if img_path_jpg.exists():
        return FileResponse(img_path_jpg, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail=f"Layer '{layer_type}' for Zone {zone_id} not found.")


@router.get("/{zone_id}/ndvi_stats")
def get_zone_ndvi_stats(zone_id: int):
    """Calculates NDVI distribution histogram for the zone."""
    ndvi_tif = ZONES_DIR / f"zone_{zone_id}_ndvi.tif"
    if not ndvi_tif.exists():
        raise HTTPException(status_code=404, detail="NDVI GeoTIFF not found.")

    with rasterio.open(ndvi_tif) as src:
        arr = src.read(1)
        valid = arr[arr != -9999.0]
        # Histogram in ranges [-0.2, 1.0] with 10 bins
        hist, bin_edges = np.histogram(valid, bins=10, range=(0.0, 1.0))

        return {
            "zone_id": zone_id,
            "mean_ndvi": float(np.round(np.mean(valid), 3)),
            "std_ndvi": float(np.round(np.std(valid), 3)),
            "min_ndvi": float(np.round(np.min(valid), 3)),
            "max_ndvi": float(np.round(np.max(valid), 3)),
            "histogram": {
                "counts": hist.tolist(),
                "bins": [float(np.round(b, 2)) for b in bin_edges.tolist()],
            },
        }

