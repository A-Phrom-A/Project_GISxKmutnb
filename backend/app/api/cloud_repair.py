"""
backend/app/api/cloud_repair.py
===============================
Endpoints for running Radar-Guided Cloud-MAE Inpainting on Satellite Imagery.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import rasterio
import numpy as np

from backend.app.core.model_manager import get_model_manager
from backend.app.db.audit import log_audit

router = APIRouter(prefix="/api/cloud-repair", tags=["Cloud Repair"])

ROOT_DIR = Path(__file__).resolve().parents[3]
ZONES_DIR = ROOT_DIR / "data" / "demo_zones"


class CloudRepairRequest(BaseModel):
    zone_id: int = Field(default=1, description="Demo Zone ID (1-4) or Pretrain Benchmark ID (101-103)")
    cloud_ratio: float = Field(default=0.55, ge=0.0, le=0.85, description="Simulated cloud ratio (0.0 to 0.85)")
    offset_x: int = Field(default=600, description="Tile X crop offset")
    offset_y: int = Field(default=600, description="Tile Y crop offset")
    mask_type: str = Field(default="block", description="Cloud mask pattern: 'block' (พื้นที่ปิด) or 'checkerboard' (ตารางหมากรุก)")
    full_extent: bool = Field(default=True, description="When True for Demo Zones 1-4, process full 5.3 x 5.3 km landscape without excessive zoom")


PRETRAIN_TILES = {
    101: ROOT_DIR / "data" / "pretrain" / "tiles" / "tile_Chanthaburi_r00000_c01152.npz",
    102: ROOT_DIR / "data" / "pretrain" / "tiles" / "tile_Chonburi_r00000_c00000.npz",
    103: ROOT_DIR / "data" / "pretrain" / "tiles" / "tile_Rayong_r00000_c00000.npz",
}


@router.post("")
def execute_cloud_repair(req: CloudRepairRequest, request: Request):
    """Executes Cloud-MAE Inpainting on the requested zone tile or pretrain benchmark tile."""
    # Branch 1: Pretrain Benchmark Tiles (Native 10m Ground Truth validation set)
    if req.zone_id in PRETRAIN_TILES:
        npz_path = PRETRAIN_TILES[req.zone_id]
        if not npz_path.exists():
            raise HTTPException(status_code=404, detail=f"Pretrain benchmark tile not found: {npz_path.name}")
        data = np.load(npz_path)
        s1_norm = data["s1"].astype(np.float32)  # (2, 256, 256)
        s2_norm = data["s2"].astype(np.float32)  # (7, 256, 256)

    # Branch 2: Demo Zones 1-4 (Sampled across Full 5.3 x 5.3 km Landscape Scale)
    else:
        s1_path = ZONES_DIR / f"zone_{req.zone_id}_s1.tif"
        s2_path = ZONES_DIR / f"zone_{req.zone_id}_s2.tif"

        if not s1_path.exists() or not s2_path.exists():
            raise HTTPException(status_code=404, detail="Zone raster data files not found.")

        # Full 5.3 km x 5.3 km area coverage without excessive zoom
        if req.full_extent:
            with rasterio.open(s1_path) as src:
                s1_raw = src.read(out_shape=(src.count, 256, 256), resampling=rasterio.enums.Resampling.bilinear)
            with rasterio.open(s2_path) as src:
                s2_raw = src.read(out_shape=(src.count, 256, 256), resampling=rasterio.enums.Resampling.bilinear)
        else:
            win_size = 1024
            with rasterio.open(s1_path) as src:
                ox = min(max(0, req.offset_x), src.width - win_size)
                oy = min(max(0, req.offset_y), src.height - win_size)
                win = rasterio.windows.Window(ox, oy, win_size, win_size)
                s1_raw = src.read(out_shape=(src.count, 256, 256), window=win, resampling=rasterio.enums.Resampling.bilinear)
            with rasterio.open(s2_path) as src:
                s2_raw = src.read(out_shape=(src.count, 256, 256), window=win, resampling=rasterio.enums.Resampling.bilinear)

        # Radiometric Calibration & Normalization:
        # S1: Reorder to [VV, VH] and normalize [-30, 0] dB
        s1_vv = s1_raw[1]
        s1_vh = s1_raw[0]
        s1_db_vv = 10.0 * np.log10(np.clip(s1_vv, 1e-6, None))
        s1_db_vh = 10.0 * np.log10(np.clip(s1_vh, 1e-6, None))
        s1_norm_vv = np.clip((s1_db_vv - (-30.0)) / 30.0, 0.0, 1.0)
        s1_norm_vh = np.clip((s1_db_vh - (-30.0)) / 30.0, 0.0, 1.0)
        s1_norm = np.stack([s1_norm_vv, s1_norm_vh], axis=0).astype(np.float32)

        # S2: Sentinel-2 PB >= 04.00 BOA_ADD_OFFSET = -1000 DN
        s2_norm = np.clip((s2_raw.astype(np.float32) - 1000.0) / 10000.0, 0.0, 1.0)

    # 4. Inference via GPU ModelManager
    manager = get_model_manager()
    res = manager.run_cloud_repair(s1_norm, s2_norm, cloud_ratio=req.cloud_ratio, mask_type=req.mask_type)

    gpu_stats = manager.get_gpu_status()

    # 5. Audit Log
    client_ip = request.client.host if request.client else "127.0.0.1"
    log_audit(
        client_ip=client_ip,
        action="cloud_repair",
        zone_id=req.zone_id,
        model_id="cloud_mae_500e",
        cloud_coverage=res["cloud_coverage_pct"],
        latency_ms=res["latency_ms"],
        vram_used_mb=gpu_stats["vram_allocated_mb"],
        details=f"PSNR: {res['psnr_db']} dB, MSE: {res['mse']}",
    )

    return {
        "status": "success",
        "zone_id": req.zone_id,
        "model": "Cloud-MAE (500e Pretrained ViT)",
        "metrics": {
            "latency_ms": res["latency_ms"],
            "cloud_coverage_pct": res["cloud_coverage_pct"],
            "psnr_db": res["psnr_db"],
            "mse": res["mse"],
            "gpu_vram_mb": gpu_stats["vram_allocated_mb"],
        },
        "images": res["images"],
        "spectral": res.get("spectral"),
    }
