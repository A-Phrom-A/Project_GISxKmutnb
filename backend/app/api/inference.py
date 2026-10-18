"""
backend/app/api/inference.py
============================
Endpoints for Oil Palm Binary Segmentation inference on GPU across 6 benchmark models.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import rasterio
import numpy as np

from backend.app.core.model_manager import get_model_manager, AVAILABLE_MODELS
from backend.app.db.audit import log_audit

router = APIRouter(prefix="/api/inference", tags=["Inference"])

ROOT_DIR = Path(__file__).resolve().parents[3]
SAM_DIR = ROOT_DIR / "sam_segment"
ZONES_DIR = ROOT_DIR / "data" / "demo_zones"


class SegmentationRequest(BaseModel):
    zone_id: int = Field(default=1, ge=1, le=4, description="Demo Zone ID (1-4)")
    model_id: str = Field(default="late_fusion_cloudmae", description="Registered model ID")
    offset_x: int = Field(default=900, description="Tile X crop offset")
    offset_y: int = Field(default=900, description="Tile Y crop offset")


@router.get("/models")
def list_available_models():
    """Lists all registered benchmark models and their metadata."""
    return list(AVAILABLE_MODELS.values())


@router.post("")
def execute_segmentation(req: SegmentationRequest, request: Request):
    """Executes multi-modal palm segmentation on GPU."""
    if req.model_id not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_id '{req.model_id}'. Available: {list(AVAILABLE_MODELS.keys())}",
        )

    s1_path = ZONES_DIR / f"zone_{req.zone_id}_s1.tif"
    s2_path = ZONES_DIR / f"zone_{req.zone_id}_s2.tif"
    gsat_path = SAM_DIR / f"{req.zone_id}.tif"

    if not s1_path.exists() or not s2_path.exists() or not gsat_path.exists():
        raise HTTPException(status_code=404, detail="Zone raster files not found.")

    tile_size = 256
    # 1. Read S1 slice (2, 256, 256)
    with rasterio.open(s1_path) as src:
        ox = min(max(0, req.offset_x), src.width - tile_size)
        oy = min(max(0, req.offset_y), src.height - tile_size)
        win = rasterio.windows.Window(ox, oy, tile_size, tile_size)
        s1_raw = src.read(window=win)

    # 2. Read S2 slice (7, 256, 256)
    with rasterio.open(s2_path) as src:
        s2_raw = src.read(window=win)

    # 3. Read GSat slice (3, 256, 256)
    with rasterio.open(gsat_path) as src:
        gsat_raw = src.read(window=win)

    # 4. Normalize
    # S1 (dB -> [0, 1])
    s1_db = 10.0 * np.log10(np.clip(s1_raw, 1e-6, None))
    s1_norm = np.clip((s1_db - (-25.0)) / 25.0, 0.0, 1.0).astype(np.float32)

    # S2 (0-10000 -> [0, 1])
    s2_norm = np.clip(s2_raw / 10000.0, 0.0, 1.0).astype(np.float32)

    # GSat (0-255 -> [0, 1])
    gsat_norm = np.clip(gsat_raw / 255.0, 0.0, 1.0).astype(np.float32)

    # 5. Inference
    manager = get_model_manager()
    res = manager.run_palm_segmentation(
        model_id=req.model_id,
        s1_arr=s1_norm,
        s2_arr=s2_norm,
        gsat_arr=gsat_norm,
    )

    gpu_stats = manager.get_gpu_status()

    # 6. Audit Log
    client_ip = request.client.host if request.client else "127.0.0.1"
    log_audit(
        client_ip=client_ip,
        action="segmentation_inference",
        zone_id=req.zone_id,
        model_id=req.model_id,
        cloud_coverage=None,
        latency_ms=res["latency_ms"],
        vram_used_mb=gpu_stats["vram_allocated_mb"],
        details=f"Palm Coverage: {res['palm_coverage_pct']}%",
    )

    return {
        "status": "success",
        "zone_id": req.zone_id,
        "model_id": req.model_id,
        "model_name": AVAILABLE_MODELS[req.model_id]["name"],
        "metrics": {
            "latency_ms": res["latency_ms"],
            "palm_coverage_pct": res["palm_coverage_pct"],
            "expected_f1": AVAILABLE_MODELS[req.model_id]["f1_score"],
            "gpu_vram_mb": gpu_stats["vram_allocated_mb"],
        },
        "images": res["images"],
    }
