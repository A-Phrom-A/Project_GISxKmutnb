"""
backend/scripts/generate_model_predictions.py
==============================================
Generates full-zone prediction heatmaps and binary masks for all registered
benchmark models on NVIDIA RTX 5060 Ti GPU.
Saves visual overlays to data/demo_zones/layers/
  - zone_{zone_id}_{model_id}_heatmap.jpg
  - zone_{zone_id}_{model_id}_mask.jpg
"""

import sys
import time
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
import torch
import matplotlib.cm as cm

# Setup Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.model_manager import get_model_manager, AVAILABLE_MODELS

DEMO_DIR = ROOT_DIR / "data" / "demo_zones"
SAM_DIR = ROOT_DIR / "sam_segment"
LAYERS_DIR = DEMO_DIR / "layers"
LAYERS_DIR.mkdir(parents=True, exist_ok=True)


def predict_full_zone(manager, model_id: str, zone_id: int):
    s1_path = DEMO_DIR / f"zone_{zone_id}_s1.tif"
    s2_path = DEMO_DIR / f"zone_{zone_id}_s2.tif"
    gsat_path = SAM_DIR / f"{zone_id}.tif"

    if not s1_path.exists() or not s2_path.exists() or not gsat_path.exists():
        print(f"[!] Missing inputs for zone {zone_id}")
        return

    print(f"[*] Inferring Zone {zone_id} with model: {model_id}...")

    with rasterio.open(s1_path) as src_s1:
        s1_raw = src_s1.read().astype(np.float32)
        H, W = src_s1.height, src_s1.width

    with rasterio.open(s2_path) as src_s2:
        s2_raw = src_s2.read().astype(np.float32)

    with rasterio.open(gsat_path) as src_gsat:
        gsat_raw = src_gsat.read().astype(np.float32)

    # Normalize
    s1_db = 10.0 * np.log10(np.clip(s1_raw, 1e-6, None))
    s1_norm = np.clip((s1_db - (-25.0)) / 25.0, 0.0, 1.0)
    s2_norm = np.clip(s2_raw / 10000.0, 0.0, 1.0)
    gsat_norm = np.clip(gsat_raw / 255.0, 0.0, 1.0)

    tile_size = 256
    stride = 224  # slight overlap
    pad_h = (tile_size - (H % stride)) % stride + (tile_size - stride)
    pad_w = (tile_size - (W % stride)) % stride + (tile_size - stride)

    s1_padded = np.pad(s1_norm, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    s2_padded = np.pad(s2_norm, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    gsat_padded = np.pad(gsat_norm, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    H_pad, W_pad = s1_padded.shape[1], s1_padded.shape[2]
    prob_map = np.zeros((H_pad, W_pad), dtype=np.float32)
    count_map = np.zeros((H_pad, W_pad), dtype=np.float32)

    patches_s1, patches_s2, patches_gsat, coords = [], [], [], []
    for y in range(0, H_pad - tile_size + 1, stride):
        for x in range(0, W_pad - tile_size + 1, stride):
            patches_s1.append(s1_padded[:, y : y + tile_size, x : x + tile_size])
            patches_s2.append(s2_padded[:, y : y + tile_size, x : x + tile_size])
            patches_gsat.append(gsat_padded[:, y : y + tile_size, x : x + tile_size])
            coords.append((y, x))

    model = manager.get_segmentation_model(model_id)
    batch_size = 16

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(patches_s1), batch_size):
            b_s1 = torch.from_numpy(np.array(patches_s1[i : i + batch_size])).float().to(manager.device)
            b_s2 = torch.from_numpy(np.array(patches_s2[i : i + batch_size])).float().to(manager.device)
            b_gsat = torch.from_numpy(np.array(patches_gsat[i : i + batch_size])).float().to(manager.device)

            if "early_fusion" in model_id:
                x12 = torch.cat([b_s1, b_s2, b_gsat], dim=1)
                logits = model(x12)
            else:
                logits = model(b_s1, b_s2, b_gsat)

            # Note: ViT models already apply sigmoid in SegDecoder, do NOT double-sigmoid!
            probs = logits.cpu().numpy()
            if probs.ndim == 4:
                probs = probs.squeeze(1)

            for j in range(probs.shape[0]):
                y, x = coords[i + j]
                prob_map[y : y + tile_size, x : x + tile_size] += probs[j]
                count_map[y : y + tile_size, x : x + tile_size] += 1.0

    dur = time.time() - t0
    final_prob = prob_map[:H, :W] / np.clip(count_map[:H, :W], 1.0, None)
    palm_ratio = (final_prob > 0.5).mean()
    print(f"    -> Done in {dur:.2f}s! Mean prob: {final_prob.mean():.3f}, Palm ratio: {palm_ratio * 100:.1f}%")

    # Generate Magma Heatmap
    heatmap_rgba = cm.magma(np.clip(final_prob, 0, 1))
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    img_hm = Image.fromarray(heatmap_rgb)
    out_hm_path = LAYERS_DIR / f"zone_{zone_id}_{model_id}_heatmap.jpg"
    img_hm.save(out_hm_path, quality=88)

    # Generate Binary Mask (Threshold 0.5)
    palm_pixels = final_prob > 0.5
    # Transparent RGBA PNG
    # Palm areas: Emerald green (RGBA: 34, 197, 94, 220)
    # Non-palm areas: Transparent (RGBA: 0, 0, 0, 0)
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[palm_pixels, 0] = 34   # R
    rgba[palm_pixels, 1] = 197  # G
    rgba[palm_pixels, 2] = 94   # B
    rgba[palm_pixels, 3] = 220  # Alpha

    img_mask_png = Image.fromarray(rgba, mode="RGBA")
    out_mask_png_path = LAYERS_DIR / f"zone_{zone_id}_{model_id}_mask.png"
    img_mask_png.save(out_mask_png_path, optimize=True)

    # Also save JPG version for compatibility
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[palm_pixels, 0] = 34
    rgb[palm_pixels, 1] = 197
    rgb[palm_pixels, 2] = 94
    img_mask_jpg = Image.fromarray(rgb, mode="RGB")
    out_mask_jpg_path = LAYERS_DIR / f"zone_{zone_id}_{model_id}_mask.jpg"
    img_mask_jpg.save(out_mask_jpg_path, quality=88)

    print(f"    -> Saved: {out_hm_path.name} & {out_mask_png_path.name}")


if __name__ == "__main__":
    mgr = get_model_manager()
    # Prioritize Zone 1 and primary models first
    for zid in range(1, 5):
        for mid in ["late_fusion_cloudmae", "cross_attn_cloudmae", "late_fusion_scratch"]:
            predict_full_zone(mgr, mid, zid)
    print("\n[+] Full zone prediction overlays generated successfully!")
