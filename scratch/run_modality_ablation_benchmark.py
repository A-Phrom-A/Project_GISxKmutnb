"""
scratch/run_modality_ablation_benchmark.py
===========================================
Empirical Modality Ablation Test on Validation Set (60 tiles):
Evaluates how much Sentinel-1 (SAR), Sentinel-2 (Optical), and GEO-Sat (RGB)
actually contribute to the Oil Palm segmentation F1 and IoU score.

Ablation Conditions tested:
1. All Sensors Active (Full Fusion: S1 + S2 + GSat)
2. Optical Only (Sentinel-2 only, S1 & GSat zeroed)
3. SAR Radar Only (Sentinel-1 only, S2 & GSat zeroed)
4. GEO-Sat Only (High-res RGB only, S1 & S2 zeroed)
5. S2 + S1 Fusion (Optical + Radar, GSat zeroed)
6. Cloudy S2 (Optical S2 corrupted with 50% cloud mask):
   - With SAR S1 active
   - Without SAR S1 (SAR also zeroed)
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from poc.dataset import PalmDataset
from lib2_vitpalm.training.trainer import compute_metrics
from lib2_vitpalm.model.late_fusion_model import ViTLateFusionModel
from lib2_vitpalm.model.vitpalm import ViTPalmModel
from lib2_vitpalm.model.cloud_mae import CloudMaskGenerator


def evaluate_condition(model, val_loader, condition_fn, device):
    total_iou = 0.0
    total_f1 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    count = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            # Apply condition modification to input channels
            imgs_mod = condition_fn(imgs)

            preds = model(imgs_mod)
            m = compute_metrics(preds, masks)

            total_iou += float(m["iou"])
            total_f1 += float(m["f1"])
            total_prec += float(m["precision"])
            total_rec += float(m["recall"])
            count += 1

    n = max(1, count)
    return {
        "F1": total_f1 / n,
        "IoU": total_iou / n,
        "Precision": total_prec / n,
        "Recall": total_rec / n
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  RUNNING EMPIRICAL MODALITY ABLATION TEST (ON 60 VALIDATION TILES)")
    print(f"  Device: {device}")
    print("=" * 70)

    val_dir = ROOT_DIR / "data" / "processed" / "split" / "val"
    val_ds = PalmDataset(str(val_dir), augmentation=None)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Load Best Late Fusion Model Checkpoint
    ckpt_path = ROOT_DIR / "checkpoints" / "benchmark_latefusion_cloudmae" / "best_model.pth"
    if not ckpt_path.exists():
        ckpt_path = ROOT_DIR / "checkpoints" / "benchmark_latefusion" / "vitpalm_best.pth"

    print(f"Loading Checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else (ckpt["model"] if "model" in ckpt else ckpt)

    model = ViTLateFusionModel(
        img_size=256,
        patch_size=16,
        embed_dim=256,
        depth=6,
        num_heads=8
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    # Define conditions (Channel layout: [0..1] = S1, [2..8] = S2, [9..11] = GSat)
    def full_fusion(x):
        return x

    def optical_s2_only(x):
        mod = x.clone()
        mod[:, 0:2, :, :] = 0.0   # zero S1
        mod[:, 9:12, :, :] = 0.0  # zero GSat
        return mod

    def sar_s1_only(x):
        mod = x.clone()
        mod[:, 2:12, :, :] = 0.0  # zero S2 and GSat
        return mod

    def gsat_only(x):
        mod = x.clone()
        mod[:, 0:9, :, :] = 0.0   # zero S1 and S2
        return mod

    def s2_plus_s1(x):
        mod = x.clone()
        mod[:, 9:12, :, :] = 0.0  # zero GSat, keep S1 + S2
        return mod

    def cloud_50_with_sar(x):
        mod = x.clone()
        B, _, H, W = x.shape
        torch.manual_seed(42)
        cloud_mask, _ = CloudMaskGenerator.generate_block_cloud_mask(
            B, H, W, patch_size=16, target_ratio=0.50, device=x.device
        )
        mod[:, 2:12, :, :] *= (1.0 - cloud_mask) # cloud blocks optical S2 and GSat, but S1 SAR penetrates!
        return mod

    def cloud_50_without_sar(x):
        mod = x.clone()
        B, _, H, W = x.shape
        torch.manual_seed(42)
        cloud_mask, _ = CloudMaskGenerator.generate_block_cloud_mask(
            B, H, W, patch_size=16, target_ratio=0.50, device=x.device
        )
        mod[:, 2:12, :, :] *= (1.0 - cloud_mask)
        mod[:, 0:2, :, :] = 0.0 # SAR also killed
        return mod

    experiments = [
        ("Full Fusion (S1 + S2 + GSat)", full_fusion),
        ("Sentinel-2 Optical Only", optical_s2_only),
        ("Sentinel-1 SAR Radar Only", sar_s1_only),
        ("GEO-Sat High-Res RGB Only", gsat_only),
        ("S2 (Optical) + S1 (SAR Radar)", s2_plus_s1),
        ("Under 50% Cloud: S1 SAR Active", cloud_50_with_sar),
        ("Under 50% Cloud: S1 SAR Zeroed", cloud_50_without_sar),
    ]

    results = []
    print("\nRunning Evaluation across conditions...")
    for name, fn in experiments:
        metrics = evaluate_condition(model, val_loader, fn, device)
        results.append({
            "Modality Condition": name,
            "Val F1 (%)": round(metrics["F1"] * 100, 2),
            "Val IoU (%)": round(metrics["IoU"] * 100, 2),
            "Precision (%)": round(metrics["Precision"] * 100, 2),
            "Recall (%)": round(metrics["Recall"] * 100, 2)
        })
        print(f"  -> {name:<35}: F1={metrics['F1']*100:.2f}% | IoU={metrics['IoU']*100:.2f}%")

    out_json = ROOT_DIR / "results" / "metrics" / "modality_ablation_val.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  FINAL ABLATION TABLE RESULTS:")
    print("=" * 70)
    for r in results:
        print(f"| {r['Modality Condition']:<35} | F1: {r['Val F1 (%)']:>6.2f}% | IoU: {r['Val IoU (%)']:>6.2f}% |")


if __name__ == "__main__":
    main()
