"""
poc/evaluate.py
================
Script to evaluate the trained ViTPalmModel on the Test set.
Calculates IoU, F1, Precision, Recall and saves overlay images.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from poc.dataset import PalmDataset
from poc.augmentation import get_validation_augmentation
from lib2_vitpalm.model.vitpalm import ViTPalmModel
from lib2_vitpalm.training.trainer import compute_metrics
import yaml

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def evaluate():
    print("=" * 60)
    print("  GISxKMUTNB - Evaluation (Test Set)")
    print("=" * 60)

    cfg_path = PROJECT_ROOT / "configs" / "baseline_config.yaml"
    cfg = load_config(cfg_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_dir = PROJECT_ROOT / "data" / "processed" / "split" / "test"
    test_ds = PalmDataset(test_dir, augmentation=get_validation_augmentation())
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Test size: {len(test_ds)}")

    model = ViTPalmModel(
        in_channels=cfg['model']['in_channels'],
        img_size=cfg['model']['img_size'],
        patch_size=cfg['model']['patch_size'],
        embed_dim=cfg['model']['embed_dim'],
        depth=cfg['model']['depth'],
        num_heads=cfg['model']['num_heads'],
        mlp_ratio=cfg['model']['mlp_ratio'],
    )

    ckpt_path = PROJECT_ROOT / "checkpoints" / "baseline" / "vitpalm_best.pth"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    out_dir = PROJECT_ROOT / "checkpoints" / "baseline" / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    threshold = cfg['training']['threshold']
    
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(test_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            pred = model(imgs)
            
            # mask out nodata
            valid = (imgs.sum(dim=1) > 0).float()
            
            m = compute_metrics(pred * valid, masks * valid, threshold)
            for k in agg:
                agg[k] += m[k]
                
            # Save overlay image for first 10 samples
            if i < 10:
                img_rgb = imgs[0, 1:4].cpu().numpy().transpose(1, 2, 0) # B,G,R to RGB roughly (S2 bands 2,3,4)
                # Normalize for display
                img_rgb = np.clip(img_rgb / 3000.0, 0, 1)
                
                pred_bin = (pred[0].cpu().numpy() > threshold).astype(float)
                mask_np = masks[0].cpu().numpy()
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img_rgb)
                axes[0].set_title("Input (RGB)")
                axes[0].axis("off")
                
                axes[1].imshow(mask_np, cmap="gray")
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")
                
                axes[2].imshow(pred_bin, cmap="jet")
                axes[2].set_title(f"Prediction (IoU: {m['iou']:.2f})")
                axes[2].axis("off")
                
                plt.tight_layout()
                plt.savefig(out_dir / f"test_sample_{i}.png")
                plt.close()

    n = len(test_loader)
    for k in agg:
        agg[k] /= n
        
    print("\nTest Results:")
    print(f"  IoU      : {agg['iou']:.4f}")
    print(f"  F1 Score : {agg['f1']:.4f}")
    print(f"  Precision: {agg['precision']:.4f}")
    print(f"  Recall   : {agg['recall']:.4f}")
    
    print(f"\nSaved sample visualizations to {out_dir}")

if __name__ == "__main__":
    evaluate()
