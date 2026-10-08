import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poc.dataset import PalmDataset
from poc.augmentation import get_validation_augmentation
from lib2_vitpalm.model.late_fusion_model import ViTLateFusionModel
from lib2_vitpalm.model.vitpalm import ViTPalmModel
from lib2_vitpalm.training.trainer import compute_metrics

def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def evaluate_model(model, loader, device, threshold=0.5):
    model.eval()
    agg = {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            pred = model(imgs)
            met = compute_metrics(pred, masks, threshold)
            for k in agg: agg[k] += met[k]
            
    # average
    for k in agg:
        agg[k] /= len(loader)
        
    return agg

def run_test_evaluation():
    cfg_path = PROJECT_ROOT / "configs" / "baseline_config.yaml"
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dir = PROJECT_ROOT / "data" / "processed" / "split" / "test"
    test_ds = PalmDataset(test_dir, augmentation=get_validation_augmentation())
    test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'], shuffle=False)
    
    models_to_test = {
        "EarlyFusion": (ViTPalmModel(
            in_channels=12,
            img_size=cfg['model']['img_size'],
            patch_size=cfg['model']['patch_size'],
            embed_dim=cfg['model']['embed_dim'],
            depth=cfg['model']['depth'],
            num_heads=cfg['model']['num_heads'],
            mlp_ratio=cfg['model']['mlp_ratio']
        ), PROJECT_ROOT / "checkpoints" / "benchmark_earlyfusion" / "vitpalm_best.pth"),
        
        "LateFusion": (ViTLateFusionModel(
            img_size=cfg['model']['img_size'],
            patch_size=cfg['model']['patch_size'],
            embed_dim=cfg['model']['embed_dim'],
            depth=cfg['model']['depth'],
            num_heads=cfg['model']['num_heads'],
            mlp_ratio=cfg['model']['mlp_ratio']
        ), PROJECT_ROOT / "checkpoints" / "benchmark_latefusion" / "vitpalm_best.pth")
    }
    
    print(f"Evaluating on Test set (N={len(test_ds)} tiles)...")
    
    for name, (model, cp_path) in models_to_test.items():
        if cp_path.exists():
            checkpoint = torch.load(cp_path, map_location=device)
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            metrics = evaluate_model(model, test_loader, device, cfg['training']['threshold'])
            print(f"[{name}] Test Metrics: F1: {metrics['f1']:.4f} | IoU: {metrics['iou']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        else:
            print(f"[{name}] Checkpoint not found: {cp_path}")

if __name__ == "__main__":
    run_test_evaluation()
