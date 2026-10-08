import sys
from pathlib import Path
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poc.dataset import PalmDataset
from poc.augmentation import get_training_augmentation, get_validation_augmentation
from lib2_vitpalm.training.trainer import Trainer
from lib2_vitpalm.model.vitpalm import ViTPalmModel
from lib2_vitpalm.model.late_fusion_model import ViTLateFusionModel
from lib2_vitpalm.model.cross_attn_model import ViTCrossAttnModel

def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_benchmark():
    print("=" * 60)
    print("  GISxKMUTNB - Fusion Strategy Benchmark")
    print("=" * 60)

    cfg_path = PROJECT_ROOT / "configs" / "baseline_config.yaml"
    cfg = load_config(cfg_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dir = PROJECT_ROOT / "data" / "processed" / "split" / "train"
    val_dir   = PROJECT_ROOT / "data" / "processed" / "split" / "val"
    
    train_ds = PalmDataset(train_dir, augmentation=get_training_augmentation())
    val_ds   = PalmDataset(val_dir, augmentation=get_validation_augmentation())
    
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False)
    
    strategies = {
        "EarlyFusion": ViTPalmModel(
            in_channels=12,
            img_size=cfg['model']['img_size'],
            patch_size=cfg['model']['patch_size'],
            embed_dim=cfg['model']['embed_dim'],
            depth=cfg['model']['depth'],
            num_heads=cfg['model']['num_heads'],
            mlp_ratio=cfg['model']['mlp_ratio']
        ),
        "LateFusion": ViTLateFusionModel(
            img_size=cfg['model']['img_size'],
            patch_size=cfg['model']['patch_size'],
            embed_dim=cfg['model']['embed_dim'],
            depth=cfg['model']['depth'],
            num_heads=cfg['model']['num_heads'],
            mlp_ratio=cfg['model']['mlp_ratio']
        ),
        "CrossAttn": ViTCrossAttnModel(
            img_size=cfg['model']['img_size'],
            patch_size=cfg['model']['patch_size'],
            embed_dim=cfg['model']['embed_dim'],
            depth=cfg['model']['depth'],
            num_heads=cfg['model']['num_heads'],
            mlp_ratio=cfg['model']['mlp_ratio']
        )
    }
    
    results = []
    
    for name, model in strategies.items():
        print(f"\n[{name}] Starting training from scratch...")
        out_dir = PROJECT_ROOT / "checkpoints" / f"benchmark_{name.lower()}"
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=out_dir,
            device=device,
            lr=float(cfg['training']['lr']),
            weight_decay=float(cfg['training']['weight_decay']),
            epochs=250,
            save_every=100, # don't save periodic
            bce_weight=cfg['training']['bce_weight'],
            dice_weight=cfg['training']['dice_weight'],
            pos_weight=cfg['training']['pos_weight'],
            threshold=cfg['training']['threshold'],
            patience=20,
            use_amp=cfg['training']['use_amp'],
            use_wandb=False
        )
        
        history = trainer.train()
        best_f1 = trainer.best_f1
        
        # Find best record
        best_record = next(r for r in history if r["val_f1"] == best_f1)
        
        results.append({
            "Strategy": name,
            "Best_Val_F1": best_record["val_f1"],
            "Best_Val_IoU": best_record["val_iou"],
            "Best_Val_Precision": best_record["val_precision"],
            "Best_Val_Recall": best_record["val_recall"],
            "Epoch": best_record["epoch"]
        })
        
        print(f"[{name}] Finished. Best F1: {best_f1:.4f}")
        
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_string())
    
    out_csv = PROJECT_ROOT / "results" / "metrics" / "fusion_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

if __name__ == "__main__":
    run_benchmark()
