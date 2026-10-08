"""
poc/train_baseline.py
=====================
Script to fine-tune the ViTPalmModel (Encoder + SegDecoder) 
on the semantic segmentation task.
"""

import sys
import yaml
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from poc.dataset import PalmDataset
from poc.augmentation import get_training_augmentation, get_validation_augmentation
from lib2_vitpalm.model.vitpalm import ViTPalmModel
from lib2_vitpalm.training.trainer import Trainer

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_finetune():
    print("=" * 60)
    print("  GISxKMUTNB - MAE Fine-Tuning")
    print("=" * 60)

    # 1. Load config
    cfg_path = PROJECT_ROOT / "configs" / "finetune_config.yaml"
    cfg = load_config(cfg_path)
    
    # 2. Setup Device & WandB
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_wandb = cfg['logging']['use_wandb']
    if use_wandb:
        try:
            import wandb
            wandb.init(project=cfg['logging']['project'], name=cfg['logging']['run_name'], config=cfg)
        except ImportError:
            print("WandB not installed. Skipping.")
            use_wandb = False
            
    # 3. Datasets & DataLoaders
    train_dir = PROJECT_ROOT / "data" / "processed" / "split" / "train"
    val_dir   = PROJECT_ROOT / "data" / "processed" / "split" / "val"
    
    train_ds = PalmDataset(train_dir, augmentation=get_training_augmentation())
    val_ds   = PalmDataset(val_dir, augmentation=get_validation_augmentation())
    
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False)
    
    # 4. Model
    model = ViTPalmModel(
        in_channels=cfg['model']['in_channels'],
        img_size=cfg['model']['img_size'],
        patch_size=cfg['model']['patch_size'],
        embed_dim=cfg['model']['embed_dim'],
        depth=cfg['model']['depth'],
        num_heads=cfg['model']['num_heads'],
        mlp_ratio=cfg['model']['mlp_ratio'],
    )
    
    # Load Pretrained MAE weights
    mae_path = PROJECT_ROOT / cfg['model']['pretrained_path']
    if mae_path.exists():
        print(f"Loading MAE pre-trained weights from {mae_path.name}...")
        model.load_mae_weights(mae_path, strict=False)
    else:
        print("WARNING: MAE weights not found. Training from scratch.")

    # 5. Trainer
    out_dir = PROJECT_ROOT / "checkpoints" / "finetune"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=out_dir,
        device=device,
        lr=float(cfg['training']['lr']),
        weight_decay=float(cfg['training']['weight_decay']),
        epochs=cfg['training']['epochs'],
        save_every=cfg['training']['save_every'],
        bce_weight=cfg['training']['bce_weight'],
        dice_weight=cfg['training']['dice_weight'],
        pos_weight=cfg['training']['pos_weight'],
        threshold=cfg['training']['threshold'],
        patience=cfg['training']['patience'],
        use_amp=cfg['training']['use_amp'],
        use_wandb=use_wandb
    )
    
    # 6. Run Training
    history = trainer.train()
    
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    train_finetune()
