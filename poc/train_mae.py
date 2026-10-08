"""
poc/train_mae.py
================
Script to run MAE pretraining on the satellite tiles.
"""

import sys
import yaml
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from poc.dataset import PalmDataset
from poc.augmentation import get_training_augmentation, get_validation_augmentation
from lib2_vitpalm.model.mae_model import MAEModel

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_mae():
    print("=" * 60)
    print("  GISxKMUTNB - MAE Pretraining")
    print("=" * 60)

    # 1. Load config
    cfg_path = PROJECT_ROOT / "configs" / "mae_config.yaml"
    cfg = load_config(cfg_path)
    
    # 2. Setup Device & AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg['training']['use_amp'] and device.type == 'cuda'
    print(f"Device: {device} | AMP: {use_amp}")
    
    # 3. WandB Logging
    use_wandb = cfg['logging']['use_wandb']
    if use_wandb:
        try:
            import wandb
            wandb.init(project=cfg['logging']['project'], name=cfg['logging']['run_name'], config=cfg)
        except ImportError:
            print("WandB not installed. Skipping.")
            use_wandb = False
            
    # 4. Datasets & DataLoaders
    train_dir = PROJECT_ROOT / "data" / "processed" / "split" / "train"
    val_dir   = PROJECT_ROOT / "data" / "processed" / "split" / "val"
    
    train_ds = PalmDataset(train_dir, augmentation=get_training_augmentation())
    val_ds   = PalmDataset(val_dir, augmentation=get_validation_augmentation())
    
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False)
    
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    # 5. Model
    model = MAEModel(**cfg['model']).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 6. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']), weight_decay=float(cfg['training']['weight_decay']))
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'], eta_min=1e-6)

    # 7. Checkpointing setup
    out_dir = PROJECT_ROOT / "checkpoints" / "mae"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = cfg['training']['epochs']
    patience = cfg['training']['patience']
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = []

    # 8. Training Loop
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # --- Train ---
        model.train()
        train_loss = 0.0
        
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                pred, ids_keep, ids_masked = model(imgs)
                # target for visible patches
                target = model.get_patch_targets(imgs, ids_keep)
                loss = criterion(pred, target)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # --- Val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                    pred, ids_keep, ids_masked = model(imgs)
                    target = model.get_patch_targets(imgs, ids_keep)
                    loss = criterion(pred, target)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        
        print(f"Epoch [{epoch:3d}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {lr_now:.2e} | Time: {elapsed:.1f}s")
              
        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now})
            
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        
        # --- Early Stopping & Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / "mae_best.pth")
            print(f"  -> Best model saved (Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            
        if epoch % cfg['training']['save_every'] == 0:
            torch.save(model.state_dict(), out_dir / f"mae_epoch_{epoch}.pth")
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break
            
    # Save History
    with open(out_dir / "mae_history.json", "w") as f:
        json.dump(history, f, indent=2)
        
    if use_wandb:
        wandb.finish()

    print("\n" + "=" * 60)
    print("  MAE Pretraining Complete!")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    train_mae()
