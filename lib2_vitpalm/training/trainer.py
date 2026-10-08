"""
lib2_vitpalm.training.trainer
================================
Trainer class สำหรับ fine-tuning ViTPalmModel บน palm detection task.

Features:
  - Train / val loop พร้อม metrics (IoU, F1, Precision, Recall)
  - Checkpoint saving (best model + periodic)
  - CosineAnnealingLR scheduler
  - Nodata-aware loss masking (บริเวณที่ sum==0 ถือว่า nodata)
  - History logging → JSON
"""

import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional

try:
    import wandb
except ImportError:
    wandb = None

from lib2_vitpalm.loss import CombinedLoss


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    pred_prob: torch.Tensor,
    target:    torch.Tensor,
    threshold: float = 0.5,
    eps:       float = 1e-6,
) -> dict:
    """
    Compute IoU, F1, Precision, Recall for binary segmentation.

    Args:
        pred_prob : (B, H, W) probability map [0, 1]
        target    : (B, H, W) binary mask {0, 1}
        threshold : decision threshold (default 0.5)
        eps       : smoothing to avoid division by zero

    Returns:
        dict with keys: iou, f1, precision, recall
    """
    pred_bin = (pred_prob > threshold).float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()

    iou       = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    return {
        "iou":       iou.item(),
        "f1":        f1.item(),
        "precision": precision.item(),
        "recall":    recall.item(),
    }


# ─── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    """
    Fine-tuning trainer สำหรับ ViTPalmModel.

    Args:
        model        : ViTPalmModel instance
        train_loader : DataLoader สำหรับ training set
        val_loader   : DataLoader สำหรับ validation set
        output_dir   : directory สำหรับ checkpoint และ history
        device       : torch device (default: auto-detect)
        lr           : learning rate (default 1e-4)
        weight_decay : AdamW weight decay (default 1e-4)
        epochs       : total training epochs (default 50)
        save_every   : save checkpoint every N epochs (default 5)
        bce_weight   : BCE loss weight (default 0.5)
        dice_weight  : Dice loss weight (default 0.5)
        pos_weight   : positive class weight in BCE (default 3.0)
        threshold    : probability threshold for binary prediction (default 0.5)
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        output_dir:   str | Path,
        device:       Optional[torch.device] = None,
        lr:           float = 1e-4,
        weight_decay: float = 1e-4,
        epochs:       int   = 50,
        save_every:   int   = 5,
        bce_weight:   float = 0.5,
        dice_weight:  float = 0.5,
        pos_weight:   float = 3.0,
        threshold:    float = 0.5,
        patience:     int   = 10,
        use_amp:      bool  = True,
        use_wandb:    bool  = False,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device    = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs    = epochs
        self.save_every = save_every
        self.threshold  = threshold
        self.patience   = patience
        self.use_amp    = use_amp
        self.use_wandb  = use_wandb

        if self.use_wandb and wandb is None:
            print("Warning: wandb is not installed. Setting use_wandb=False")
            self.use_wandb = False

        self.model.to(self.device)

        self.criterion = CombinedLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp and self.device.type == 'cuda')

        self.history:   list[dict] = []
        self.best_f1:   float      = 0.0
        self.best_epoch: int       = 0
        self.epochs_no_improve: int = 0

    # ── Private helpers ───────────────────────────────────────────────────────

    def _nodata_mask(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return valid pixel mask: False where all bands == 0 (nodata)."""
        return imgs.sum(dim=1) > 0   # (B, H, W)

    def _train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        agg = {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        for imgs, masks in self.train_loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.use_amp and self.device.type == 'cuda', dtype=torch.bfloat16):
                pred = self.model(imgs)          # (B, H, W) prob
                
            # Compute loss outside autocast in float32
            pred = pred.float()
            valid = self._nodata_mask(imgs).float()
            loss  = self.criterion(pred * valid, masks * valid)

            self.scaler.scale(loss).backward()
            
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            m = compute_metrics(pred.detach(), masks, self.threshold)
            for k in agg:
                agg[k] += m[k]

        n = len(self.train_loader)
        return {"loss": total_loss / n, **{k: v / n for k, v in agg.items()}}

    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        agg = {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        for imgs, masks in self.val_loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp and self.device.type == 'cuda', dtype=torch.bfloat16):
                pred  = self.model(imgs)
                
            pred = pred.float()
            valid = self._nodata_mask(imgs).float()
            loss  = self.criterion(pred * valid, masks * valid)

            total_loss += loss.item()
            m = compute_metrics(pred, masks, self.threshold)
            for k in agg:
                agg[k] += m[k]

        n = len(self.val_loader)
        return {"loss": total_loss / n, **{k: v / n for k, v in agg.items()}}

    def _save_checkpoint(self, epoch: int, tag: str) -> Path:
        path = self.output_dir / f"vitpalm_{tag}.pth"
        torch.save({
            "epoch":      epoch,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
            "best_f1":    self.best_f1,
        }, path)
        return path

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> list[dict]:
        """
        Run full training loop.

        Returns:
            history: list of per-epoch metric dicts
        """
        print(f"Training on {self.device}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val   batches: {len(self.val_loader)}")
        print(f"  Epochs       : {self.epochs}")
        print(f"  Params       : {self.model.num_parameters:,}")
        print("-" * 60)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_m = self._train_epoch()
            val_m   = self._val_epoch()
            self.scheduler.step()

            elapsed = time.time() - t0
            lr_now  = self.scheduler.get_last_lr()[0]

            record = {
                "epoch":          epoch,
                "lr":             lr_now,
                "train_loss":     train_m["loss"],
                "train_iou":      train_m["iou"],
                "train_f1":       train_m["f1"],
                "train_precision": train_m["precision"],
                "train_recall":   train_m["recall"],
                "val_loss":       val_m["loss"],
                "val_iou":        val_m["iou"],
                "val_f1":         val_m["f1"],
                "val_precision":  val_m["precision"],
                "val_recall":     val_m["recall"],
                "elapsed_s":      elapsed,
            }
            self.history.append(record)

            print(
                f"Epoch [{epoch:3d}/{self.epochs}]  "
                f"loss={val_m['loss']:.4f}  "
                f"IoU={val_m['iou']:.4f}  "
                f"F1={val_m['f1']:.4f}  "
                f"Prec={val_m['precision']:.4f}  "
                f"Rec={val_m['recall']:.4f}  "
                f"lr={lr_now:.2e}  ({elapsed:.1f}s)"
            )
            
            if self.use_wandb:
                wandb.log(record)

            # Save best model and check early stopping
            if val_m["f1"] > self.best_f1:
                self.best_f1    = val_m["f1"]
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                path = self._save_checkpoint(epoch, "best")
                print(f"  -> Best F1={self.best_f1:.4f} saved: {path.name}")
            else:
                self.epochs_no_improve += 1
                
            # Periodic checkpoint
            if epoch % self.save_every == 0 or epoch == self.epochs:
                self._save_checkpoint(epoch, f"epoch_{epoch:03d}")
                
            # Early stopping check
            if self.epochs_no_improve >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience={self.patience})")
                break

        # Save history
        hist_path = self.output_dir / "train_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print("-" * 60)
        print(f"Training complete. Best F1={self.best_f1:.4f} @ epoch {self.best_epoch}")
        print(f"History saved: {hist_path}")
        return self.history
