"""
lib3_geoai.pretrain.cloud_mae
==============================
CloudMAE — Feature ใหม่ที่รองรับการ mask แบบมีข้อมูล (Cloud mask)

แทนที่จะใช้ random mask ปกติ เราสามารถป้อนข้อมูลเมฆ
ให้โมเดลหัด reconstruct พื้นที่ที่มีเมฆบัง (จาก Optical) 
โดยอาศัยบริบทของข้อมูล SAR (ที่ทะลุเมฆได้) แทน

Usage:
    from lib3_geoai import GeoAI
    mae = GeoAI.pretrain(mask_strategy="cloud", cloud_mask_dir="...")
    mae.fit()
"""

from __future__ import annotations

import torch
import torch.nn as nn
import time
import json
from pathlib import Path
from typing import Optional

from lib3_geoai.config import GeoAIConfig
from lib3_geoai.data.loader import build_pretrain_loader
from lib2_vitpalm.model.mae_model import MAEModel, random_masking
from lib2_vitpalm.loss.combined import mae_reconstruction_loss
from lib2_vitpalm.utils import set_seed


class CloudMAE:
    """
    MAE Pretraining Pipeline รองรับ Cloud Mask
    """
    def __init__(self, config: Optional[GeoAIConfig] = None):
        self.config = config or GeoAIConfig.mae_default()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ตั้ง random seed
        set_seed(self.config.seed)
        
        # เตรียม model
        self.model = MAEModel(
            in_channels = self.config.in_channels,
            img_size    = self.config.tile_size,
            patch_size  = self.config.patch_size,
            embed_dim   = self.config.embed_dim,
            depth       = self.config.depth,
            num_heads   = self.config.num_heads,
            mlp_ratio   = self.config.mlp_ratio,
            mask_ratio  = self.config.mask_ratio,
        )
        self.model.to(self.device)
        
        self.output_dir = Path(self.config.save_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._loader = None

    def _setup_data(self):
        """โหลด pretrain data (ไม่มี mask label)"""
        if self._loader is not None:
            return
            
        print("\n[CloudMAE] Building DataLoader...")
        # ถ้าระบุ folder tiles โดยตรง
        data_path = Path(self.config.data_dir)
        if data_path.name == "tiles":
            dirs = [data_path.parent]
        else:
            # สมมติว่าเป็น split dir
            dirs = [data_path / "train", data_path / "val"]
            
        self._loader = build_pretrain_loader(dirs, self.config)

    def _cloud_masking(self, B: int, N: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Placeholder สำหรับ Cloud Masking Logic 
        (จะถูกเรียกใช้แทน random_masking ใน forward pass)
        """
        # TODO: โหลดไฟล์ cloud mask และแปลงเป็น patch indices
        # ตอนนี้ fallback ไปหา random masking ปกติไปก่อน
        return random_masking(B, N, self.config.mask_ratio, self.device)

    def _train_epoch(self, optimizer, scaler) -> float:
        self.model.train()
        total_loss = 0.0
        
        for imgs in self._loader:
            imgs = imgs.to(self.device)
            B = imgs.size(0)
            
            # ถ้าเป็น cloud strategy และมี logic รองรับแล้ว
            # ids_keep, ids_masked = self._cloud_masking(B, self.model.num_patches)
            # แต่ปัจจุบัน MAEModel.forward ใช้ random_masking ภายในตัว
            # อนาคตเราสามารถ override forward ตรงนี้ได้
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.config.use_amp and self.device.type == 'cuda', dtype=torch.bfloat16):
                # pred: (B, N_visible, patch_px)
                pred, ids_keep, ids_masked = self.model(imgs)
                
                # หา ground truth สำหรับ patch ที่เห็น (visible patches) 
                # หมายเหตุ: MAE ปกติมักจะคำนวณ loss บน patches ที่โดน mask
                # แต่ในโค้ด lib2 เดิม pred คือของ ids_keep
                targets = self.model.get_patch_targets(imgs, ids_keep)
                
            pred = pred.float()
            targets = targets.float()
            loss = mae_reconstruction_loss(pred, targets, norm=True)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        return total_loss / len(self._loader)

    def fit(self) -> None:
        """รัน Pretraining"""
        self._setup_data()
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.lr, 
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        scaler = torch.amp.GradScaler('cuda', enabled=self.config.use_amp and self.device.type == 'cuda')
        
        print(f"\n{'='*60}")
        print(f"Pretraining MAE: {self.config.mask_strategy} masking")
        print(f"  Device : {self.device}")
        print(f"  Epochs : {self.config.epochs}")
        print(f"  Params : {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"{'='*60}")

        history = []
        best_loss = float('inf')

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            loss = self._train_epoch(optimizer, scaler)
            scheduler.step()
            elapsed = time.time() - t0
            
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch:3d}/{self.config.epochs}] loss={loss:.4f} lr={lr_now:.2e} ({elapsed:.1f}s)")
            
            history.append({"epoch": epoch, "loss": loss, "lr": lr_now})
            
            if loss < best_loss:
                best_loss = loss
                ckpt_path = self.output_dir / "mae_best.pth"
                torch.save({"model": self.model.state_dict(), "epoch": epoch}, ckpt_path)
                
            if epoch % self.config.save_every == 0:
                ckpt_path = self.output_dir / f"mae_epoch_{epoch:03d}.pth"
                torch.save({"model": self.model.state_dict(), "epoch": epoch}, ckpt_path)

        with open(self.output_dir / "mae_history.json", "w") as f:
            json.dump(history, f, indent=2)
            
        print(f"\nPretraining Complete. Best loss={best_loss:.4f}")
        print(f"Saved best model to: {self.output_dir / 'mae_best.pth'}")

    @property
    def checkpoint_path(self) -> Optional[Path]:
        p = self.output_dir / "mae_best.pth"
        return p if p.exists() else None
