"""
lib3_geoai.training.engine
============================
TrainingEngine — wrapper รอบ lib2_vitpalm Trainer ที่ยืดหยุ่นขึ้น

เพิ่มเติมจาก lib2 Trainer:
  - load_pretrained_weights()   : โหลด MAE weights ก่อน train
  - fine_tune()                 : freeze encoder แล้ว train decoder ก่อน
  - predict_tile()              : inference บน single tile พร้อม output_mode
  - format_output()             : แปลง prob map ตาม output_mode config
"""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional

from lib3_geoai.config import GeoAIConfig
from lib2_vitpalm.training.trainer import Trainer
from lib2_vitpalm.utils import set_seed

try:
    import wandb
except ImportError:
    wandb = None


class TrainingEngine:
    """
    High-level training engine สำหรับ GeoAI pipeline.

    Args:
        model        : nn.Module (จาก build_model())
        train_loader : DataLoader สำหรับ training
        val_loader   : DataLoader สำหรับ validation
        config       : GeoAIConfig

    Usage:
        engine = TrainingEngine(model, train_loader, val_loader, config)
        engine.load_pretrained_weights("checkpoints/mae/mae_best.pth")
        history = engine.train()
        results = engine.evaluate(test_loader)
    """

    def __init__(
        self,
        model:        "torch.nn.Module",
        train_loader: DataLoader,
        val_loader:   DataLoader,
        config:       GeoAIConfig,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history: list[dict] = []

        # ตั้ง random seed
        set_seed(config.seed)

        # สร้าง save_dir
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

        # สร้าง lib2 Trainer
        self._trainer = Trainer(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            output_dir   = config.save_dir,
            lr           = config.lr,
            weight_decay = config.weight_decay,
            epochs       = config.epochs,
            save_every   = config.save_every,
            bce_weight   = config.bce_weight,
            dice_weight  = config.dice_weight,
            pos_weight   = config.pos_weight,
            threshold    = config.threshold,
            patience     = config.patience,
            use_amp      = config.use_amp,
            use_wandb    = config.use_wandb,
        )

    # ── Pretrained Weights ──────────────────────────────────────────────────

    def load_pretrained_weights(
        self,
        ckpt_path: str | Path,
        strict: bool = False,
    ) -> None:
        """
        โหลด MAE pretrained encoder weights เข้า model.
        รองรับเฉพาะ ViTPalmModel (Early Fusion).
        ViTLateFusionModel / ViTCrossAttnModel: โหลดให้ encoder_s2 (primary).

        Args:
            ckpt_path : path ไปยัง checkpoint (.pth)
            strict    : ถ้า True → error เมื่อ key ไม่ตรง
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = self.model
        model_name = model.__class__.__name__

        # ViTPalmModel มี load_mae_weights() อยู่แล้ว
        if hasattr(model, "load_mae_weights"):
            print(f"  Loading MAE weights → {model_name}.encoder")
            model.load_mae_weights(ckpt_path, strict=strict)

        # Late / Cross Fusion: โหลดให้ encoder_s2 (primary modality)
        elif hasattr(model, "encoder_s2"):
            print(f"  Loading MAE weights → {model_name}.encoder_s2 (S2 primary)")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = {
                k.replace("encoder.", ""): v
                for k, v in ckpt.items()
                if k.startswith("encoder.")
            }
            missing, unexpected = model.encoder_s2.load_state_dict(state_dict, strict=strict)
            print(f"  Loaded {len(state_dict) - len(missing)} / {len(state_dict)} keys")
        else:
            print(f"  Warning: {model_name} ไม่รองรับการโหลด MAE weights")

    # ── Fine-Tune Helpers ───────────────────────────────────────────────────

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters — ใช้เมื่อ fine-tune decoder ก่อน."""
        frozen = 0
        for name, param in self.model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False
                frozen += 1
        print(f"  Frozen {frozen} encoder parameters")

    def unfreeze_all(self) -> None:
        """Unfreeze ทุก parameter."""
        for param in self.model.parameters():
            param.requires_grad = True
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Unfrozen all parameters ({n:,} trainable)")

    def fine_tune(
        self,
        epochs: int = 30,
        lr: float = 1e-5,
        freeze_encoder_epochs: int = 5,
    ) -> list[dict]:
        """
        Fine-tuning strategy:
          Phase 1: freeze encoder, train decoder only (freeze_encoder_epochs)
          Phase 2: unfreeze all, train end-to-end (epochs - freeze_encoder_epochs)

        Args:
            epochs                : total fine-tune epochs
            lr                    : fine-tune learning rate (ต่ำกว่า initial)
            freeze_encoder_epochs : epochs ที่ freeze encoder ก่อน

        Returns:
            history: list of metric dicts
        """
        print(f"Fine-tune Phase 1: freeze encoder for {freeze_encoder_epochs} epochs")
        self.freeze_encoder()

        # สร้าง trainer ใหม่ด้วย lr ต่ำ + epochs น้อย
        trainer_p1 = Trainer(
            model        = self.model,
            train_loader = self.train_loader,
            val_loader   = self.val_loader,
            output_dir   = Path(self.config.save_dir) / "finetune",
            lr           = lr * 10,    # decoder learning rate
            epochs       = freeze_encoder_epochs,
            patience     = freeze_encoder_epochs,
            use_amp      = self.config.use_amp,
            pos_weight   = self.config.pos_weight,
            bce_weight   = self.config.bce_weight,
            dice_weight  = self.config.dice_weight,
        )
        history = trainer_p1.train()

        print(f"\nFine-tune Phase 2: unfreeze all for {epochs - freeze_encoder_epochs} epochs")
        self.unfreeze_all()
        trainer_p2 = Trainer(
            model        = self.model,
            train_loader = self.train_loader,
            val_loader   = self.val_loader,
            output_dir   = Path(self.config.save_dir) / "finetune",
            lr           = lr,
            epochs       = epochs - freeze_encoder_epochs,
            patience     = self.config.patience,
            use_amp      = self.config.use_amp,
            pos_weight   = self.config.pos_weight,
            bce_weight   = self.config.bce_weight,
            dice_weight  = self.config.dice_weight,
        )
        history += trainer_p2.train()
        return history

    # ── Main Train ───────────────────────────────────────────────────────────

    def train(self) -> list[dict]:
        """รัน training loop ปกติ."""
        print(f"\n{'='*60}")
        print(f"Training: {self.model.__class__.__name__}")
        print(f"  Device : {self.device}")
        print(self.config.summary())
        print(f"{'='*60}")
        self.history = self._trainer.train()
        return self.history

    # ── Evaluation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        ckpt_path: Optional[str | Path] = None,
    ) -> dict:
        """
        ประเมินผลบน test set.

        Args:
            test_loader : DataLoader สำหรับ test set
            ckpt_path   : โหลด checkpoint ก่อน evaluate (optional)

        Returns:
            dict: iou, f1, precision, recall
        """
        from lib2_vitpalm.training.trainer import compute_metrics

        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(ckpt["model"])
            print(f"  Loaded checkpoint: {ckpt_path}")

        self.model.eval()
        self.model.to(self.device)

        agg = {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            pred = self.model(imgs).float()
            m = compute_metrics(pred, masks, self.config.threshold)
            for k in agg:
                agg[k] += m[k]

        n = len(test_loader)
        results = {k: v / n for k, v in agg.items()}
        print("\n=== Test Results ===")
        for k, v in results.items():
            print(f"  {k:12s}: {v:.4f}")
        return results

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_tile(
        self,
        tile: "np.ndarray | torch.Tensor",
        ckpt_path: Optional[str | Path] = None,
    ) -> "np.ndarray":
        """
        Predict บน single tile.

        Args:
            tile      : (C, H, W) numpy array หรือ torch Tensor
            ckpt_path : โหลด checkpoint ก่อน predict (optional)

        Returns:
            output ตาม config.output_mode:
              "probability" → (H, W) float [0.0, 1.0]
              "binary"      → (H, W) uint8 {0, 1}
              "both"        → dict {"probability": ..., "binary": ...}
        """
        import numpy as np

        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(ckpt["model"])

        self.model.eval()
        self.model.to(self.device)

        if isinstance(tile, np.ndarray):
            tile = torch.from_numpy(tile).float()
        if tile.dim() == 3:
            tile = tile.unsqueeze(0)   # (1, C, H, W)

        tile = tile.to(self.device)
        prob = self.model(tile).squeeze(0).cpu().numpy()  # (H, W)

        return self._format_output(prob)

    def _format_output(self, prob: "np.ndarray") -> "np.ndarray | dict":
        """แปลง prob map ตาม output_mode."""
        import numpy as np
        mode = self.config.output_mode
        binary = (prob > self.config.threshold).astype(np.uint8)

        if mode == "probability":
            return prob
        elif mode == "binary":
            return binary
        elif mode == "both":
            return {"probability": prob, "binary": binary}
        else:
            raise ValueError(f"Unknown output_mode '{mode}'")
