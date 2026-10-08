"""
lib3_geoai.pipeline.palm
==========================
PalmPipeline — pipeline สำหรับ Palm Oil Detection

Use Case 1: ตรวจจับสวนปาล์มในพื้นที่ใกล้เคียงชลบุรี
Use Case 1.1: เพิ่มข้อมูล palm ใหม่ เพื่อขยายพื้นที่ใช้งาน

Usage:
    from lib3_geoai import GeoAI

    palm = GeoAI.palm(
        data_dir   = "data/processed/split",
        fusion     = "late",
        pretrained = "checkpoints/mae/mae_best.pth",
    )
    palm.fit(epochs=100)
    palm.add_data("data/new_region/")
    palm.fine_tune(epochs=30)
    prob = palm.predict("tile.npy")
    palm.evaluate()
    palm.visualize(n=5)
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Literal

from lib3_geoai.config import GeoAIConfig
from lib3_geoai.data.loader import build_dataloaders
from lib3_geoai.models.factory import build_model, model_summary
from lib3_geoai.training.engine import TrainingEngine


class PalmPipeline:
    """
    Pipeline สำหรับ Palm Oil Segmentation.

    Args:
        config     : GeoAIConfig (ถ้าไม่ระบุจะใช้ palm_default())
        pretrained : path ไปยัง MAE checkpoint (optional)
    """

    def __init__(
        self,
        config:     Optional[GeoAIConfig] = None,
        pretrained: Optional[str | Path]  = None,
    ):
        self.config     = config or GeoAIConfig.palm_default()
        self.pretrained = Path(pretrained) if pretrained else None
        self._engine: Optional[TrainingEngine] = None
        self._loaders: dict = {}
        self._extra_data_dirs: list[Path] = []

    # ── Setup ────────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """สร้าง model + dataloaders + engine (lazy init)."""
        if self._engine is not None:
            return

        print(f"\n[PalmPipeline] Setup")
        print(f"  fusion   = {self.config.fusion}")
        print(f"  data_dir = {self.config.data_dir}")

        # DataLoaders
        print("\n[PalmPipeline] Building DataLoaders...")
        self._loaders = build_dataloaders(
            config         = self.config,
            extra_data_dirs = self._extra_data_dirs if self._extra_data_dirs else None,
        )

        if "train" not in self._loaders:
            raise RuntimeError(f"ไม่พบ train/ ใน {self.config.data_dir}")

        # Model
        print(f"\n[PalmPipeline] Building Model...")
        model = build_model(self.config)
        model_summary(model, self.config.fusion)

        # Engine
        self._engine = TrainingEngine(
            model        = model,
            train_loader = self._loaders["train"],
            val_loader   = self._loaders.get("val", self._loaders["train"]),
            config       = self.config,
        )

        # Load pretrained MAE weights ถ้ามี
        if self.pretrained and self.pretrained.exists():
            print(f"\n[PalmPipeline] Loading pretrained weights: {self.pretrained.name}")
            self._engine.load_pretrained_weights(self.pretrained)

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(
        self,
        epochs: Optional[int] = None,
        lr:     Optional[float] = None,
    ) -> list[dict]:
        """
        เทรน model.

        Args:
            epochs : override จำนวน epoch (optional)
            lr     : override learning rate (optional)
        """
        if epochs: self.config.epochs = epochs
        if lr:     self.config.lr = lr

        self._setup()
        return self._engine.train()

    def add_data(
        self,
        data_dir: str | Path,
        strategy: Literal["finetune", "retrain"] = "finetune",
    ) -> "PalmPipeline":
        """
        เพิ่มข้อมูล palm ใหม่ (สำหรับขยายพื้นที่ใช้งาน)

        Args:
            data_dir : directory ที่มี tiles/ และ masks/
            strategy : "finetune" → fine-tune บน data ใหม่อย่างเดียว
                       "retrain"  → merge กับ data เดิม แล้ว retrain ทั้งหมด

        Returns:
            self (chain calls ได้)
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"ไม่พบ directory: {data_dir}")

        if strategy == "retrain":
            self._extra_data_dirs.append(data_dir)
            self._engine = None   # reset เพื่อ rebuild ด้วย data ใหม่
            print(f"  add_data (retrain): {data_dir.name} จะถูก merge เมื่อ fit()")
        else:
            # finetune: fine-tune บน data ใหม่โดยตรง
            self._setup()
            print(f"\n[PalmPipeline] Fine-tuning on new data: {data_dir.name}")
            new_config = GeoAIConfig(
                **{k: v for k, v in self.config.__dict__.items()
                   if k not in ("data_dir", "save_dir")}
            )
            new_config.data_dir = data_dir
            new_config.save_dir = Path(self.config.save_dir) / "finetune_new"
            new_loaders = build_dataloaders(new_config)
            self._engine.train_loader = new_loaders.get("train", self._engine.train_loader)
            self._engine.fine_tune(epochs=30, lr=self.config.lr * 0.1)

        return self

    def fine_tune(
        self,
        epochs: int = 30,
        lr:     float = 1e-5,
        freeze_encoder_epochs: int = 5,
    ) -> list[dict]:
        """Fine-tune model ที่เทรนแล้ว (two-phase)."""
        self._setup()
        return self._engine.fine_tune(
            epochs = epochs,
            lr = lr,
            freeze_encoder_epochs = freeze_encoder_epochs,
        )

    def evaluate(
        self,
        ckpt_path: Optional[str | Path] = None,
    ) -> dict:
        """ประเมินผลบน test set."""
        self._setup()
        if "test" not in self._loaders:
            print("  Warning: ไม่พบ test/ — ใช้ val/ แทน")
        test_loader = self._loaders.get("test", self._loaders.get("val"))
        return self._engine.evaluate(test_loader, ckpt_path)

    def predict(
        self,
        tile: "str | Path | np.ndarray",
        ckpt_path: Optional[str | Path] = None,
    ) -> "np.ndarray | dict":
        """
        Predict บน single tile.

        Args:
            tile      : path ไปยัง .npy file หรือ numpy array (C, H, W)
            ckpt_path : โหลด checkpoint (optional)

        Returns:
            ตาม config.output_mode: "probability" | "binary" | "both"
        """
        self._setup()

        if isinstance(tile, (str, Path)):
            tile = np.load(tile).astype(np.float32)

        return self._engine.predict_tile(tile, ckpt_path)

    def visualize(
        self,
        n:          int = 5,
        save_dir:   Optional[str | Path] = None,
        ckpt_path:  Optional[str | Path] = None,
    ) -> None:
        """
        แสดง/บันทึก 4-panel figure: RGB / True Mask / Predicted / TP-FP-FN

        Args:
            n        : จำนวน sample ที่แสดง
            save_dir : directory สำหรับบันทึกภาพ (None = แสดงเท่านั้น)
            ckpt_path: โหลด checkpoint (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("  Error: ต้องติดตั้ง matplotlib → pip install matplotlib")
            return

        self._setup()
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self._engine.model.load_state_dict(ckpt["model"])

        loader = self._loaders.get("test", self._loaders.get("val"))
        if loader is None:
            print("  Error: ไม่พบ test/val loader")
            return

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        model  = self._engine.model
        device = self._engine.device
        model.eval()
        model.to(device)

        count = 0
        with torch.no_grad():
            for imgs, masks in loader:
                for i in range(imgs.shape[0]):
                    if count >= n:
                        break

                    img  = imgs[i].numpy()    # (C, H, W)
                    mask = masks[i].numpy()   # (H, W)
                    pred_prob = model(imgs[i:i+1].to(device)).squeeze().cpu().numpy()
                    pred_bin  = (pred_prob > self.config.threshold).astype(np.uint8)

                    # RGB: bands index 9,10,11 = GSat R,G,B
                    rgb = img[9:12].transpose(1, 2, 0)
                    rgb = np.clip(rgb, 0, 1)

                    # TP/FP/FN overlay
                    overlay = np.zeros((*mask.shape, 3))
                    overlay[..., 1] = ((pred_bin == 1) & (mask == 1)).astype(float)  # TP=green
                    overlay[..., 0] = ((pred_bin == 1) & (mask == 0)).astype(float)  # FP=red
                    overlay[..., 2] = ((pred_bin == 0) & (mask == 1)).astype(float)  # FN=blue

                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                    axes[0].imshow(rgb);           axes[0].set_title("RGB (GSat)")
                    axes[1].imshow(mask, cmap="Greens"); axes[1].set_title("True Mask")
                    axes[2].imshow(pred_prob, cmap="RdYlGn", vmin=0, vmax=1)
                    axes[2].set_title("Predicted (prob)")
                    axes[3].imshow(overlay);       axes[3].set_title("TP/FP/FN")

                    patches = [
                        mpatches.Patch(color="green", label="TP"),
                        mpatches.Patch(color="red",   label="FP"),
                        mpatches.Patch(color="blue",  label="FN"),
                    ]
                    axes[3].legend(handles=patches, loc="lower right", fontsize=8)
                    for ax in axes:
                        ax.axis("off")

                    plt.tight_layout()
                    if save_dir:
                        path = save_dir / f"visualization_{count:03d}.png"
                        plt.savefig(path, dpi=150, bbox_inches="tight")
                        print(f"  Saved: {path}")
                    else:
                        plt.show()
                    plt.close()
                    count += 1

                if count >= n:
                    break

        print(f"  Visualized {count} samples")

    @property
    def best_checkpoint(self) -> Optional[Path]:
        """Path ไปยัง best checkpoint."""
        p = Path(self.config.save_dir) / "vitpalm_best.pth"
        return p if p.exists() else None
