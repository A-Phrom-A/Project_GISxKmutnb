"""
lib3_geoai.pipeline.custom
============================
CustomPipeline — pipeline สำหรับ task ทั่วไป (ไม่จำกัดแค่ palm)

Use Case 2: เทรนกับข้อมูลและ task ใหม่
  - กำหนด in_channels ได้เอง
  - กำหนด pos_weight สำหรับ class imbalance ได้เอง
  - output_mode เลือกได้ (binary / probability / both)

Usage:
    from lib3_geoai import GeoAI

    model = GeoAI.custom(
        data_dir    = "data/my_dataset/split",
        in_channels = 7,           # เช่น S2 เท่านั้น
        fusion      = "early",
        pos_weight  = 3.0,
        output_mode = "both",
    )
    model.fit(epochs=100)
    result = model.predict("tile.npy")
    print(result["probability"])
    print(result["binary"])
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Literal

from lib3_geoai.config import GeoAIConfig
from lib3_geoai.data.loader import build_dataloaders
from lib3_geoai.models.factory import build_model, model_summary
from lib3_geoai.training.engine import TrainingEngine


class CustomPipeline:
    """
    Pipeline สำหรับ task ทั่วไป — ยืดหยุ่นกว่า PalmPipeline.

    Args:
        config     : GeoAIConfig (ถ้าไม่ระบุจะสร้างจาก kwargs)
        pretrained : path ไปยัง MAE checkpoint (optional)

    ความแตกต่างจาก PalmPipeline:
        - ไม่ hardcode pos_weight
        - ไม่ assume 12 bands
        - output_mode เลือกได้ชัดเจน
    """

    def __init__(
        self,
        config:     Optional[GeoAIConfig] = None,
        pretrained: Optional[str | Path]  = None,
    ):
        self.config     = config or GeoAIConfig()
        self.pretrained = Path(pretrained) if pretrained else None
        self._engine: Optional[TrainingEngine] = None
        self._loaders: dict = {}

    # ── Setup ────────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        if self._engine is not None:
            return

        print(f"\n[CustomPipeline] Setup")
        print(f"  task        = segmentation")
        print(f"  fusion      = {self.config.fusion}")
        print(f"  in_channels = {self.config.in_channels}")
        print(f"  output_mode = {self.config.output_mode}")
        print(f"  data_dir    = {self.config.data_dir}")

        print("\n[CustomPipeline] Building DataLoaders...")
        self._loaders = build_dataloaders(self.config)

        if "train" not in self._loaders:
            raise RuntimeError(f"ไม่พบ train/ ใน {self.config.data_dir}")

        print(f"\n[CustomPipeline] Building Model...")
        model = build_model(self.config)
        model_summary(model, self.config.fusion)

        self._engine = TrainingEngine(
            model        = model,
            train_loader = self._loaders["train"],
            val_loader   = self._loaders.get("val", self._loaders["train"]),
            config       = self.config,
        )

        if self.pretrained and self.pretrained.exists():
            print(f"\n[CustomPipeline] Loading pretrained: {self.pretrained.name}")
            self._engine.load_pretrained_weights(self.pretrained)

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(
        self,
        epochs: Optional[int] = None,
        lr:     Optional[float] = None,
    ) -> list[dict]:
        """เทรน model."""
        if epochs: self.config.epochs = epochs
        if lr:     self.config.lr = lr
        self._setup()
        return self._engine.train()

    def fine_tune(
        self,
        epochs:               int   = 30,
        lr:                   float = 1e-5,
        freeze_encoder_epochs: int  = 5,
    ) -> list[dict]:
        """Fine-tune แบบ two-phase."""
        self._setup()
        return self._engine.fine_tune(epochs, lr, freeze_encoder_epochs)

    def evaluate(
        self,
        ckpt_path: Optional[str | Path] = None,
    ) -> dict:
        """ประเมินผลบน test set."""
        self._setup()
        test_loader = self._loaders.get("test", self._loaders.get("val"))
        return self._engine.evaluate(test_loader, ckpt_path)

    def predict(
        self,
        tile:      "str | Path | np.ndarray",
        ckpt_path: Optional[str | Path] = None,
    ) -> "np.ndarray | dict":
        """
        Predict บน single tile.

        Returns:
            ตาม config.output_mode:
              "probability" → (H, W) float [0, 1]
              "binary"      → (H, W) uint8 {0, 1}
              "both"        → {"probability": ..., "binary": ...}
        """
        self._setup()

        if isinstance(tile, (str, Path)):
            tile = np.load(tile).astype(np.float32)

        return self._engine.predict_tile(tile, ckpt_path)

    @property
    def best_checkpoint(self) -> Optional[Path]:
        p = Path(self.config.save_dir) / "vitpalm_best.pth"
        return p if p.exists() else None
