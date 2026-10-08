"""
lib3_geoai.data.dataset
========================
GeoDataset — generalized dataset สำหรับ GEO + AI tasks

รองรับ:
  - Supervised mode: อ่านคู่ (tile, mask) สำหรับ segmentation
  - Pretrain mode : อ่านแค่ tile (ไม่ต้องมี mask) สำหรับ MAE pretraining
  - Custom bands  : กำหนด in_channels ได้ (ไม่จำกัดแค่ 12 bands)
"""

from __future__ import annotations

import glob
import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Callable


class GeoDataset(Dataset):
    """
    Dataset สำหรับ GEO satellite tile segmentation หรือ pretraining.

    โครงสร้าง directory ที่รองรับ:
        split_dir/
        ├── tiles/
        │   ├── tile_r0000_c0000.npy   (C, H, W) float32
        │   └── ...
        └── masks/                     (optional — ไม่ต้องมีสำหรับ pretrain)
            ├── mask_r0000_c0000.npy   (H, W) uint8
            └── ...

    Args:
        split_dir   : path ไปยัง split directory (train/ val/ หรือ test/)
        augmentation: albumentations Compose pipeline (optional)
        pretrain    : ถ้า True → ไม่โหลด mask, return แค่ tile tensor
        in_channels : จำนวน channels ที่ต้องการ (None = ใช้ทั้งหมด)
    """

    def __init__(
        self,
        split_dir:   str | Path,
        augmentation: Optional[Callable] = None,
        pretrain:    bool = False,
        in_channels: Optional[int] = None,
    ):
        super().__init__()
        self.split_dir   = Path(split_dir)
        self.augmentation = augmentation
        self.pretrain    = pretrain
        self.in_channels = in_channels

        self.tile_dir = self.split_dir / "tiles"
        self.mask_dir = self.split_dir / "masks"

        if not self.tile_dir.exists():
            raise FileNotFoundError(f"Tile directory not found: {self.tile_dir}")

        tile_paths = sorted(self.tile_dir.glob("*.npy"))
        if not tile_paths:
            raise RuntimeError(f"No .npy tiles found in: {self.tile_dir}")

        if pretrain:
            # Pretrain mode: ไม่ต้องการ mask
            self.samples = [(tp, None) for tp in tile_paths]
        else:
            # Supervised mode: จับคู่ tile ↔ mask
            self.samples = []
            missing_masks = []
            for tp in tile_paths:
                name = tp.stem  # tile_r0000_c0000
                parts = name.split("_")
                if len(parts) >= 3:
                    r_str, c_str = parts[1], parts[2]
                    mp = self.mask_dir / f"mask_{r_str}_{c_str}.npy"
                else:
                    mp = self.mask_dir / (name.replace("tile", "mask") + ".npy")

                if mp.exists():
                    self.samples.append((tp, mp))
                else:
                    missing_masks.append(tp.name)

            if not self.samples:
                raise RuntimeError(
                    f"No tile/mask pairs found in {split_dir}. "
                    f"Check mask directory: {self.mask_dir}"
                )
            if missing_masks:
                print(f"  Warning: {len(missing_masks)} tiles have no matching mask (skipped)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        tile_path, mask_path = self.samples[idx]

        # โหลด tile
        image = np.load(tile_path).astype(np.float32)  # (C, H, W)

        # ตัด channels ถ้าระบุ in_channels
        if self.in_channels is not None and image.shape[0] != self.in_channels:
            image = image[:self.in_channels]

        if self.pretrain:
            # Pretrain mode: return แค่ image
            image_t = torch.from_numpy(np.ascontiguousarray(image))
            return image_t

        # Supervised mode: โหลด mask
        mask = np.load(mask_path).astype(np.float32)  # (H, W)

        # Albumentations รับ (H, W, C)
        if self.augmentation is not None:
            image_hwc = image.transpose(1, 2, 0)       # (H, W, C)
            aug = self.augmentation(image=image_hwc, mask=mask)
            image = aug["image"].transpose(2, 0, 1)    # (C, H, W)
            mask  = aug["mask"]

        image_t = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask_t  = torch.from_numpy(np.ascontiguousarray(mask)).float()
        return image_t, mask_t

    def __repr__(self) -> str:
        mode = "pretrain" if self.pretrain else "supervised"
        return (
            f"GeoDataset({mode}, "
            f"n={len(self)}, "
            f"dir='{self.split_dir.name}')"
        )
