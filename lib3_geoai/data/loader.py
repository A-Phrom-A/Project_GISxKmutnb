"""
lib3_geoai.data.loader
========================
build_dataloaders() — สร้าง DataLoader ครบทั้ง train/val/test
จาก GeoAIConfig เพียง object เดียว.
"""

from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional

from lib3_geoai.config import GeoAIConfig
from lib3_geoai.data.dataset import GeoDataset
from lib3_geoai.data.augmentation import get_augmentation


def build_dataloaders(
    config: GeoAIConfig,
    pretrain: bool = False,
    extra_data_dirs: Optional[list[str | Path]] = None,
) -> dict[str, DataLoader]:
    """
    สร้าง DataLoader สำหรับ train, val, test splits.

    Args:
        config          : GeoAIConfig object
        pretrain        : ถ้า True → ใช้ GeoDataset ใน pretrain mode (ไม่ต้องมี mask)
        extra_data_dirs : list ของ directories เพิ่มเติม (สำหรับ add_data())
                          จะ concat เข้ากับ train set

    Returns:
        dict ที่มี keys: "train", "val", "test"
        (ถ้าไม่มี folder → ข้ามไป ไม่ error)
    """
    data_dir = Path(config.data_dir)
    loaders  = {}

    splits = {
        "train": (data_dir / "train",  config.augment and not pretrain),
        "val":   (data_dir / "val",    False),
        "test":  (data_dir / "test",   False),
    }

    for split_name, (split_path, do_aug) in splits.items():
        if not split_path.exists():
            continue

        aug = get_augmentation("train" if do_aug else "val")

        dataset = GeoDataset(
            split_dir   = split_path,
            augmentation = aug,
            pretrain    = pretrain,
            in_channels = config.in_channels if config.in_channels != 12 else None,
        )

        # Extra data สำหรับ train set เท่านั้น
        if split_name == "train" and extra_data_dirs:
            from torch.utils.data import ConcatDataset
            extra_datasets = []
            for d in extra_data_dirs:
                d = Path(d)
                if d.exists():
                    extra_aug = get_augmentation("train") if do_aug else get_augmentation("val")
                    extra_ds  = GeoDataset(
                        split_dir    = d,
                        augmentation = extra_aug,
                        pretrain     = pretrain,
                        in_channels  = config.in_channels if config.in_channels != 12 else None,
                    )
                    extra_datasets.append(extra_ds)
                    print(f"  add_data: +{len(extra_ds)} samples from {d.name}")
            if extra_datasets:
                dataset = ConcatDataset([dataset] + extra_datasets)

        shuffle = (split_name == "train")
        loader  = DataLoader(
            dataset,
            batch_size  = config.batch_size,
            shuffle     = shuffle,
            num_workers = config.num_workers,
            pin_memory  = True,
            drop_last   = (split_name == "train"),
        )

        loaders[split_name] = loader
        n = len(dataset) if hasattr(dataset, "__len__") else "?"
        print(f"  {split_name:5s}: {n} samples → {len(loader)} batches")

    return loaders


def build_pretrain_loader(
    tile_dirs: list[str | Path],
    config: GeoAIConfig,
) -> DataLoader:
    """
    สร้าง DataLoader สำหรับ MAE pretraining.
    รับ tile directories หลายอัน (ไม่ต้องแบ่ง split).

    Args:
        tile_dirs : list ของ directories ที่มี tiles/ folder
        config    : GeoAIConfig

    Returns:
        DataLoader สำหรับ pretraining
    """
    from torch.utils.data import ConcatDataset

    datasets = []
    for d in tile_dirs:
        d = Path(d)
        if not d.exists():
            print(f"  Warning: {d} ไม่มี — ข้ามไป")
            continue
        ds = GeoDataset(split_dir=d, pretrain=True)
        datasets.append(ds)
        print(f"  pretrain data: {len(ds)} tiles from {d.name}")

    if not datasets:
        raise RuntimeError("ไม่พบ tile directories ที่ใช้ได้สำหรับ pretraining")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    loader   = DataLoader(
        combined,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    print(f"  pretrain total: {len(combined)} tiles → {len(loader)} batches")
    return loader
