"""
lib3_geoai.data.augmentation
==============================
Augmentation pipelines สำหรับ satellite imagery segmentation.

ข้อสังเกต: ไม่ใช้ RandomBrightnessContrast แบบ strong เพราะ
band ใน satellite imagery แทนค่าทางกายภาพ (reflectance/backscatter)
ไม่ใช่ค่าสี RGB ทั่วไป
"""

from __future__ import annotations
import albumentations as A


def get_training_augmentation() -> A.Compose:
    """
    Augmentation pipeline สำหรับ training set.
    - Spatial: flip, rotate (sync กับ mask อัตโนมัติ)
    - Pixel:   noise เล็กน้อย + multiplicative (เลียนแบบ sensor variability)
    """
    return A.Compose([
        # Spatial — sync กับ mask
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),

        # Pixel — image only
        A.GaussNoise(std_range=(0.005, 0.02), p=0.2),
        A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.2, elementwise=True),
    ])


def get_validation_augmentation() -> A.Compose:
    """Augmentation สำหรับ validation/test — ไม่มี augmentation."""
    return A.Compose([])


def get_augmentation(mode: str = "train") -> A.Compose:
    """
    Helper สำหรับขอ augmentation pipeline ตาม mode.

    Args:
        mode: "train" | "val" | "test"

    Returns:
        albumentations.Compose pipeline
    """
    if mode == "train":
        return get_training_augmentation()
    elif mode in ("val", "test", "inference"):
        return get_validation_augmentation()
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: 'train', 'val', 'test'")
