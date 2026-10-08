"""
tests/test_dataset.py
=====================
Tests the PalmDataset, Augmentation pipeline, and DataLoader.
Also calculates the class balance across the training set to find `pos_weight`.
"""

import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from poc.dataset import PalmDataset
from poc.augmentation import get_training_augmentation, get_validation_augmentation

def test_dataset_and_dataloader():
    train_dir = os.path.join(PROJECT_ROOT, "data", "processed", "split", "train")
    if not os.path.exists(train_dir):
        print("Skipping dataset test because split data doesn't exist yet.")
        return

    # Initialize Dataset
    aug = get_training_augmentation()
    ds = PalmDataset(train_dir, augmentation=aug)
    print(f"Dataset Size (Train): {len(ds)}")
    
    assert len(ds) > 0, "Dataset is empty"
    
    # Test __getitem__
    img, mask = ds[0]
    print(f"Sample 0 - Image shape: {img.shape}, dtype: {img.dtype}")
    print(f"Sample 0 - Mask shape: {mask.shape}, dtype: {mask.dtype}")
    
    assert img.shape == (12, 256, 256)
    assert mask.shape == (256, 256)
    assert not torch.isnan(img).any()
    assert not torch.isnan(mask).any()

    # Test DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch_img, batch_mask = next(iter(loader))
    
    print(f"Batch - Image shape: {batch_img.shape}")
    print(f"Batch - Mask shape: {batch_mask.shape}")
    
    assert batch_img.shape == (4, 12, 256, 256)
    assert batch_mask.shape == (4, 256, 256)

def calculate_class_balance():
    train_dir = os.path.join(PROJECT_ROOT, "data", "processed", "split", "train")
    if not os.path.exists(train_dir):
        return

    ds = PalmDataset(train_dir, augmentation=None)
    
    total_palm = 0
    total_bg = 0
    
    for i in range(len(ds)):
        _, mask = ds[i]
        total_palm += int(mask.sum().item())
        total_bg += int((1 - mask).sum().item())
        
    print("\n--- Class Balance (Train Set) ---")
    print(f"Total Palm Pixels: {total_palm:,}")
    print(f"Total Background Pixels: {total_bg:,}")
    
    if total_palm > 0:
        pos_weight = total_bg / total_palm
        print(f"Recommended BCE pos_weight: {pos_weight:.2f}")
    else:
        print("Warning: No palm pixels found in train set!")

if __name__ == "__main__":
    test_dataset_and_dataloader()
    calculate_class_balance()
    print("\nAll tests passed!")
