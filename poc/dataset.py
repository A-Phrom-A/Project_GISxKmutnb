"""
poc/dataset.py
==============
PyTorch Dataset for Palm Detection.

Reads .npy tiles and masks from the disk, applies data augmentation, 
and yields PyTorch Tensors.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

class PalmDataset(Dataset):
    """
    Dataset for Oil Palm Segmentation.
    
    Args:
        split_dir (str): Path to the split directory (e.g. 'data/processed/split/train')
        augmentation (albumentations.Compose, optional): Augmentation pipeline.
    """
    def __init__(self, split_dir: str, augmentation: A.Compose = None):
        super().__init__()
        self.split_dir = split_dir
        self.augmentation = augmentation
        
        self.tile_dir = os.path.join(split_dir, "tiles")
        self.mask_dir = os.path.join(split_dir, "masks")
        
        # Load all tile file paths
        self.tile_paths = sorted(glob.glob(os.path.join(self.tile_dir, "*.npy")))
        
        # Validate that each tile has a corresponding mask
        self.samples = []
        for tp in self.tile_paths:
            basename = os.path.basename(tp) # tile_r0000_c0000.npy
            r_str = basename.split('_')[1]
            c_str = basename.split('_')[2]
            mask_name = f"mask_{r_str}_{c_str}"
            mp = os.path.join(self.mask_dir, mask_name)
            
            if os.path.exists(mp):
                self.samples.append((tp, mp))
                
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid tile/mask pairs found in {split_dir}")
            
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile_path, mask_path = self.samples[idx]
        
        # Load arrays
        image = np.load(tile_path)  # (12, 256, 256) float32
        mask = np.load(mask_path)   # (256, 256) uint8
        
        # Albumentations expects HWC for images
        image = image.transpose(1, 2, 0) # (256, 256, 12)
        
        if self.augmentation is not None:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Convert back to CHW for PyTorch
        image = image.transpose(2, 0, 1) # (12, 256, 256)
        
        # Convert to torch Tensors
        # Ensure contiguous memory and correct dtypes
        image_t = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).float() # Loss function usually expects float
        
        return image_t, mask_t

