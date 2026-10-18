"""
projects.cross_band_recovery.training.dataset
============================================
PyTorch Dataset and DataLoader for Paired Sentinel-1 SAR and Sentinel-2 Optical Tiles.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "tiles"


class CrossBandTileDataset(Dataset):
    """
    Loads pre-tiled Sentinel-1 and Sentinel-2 paired numpy archives (.npz).
    Each archive contains:
      - s2_raw: (7, 128, 128) float32 in [0, 1]
      - s1_lee_db: (2, 128, 128) float32 in dB [-35, +5]
      - s1_raw_db: (2, 128, 128) float32 in dB [-35, +5]
      - bbox: (4,) float32
    """
    def __init__(
        self,
        split_dir: Path,
        use_lee_filter: bool = True,
        augment: bool = False,
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        self.split_dir = Path(split_dir)
        self.use_lee_filter = use_lee_filter
        self.augment = augment
        
        self.files = sorted(list(self.split_dir.glob("*.npz")))
        if max_samples is not None and len(self.files) > max_samples:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(self.files), max_samples, replace=False)
            self.files = [self.files[i] for i in sorted(indices)]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filepath = self.files[idx]
        with np.load(filepath) as data:
            s2 = data["s2_raw"].astype(np.float32)  # (7, 128, 128)
            s1_key = "s1_lee_db" if self.use_lee_filter else "s1_raw_db"
            s1 = data[s1_key].astype(np.float32)    # (2, 128, 128)
            bbox = data["bbox"].astype(np.float32)

        # Spatial data augmentations (flips & rotations)
        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                s2 = np.flip(s2, axis=2).copy()
                s1 = np.flip(s1, axis=2).copy()
            # Vertical flip
            if np.random.rand() > 0.5:
                s2 = np.flip(s2, axis=1).copy()
                s1 = np.flip(s1, axis=1).copy()
            # Rotations (0, 90, 180, 270 deg)
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                s2 = np.rot90(s2, k=rot_k, axes=(1, 2)).copy()
                s1 = np.rot90(s1, k=rot_k, axes=(1, 2)).copy()

        return {
            "s2": torch.from_numpy(s2),
            "s1": torch.from_numpy(s1),
            "tile_id": filepath.stem,
            "bbox": torch.from_numpy(bbox)
        }


def create_dataloader(
    split: str,
    batch_size: int = 16,
    shuffle: Optional[bool] = None,
    augment: Optional[bool] = None,
    use_lee_filter: bool = True,
    max_samples: Optional[int] = None,
    num_workers: int = 0
) -> DataLoader:
    """
    Convenience factory to create DataLoader for train, val, or test splits.
    """
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    is_train = (split == "train")
    if shuffle is None:
        shuffle = is_train
    if augment is None:
        augment = is_train

    dataset = CrossBandTileDataset(
        split_dir=split_dir,
        use_lee_filter=use_lee_filter,
        augment=augment,
        max_samples=max_samples
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train
    )
