"""
lib2_vitpalm.utils
====================
Shared utilities for lib2_vitpalm.

Includes reproducibility helpers (seed control) and common functions.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Sets seed for: Python random, NumPy, PyTorch CPU/CUDA,
    and enables deterministic cuDNN behavior.

    Args:
        seed: Integer seed value (default 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic behavior (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"  Seed set to {seed} (deterministic mode)")


def get_device() -> torch.device:
    """
    Auto-detect best available device.

    Returns:
        torch.device — 'cuda' if GPU available, else 'cpu'
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"  Device: {name} ({vram:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("  Device: CPU (no CUDA GPU detected)")
    return device
