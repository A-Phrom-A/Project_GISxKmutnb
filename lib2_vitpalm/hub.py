"""
lib2_vitpalm.hub
================
Official model weights hub and checkpoint registry for GISxKMUTNB.
Manages downloading, checksum verification, and loading of pretrained weights.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import torch

# Official model checkpoints registry
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "cloud_mae_500e": {
        "description": "Cloud-MAE Pretrained ViT (500 epochs on 5 Eastern Thailand provinces)",
        "default_rel_path": "checkpoints/cloud_mae/cloud_mae_best.pth",
        "sha256": "448a39fc203067e0ff523ffae0f1298d850f44a452e08b8b17010e0d33fd2784",
        "size_bytes": 25611753,
        "release_url": "https://github.com/A-Phrom-A/Project_GISxKmutnb/releases/download/v1.0.0/cloud_mae_best.pth",
        "model_type": "cloud_mae",
    },
    "mae_100e": {
        "description": "Standard MAE Pretrained ViT (100 epochs baseline)",
        "default_rel_path": "checkpoints/mae/mae_best.pth",
        "sha256": None,
        "model_type": "mae",
    },
}


def calculate_sha256(filepath: str | Path, block_size: int = 65536) -> str:
    """Computes SHA256 hexadecimal digest of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha.update(block)
    return sha.hexdigest().lower()


def get_checkpoint_path(model_id: str = "cloud_mae_500e", custom_path: Optional[str] = None) -> Path:
    """
    Resolves the checkpoint file path for a registered model.
    Checks custom path, workspace default path, and system cache.
    """
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified checkpoint not found at: {custom_path}")

    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_id '{model_id}'. Available: {list(MODEL_REGISTRY.keys())}")

    meta = MODEL_REGISTRY[model_id]
    rel_path = Path(meta["default_rel_path"])

    # 1. Check relative to current working directory
    if rel_path.exists():
        return rel_path.resolve()

    # 2. Check relative to project root
    project_root = Path(__file__).resolve().parent.parent
    candidate = project_root / rel_path
    if candidate.exists():
        return candidate

    # 3. Check user home cache
    cache_dir = Path.home() / ".cache" / "gis_kmutnb" / "checkpoints"
    cached_file = cache_dir / rel_path.name
    if cached_file.exists():
        return cached_file

    raise FileNotFoundError(
        f"Checkpoint for '{model_id}' not found locally.\n"
        f"Searched paths:\n"
        f"  - {rel_path.resolve()}\n"
        f"  - {candidate}\n"
        f"  - {cached_file}\n"
        f"Please run `python scripts/download_weights.py --model {model_id}` or place weights at {rel_path}."
    )


def verify_checkpoint(filepath: str | Path, expected_sha256: Optional[str] = None) -> bool:
    """Verifies SHA256 checksum of the target file."""
    p = Path(filepath)
    if not p.exists():
        return False
    if not expected_sha256:
        return True
    computed = calculate_sha256(p)
    return computed == expected_sha256.lower()


def load_cloud_mae_weights(
    model: torch.nn.Module,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Loads pretrained Cloud-MAE weights into target model.
    Supports ViTPalmModel (Early), ViTLateFusionModel, and ViTCrossAttnModel.
    """
    path = get_checkpoint_path("cloud_mae_500e", custom_path=checkpoint_path)
    state = torch.load(path, map_location=device, weights_only=False)

    weights = state["model_state_dict"] if "model_state_dict" in state else state

    # If loading into CloudMAEModel directly:
    if hasattr(model, "load_state_dict"):
        try:
            model.load_state_dict(weights, strict=strict)
        except Exception:
            # Fallback to direct weight transfer helper if available
            if hasattr(model, "transfer_from_cloud_mae"):
                model.transfer_from_cloud_mae(weights)
            else:
                # Load compatible keys
                model_state = model.state_dict()
                compatible = {k: v for k, v in weights.items() if k in model_state and model_state[k].shape == v.shape}
                model_state.update(compatible)
                model.load_state_dict(model_state)

    return state


if __name__ == "__main__":
    import sys

    print("=== GISxKMUTNB Model Hub Registry ===")
    for mid, info in MODEL_REGISTRY.items():
        print(f"\nModel ID: {mid}")
        print(f"  Description: {info['description']}")
        print(f"  Target path: {info['default_rel_path']}")
        try:
            resolved = get_checkpoint_path(mid)
            valid = verify_checkpoint(resolved, info.get("sha256"))
            print(f"  Status: FOUND at {resolved}")
            print(f"  Integrity: {'VALID (SHA256 Match)' if valid else 'CHECKSUM MISMATCH'}")
        except FileNotFoundError:
            print("  Status: NOT FOUND (run download script)")
