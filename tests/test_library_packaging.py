"""
tests/test_library_packaging.py
===============================
Automated unit tests for gis-kmutnb library packaging, model hub,
reconstruction inference, and multi-modal models.
"""

import pytest
import torch
from pathlib import Path


def test_public_imports():
    """Verify that all major modules and exports are accessible."""
    import lib1_fusion
    import lib2_vitpalm
    import lib3_geoai
    from lib2_vitpalm import (
        CloudMAEModel,
        ViTLateFusionModel,
        ViTCrossAttnModel,
        ViTPalmModel,
        MAEModel,
    )
    from lib2_vitpalm.hub import MODEL_REGISTRY, get_checkpoint_path, verify_checkpoint

    assert lib1_fusion.__version__ == "1.0.0"
    assert lib2_vitpalm.__version__ == "1.0.0"
    assert "cloud_mae_500e" in MODEL_REGISTRY


def test_model_hub_registry():
    """Verify hub paths and SHA-256 verification logic."""
    from lib2_vitpalm.hub import MODEL_REGISTRY, get_checkpoint_path, verify_checkpoint

    info = MODEL_REGISTRY["cloud_mae_500e"]
    path = get_checkpoint_path("cloud_mae_500e")
    assert path.exists()

    valid = verify_checkpoint(path, info["sha256"])
    assert valid is True


def test_cloud_mae_reconstruct():
    """Verify Cloud-MAE reconstruct method on CPU."""
    from lib2_vitpalm import CloudMAEModel

    model = CloudMAEModel(s1_channels=2, s2_channels=7, embed_dim=256, decoder_dim=128).eval()
    s1 = torch.randn(1, 2, 256, 256)
    s2 = torch.randn(1, 7, 256, 256)
    mask = (torch.rand(1, 1, 256, 256) > 0.5).float()

    with torch.no_grad():
        recon = model.reconstruct(s1, s2, mask_pixel=mask)

    assert recon.shape == (1, 7, 256, 256)


def test_late_fusion_dual_input():
    """Verify ViTLateFusionModel supports both 12-band and 3-tensor inputs."""
    from lib2_vitpalm import ViTLateFusionModel

    model = ViTLateFusionModel().eval()
    s1 = torch.randn(1, 2, 256, 256)
    s2 = torch.randn(1, 7, 256, 256)
    gsat = torch.randn(1, 3, 256, 256)
    x12 = torch.cat([s1, s2, gsat], dim=1)

    with torch.no_grad():
        out_sep = model(s1, s2, gsat)
        out_12 = model(x12)

    assert out_sep.shape == (1, 256, 256)
    assert out_12.shape == (1, 256, 256)


def test_cross_attn_dual_input():
    """Verify ViTCrossAttnModel supports both 12-band and 3-tensor inputs."""
    from lib2_vitpalm import ViTCrossAttnModel

    model = ViTCrossAttnModel().eval()
    s1 = torch.randn(1, 2, 256, 256)
    s2 = torch.randn(1, 7, 256, 256)
    gsat = torch.randn(1, 3, 256, 256)
    x12 = torch.cat([s1, s2, gsat], dim=1)

    with torch.no_grad():
        out_sep = model(s1, s2, gsat)
        out_12 = model(x12)

    assert out_sep.shape == (1, 256, 256)
    assert out_12.shape == (1, 256, 256)


if __name__ == "__main__":
    test_public_imports()
    test_model_hub_registry()
    test_cloud_mae_reconstruct()
    test_late_fusion_dual_input()
    test_cross_attn_dual_input()
    print("All packaging & hub unit tests passed successfully!")
