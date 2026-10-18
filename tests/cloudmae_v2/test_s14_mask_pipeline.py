"""
tests/cloudmae_v2/test_s14_mask_pipeline.py
===========================================
Unit tests for Phase 1 - Mask Pipeline (CHK 1.4a)
Verifies:
1. hard_mask subset of dilated envelope
2. soft_mask values bounded strictly in [0.0, 1.0]
3. dimensions exactly match input tile
4. type_mask discrete categories in {0, 1, 2, 3}
5. cloud coverage binning logic conforms to <=20% (margin <=25%)
"""

import numpy as np
import pytest
import torch
from lib2_vitpalm.mask_pipeline import CloudMaskPipeline, classify_cloud_bin, compute_spectral_cloud_prob


def test_classify_cloud_bin():
    # Test boundary constraints
    assert classify_cloud_bin(1.5) == ("0-2% (Clear)", False)
    assert classify_cloud_bin(3.0) == ("2-5%", True)
    assert classify_cloud_bin(8.5) == ("5-10%", True)
    assert classify_cloud_bin(15.0) == ("10-20%", True)
    assert classify_cloud_bin(22.0) == ("20-25% (Margin)", True)
    assert classify_cloud_bin(25.1) == (">25% (Rejected)", False)
    assert classify_cloud_bin(50.0) == (">25% (Rejected)", False)


def test_chk_14a_mask_properties():
    pipeline = CloudMaskPipeline(prob_threshold=0.40, dilation_radius_px=3)
    
    # Create synthetic optical tile (7, 256, 256)
    np.random.seed(42)
    s2_optical = np.random.uniform(500, 2500, size=(7, 256, 256)).astype(np.float32)
    
    # Inject bright cloud region in center
    s2_optical[:, 80:140, 80:140] = 5000.0  # Bright cloud reflectance
    
    # Synthetic SCL with cloud and shadow
    scl = np.full((256, 256), 4, dtype=np.int32) # vegetation
    scl[80:140, 80:140] = 9  # High prob cloud
    scl[70:80, 80:140] = 3   # Cloud shadow
    
    result = pipeline.generate(s2_optical, scl=scl)
    
    hard_mask = result["hard_mask"]
    soft_mask = result["soft_mask"]
    type_mask = result["type_mask"]
    
    # 1. Output dimensions match tile
    assert hard_mask.shape == (256, 256)
    assert soft_mask.shape == (256, 256)
    assert type_mask.shape == (256, 256)
    
    # 2. soft_mask in [0.0, 1.0]
    assert np.all(soft_mask >= 0.0)
    assert np.all(soft_mask <= 1.0)
    
    # 3. hard_mask is binary {0, 1}
    assert set(np.unique(hard_mask)).issubset({0, 1})
    
    # 4. type_mask in {0, 1, 2, 3}
    assert set(np.unique(type_mask)).issubset({0, 1, 2, 3})
    
    # 5. Cloud center must be detected
    assert np.all(hard_mask[85:135, 85:135] == 1)
    
    # 6. Dilation: region around cloud must be enveloped
    assert hard_mask[79, 80] == 1  # dilated boundary pixel
    
    # 7. Coverage calculation
    assert result["coverage_pct"] > 0.0
    assert isinstance(result["coverage_bin"], str)
    assert isinstance(result["is_valid_for_repair"], bool)
