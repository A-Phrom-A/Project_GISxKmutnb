"""
tests/test_lib1_fusion.py
==========================
Unit tests + Integration test สำหรับ lib1_fusion

ทดสอบ:
  1. Processors: S1Processor, S2Processor, GSatProcessor
  2. Utils: normalization functions, validation, stacking
  3. Fusion Strategies: early_fusion, LateFusion, CrossModalAttention
  4. Integration: raw arrays → fused (12, H, W)
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─── 1. Test Utils ────────────────────────────────────────────────────────────

def test_normalize_minmax():
    from lib1_fusion.utils import normalize_minmax

    arr = np.array([0.0, 50.0, 100.0, 200.0, 255.0], dtype=np.float32)
    out = normalize_minmax(arr, vmin=0.0, vmax=255.0)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert abs(out[0] - 0.0) < 1e-6
    assert abs(out[-1] - 1.0) < 1e-6
    print("  [OK] normalize_minmax")


def test_normalize_minmax_constant():
    from lib1_fusion.utils import normalize_minmax

    arr = np.full((10,), 5.0, dtype=np.float32)
    out = normalize_minmax(arr)
    assert np.all(out == 0.0), "Constant array should return all zeros"
    print("  [OK] normalize_minmax (constant input)")


def test_normalize_sar_db():
    from lib1_fusion.utils import normalize_sar_db

    arr = np.array([-30.0, -25.0, -12.5, 0.0, 5.0], dtype=np.float32)
    out = normalize_sar_db(arr, db_min=-25.0, db_max=0.0)
    assert out.dtype == np.float32
    assert abs(out[1] - 0.0) < 1e-6, f"Expected 0.0 at -25dB, got {out[1]}"
    assert abs(out[3] - 1.0) < 1e-6, f"Expected 1.0 at  0dB, got {out[3]}"
    assert out[0] == 0.0, "Below db_min should clip to 0"
    assert out[4] == 1.0, "Above db_max should clip to 1"
    print("  [OK] normalize_sar_db")


def test_normalize_s2_reflectance():
    from lib1_fusion.utils import normalize_s2_reflectance

    arr = np.array([0, 2500, 5000, 10000, 15000], dtype=np.float32)
    out = normalize_s2_reflectance(arr)
    assert abs(out[0] - 0.0) < 1e-6
    assert abs(out[2] - 0.5) < 1e-6
    assert abs(out[3] - 1.0) < 1e-6
    assert out[4] == 1.0, "Above 10000 should clip to 1"
    print("  [OK] normalize_s2_reflectance")


def test_validate_spatial_shape():
    from lib1_fusion.utils import validate_spatial_shape

    a = np.zeros((100, 200))
    b = np.zeros((100, 200))
    c = np.zeros((100, 200))
    result = validate_spatial_shape([a, b, c], ["A", "B", "C"])
    assert result == (100, 200)
    print("  [OK] validate_spatial_shape (matching)")


def test_validate_spatial_shape_mismatch():
    from lib1_fusion.utils import validate_spatial_shape

    a = np.zeros((100, 200))
    b = np.zeros((100, 300))
    try:
        validate_spatial_shape([a, b], ["A", "B"])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "mismatch" in str(e).lower()
    print("  [OK] validate_spatial_shape (mismatch raises ValueError)")


def test_stack_bands():
    from lib1_fusion.utils import stack_bands

    a = np.ones((64, 64), dtype=np.float32)
    b = np.ones((64, 64), dtype=np.float32) * 2
    out = stack_bands(a, b)
    assert out.shape == (2, 64, 64)
    assert out.dtype == np.float32
    assert out[0, 0, 0] == 1.0
    assert out[1, 0, 0] == 2.0
    print("  [OK] stack_bands")


# ─── 2. Test Processors ──────────────────────────────────────────────────────

def test_s1_processor():
    from lib1_fusion import S1Processor

    proc = S1Processor(db_min=-25.0, db_max=0.0)
    H, W = 128, 128
    vv = np.random.uniform(-30, 5, (H, W)).astype(np.float32)
    vh = np.random.uniform(-30, 5, (H, W)).astype(np.float32)

    out = proc.process(vv, vh)
    assert out.shape == (2, H, W), f"Expected (2,{H},{W}), got {out.shape}"
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    print("  [OK] S1Processor.process")


def test_s2_processor():
    from lib1_fusion import S2Processor

    proc = S2Processor()
    H, W = 128, 128
    bands = [np.random.uniform(0, 10000, (H, W)).astype(np.float32) for _ in range(7)]

    out = proc.process(*bands)
    assert out.shape == (7, H, W), f"Expected (7,{H},{W}), got {out.shape}"
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    print("  [OK] S2Processor.process")


def test_gsat_processor():
    from lib1_fusion import GSatProcessor

    proc = GSatProcessor()
    H, W = 128, 128

    # uint8 input
    r = np.random.randint(0, 256, (H, W), dtype=np.uint8)
    g = np.random.randint(0, 256, (H, W), dtype=np.uint8)
    b = np.random.randint(0, 256, (H, W), dtype=np.uint8)

    out = proc.process(r, g, b)
    assert out.shape == (3, H, W), f"Expected (3,{H},{W}), got {out.shape}"
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    print("  [OK] GSatProcessor.process (uint8)")

    # float input
    r_f = np.random.uniform(0, 1, (H, W)).astype(np.float32)
    g_f = np.random.uniform(0, 1, (H, W)).astype(np.float32)
    b_f = np.random.uniform(0, 1, (H, W)).astype(np.float32)
    out_f = proc.process(r_f, g_f, b_f)
    assert out_f.shape == (3, H, W)
    assert out_f.min() >= 0.0
    assert out_f.max() <= 1.0
    print("  [OK] GSatProcessor.process (float)")


# ─── 3. Test Fusion Strategies ───────────────────────────────────────────────

def test_early_fusion():
    from lib1_fusion.fusion.early_fusion import early_fusion

    H, W = 64, 64
    s1   = np.random.rand(2, H, W).astype(np.float32)
    s2   = np.random.rand(7, H, W).astype(np.float32)
    gsat = np.random.rand(3, H, W).astype(np.float32)

    out = early_fusion(s1, s2, gsat)
    assert out.shape == (12, H, W), f"Expected (12,{H},{W}), got {out.shape}"
    assert out.dtype == np.float32
    # Verify band order: S1 first, then S2, then Gsat
    assert np.allclose(out[:2], s1)
    assert np.allclose(out[2:9], s2)
    assert np.allclose(out[9:12], gsat)
    print("  [OK] early_fusion")


def test_early_fusion_shape_mismatch():
    from lib1_fusion.fusion.early_fusion import early_fusion

    s1   = np.random.rand(2, 64, 64).astype(np.float32)
    s2   = np.random.rand(7, 32, 32).astype(np.float32)   # different size!
    gsat = np.random.rand(3, 64, 64).astype(np.float32)

    try:
        early_fusion(s1, s2, gsat)
        assert False, "Should have raised ValueError"
    except (ValueError, AssertionError):
        pass
    print("  [OK] early_fusion (shape mismatch raises error)")


def test_early_fusion_wrong_bands():
    from lib1_fusion.fusion.early_fusion import early_fusion

    s1   = np.random.rand(3, 64, 64).astype(np.float32)   # wrong! should be 2
    s2   = np.random.rand(7, 64, 64).astype(np.float32)
    gsat = np.random.rand(3, 64, 64).astype(np.float32)

    try:
        early_fusion(s1, s2, gsat)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    print("  [OK] early_fusion (wrong band count raises error)")


def test_late_fusion_concat():
    from lib1_fusion.fusion.late_fusion import LateFusion, LateFusionConfig

    config = LateFusionConfig(merge_strategy="concat")
    lf = LateFusion(config)

    H, W = 64, 64
    f1 = np.random.rand(16, H, W).astype(np.float32)
    f2 = np.random.rand(32, H, W).astype(np.float32)
    f3 = np.random.rand(8, H, W).astype(np.float32)

    out = lf.merge([f1, f2, f3])
    assert out.shape == (56, H, W), f"Expected (56,{H},{W}), got {out.shape}"
    print("  [OK] LateFusion (concat)")


def test_late_fusion_mean():
    from lib1_fusion.fusion.late_fusion import LateFusion, LateFusionConfig

    config = LateFusionConfig(merge_strategy="mean")
    lf = LateFusion(config)

    H, W = 64, 64
    f1 = np.ones((16, H, W), dtype=np.float32)
    f2 = np.ones((16, H, W), dtype=np.float32) * 3.0

    out = lf.merge([f1, f2])
    assert out.shape == (16, H, W)
    assert abs(out.mean() - 2.0) < 1e-5, f"Mean should be 2.0, got {out.mean()}"
    print("  [OK] LateFusion (mean)")


def test_cross_modal_attention_validate():
    from lib1_fusion.fusion.cross_modal_attention import CrossModalAttention

    cma = CrossModalAttention()

    H, W = 64, 64
    f1 = np.random.rand(16, H, W).astype(np.float32)
    f2 = np.random.rand(32, H, W).astype(np.float32)
    f3 = np.random.rand(8, H, W).astype(np.float32)
    cma.validate_inputs([f1, f2, f3])  # should not raise
    print("  [OK] CrossModalAttention.validate_inputs (valid)")

    # test mismatch
    f_bad = np.random.rand(16, 32, 32).astype(np.float32)
    try:
        cma.validate_inputs([f1, f_bad])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  [OK] CrossModalAttention.validate_inputs (mismatch raises ValueError)")


def test_cross_modal_attention_config():
    from lib1_fusion.fusion.cross_modal_attention import CrossModalAttention, CrossModalConfig

    cfg = CrossModalConfig(num_heads=4, embed_dim=128, dropout=0.2)
    cma = CrossModalAttention(cfg)
    d = cma.get_config_dict()
    assert d["num_heads"] == 4
    assert d["embed_dim"] == 128
    assert d["dropout"] == 0.2
    print("  [OK] CrossModalAttention.get_config_dict")


# ─── 4. Integration Test ─────────────────────────────────────────────────────

def test_integration_raw_to_fused():
    """
    End-to-end: raw synthetic data → processors → early_fusion → (12, H, W)
    """
    from lib1_fusion import S1Processor, S2Processor, GSatProcessor
    from lib1_fusion.fusion.early_fusion import early_fusion

    H, W = 256, 256

    # Simulate raw sensor data
    s1_vv = np.random.uniform(-25, 0, (H, W)).astype(np.float32)
    s1_vh = np.random.uniform(-25, 0, (H, W)).astype(np.float32)

    s2_bands = [np.random.uniform(0, 10000, (H, W)).astype(np.float32) for _ in range(7)]

    gsat_r = np.random.randint(0, 256, (H, W), dtype=np.uint8)
    gsat_g = np.random.randint(0, 256, (H, W), dtype=np.uint8)
    gsat_b = np.random.randint(0, 256, (H, W), dtype=np.uint8)

    # Process each modality
    s1_stack   = S1Processor().process(s1_vv, s1_vh)         # (2, H, W)
    s2_stack   = S2Processor().process(*s2_bands)            # (7, H, W)
    gsat_stack = GSatProcessor().process(gsat_r, gsat_g, gsat_b)  # (3, H, W)

    assert s1_stack.shape == (2, H, W)
    assert s2_stack.shape == (7, H, W)
    assert gsat_stack.shape == (3, H, W)

    # Fuse
    fused = early_fusion(s1_stack, s2_stack, gsat_stack)
    assert fused.shape == (12, H, W), f"Expected (12,{H},{W}), got {fused.shape}"
    assert fused.dtype == np.float32
    assert fused.min() >= 0.0
    assert fused.max() <= 1.0

    print("  [OK] Integration: raw -> processors -> early_fusion -> (12, 256, 256)")


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  lib1_fusion — Unit Tests + Integration Test")
    print("=" * 60)

    print("\n--- Utils ---")
    test_normalize_minmax()
    test_normalize_minmax_constant()
    test_normalize_sar_db()
    test_normalize_s2_reflectance()
    test_validate_spatial_shape()
    test_validate_spatial_shape_mismatch()
    test_stack_bands()

    print("\n--- Processors ---")
    test_s1_processor()
    test_s2_processor()
    test_gsat_processor()

    print("\n--- Fusion Strategies ---")
    test_early_fusion()
    test_early_fusion_shape_mismatch()
    test_early_fusion_wrong_bands()
    test_late_fusion_concat()
    test_late_fusion_mean()
    test_cross_modal_attention_validate()
    test_cross_modal_attention_config()

    print("\n--- Integration ---")
    test_integration_raw_to_fused()

    print("\n" + "=" * 60)
    print("  All tests PASSED!")
    print("=" * 60)
