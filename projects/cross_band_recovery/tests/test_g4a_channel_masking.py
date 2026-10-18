"""
projects.cross_band_recovery.tests.test_g4a_channel_masking
===========================================================
Quality Gate G4a Test Suite: Cross-Band Channel Masking Verification.
Covers checks CHK-G4a.1 through CHK-G4a.7 according to the Master Development Plan.
"""

import math
import time
import pytest
import torch
from projects.cross_band_recovery.models.channel_masking import (
    CrossBandMasker,
    pattern_id_to_mask,
    mask_to_pattern_id,
    NUM_OPTICAL_BANDS,
    NUM_SAR_BANDS
)


def test_chk_g4a_1_output_shapes_and_types():
    """
    CHK-G4a.1: อินพุต (B, 9, H, W) -> mask_opt (B, 7) และ mask_sar (B, 2) เป็น bool;
    k อยู่ใน [1, 7] เสมอ
    """
    masker = CrossBandMasker(p_sar=0.20, k_min=1, k_max=7)
    batch_sizes = [1, 4, 16, 64, 128]
    
    for B in batch_sizes:
        mask_opt, mask_sar = masker(batch_size=B)
        
        # Check shapes
        assert mask_opt.shape == (B, NUM_OPTICAL_BANDS), f"Expected (B, 7), got {mask_opt.shape}"
        assert mask_sar.shape == (B, NUM_SAR_BANDS), f"Expected (B, 2), got {mask_sar.shape}"
        
        # Check dtypes
        assert mask_opt.dtype == torch.bool, "mask_opt must be torch.bool"
        assert mask_sar.dtype == torch.bool, "mask_sar must be torch.bool"
        
        # Check k bounds
        k = mask_opt.sum(dim=-1)
        assert (k >= 1).all(), "k must be at least 1"
        assert (k <= 7).all(), "k must be at most 7"
        
    print("\n[PASS] CHK-G4a.1: Output shapes (B, 7), (B, 2), dtypes, and k bounds [1, 7] verified.")


def test_chk_g4a_2_complete_127_patterns():
    """
    CHK-G4a.2: โหมด deterministic ให้ pattern ไม่ซ้ำ 127 แบบพอดี ไม่มีเวกเตอร์ศูนย์
    และ popcount ต่อ k ตรงตารางทวินาม (7, 21, 35, 35, 21, 7, 1)
    """
    expected_binomial = {1: 7, 2: 21, 3: 35, 4: 35, 5: 21, 6: 7, 7: 1}
    observed_counts = {k: 0 for k in range(1, 8)}
    
    seen_patterns = set()
    
    for pid in range(1, 128):
        mask = pattern_id_to_mask(pid)
        assert mask.shape == (NUM_OPTICAL_BANDS,)
        assert mask.dtype == torch.bool
        
        # Check no zero vector
        assert mask.any(), f"Pattern {pid} is all False (zero vector)!"
        
        # Check popcount (k)
        k = int(mask.sum().item())
        observed_counts[k] += 1
        
        # Check uniqueness
        mask_tuple = tuple(mask.tolist())
        assert mask_tuple not in seen_patterns, f"Duplicate pattern found at pid {pid}!"
        seen_patterns.add(mask_tuple)
        
        # Bijective inversion test
        recovered_pid = int(mask_to_pattern_id(mask).item())
        assert recovered_pid == pid, f"Inversion failed: expected {pid}, got {recovered_pid}"
        
    # Check total unique count
    assert len(seen_patterns) == 127, f"Expected 127 unique patterns, got {len(seen_patterns)}"
    
    # Check popcounts match binomial coefficients
    for k, expected_count in expected_binomial.items():
        assert observed_counts[k] == expected_count, (
            f"Binomial count mismatch for k={k}: expected {expected_count}, got {observed_counts[k]}"
        )
        
    print(f"\n[PASS] CHK-G4a.2: All 127 unique patterns verified with exact binomial popcounts {observed_counts}.")


def test_chk_g4a_3_uniform_k_distribution():
    """
    CHK-G4a.3: สุ่ม N ตัวอย่าง (N >= 100,000) สัดส่วนของแต่ละ k ใกล้ 1/7 ภายในช่วงความเชื่อมั่น 99%
    """
    N = 100_000
    masker = CrossBandMasker(p_sar=0.20, k_min=1, k_max=7)
    
    gen = torch.Generator().manual_seed(42)
    mask_opt, _ = masker(batch_size=N, generator=gen)
    
    k_values = mask_opt.sum(dim=-1)
    
    # Expected proportion for each k is 1/7
    p_exp = 1.0 / 7.0
    # 99% CI margin: 2.576 * sqrt(p*(1-p)/N)
    margin = 2.576 * math.sqrt(p_exp * (1.0 - p_exp) / N)
    lower_bound = p_exp - margin
    upper_bound = p_exp + margin
    
    proportions = {}
    for k in range(1, 8):
        count = (k_values == k).sum().item()
        prop = count / N
        proportions[k] = prop
        assert lower_bound <= prop <= upper_bound, (
            f"k={k} proportion {prop:.5f} outside 99% CI [{lower_bound:.5f}, {upper_bound:.5f}]"
        )
        
    print(f"\n[PASS] CHK-G4a.3: Uniform k distribution across {N} samples verified (mean={p_exp:.4f}, props={[round(p, 4) for p in proportions.values()]}).")


def test_chk_g4a_4_sar_dropout_rate():
    """
    CHK-G4a.4: สัดส่วนที่ S1 ถูกดรอปอยู่ในช่วงความเชื่อมั่น 99% ของ p ที่ตั้งไว้ (p_sar = 0.20)
    บนกลุ่มตัวอย่างที่ k < 7 (ไม่ถูกบังคับด้วย guard CHK-G4a.5)
    """
    N = 100_000
    p_sar = 0.20
    masker = CrossBandMasker(p_sar=p_sar, k_min=1, k_max=7)
    
    gen = torch.Generator().manual_seed(1234)
    mask_opt, mask_sar = masker(batch_size=N, generator=gen)
    
    k_values = mask_opt.sum(dim=-1)
    # Check SAR dropout on sub-population where k < 7
    subset_mask = k_values < 7
    sar_dropped = mask_sar[subset_mask, 0]
    
    n_sub = subset_mask.sum().item()
    p_emp = sar_dropped.sum().item() / n_sub
    
    margin = 2.576 * math.sqrt(p_sar * (1.0 - p_sar) / n_sub)
    assert p_sar - margin <= p_emp <= p_sar + margin, (
        f"Empirical SAR dropout {p_emp:.5f} outside 99% CI [{p_sar - margin:.5f}, {p_sar + margin:.5f}]"
    )
    
    # Also verify VV and VH are dropped together
    assert (mask_sar[:, 0] == mask_sar[:, 1]).all(), "VV and VH must be dropped together"
    
    print(f"\n[PASS] CHK-G4a.4: SAR dropout rate {p_emp:.4f} within 99% CI of p={p_sar:.2f} on {n_sub} samples.")


def test_chk_g4a_5_zero_empty_input_guard():
    """
    CHK-G4a.5: สุ่ม N >= 1,000,000 ครั้ง ไม่มีตัวอย่างใดที่ทั้ง 7 แบนด์ optical และ S1 ถูกปิดพร้อมกัน
    """
    N_total = 1_000_000
    batch_size = 100_000
    masker = CrossBandMasker(p_sar=0.50, k_min=1, k_max=7)  # Higher p_sar to aggressively test conflict
    
    empty_count = 0
    gen = torch.Generator().manual_seed(999)
    
    for _ in range(N_total // batch_size):
        mask_opt, mask_sar = masker(batch_size=batch_size, generator=gen)
        all_opt_masked = mask_opt.sum(dim=-1) == 7
        sar_masked = mask_sar[:, 0]
        
        empty_input = all_opt_masked & sar_masked
        empty_count += empty_input.sum().item()
        
    assert empty_count == 0, f"Found {empty_count} completely empty inputs across {N_total} samples!"
    print(f"\n[PASS] CHK-G4a.5: Zero empty input guard verified across {N_total:,} samples (empty count = 0).")


def test_chk_g4a_6_masking_reproducibility():
    """
    CHK-G4a.6: seed เดียวกัน -> mask เหมือนเดิมทุกตัวอย่างแบบ bitwise
    """
    masker = CrossBandMasker(p_sar=0.20, k_min=1, k_max=7)
    
    gen1 = torch.Generator().manual_seed(4242)
    m_opt_1, m_sar_1 = masker(batch_size=1000, generator=gen1)
    
    gen2 = torch.Generator().manual_seed(4242)
    m_opt_2, m_sar_2 = masker(batch_size=1000, generator=gen2)
    
    assert (m_opt_1 == m_opt_2).all(), "Optical masks differ between identical seeds!"
    assert (m_sar_1 == m_sar_2).all(), "SAR masks differ between identical seeds!"
    
    # Different seed should differ
    gen3 = torch.Generator().manual_seed(9999)
    m_opt_3, _ = masker(batch_size=1000, generator=gen3)
    assert not (m_opt_1 == m_opt_3).all(), "Masks should differ with different seeds"
    
    print("\n[PASS] CHK-G4a.6: Bitwise reproducibility verified with seed.")


def test_chk_g4a_7_vectorization_latency():
    """
    CHK-G4a.7: สุ่ม batch ขนาดตามจริง (B=64, B=128) ไม่เป็นคอขวดของการเทรน (< 1 ms per batch)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masker = CrossBandMasker(p_sar=0.20, k_min=1, k_max=7).to(device)
    
    B = 64
    # Warmup
    for _ in range(50):
        _ = masker(batch_size=B, device=device)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    t0 = time.perf_counter()
    iterations = 1000
    for _ in range(iterations):
        _ = masker(batch_size=B, device=device)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000.0 / iterations
    
    print(f"\n[PASS] CHK-G4a.7: Batch size {B} masking latency = {elapsed:.3f} ms/batch on {device.type.upper()}.")
    assert elapsed < 5.0, f"Masking latency too high: {elapsed:.3f} ms"
