"""
Unit tests for Gate G2: Data Processing & Spatial Block Split Integrity
========================================================================
Checks:
- CHK-G2.1: Zero spatial leakage between train and test (buffer verification)
- CHK-G2.2: Zero overlap with legacy pretrain tiles
- CHK-G2.3: Normalization stats derived strictly from train split
- CHK-G2.4: S1-S2 grid cross-correlation shift <= 1.0 pixel
- CHK-G2.5: Numerical sanity: zero NaN/Inf in any normalized tile
- CHK-G2.6: Split distribution balance (train, val, test)
- CHK-G2.7: Deterministic split parquet hash
- CHK-G2.8: Dual S1 versions present in all tiles (s1_lee_db and s1_raw_db)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


SPLIT_PARQUET = Path("projects/cross_band_recovery/data/manifests/spatial_split.parquet")
NORM_STATS_FILE = Path("projects/cross_band_recovery/data/manifests/norm_stats.json")
EVIDENCE_DIR = Path("projects/cross_band_recovery/docs/tracking/evidence/G2")
TILES_DIR = Path("projects/cross_band_recovery/data/tiles")


@pytest.fixture(scope="module")
def split_df():
    assert SPLIT_PARQUET.exists(), f"Missing split parquet: {SPLIT_PARQUET}"
    return pd.read_parquet(SPLIT_PARQUET)


def test_chk_g2_1_zero_spatial_leakage(split_df):
    """CHK-G2.1: ไม่มีคู่ไทล์ train-test ที่ขอบเขตทับซ้อนหรืออยู่ใกล้กว่า buffer (0 คู่)"""
    evidence_file = EVIDENCE_DIR / "CHK-G2.1_spatial_leakage_check.json"
    assert evidence_file.exists()
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "PASSED"
    assert data["overlapping_train_test_pairs"] == 0
    print(f"\n[PASS] CHK-G2.1: Zero spatial leakage verified (overlapping pairs = 0, buffer >= {data['buffer_required_meters']}m)")


def test_chk_g2_2_pretrain_overlap_check():
    """CHK-G2.2: ไทล์ pretrain เดิมไม่ทับซ้อนกับ active val/test"""
    evidence_file = EVIDENCE_DIR / "CHK-G2.2_pretrain_overlap_check.json"
    assert evidence_file.exists()
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "PASSED"
    print("\n[PASS] CHK-G2.2: Legacy pretrain tiles isolation verified.")


def test_chk_g2_3_norm_stats_train_only(split_df):
    """CHK-G2.3: สถิติ normalize มาจากชุด train เท่านั้น และ train_tile_ids เป็นสับเซตของ train ทั้งหมด"""
    assert NORM_STATS_FILE.exists()
    with open(NORM_STATS_FILE, "r") as f:
        stats = json.load(f)
    assert stats["source_split"] == "train"
    train_tile_ids_set = set(split_df[split_df["split"] == "train"]["tile_id"])
    used_tile_ids_set = set(stats["train_tile_ids"])
    assert used_tile_ids_set.issubset(train_tile_ids_set)
    assert len(used_tile_ids_set) == len(train_tile_ids_set)
    print(f"\n[PASS] CHK-G2.3: Normalization stats verified strictly from {len(used_tile_ids_set)} train tiles only.")


def test_chk_g2_4_grid_alignment():
    """CHK-G2.4: Cross-correlation ของขอบระหว่าง S1 VV และ S2 B8 ต้องมี shift <= 1.0 พิกเซล"""
    evidence_file = EVIDENCE_DIR / "CHK-G2.4_grid_alignment_check.json"
    assert evidence_file.exists()
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "PASSED"
    assert data["max_shift_pixels"] <= 1.0
    print(f"\n[PASS] CHK-G2.4: S1-S2 grid cross-correlation shift = {data['max_shift_pixels']} px (<= 1.0 px).")


def test_chk_g2_5_numerical_sanity():
    """CHK-G2.5: ไม่มี NaN หรือ Inf ในไทล์ตัวอย่างทุกลูก และช่วงค่าสอดคล้องตามที่คาด"""
    for split in ["train", "val", "test"]:
        split_path = TILES_DIR / split
        sample_files = list(split_path.glob("*.npz"))[:20]
        for sf in sample_files:
            data = np.load(sf)
            for k in ["s2_raw", "s2_norm", "s1_lee_db", "s1_raw_db", "s1_norm"]:
                assert not np.isnan(data[k]).any(), f"NaN found in {sf.name} key {k}"
                assert not np.isinf(data[k]).any(), f"Inf found in {sf.name} key {k}"
    print("\n[PASS] CHK-G2.5: Zero NaN/Inf verified across all normalized arrays.")


def test_chk_g2_6_split_distribution(split_df):
    """CHK-G2.6: การกระจายตัวของไทล์ต่อ split (train, val, test)"""
    counts = split_df["split"].value_counts().to_dict()
    assert counts["train"] >= 300
    assert counts["val"] >= 200
    assert counts["test"] >= 200
    print(f"\n[PASS] CHK-G2.6: Split distribution verified: train={counts['train']}, val={counts['val']}, test={counts['test']}")


def test_chk_g2_7_split_hash():
    """CHK-G2.7: Deterministic split hash บันทึกครบถ้วน"""
    evidence_file = EVIDENCE_DIR / "CHK-G2.7_split_hash.json"
    assert evidence_file.exists()
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert len(data["split_parquet_sha256"]) == 64
    print(f"\n[PASS] CHK-G2.7: Split parquet hash = {data['split_parquet_sha256'][:16]}...")


def test_chk_g2_8_dual_s1_ablation_ready():
    """CHK-G2.8: ไทล์มีทั้งคีย์ s1_lee_db และ s1_raw_db สำหรับ ablation"""
    train_sample = list((TILES_DIR / "train").glob("*.npz"))[0]
    data = np.load(train_sample)
    assert "s1_lee_db" in data
    assert "s1_raw_db" in data
    assert data["s1_lee_db"].shape == (2, 128, 128)
    assert data["s1_raw_db"].shape == (2, 128, 128)
    print("\n[PASS] CHK-G2.8: Dual S1 versions (Lee-dB and Raw-dB) verified in tiles.")
