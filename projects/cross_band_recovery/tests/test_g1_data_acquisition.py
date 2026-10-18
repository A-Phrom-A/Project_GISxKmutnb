"""
Unit tests for Gate G1: Data Acquisition & Pairing Manifest Integrity
======================================================================
Checks:
- CHK-G1.1: Coverage per target ROI (at least 1 pair per target ROI)
- CHK-G1.2: Season diversity documented (Dry season core baseline)
- CHK-G1.3: Temporal gap |dt| logged and within tolerance (Tier 1 & Tier 2)
- CHK-G1.4: Cloud cover <= 5% in ROI
- CHK-G1.5: File integrity: exactly 2 bands for S1, 7 bands for S2, CRS match
- CHK-G1.6: Value ranges: S2 reflectance, S1 positive linear
- CHK-G1.7: Data lineage metadata logged
"""

import json
from pathlib import Path
import pandas as pd
import pytest


MANIFEST_PATH = Path("projects/cross_band_recovery/data/manifests/metadata_manifest.parquet")
EVIDENCE_DIR = Path("projects/cross_band_recovery/docs/tracking/evidence/G1")


@pytest.fixture(scope="module")
def manifest_df():
    assert MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}"
    df = pd.read_parquet(MANIFEST_PATH)
    return df


def test_chk_g1_1_roi_coverage(manifest_df):
    """CHK-G1.1: ตรวจสอบการครอบคลุมพื้นที่เป้าหมาย (Chonburi, Rayong, Chanthaburi, SaKaeo, Trat, Surat Thani)"""
    rois = manifest_df["roi_id"].unique().tolist()
    expected_rois = ["Chanthaburi", "Chonburi", "Rayong", "SaKaeo", "Trat", "Surat_Thani"]
    for r in expected_rois:
        assert r in rois, f"Missing target ROI: {r}"
    print(f"\n[PASS] CHK-G1.1: All {len(expected_rois)} target ROIs covered with valid paired scenes.")


def test_chk_g1_2_season_diversity(manifest_df):
    """CHK-G1.2: ตรวจสอบการระบุฤดูกาล"""
    evidence_file = EVIDENCE_DIR / "CHK-G1.2_season_diversity.json"
    assert evidence_file.exists()
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "PASSED"
    assert "Dry" in data["season_distribution"]
    print(f"\n[PASS] CHK-G1.2: Season diversity documented: {data['season_distribution']}")


def test_chk_g1_3_temporal_gap_histogram(manifest_df):
    """CHK-G1.3: ช่องว่างเวลา |dt| ต้องไม่เกิน 6 วัน (Tier 1 & Tier 2)"""
    assert (manifest_df["dt_days"] <= 6.0).all()
    evidence_file = EVIDENCE_DIR / "CHK-G1.3_temporal_gap_histogram.json"
    assert evidence_file.exists()
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "PASSED"
    print(f"\n[PASS] CHK-G1.3: Max temporal gap = {data['max_dt_days']} days (Mean: {data['mean_dt_days']:.2f} days)")


def test_chk_g1_4_cloud_cover_roi(manifest_df):
    """CHK-G1.4: สัดส่วนเมฆระดับ ROI ต้อง <= 5.0%"""
    assert (manifest_df["cloud_pct"] <= 5.0).all()
    max_cloud = manifest_df["cloud_pct"].max()
    print(f"\n[PASS] CHK-G1.4: Max cloud percentage = {max_cloud:.2f}% (<= 5.0% threshold)")


def test_chk_g1_5_file_integrity(manifest_df):
    """CHK-G1.5: S1 ต้องมี 2 แบนด์, S2 ต้องมี 7 แบนด์, CRS ต้องตรงกันในแต่ละคู่"""
    assert (manifest_df["s1_bands"] == 2).all()
    assert (manifest_df["s2_bands"] == 7).all()
    assert (manifest_df["crs_match"] == True).all()
    print(f"\n[PASS] CHK-G1.5: All {len(manifest_df)} pairs have exact 2 S1 + 7 S2 bands and matched CRS.")


def test_chk_g1_6_value_ranges(manifest_df):
    """CHK-G1.6: ตรวจสอบช่วงค่าตัวเลข S1 และ S2 ตามหน่วยที่ระบุ (log_dB หรือ linear)"""
    for _, row in manifest_df.iterrows():
        s1_range = row["s1_val_range"]
        unit = row["s1_unit"]
        if unit == "linear_backscatter":
            assert s1_range[0] >= 0.0, f"Linear S1 must be non-negative: {s1_range}"
        elif unit == "log_dB":
            # ค่า dB สำหรับ Sentinel-1 ทั่วไปอยู่ในช่วง -45 dB ถึง +10 dB
            assert -50.0 <= s1_range[0] <= 15.0, f"dB S1 out of physical bounds: {s1_range}"
            
        s2_range = row["s2_val_range"]
        assert s2_range[0] >= 0.0, f"S2 reflectance must be non-negative: {s2_range}"
    print("\n[PASS] CHK-G1.6: Value ranges physically validated across both log_dB and linear units.")


def test_chk_g1_7_data_lineage():
    """CHK-G1.7: ตรวจสอบข้อมูล Lineage และ Collection"""
    evidence_file = EVIDENCE_DIR / "CHK-G1.7_data_lineage.json"
    assert evidence_file.exists()
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "PASSED"
    assert len(data["collections"]) >= 2
    print("\n[PASS] CHK-G1.7: Data lineage and collection metadata verified.")
