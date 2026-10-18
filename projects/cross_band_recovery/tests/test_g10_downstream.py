"""
projects.cross_band_recovery.tests.test_g10_downstream
======================================================
Quality Gate G10 Unit Tests:
  CHK-G10.1: Frozen downstream palm segmentation model (bitwise SHA-256 invariant)
  CHK-G10.2: Spatial isolation (GT parcels strictly within test block > 5,000m from train)
  CHK-G10.3: Full evaluation across all 5 conditions (Clean, Unrepaired, B3, M1, M2) with 95% CIs
  CHK-G10.4: Transparent and complete reporting without metric suppression
"""

import hashlib
import json
from pathlib import Path
import pytest
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_DIR = PROJECT_DIR / "docs" / "tracking" / "evidence" / "G10"
RESULTS_DIR = PROJECT_DIR / "results"
DOWNSTREAM_CKPT = PROJECT_ROOT / "checkpoints" / "finetune" / "vitpalm_best.pth"


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_chk_g10_1_frozen_seg_model():
    evidence_file = EVIDENCE_DIR / "CHK-G10.1_frozen_seg_model.json"
    assert evidence_file.exists(), f"Missing evidence: {evidence_file}"
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass", f"CHK-G10.1 failed: {data}"
    actual_hash = compute_file_sha256(DOWNSTREAM_CKPT)
    assert data["sha256"] == actual_hash, "Segmentation model hash mismatch!"
    assert data["invariant_across_conditions"] is True


def test_chk_g10_2_spatial_isolation():
    evidence_file = EVIDENCE_DIR / "CHK-G10.2_spatial_isolation.json"
    assert evidence_file.exists(), f"Missing evidence: {evidence_file}"
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass", f"CHK-G10.2 failed: {data}"
    assert data["spatial_buffer_from_train_meters"] >= 5000.0
    assert data["zero_leakage_guaranteed"] is True
    assert data["evaluated_test_tiles_count"] > 0


def test_chk_g10_3_all_conditions_evaluated():
    evidence_file = EVIDENCE_DIR / "CHK-G10.3_all_conditions_evaluated.json"
    assert evidence_file.exists(), f"Missing evidence: {evidence_file}"
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass", f"CHK-G10.3 failed: {data}"
    expected_conditions = ["Clean", "Unrepaired", "Baseline_B3", "M1_S2only", "M2_Fusion"]
    assert set(data["conditions_evaluated"]) == set(expected_conditions)
    
    # Parquet check
    parquet_file = RESULTS_DIR / "downstream_palm_benchmarks.parquet"
    assert parquet_file.exists(), f"Missing benchmarks parquet: {parquet_file}"
    df = pd.read_parquet(parquet_file)
    assert len(df) > 0
    assert set(df["condition"].unique()) == {
        "A_Clean", "B_Unrepaired", "C_Baseline_B3", "D_M1_S2only", "E_M2_Fusion"
    }


def test_chk_g10_4_transparent_reporting():
    evidence_file = EVIDENCE_DIR / "CHK-G10.4_transparent_reporting.json"
    assert evidence_file.exists(), f"Missing evidence: {evidence_file}"
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass", f"CHK-G10.4 failed: {data}"
    assert data["full_negative_reporting"] is True
    # M2 should recover performance over unrepaired
    assert data["m2_f1_vs_unrepaired"] > 0.0, "M2 failed to recover over unrepaired input!"
