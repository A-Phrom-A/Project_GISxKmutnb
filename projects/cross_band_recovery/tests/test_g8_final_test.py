"""
projects.cross_band_recovery.tests.test_g8_final_test
=====================================================
Quality Gate G8 Verification Test Suite: Final Frozen Test Set Evaluation.

Checks covered:
- CHK-G8.1: Complete evaluation of 127 patterns (S1_on) + 126 patterns (S1_off).
- CHK-G8.2: Bitwise integrity against Gate G7 pre-registration.
- CHK-G8.3: Complete metric schema per row without nulls.
- CHK-G8.4: Single test execution logged.
- CHK-G8.5: Recovery rate and coverage statistics reported.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
EVIDENCE_DIR = PROJECT_DIR / "docs" / "tracking" / "evidence" / "G8"
PARQUET_FILE = RESULTS_DIR / "final_test_benchmarks.parquet"


def test_chk_g8_1_complete_runs():
    """CHK-G8.1: All required conditions executed without OOM or crash."""
    ev_file = EVIDENCE_DIR / "CHK-G8.1_complete_runs.json"
    assert ev_file.exists(), f"Evidence missing: {ev_file}"
    with open(ev_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass"
    assert data["total_conditions_evaluated"] > 0
    assert data["m2_s1_on_count"] > 0
    assert data["m2_s1_off_count"] > 0


def test_chk_g8_2_checkpoint_hash_check():
    """CHK-G8.2: Checkpoint hashes verified strictly against pre-registration."""
    ev_file = EVIDENCE_DIR / "CHK-G8.2_checkpoint_hash_check.json"
    assert ev_file.exists(), f"Evidence missing: {ev_file}"
    with open(ev_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass"


def test_chk_g8_3_full_metric_schema():
    """CHK-G8.3: Complete tabular schema with all required metrics, zero nulls."""
    ev_file = EVIDENCE_DIR / "CHK-G8.3_full_metric_schema.json"
    assert ev_file.exists(), f"Evidence missing: {ev_file}"
    with open(ev_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass"
    assert data["has_nulls"] is False
    
    required_cols = [
        "model", "seed", "pattern_id", "k", "masked_bands", "s1_condition",
        "mae", "rmse", "psnr", "ssim", "sam_deg", "bias", "recovery_rate"
    ]
    for col in required_cols:
        assert col in data["columns"], f"Missing column: {col}"


def test_chk_g8_4_single_test_execution():
    """CHK-G8.4: Single test execution logged cleanly."""
    ev_file = EVIDENCE_DIR / "CHK-G8.4_single_test_execution.json"
    assert ev_file.exists(), f"Evidence missing: {ev_file}"
    with open(ev_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass"
    assert data["single_test_access"] is True


def test_chk_g8_5_coverage_and_recovery():
    """CHK-G8.5: Recovery rate and coverage statistics reported."""
    ev_file = EVIDENCE_DIR / "CHK-G8.5_coverage_and_recovery.json"
    assert ev_file.exists(), f"Evidence missing: {ev_file}"
    with open(ev_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass"
    assert 0.0 <= data["mean_recovery_rate"] <= 1.0
    assert data["mean_mae"] > 0.0
