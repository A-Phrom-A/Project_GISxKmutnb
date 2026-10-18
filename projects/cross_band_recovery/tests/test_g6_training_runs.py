"""
projects.cross_band_recovery.tests.test_g6_training_runs
========================================================
Quality Gate G6 Verification Test Suite: Main Model Training Runs.

Checks covered:
- CHK-G6.1: Loss convergence without oscillation or NaNs.
- CHK-G6.2: Early stopping & healthy train/val gap.
- CHK-G6.3: Consistency across 3 seeds (M1 and M2).
- CHK-G6.4: Zero test set touch during training and model selection.
- CHK-G6.5: M2 S1_off graceful degradation.
- CHK-G6.6: S1 shuffle test for shortcut detection.
- CHK-G6.7: Complete checkpoint saving & SHA-256 hashes.
"""

import json
from pathlib import Path
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = PROJECT_DIR / "docs" / "tracking" / "evidence" / "G6"
CHECKPOINTS_DIR = PROJECT_DIR / "results" / "checkpoints"


def test_chk_g6_1_convergence():
    """CHK-G6.1: Both M1 and M2 converge properly without NaNs."""
    ev_file = EVIDENCE_DIR / "CHK-G6.1_convergence_report.json"
    assert ev_file.exists(), f"Evidence missing: {ev_file}"
    with open(ev_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass"
    assert data["m1_convergence"] is True
    assert data["m2_convergence"] is True
    assert data["zero_nan_encountered"] is True


def test_chk_g6_2_no_severe_overfitting():
    """CHK-G6.2: Generalization gap between train and val is bounded."""
    ev_file = EVIDENCE_DIR / "CHK-G6.2_overfitting_check.json"
    assert ev_file.exists(), f"Evidence missing: {ev_file}"
    with open(ev_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass"
    assert data["early_stopping_used_val"] is True
    for gap in data["m1_gap"] + data["m2_gap"]:
        # Healthy generalization: validation loss gap < 0.05
        assert gap < 0.05, f"Excessive overfitting gap: {gap}"


def test_chk_g6_3_multi_seed_consistency():
    """CHK-G6.3: Consistency across 3 seeds (low std)."""
    ev_file = EVIDENCE_DIR / "CHK-G6.3_multi_seed_consistency.json"
    assert ev_file.exists(), f"Evidence missing: {ev_file}"
    with open(ev_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass"
    assert len(data["seeds"]) == 3
    assert data["m1_val_std"] < 0.015, f"High variance across M1 seeds: std={data['m1_val_std']}"
    assert data["m2_val_std"] < 0.015, f"High variance across M2 seeds: std={data['m2_val_std']}"


def test_chk_g6_4_zero_test_touch():
    """CHK-G6.4: Zero test set touch during training."""
    ev_file = EVIDENCE_DIR / "CHK-G6.4_zero_test_touch.json"
    assert ev_file.exists(), f"Evidence missing: {ev_file}"
    with open(ev_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass"
    assert data["test_split_accessed"] is False
    assert "test" not in data["splits_loaded"]


def test_chk_g6_5_graceful_degradation():
    """CHK-G6.5: M2 without S1 does not catastrophically collapse."""
    ev_file = EVIDENCE_DIR / "CHK-G6.5_graceful_degradation.json"
    assert ev_file.exists(), f"Evidence missing: {ev_file}"
    with open(ev_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass"
    diag = data["diagnostics"]
    assert diag["graceful_degradation_passed"] is True
    assert diag["degradation_ratio"] < 2.5


def test_chk_g6_6_shortcut_detection():
    """CHK-G6.6: S1 shuffle test to assess shortcut vs SAR utilization."""
    ev_file = EVIDENCE_DIR / "CHK-G6.6_shortcut_detection.json"
    assert ev_file.exists(), f"Evidence missing: {ev_file}"
    with open(ev_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass"
    assert "sar_sensitivity_delta" in data


def test_chk_g6_7_artifacts_and_hashes():
    """CHK-G6.7: Checkpoints and SHA-256 hashes exist for all runs."""
    ev_file = EVIDENCE_DIR / "CHK-G6.7_artifacts_and_hashes.json"
    assert ev_file.exists(), f"Evidence missing: {ev_file}"
    with open(ev_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass"
    artifacts = data["artifacts"]
    assert len(artifacts) == 6  # 3 seeds * 2 models
    
    for run_name, meta in artifacts.items():
        chkpt_path = Path(meta["checkpoint"])
        assert chkpt_path.exists(), f"Checkpoint file missing: {chkpt_path}"
        assert len(meta["sha256"]) == 64, f"Invalid SHA-256 for {run_name}"
        assert meta["best_val_loss"] > 0.0
