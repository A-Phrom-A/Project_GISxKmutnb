"""
projects.cross_band_recovery.tests.test_g9_xai
==============================================
Quality Gate G9 Unit Tests:
  CHK-G9.1: Additivity / Completeness of feature attributions (relative error <= 5.0%)
  CHK-G9.2: Model Parameter Randomization Sanity Check (Pearson r < 0.30)
  CHK-G9.3: Stability across baseline backgrounds & multi-seed runs (Spearman rho > 0.50)
  CHK-G9.4: Channel attribution rank matches empirical ablation delta-MAE (Spearman rho > 0.70)
  CHK-G9.5: Multi-collinear feature grouping (RGB, NIR, SWIR, SAR shares report)
"""

import json
from pathlib import Path
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = PROJECT_DIR / "docs" / "tracking" / "evidence" / "G9"
RESULTS_DIR = PROJECT_DIR / "results"


def test_chk_g9_1_additivity_completeness():
    evidence_file = EVIDENCE_DIR / "CHK-G9.1_additivity_completeness.json"
    assert evidence_file.exists(), f"Missing evidence: {evidence_file}"
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass", f"CHK-G9.1 failed: {data}"
    assert data["max_relative_error"] <= 0.05, f"Relative error {data['max_relative_error']} exceeded 5.0%"
    assert len(data["details"]) >= 3, "Must evaluate at least 3 distinct target spectral bands"


def test_chk_g9_2_randomization_sanity():
    evidence_file = EVIDENCE_DIR / "CHK-G9.2_randomization_sanity.json"
    assert evidence_file.exists(), f"Missing evidence: {evidence_file}"
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass", f"CHK-G9.2 failed: {data}"
    assert data["mean_pearson_r"] < 0.30, f"Pearson correlation {data['mean_pearson_r']} >= 0.30"
    for item in data["details"]:
        assert item["pearson_r"] < 0.30, f"Target {item['target_band']} failed randomization test (r={item['pearson_r']})"


def test_chk_g9_3_background_and_seed_stability():
    evidence_file = EVIDENCE_DIR / "CHK-G9.3_background_stability.json"
    assert evidence_file.exists(), f"Missing evidence: {evidence_file}"
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass", f"CHK-G9.3 failed: {data}"
    assert data["background_rank_rho"] > 0.50, f"Background rank correlation {data['background_rank_rho']} <= 0.50"
    assert data["multi_seed_mean_rho"] > 0.50, f"Multi-seed correlation {data['multi_seed_mean_rho']} <= 0.50"


def test_chk_g9_4_ablation_rank_correlation():
    evidence_file = EVIDENCE_DIR / "CHK-G9.4_ablation_rank_correlation.json"
    assert evidence_file.exists(), f"Missing evidence: {evidence_file}"
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass", f"CHK-G9.4 failed: {data}"
    assert data["spearman_rho"] > 0.70, f"Ablation rank correlation {data['spearman_rho']} <= 0.70"
    assert len(data["evaluated_channels"]) >= 5, "Must evaluate across at least 5 channels"


def test_chk_g9_5_feature_grouping():
    evidence_file = EVIDENCE_DIR / "CHK-G9.5_feature_grouping.json"
    assert evidence_file.exists(), f"Missing evidence: {evidence_file}"
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass", f"CHK-G9.5 failed: {data}"
    shares = data["group_attribution_shares"]
    assert "RGB" in shares and "NIR_RedEdge" in shares and "SWIR" in shares and "SAR" in shares
    total_share = sum(shares.values())
    assert abs(total_share - 1.0) < 1e-4, f"Group shares sum to {total_share}, expected 1.0"
    assert "collinearity_note" in data, "Collinearity caveats must be documented"
