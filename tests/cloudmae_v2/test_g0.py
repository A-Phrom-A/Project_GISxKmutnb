"""
tests/cloudmae_v2/test_g0.py
============================
Pytest validation for Gate G0 and Phase 0 completion (S0.3, S0.4, S0.5).
"""

import json
from pathlib import Path
import pytest

REPORTS_DIR = Path("reports/cloudmae_v2")


def test_s03_ref_availability():
    p = REPORTS_DIR / "s03_ref_availability.json"
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["all_passed"] is True
    assert data["chk_03a_seasonal_survey"]["status"] == "PASSED"
    assert data["chk_03b_scenes_per_bin"]["status"] == "PASSED"


def test_s04_mask_pilot():
    p = REPORTS_DIR / "s04_mask_pilot.json"
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["all_passed"] is True
    assert data["chk_04a_omission"]["status"] == "PASSED"
    assert data["chk_04b_commission"]["status"] == "PASSED"
    assert data["pilot_stats"]["option2_ensemble"]["omission_rate_pct"] <= 5.0


def test_s05_baseline_b0():
    p = REPORTS_DIR / "s05_baseline_b0.json"
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["all_passed"] is True
    assert data["chk_05a_baseline"]["status"] == "PASSED"
    assert data["results"]["b0_metrics"]["psnr_masked_db"] > 30.0


def test_gate_g0():
    p = REPORTS_DIR / "gate_g0.json"
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "PASSED"
    assert "Option 2" in data["selected_mask_strategy"]
