"""
tests/cloudmae_v2/test_s10_production.py
========================================
Pytest validation for Phase 1 - Production Mask Pipeline.
"""

import json
from pathlib import Path
import pytest

REPORTS_DIR = Path("reports/cloudmae_v2")


def test_s10_mask_production_report():
    p = REPORTS_DIR / "s10_mask_production.json"
    assert p.exists(), "s10_mask_production.json must exist"
    
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["all_passed"] is True
    assert data["chk_14a"]["status"] == "PASSED"
    assert data["chk_14b"]["status"] == "PASSED"
    assert data["chk_14c"]["status"] == "PASSED"
    assert data["chk_14d"]["status"] == "PASSED"
    assert "Option 2" in data["strategy"]
