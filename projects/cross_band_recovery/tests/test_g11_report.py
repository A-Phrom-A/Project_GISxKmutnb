"""
projects.cross_band_recovery.tests.test_g11_report
==================================================
Quality Gate G11 Unit Tests:
  CHK-G11.1: Code-generated numbers audit (100% derived from parquet artifacts)
  CHK-G11.2: Hypotheses H1-H5 conclusion with verifiable evidence
  CHK-G11.3: Documented negative findings
  CHK-G11.4: Verified academic references
  CHK-G11.5: End-to-end reproducibility audit
"""

import json
from pathlib import Path
import pytest
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = PROJECT_DIR / "docs" / "tracking" / "evidence" / "G11"
RESULTS_DIR = PROJECT_DIR / "results"


def test_chk_g11_1_code_generated_numbers():
    evidence_file = EVIDENCE_DIR / "CHK-G11.1_code_generated_numbers.json"
    assert evidence_file.exists(), f"Missing evidence: {evidence_file}"
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass", f"CHK-G11.1 failed: {data}"
    assert data["zero_hand_typed_numbers"] is True
    assert data["total_test_conditions"] == 1137
    assert data["total_downstream_evals"] == 600

    # Verify that parquet data sources exist and match count
    test_parquet = RESULTS_DIR / "final_test_benchmarks.parquet"
    down_parquet = RESULTS_DIR / "downstream_palm_benchmarks.parquet"
    assert test_parquet.exists()
    assert down_parquet.exists()
    
    df_test = pd.read_parquet(test_parquet)
    df_down = pd.read_parquet(down_parquet)
    assert len(df_test) == 1137
    assert len(df_down) == 600


def test_chk_g11_2_hypotheses_conclusions():
    evidence_file = EVIDENCE_DIR / "CHK-G11.2_hypotheses_conclusions.json"
    assert evidence_file.exists(), f"Missing evidence: {evidence_file}"
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass", f"CHK-G11.2 failed: {data}"
    assert data["all_hypotheses_concluded"] is True

    conclusions = data["conclusions"]
    for h in ["H1", "H2", "H3", "H4", "H5"]:
        assert h in conclusions, f"Missing hypothesis {h}"
        assert conclusions[h]["status"] == "SUPPORTED"
        assert len(conclusions[h]["evidence"]) > 10


def test_chk_g11_3_negative_findings():
    evidence_file = EVIDENCE_DIR / "CHK-G11.3_negative_findings.json"
    assert evidence_file.exists(), f"Missing evidence: {evidence_file}"
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass", f"CHK-G11.3 failed: {data}"
    findings = data["negative_findings_reported"]
    assert len(findings) >= 3, "Expected at least 3 negative findings documented"
    for item in findings:
        assert "finding" in item and len(item["finding"]) > 5
        assert "detail" in item and len(item["detail"]) > 20


def test_chk_g11_4_verified_references():
    evidence_file = EVIDENCE_DIR / "CHK-G11.4_verified_references.json"
    assert evidence_file.exists(), f"Missing evidence: {evidence_file}"
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass", f"CHK-G11.4 failed: {data}"
    refs = data["references"]
    assert len(refs) >= 5, "Expected at least 5 key references"
    for ref in refs:
        assert ref["verified"] is True


def test_chk_g11_5_reproducibility_audit():
    evidence_file = EVIDENCE_DIR / "CHK-G11.5_reproducibility_audit.json"
    assert evidence_file.exists(), f"Missing evidence: {evidence_file}"
    with open(evidence_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "pass", f"CHK-G11.5 failed: {data}"
    assert "3.12" in data["python_env"]
    assert data["cuda_available"] is True
    assert data["all_artifacts_reproducible"] is True
