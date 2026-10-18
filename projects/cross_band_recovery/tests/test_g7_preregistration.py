"""
projects.cross_band_recovery.tests.test_g7_preregistration
==========================================================
Quality Gate G7 Verification Test Suite: Pre-Registration Freeze Protocol.

Checks covered:
- CHK-G7.1: Provisional thresholds permanently frozen with empirical grounding.
- CHK-G7.2: Thresholds derived from validation set and baselines prior to test.
- CHK-G7.3: Complete cryptographic SHA-256 hashes recorded and verified.
- CHK-G7.4: Single primary metric strictly declared for RQ1/RQ2.
"""

import hashlib
import json
from pathlib import Path
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
PREREG_FILE = PROJECT_DIR / "docs" / "tracking" / "preregistration.md"
EVIDENCE_DIR = PROJECT_DIR / "docs" / "tracking" / "evidence" / "G7"
CHECKPOINTS_DIR = PROJECT_DIR / "results" / "checkpoints"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_chk_g7_1_frozen_thresholds_defined():
    """CHK-G7.1: All provisional thresholds are defined and non-empty."""
    assert PREREG_FILE.exists(), f"Pre-registration file missing: {PREREG_FILE}"
    content = PREREG_FILE.read_text(encoding="utf-8")
    
    assert "Recovery Rate Tolerance" in content
    assert "0.050" in content
    assert "Graceful Degradation Margin" in content
    assert "5.0" in content
    assert "SHAP Additivity Tolerance" in content


def test_chk_g7_2_thresholds_derived_from_val_and_baselines():
    """CHK-G7.2: Criteria are explicitly grounded in Phase 3 & Phase 6 validation results."""
    content = PREREG_FILE.read_text(encoding="utf-8")
    assert "MAE" in content
    assert "B3" in content or "RandomForest" in content
    assert "Validation" in content


def test_chk_g7_3_cryptographic_hashes_integrity():
    """CHK-G7.3: Cryptographic hashes in preregistration match current active files."""
    content = PREREG_FILE.read_text(encoding="utf-8")
    
    # 1. Checkpoints
    seeds = [42, 101, 2024]
    for m in ["M1_S2only", "M2_Fusion"]:
        for s in seeds:
            chkpt_name = f"{m}_seed{s}_best.pt"
            chkpt_path = CHECKPOINTS_DIR / chkpt_name
            assert chkpt_path.exists(), f"Checkpoint {chkpt_name} missing!"
            h = compute_sha256(chkpt_path)
            assert h in content, f"Hash for {chkpt_name} ({h}) not found in preregistration.md!"
            
    # 2. Manifests & Configs
    split_path = PROJECT_DIR / "data" / "manifests" / "spatial_split.parquet"
    assert split_path.exists()
    assert compute_sha256(split_path) in content
    
    config_path = PROJECT_DIR / "configs" / "base.yaml"
    assert config_path.exists()
    assert compute_sha256(config_path) in content
    
    # 3. Evaluation script
    eval_script = PROJECT_DIR / "scripts" / "04_evaluate_final_test.py"
    assert eval_script.exists()
    assert compute_sha256(eval_script) in content


def test_chk_g7_4_single_primary_metric_declared():
    """CHK-G7.4: Single primary metric is unequivocally declared for RQ1/RQ2."""
    content = PREREG_FILE.read_text(encoding="utf-8")
    assert "Single Metric for RQ1 and RQ2" in content
    assert "Masked Spectral Mean Absolute Error" in content
    assert "Composite Spectral Angle Mapper" in content
