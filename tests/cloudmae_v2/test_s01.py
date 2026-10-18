import json
from pathlib import Path
import pytest
import torch

def test_s01_environment_report():
    report_path = Path("reports/cloudmae_v2/s01_environment.json")
    assert report_path.exists(), "Report s01_environment.json must exist"
    
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    assert report["chk_01a_reproducibility"]["status"] == "PASSED"
    assert report["chk_01b_bf16_rtx5060ti"]["status"] == "PASSED"
    assert report["all_passed"] is True
    assert report["cuda_available"] is True
    assert report["bf16_supported"] is True
