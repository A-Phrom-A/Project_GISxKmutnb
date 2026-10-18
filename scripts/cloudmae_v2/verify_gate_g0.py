"""
scripts/cloudmae_v2/verify_gate_g0.py
====================================
Validates all Gate G0 conditions before transitioning to Phase 1.
"""

import os
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


def verify_gate_g0():
    reports_dir = ROOT_DIR / "reports/cloudmae_v2"
    
    req_files = [
        "s01_environment.json",
        "s02_inventory.json",
        "s03_ref_availability.json",
        "s04_mask_pilot.json",
        "s05_baseline_b0.json"
    ]
    
    details = {}
    all_ok = True
    
    for f_name in req_files:
        f_path = reports_dir / f_name
        if not f_path.exists():
            details[f_name] = {"exists": False, "passed": False}
            all_ok = False
            continue
            
        with open(f_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            passed = data.get("all_passed", False)
            details[f_name] = {"exists": True, "passed": passed}
            if not passed:
                all_ok = False

    gate_result = {
        "gate": "Gate G0 - Phase 0 Completion",
        "status": "PASSED" if all_ok else "FAILED",
        "selected_mask_strategy": "Option 2: SCL + Optical Cloud Probability Ensemble (No local L1C required)",
        "phase0_reports": details,
        "unlocked_next_phase": "Phase 1: Production Mask Pipeline & Phase 2: Real Pair Mining"
    }
    
    out_path = reports_dir / "gate_g0.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gate_result, f, indent=2, ensure_ascii=False)
        
    print(f"Gate G0 Verification: {gate_result['status']}")
    print(f"Strategy: {gate_result['selected_mask_strategy']}")
    print(f"Saved to: {out_path}")
    return gate_result


if __name__ == "__main__":
    verify_gate_g0()
