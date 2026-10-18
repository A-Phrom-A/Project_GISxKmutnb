import json
from pathlib import Path
import pandas as pd
import pytest

def test_s02_inventory_manifest_and_report():
    report_path = Path("reports/cloudmae_v2/s02_inventory.json")
    manifest_path = Path("data/cloudmae_v2/manifests/scenes.parquet")
    
    assert report_path.exists(), "Report s02_inventory.json must exist"
    assert manifest_path.exists(), "Manifest scenes.parquet must exist"
    
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    assert report["all_passed"] is True
    assert report["chk_02a_scene_inventory"]["status"] == "PASSED"
    assert report["chk_02b_schema_resolution"]["status"] == "PASSED"
    assert report["chk_02c_nodata_check"]["status"] == "PASSED"
    assert report["chk_02d_s1_s2_pairing"]["status"] == "PASSED"
    assert report["chk_02e_palm_layer"]["status"] == "PASSED"
    assert report["pretrain_tiles_count"] == 8996
    assert report["chk_02e_palm_layer"]["features_count"] == 4726
    
    df = pd.read_parquet(manifest_path)
    assert len(df) > 0
    assert "scene_id" in df.columns
    assert "bands" in df.columns
    assert "crs" in df.columns
