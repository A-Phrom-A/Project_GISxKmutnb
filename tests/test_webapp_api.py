"""
tests/test_webapp_api.py
========================
Automated regression tests for FastAPI backend and GPU inference endpoints.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_check():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_list_zones():
    res = client.get("/api/zones")
    assert res.status_code == 200
    zones = res.json()
    assert len(zones) >= 4
    assert zones[0]["zone_id"] == 1


def test_zone_geojson():
    res = client.get("/api/zones/1/palms")
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 1000


def test_available_models():
    res = client.get("/api/inference/models")
    assert res.status_code == 200
    models = res.json()
    assert len(models) == 6


def test_admin_stats():
    res = client.get("/api/admin/stats")
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert "gpu" in data


def test_cloud_repair_endpoint():
    res = client.post("/api/cloud-repair", json={"zone_id": 1, "cloud_ratio": 0.5, "offset_x": 900, "offset_y": 900})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "reconstructed" in data["images"]


def test_segmentation_inference_endpoint():
    res = client.post("/api/inference", json={"zone_id": 1, "model_id": "late_fusion_cloudmae", "offset_x": 900, "offset_y": 900})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "probability_heatmap" in data["images"]


def test_audit_logs():
    res = client.get("/api/admin/logs?limit=5")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_research_endpoints():
    # 1. Research Overview
    res = client.get("/api/research/overview")
    assert res.status_code == 200
    data = res.json()
    assert "hypotheses" in data
    assert len(data["hypotheses"]) == 5

    # 2. Benchmarks
    res = client.get("/api/research/benchmarks")
    assert res.status_code == 200
    b_data = res.json()
    assert len(b_data["models"]) >= 2
    assert len(b_data["k_progression"]) >= 6

    # 3. Downstream
    res = client.get("/api/research/downstream")
    assert res.status_code == 200
    d_data = res.json()
    assert len(d_data["conditions"]) == 5

    # 4. Quality Gates
    res = client.get("/api/research/quality-gates")
    assert res.status_code == 200
    gates = res.json()
    assert len(gates) == 13

    # 5. Serve HTML
    res = client.get("/research")
    assert res.status_code == 200
    assert "Cross-Band" in res.text


if __name__ == "__main__":
    test_health_check()
    test_list_zones()
    test_zone_geojson()
    test_available_models()
    test_admin_stats()
    test_cloud_repair_endpoint()
    test_segmentation_inference_endpoint()
    test_audit_logs()
    test_research_endpoints()
    print("All web app API tests passed successfully!")
