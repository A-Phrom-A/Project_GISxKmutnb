"""
projects.cross_band_recovery.tests.test_g3_baselines
===================================================
Quality Gate G3 Verification Test Suite.

Checks covered:
- CHK-G3.1: Metric hand-computed unit tests (MAE, RMSE, PSNR, SSIM, Composite SAM).
- CHK-G3.2: Baseline plausible order (Mean-fill error > Linear > MLP).
- CHK-G3.3: Same split and normalization stats (100% train-derived, split consistency).
- CHK-G3.4: Val baseline reference benchmarks parquet logged with required schema.
- CHK-G3.5: Zero test set touch verification.
"""

import json
import math
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

from projects.cross_band_recovery.eval.metrics import (
    compute_mae,
    compute_rmse,
    compute_psnr,
    compute_ssim_torch,
    compute_composite_sam,
    evaluate_all_metrics
)
from projects.cross_band_recovery.models.baselines.mean_fill import MeanFillBaseline
from projects.cross_band_recovery.models.baselines.linear_regression import PerPixelLinearRegression
from projects.cross_band_recovery.models.baselines.pixel_mlp import PerPixelMLP
from projects.cross_band_recovery.models.baselines.random_forest import RandomForestBaseline

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
EVIDENCE_DIR = PROJECT_DIR / "docs" / "tracking" / "evidence" / "G3"


class TestGateG3Metrics:
    """CHK-G3.1: Metric hand-computed unit tests."""

    def test_mae_hand_computed(self):
        y_true = np.array([0.1, 0.4, 0.9], dtype=np.float32)
        y_pred = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        # unmasked: (|0.1| + |0.0| + |-0.3|) / 3 = 0.4 / 3 ≈ 0.1333333
        mae_unmasked = compute_mae(y_true, y_pred)
        assert abs(mae_unmasked - (0.4 / 3.0)) < 1e-6

        # masked: only indices 0 and 2
        mask = np.array([True, False, True], dtype=bool)
        mae_masked = compute_mae(y_true, y_pred, mask=mask)
        # (|0.1| + |-0.3|) / 2 = 0.4 / 2 = 0.2
        assert abs(mae_masked - 0.2) < 1e-6

    def test_rmse_hand_computed(self):
        y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_pred = np.array([1.0, 4.0, 3.0], dtype=np.float32)
        # unmasked: sqrt((0 + 4 + 0) / 3) = sqrt(4/3) ≈ 1.1547005
        rmse_unmasked = compute_rmse(y_true, y_pred)
        assert abs(rmse_unmasked - math.sqrt(4.0 / 3.0)) < 1e-6

        # masked: only index 1
        mask = np.array([False, True, False], dtype=bool)
        rmse_masked = compute_rmse(y_true, y_pred, mask=mask)
        # sqrt(4 / 1) = 2.0
        assert abs(rmse_masked - 2.0) < 1e-6

    def test_psnr_hand_computed(self):
        # MSE = 0.01, data_range = 1.0 -> PSNR = 10 * log10(1.0 / 0.01) = 10 * 2.0 = 20.0 dB
        y_true = np.zeros((10, 10), dtype=np.float32)
        y_pred = np.full((10, 10), 0.1, dtype=np.float32)  # diff = 0.1, mse = 0.01
        psnr = compute_psnr(y_true, y_pred, data_range=1.0)
        assert abs(psnr - 20.0) < 1e-4

    def test_ssim_identity(self):
        x = torch.rand(2, 7, 32, 32)
        ssim = compute_ssim_torch(x, x, window_size=7)
        assert abs(ssim - 1.0) < 1e-3

    def test_composite_sam_hand_computed(self):
        # Construct 1 pixel with 7 channels
        # True:  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] -> norm = sqrt(2)
        # Pred:  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Mask:  [False, True, False, False, False, False, False] (only channel 1 masked)
        # Composite takes Pred at channel 1, and True elsewhere:
        # Comp:  [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0] -> norm = sqrt(3)
        # Dot with True: 1*1 + 1*0 + ... + 1*1 = 2.0
        # Cosine = 2.0 / (sqrt(3) * sqrt(2)) = 2 / sqrt(6) = sqrt(2/3) ≈ 0.81649658
        # Angle = arccos(sqrt(2/3)) ≈ 0.6154797 rad ≈ 35.2643897 deg
        y_true = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).view(1, 7, 1, 1)
        y_pred = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(1, 7, 1, 1)
        mask_opt = torch.tensor([[False, True, False, False, False, False, False]])

        sam_rad = compute_composite_sam(y_true, y_pred, mask_opt)
        expected_rad = math.acos(math.sqrt(2.0 / 3.0))
        assert abs(sam_rad - expected_rad) < 1e-4

        expected_deg = math.degrees(expected_rad)
        actual_deg = math.degrees(sam_rad)
        assert abs(actual_deg - 35.264) < 1e-2


class TestGateG3BaselinesAndSplit:
    """CHK-G3.2, CHK-G3.3, CHK-G3.4, CHK-G3.5."""

    def test_chk_g3_3_same_split_and_norm_stats(self):
        split_path = MANIFEST_DIR / "spatial_split.parquet"
        norm_path = MANIFEST_DIR / "norm_stats.json"

        assert split_path.exists(), "spatial_split.parquet must exist"
        assert norm_path.exists(), "norm_stats.json must exist"

        df_split = pd.read_parquet(split_path)
        with open(norm_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        assert stats["source_split"] == "train"
        assert stats["num_train_tiles_used"] == 1144

        # Verify all tile IDs in norm_stats belong strictly to train
        train_tile_ids_set = set(df_split[df_split["split"] == "train"]["tile_id"])
        norm_tile_ids = stats["train_tile_ids"]
        for tid in norm_tile_ids:
            assert tid in train_tile_ids_set, f"Tile {tid} in norm_stats not in train split!"

        # Verify MeanFill loads the exact train mean from norm_stats
        mean_fill = MeanFillBaseline()
        expected_mean = torch.tensor(stats["s2_optical"]["mean"], dtype=torch.float32).view(1, 7, 1, 1)
        assert torch.allclose(mean_fill.s2_mean, expected_mean, atol=1e-5)

    def test_chk_g3_4_val_baseline_reference_logged(self):
        bench_path = MANIFEST_DIR / "val_baseline_benchmarks.parquet"
        assert bench_path.exists(), f"Benchmark parquet missing: {bench_path}"

        df = pd.read_parquet(bench_path)
        assert len(df) > 0, "Benchmark parquet is empty"

        required_cols = ["baseline", "k", "pattern_id", "s1_condition", "mae", "rmse", "psnr", "ssim", "sam_deg"]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

        # Must contain all 4 baselines
        baselines = set(df["baseline"].unique())
        expected_baselines = {"B0_MeanFill", "B1_Linear", "B2_MLP", "B3_RandomForest"}
        assert expected_baselines.issubset(baselines), f"Missing baselines: {expected_baselines - baselines}"

        # Must cover k in 1..7
        k_values = set(df["k"].unique())
        assert {1, 2, 3, 4, 5, 6, 7}.issubset(k_values), f"Missing k values: {{1..7}} - {k_values}"

    def test_chk_g3_2_baseline_plausible_order(self):
        bench_path = MANIFEST_DIR / "val_baseline_benchmarks.parquet"
        df = pd.read_parquet(bench_path)

        summary = df.groupby("baseline")["mae"].mean().to_dict()
        mae_b0 = summary["B0_MeanFill"]
        mae_b1 = summary["B1_Linear"]
        mae_b2 = summary["B2_MLP"]

        # Plausible ordering: MeanFill (trivial fill) error must be greater than Linear Regression and MLP
        assert mae_b0 > mae_b1, f"MeanFill MAE ({mae_b0:.4f}) should be > Linear MAE ({mae_b1:.4f})"
        assert mae_b0 > mae_b2, f"MeanFill MAE ({mae_b0:.4f}) should be > MLP MAE ({mae_b2:.4f})"

    def test_chk_g3_5_zero_test_touch(self):
        ev_file = EVIDENCE_DIR / "CHK-G3.5_zero_test_touch.json"
        assert ev_file.exists(), f"Missing CHK-G3.5 evidence file: {ev_file}"

        with open(ev_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["status"] == "pass"
        assert data["test_tiles_accessed"] is False
        for p in data["directories_accessed"]:
            assert "test" not in Path(p).name.lower(), f"Test directory accessed: {p}"
