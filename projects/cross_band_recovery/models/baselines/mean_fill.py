"""
projects.cross_band_recovery.models.baselines.mean_fill
=======================================================
Baseline B0: Empirical Mean-Fill Estimator.
Replaces masked bands with the empirical mean of the corresponding band
derived strictly from the training split (norm_stats.json).
"""

import json
from pathlib import Path
from typing import Optional, Union, Dict
import torch
import torch.nn as nn
import numpy as np


class MeanFillBaseline(nn.Module):
    def __init__(self, norm_stats_path: Optional[Union[str, Path]] = None):
        super().__init__()
        if norm_stats_path is None:
            norm_stats_path = Path(__file__).resolve().parents[2] / "data" / "manifests" / "norm_stats.json"
            
        with open(norm_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
            
        # (7,) optical mean
        if "s2_optical" in stats and "mean" in stats["s2_optical"]:
            s2_mean = stats["s2_optical"]["mean"]
        elif "s2_mean" in stats:
            s2_mean = stats["s2_mean"]
        else:
            raise KeyError("Neither 's2_optical.mean' nor 's2_mean' found in norm_stats.json")
        self.register_buffer("s2_mean", torch.tensor(s2_mean, dtype=torch.float32).view(1, 7, 1, 1))

    def forward(
        self,
        opt: torch.Tensor,
        sar: Optional[torch.Tensor] = None,
        mask_opt: Optional[torch.Tensor] = None,
        mask_sar: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        opt: (B, 7, H, W)
        mask_opt: (B, 7) bool (True = MASKED)
        Returns:
            pred_opt: (B, 7, H, W) with masked bands filled with train mean.
        """
        pred_opt = opt.clone()
        if mask_opt is None:
            return pred_opt
            
        B, C, H, W = opt.shape
        mean_expanded = self.s2_mean.expand(B, 7, H, W)
        mask_expanded = mask_opt.unsqueeze(-1).unsqueeze(-1).expand(B, 7, H, W)
        
        pred_opt = torch.where(mask_expanded, mean_expanded, opt)
        return pred_opt
