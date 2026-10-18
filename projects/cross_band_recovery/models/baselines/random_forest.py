"""
projects.cross_band_recovery.models.baselines.random_forest
===========================================================
Baseline B3: Random Forest Regressor on Pixel Features.
Fitted on subsampled pixel features from the training split.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor


class RandomForestBaseline(nn.Module):
    def __init__(self, n_estimators: int = 50, max_depth: int = 12):
        super().__init__()
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit_from_pixels(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        X_train: (N_pixels, 18)
        y_train: (N_pixels, 7)
        """
        self.rf.fit(X_train, y_train)
        self.is_fitted = True

    def forward(
        self,
        opt: torch.Tensor,
        sar: Optional[torch.Tensor] = None,
        mask_opt: Optional[torch.Tensor] = None,
        mask_sar: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("RandomForestBaseline must be fitted before calling forward!")
            
        B, C, H, W = opt.shape
        device = opt.device
        
        if sar is None:
            sar = torch.zeros(B, 2, H, W, device=device, dtype=opt.dtype)
            mask_sar = torch.ones(B, 2, dtype=torch.bool, device=device)
        elif mask_sar is None:
            mask_sar = torch.zeros(B, 2, dtype=torch.bool, device=device)
            
        if mask_opt is None:
            mask_opt = torch.zeros(B, 7, dtype=torch.bool, device=device)
            
        mask_opt_exp = mask_opt.unsqueeze(-1).unsqueeze(-1).expand(B, 7, H, W)
        mask_sar_exp = mask_sar.unsqueeze(-1).unsqueeze(-1).expand(B, 2, H, W)
        
        opt_input = torch.where(mask_opt_exp, torch.zeros_like(opt), opt)
        sar_input = torch.where(mask_sar_exp, torch.zeros_like(sar), sar)
        
        opt_ind = (~mask_opt_exp).to(dtype=opt.dtype)
        sar_ind = (~mask_sar_exp).to(dtype=opt.dtype)
        
        feat = torch.cat([opt_input, sar_input, opt_ind, sar_ind], dim=1)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, 18).cpu().numpy()
        
        preds_flat = self.rf.predict(feat_flat)
        pred_opt = torch.from_numpy(preds_flat).to(device=device, dtype=opt.dtype)
        pred_opt = pred_opt.view(B, H, W, 7).permute(0, 3, 1, 2).contiguous()
        
        out_opt = torch.where(mask_opt_exp, pred_opt, opt)
        return out_opt
