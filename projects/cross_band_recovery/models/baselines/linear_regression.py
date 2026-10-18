"""
projects.cross_band_recovery.models.baselines.linear_regression
==============================================================
Baseline B1: Per-Pixel Ridge Linear Regression.
Maps visible optical and SAR channels directly to masked optical channels.
"""

from typing import Optional
import torch
import torch.nn as nn


class PerPixelLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # 18 input features -> 7 optical channels (pure linear model)
        self.linear = nn.Linear(18, 7, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(
        self,
        opt: torch.Tensor,
        sar: Optional[torch.Tensor] = None,
        mask_opt: Optional[torch.Tensor] = None,
        mask_sar: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, 18)
        
        pred_flat = self.linear(feat_flat)
        pred_opt = pred_flat.view(B, H, W, 7).permute(0, 3, 1, 2).contiguous()
        
        out_opt = torch.where(mask_opt_exp, pred_opt, opt)
        return out_opt
