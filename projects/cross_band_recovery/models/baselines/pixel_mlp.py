"""
projects.cross_band_recovery.models.baselines.pixel_mlp
======================================================
Baseline B2: Per-Pixel Multi-Layer Perceptron (No Spatial Context).
Receives observed optical bands + SAR bands + mask indicators per pixel.
Evaluates cross-spectral correlations in isolation without spatial context.
"""

from typing import Optional
import torch
import torch.nn as nn


class PerPixelMLP(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        # 7 optical + 2 SAR + 7 optical mask indicators + 2 SAR mask indicators = 18 features
        in_dim = 7 + 2 + 7 + 2
        out_dim = 7
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        opt: torch.Tensor,
        sar: Optional[torch.Tensor] = None,
        mask_opt: Optional[torch.Tensor] = None,
        mask_sar: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        opt: (B, 7, H, W)
        sar: (B, 2, H, W) or zeros
        mask_opt: (B, 7) bool
        mask_sar: (B, 2) bool
        """
        B, C, H, W = opt.shape
        device = opt.device
        
        if sar is None:
            sar = torch.zeros(B, 2, H, W, device=device, dtype=opt.dtype)
            mask_sar = torch.ones(B, 2, dtype=torch.bool, device=device)
        elif mask_sar is None:
            mask_sar = torch.zeros(B, 2, dtype=torch.bool, device=device)
            
        if mask_opt is None:
            mask_opt = torch.zeros(B, 7, dtype=torch.bool, device=device)
            
        # Zero out masked input channels
        mask_opt_exp = mask_opt.unsqueeze(-1).unsqueeze(-1).expand(B, 7, H, W)
        mask_sar_exp = mask_sar.unsqueeze(-1).unsqueeze(-1).expand(B, 2, H, W)
        
        opt_input = torch.where(mask_opt_exp, torch.zeros_like(opt), opt)
        sar_input = torch.where(mask_sar_exp, torch.zeros_like(sar), sar)
        
        # Floating point mask indicators: 1.0 = visible, 0.0 = masked
        opt_ind = (~mask_opt_exp).to(dtype=opt.dtype)
        sar_ind = (~mask_sar_exp).to(dtype=opt.dtype)
        
        # Concat all 18 features: (B, 18, H, W)
        feat = torch.cat([opt_input, sar_input, opt_ind, sar_ind], dim=1)
        
        # Permute to (B*H*W, 18)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, 18)
        
        # MLP forward
        pred_flat = self.net(feat_flat)
        
        # Reshape back to (B, 7, H, W)
        pred_opt = pred_flat.view(B, H, W, 7).permute(0, 3, 1, 2).contiguous()
        
        # Composite: keep original pixels for visible bands, take prediction for masked bands
        out_opt = torch.where(mask_opt_exp, pred_opt, opt)
        return out_opt
