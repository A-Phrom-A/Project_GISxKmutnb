"""
projects.cross_band_recovery.training.losses
===========================================
Unified Loss Functions for Cross-Band Satellite Reconstruction.

Components:
1. MaskedL1Loss: Reconstruction L1 loss strictly evaluated on masked optical channels.
2. CompositeSAMLoss: Spectral Angle Mapper computed on composite spectrum (Appendix 11.1).
3. CrossBandRecoveryLoss: Composite multi-objective loss combining L1 and SAM.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedL1Loss(nn.Module):
    """
    Mean Absolute Error computed strictly on masked optical bands.
    Guarantees that unmasked bands have zero gradient.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask_opt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            y_pred: Predicted optical imagery (B, 7, H, W).
            y_true: Ground truth optical imagery (B, 7, H, W).
            mask_opt: Optical masking boolean tensor (B, 7) where True = MASKED.
        Returns:
            loss: Scalar L1 loss across all masked pixels.
            per_band_loss: (7,) L1 loss for each optical band.
        """
        if isinstance(y_pred, dict):
            y_pred = y_pred["pred_opt"]
            
        B, C, H, W = y_pred.shape
        # Expand mask_opt to (B, C, H, W)
        mask_exp = mask_opt.unsqueeze(-1).unsqueeze(-1).expand_as(y_true).float()
        
        diff = torch.abs(y_pred - y_true) * mask_exp
        total_masked_elements = mask_exp.sum()
        
        if total_masked_elements > 0:
            loss = diff.sum() / (total_masked_elements + self.eps)
        else:
            loss = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype, requires_grad=True)
            
        # Per band average loss
        band_weights = mask_exp.sum(dim=(0, 2, 3))
        band_diffs = diff.sum(dim=(0, 2, 3))
        per_band_loss = band_diffs / (band_weights + self.eps)
        
        return loss, per_band_loss


class CompositeSAMLoss(nn.Module):
    """
    Composite Spectral Angle Mapper (SAM) Loss (Appendix 11.1).
    Constructs composite spectrum:
      V_composite[b] = y_pred[b] if b is masked else y_true[b]
      SAM = arccos( clamp( (V_composite . y_true) / (||V_composite|| * ||y_true|| + eps), -1 + eps2, 1 - eps2 ) )
    
    Safe for backpropagation across all k in [1..7] without NaN/Inf gradients.
    """
    def __init__(self, eps: float = 1e-7, eps2: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.eps2 = eps2

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask_opt: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            y_pred: (B, 7, H, W)
            y_true: (B, 7, H, W)
            mask_opt: (B, 7) bool (True = MASKED)
        Returns:
            Scalar SAM loss in radians.
        """
        if isinstance(y_pred, dict):
            y_pred = y_pred["pred_opt"]
        mask_exp = mask_opt.unsqueeze(-1).unsqueeze(-1).expand_as(y_true)
        
        # Form composite spectrum
        v_composite = torch.where(mask_exp, y_pred, y_true)
        
        # Channel-wise dot product: (B, H, W)
        dot_product = (v_composite * y_true).sum(dim=1)
        
        # Norms along channel dimension
        norm_comp = torch.norm(v_composite, p=2, dim=1)
        norm_true = torch.norm(y_true, p=2, dim=1)
        
        cosine = dot_product / (norm_comp * norm_true + self.eps)
        clamped_cosine = torch.clamp(cosine, -1.0 + self.eps2, 1.0 - self.eps2)
        
        sam_map = torch.acos(clamped_cosine)
        return sam_map.mean()


class CrossBandRecoveryLoss(nn.Module):
    """
    Combined Multi-Objective Loss:
      L_total = l1_weight * L_masked_l1 + sam_weight * L_composite_sam
    """
    def __init__(self, l1_weight: float = 1.0, sam_weight: float = 0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.sam_weight = sam_weight
        self.l1_loss = MaskedL1Loss()
        self.sam_loss = CompositeSAMLoss()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask_opt: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if isinstance(y_pred, dict):
            y_pred = y_pred["pred_opt"]
        loss_l1, per_band_l1 = self.l1_loss(y_pred, y_true, mask_opt)
        loss_sam = self.sam_loss(y_pred, y_true, mask_opt)
        
        total_loss = self.l1_weight * loss_l1 + self.sam_weight * loss_sam
        
        return {
            "loss": total_loss,
            "loss_l1": loss_l1,
            "loss_sam": loss_sam,
            "per_band_l1": per_band_l1
        }
