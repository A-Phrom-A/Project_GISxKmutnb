"""
lib2_vitpalm.loss.combined
============================
Loss functions สำหรับ palm binary segmentation.

ให้เลือกใช้:
  - BCE + Dice (combined_loss)   ← แนะนำ สำหรับ imbalanced dataset
  - Dice only   (dice_loss)
  - BCE only    ผ่าน nn.BCELoss หรือ nn.BCEWithLogitsLoss โดยตรง

หมายเหตุ:
  - ฟังก์ชันทั้งหมดรับ pred ที่ผ่าน sigmoid แล้ว (prob map [0,1])
  - ยกเว้น bce_with_logits_loss ที่รับ raw logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft Dice Loss สำหรับ binary segmentation.

    Args:
        pred   : (B, H, W) probability map [0, 1] — หลัง sigmoid แล้ว
        target : (B, H, W) binary mask {0.0, 1.0}
        smooth : smoothing constant ป้องกัน division by zero

    Returns:
        Scalar loss ∈ [0, 1]
    """
    pred_flat   = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )


def bce_loss(
    pred:       torch.Tensor,
    target:     torch.Tensor,
    pos_weight: float | None = None,
) -> torch.Tensor:
    """
    Binary Cross-Entropy Loss (รับ prob map ที่ผ่าน sigmoid แล้ว).

    Args:
        pred       : (B, H, W) probability map [0, 1]
        target     : (B, H, W) binary mask {0.0, 1.0}
        pos_weight : weight สำหรับ positive class (ใช้เมื่อ palm พื้นที่น้อย)
                     เช่น pos_weight=3.0 ทำให้ palm pixels มี weight 3×

    Returns:
        Scalar loss
    """
    if pos_weight is not None:
        weight = torch.ones_like(target)
        weight[target == 1] = pos_weight
        return F.binary_cross_entropy(pred, target, weight=weight)
    return F.binary_cross_entropy(pred, target)


def combined_loss(
    pred:       torch.Tensor,
    target:     torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    pos_weight: float | None = 3.0,
) -> torch.Tensor:
    """
    Combined BCE + Dice Loss — แนะนำสำหรับ imbalanced palm segmentation.

    Args:
        pred        : (B, H, W) probability map [0, 1] — หลัง sigmoid
        target      : (B, H, W) binary mask {0.0, 1.0}
        bce_weight  : weight ของ BCE component (default 0.5)
        dice_weight : weight ของ Dice component (default 0.5)
        pos_weight  : positive class weight สำหรับ BCE (default 3.0)
                      set None เพื่อปิด

    Returns:
        Scalar combined loss
    """
    bce  = bce_loss(pred, target, pos_weight=pos_weight)
    dice = dice_loss(pred, target)
    return bce_weight * bce + dice_weight * dice


def mae_reconstruction_loss(
    pred:    torch.Tensor,
    target:  torch.Tensor,
    norm:    bool = True,
) -> torch.Tensor:
    """
    MSE reconstruction loss สำหรับ MAE pretraining.

    Args:
        pred   : (B, N_masked, patch_px) — predicted pixel values
        target : (B, N_masked, patch_px) — ground truth pixel values
        norm   : normalize loss per patch (default True — ตาม MAE paper)

    Returns:
        Scalar MSE loss
    """
    if norm:
        # Normalize target per patch (mean=0, std=1) ตาม MAE paper
        mean   = target.mean(dim=-1, keepdim=True)
        var    = target.var(dim=-1, keepdim=True, unbiased=False)
        target = (target - mean) / (var + 1e-6).sqrt()
    return F.mse_loss(pred, target)


class CombinedLoss(nn.Module):
    """
    nn.Module wrapper สำหรับ combined_loss — ใช้กับ Trainer ได้สะดวก

    Args:
        bce_weight  : float — BCE component weight (default 0.5)
        dice_weight : float — Dice component weight (default 0.5)
        pos_weight  : float — positive class weight (default 3.0)
    """

    def __init__(
        self,
        bce_weight:  float = 0.5,
        dice_weight: float = 0.5,
        pos_weight:  float = 3.0,
    ):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight  = pos_weight

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return combined_loss(
            pred, target,
            bce_weight=self.bce_weight,
            dice_weight=self.dice_weight,
            pos_weight=self.pos_weight,
        )
