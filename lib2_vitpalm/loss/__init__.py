"""
lib2_vitpalm.loss
==================
Loss functions สำหรับ ViTPalm training.

Public API:
    dice_loss               — Soft Dice Loss (functional)
    bce_loss                — Binary Cross-Entropy (functional, รับ prob)
    combined_loss           — BCE + Dice (functional)
    mae_reconstruction_loss — MSE reconstruction loss สำหรับ MAE
    CombinedLoss            — nn.Module wrapper สำหรับ combined_loss
"""

from lib2_vitpalm.loss.combined import (
    dice_loss,
    bce_loss,
    combined_loss,
    mae_reconstruction_loss,
    CombinedLoss,
)

__all__ = [
    "dice_loss",
    "bce_loss",
    "combined_loss",
    "mae_reconstruction_loss",
    "CombinedLoss",
]
