"""
lib2_vitpalm
=============
ViTPalm — Multi-Modal Satellite Palm Detection Library

Library ที่ 2 ของโปรเจค GISxKMUTNB
รองรับ:
  1. Self-supervised pretraining ด้วย Masked Autoencoder (MAE)
  2. Fine-tuning สำหรับ palm binary segmentation

Input: 12-band fused satellite tile (S1 VV+VH | S2 x7 | Gsat RGB)
       shape (B, 12, 256, 256), float32, [0, 1]

Output: Palm probability map (B, 256, 256) ∈ [0, 1]

Quick start:
    from lib2_vitpalm import ViTPalmModel, MAEModel, Trainer, CombinedLoss

    # Fine-tuning
    model = ViTPalmModel()
    model.load_mae_weights("checkpoints/mae_pretrained.pth")

    # Training
    trainer = Trainer(model, train_loader, val_loader, output_dir="checkpoints/")
    history = trainer.train()
"""

from lib2_vitpalm.model    import ViTPalmModel, MAEModel, ViTEncoder, SegDecoder, PatchEmbed
from lib2_vitpalm.loss     import CombinedLoss, combined_loss, dice_loss, mae_reconstruction_loss
from lib2_vitpalm.training import Trainer, compute_metrics

__version__ = "0.1.0"

__all__ = [
    # Model
    "ViTPalmModel",
    "MAEModel",
    "ViTEncoder",
    "SegDecoder",
    "PatchEmbed",
    # Loss
    "CombinedLoss",
    "combined_loss",
    "dice_loss",
    "mae_reconstruction_loss",
    # Training
    "Trainer",
    "compute_metrics",
]
