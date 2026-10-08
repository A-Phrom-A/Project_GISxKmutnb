"""
lib2_vitpalm.training
======================
Training utilities สำหรับ ViTPalm.

Public API:
    Trainer        — fine-tuning train/val loop สำหรับ ViTPalmModel
    compute_metrics — IoU, F1, Precision, Recall สำหรับ binary segmentation
"""

from lib2_vitpalm.training.trainer import Trainer, compute_metrics

__all__ = [
    "Trainer",
    "compute_metrics",
]
