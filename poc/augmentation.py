"""
poc/augmentation.py
===================
Data Augmentation pipeline using Albumentations for semantic segmentation.

Includes spatial augmentations (flips, rotations) which are applied identically 
to both the image and the mask, and pixel-level augmentations (noise, etc.) 
applied only to the image.
"""

import albumentations as A
import numpy as np

def get_training_augmentation() -> A.Compose:
    """
    Returns the augmentation pipeline for training data.
    """
    return A.Compose([
        # Spatial: Applied to both image and mask
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        
        # Pixel-level: Applied only to image
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        
        # We don't use RandomBrightnessContrast heavily here because the channels 
        # represent specific physical reflectances, not standard RGB.
        # But a small random multiplier can act as "illumination" augmentation.
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.2, elementwise=True),
    ])

def get_validation_augmentation() -> A.Compose:
    """
    Returns the augmentation pipeline for validation/test data (no aug).
    """
    return A.Compose([])
