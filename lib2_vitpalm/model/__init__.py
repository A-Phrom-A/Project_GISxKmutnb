"""
lib2_vitpalm.model
==================
Model components for ViTPalm.

Public API:
    PatchEmbed     — image → patch token sequence
    ViTEncoder     — transformer encoder (supports MAE masking)
    SegDecoder     — token sequence → palm probability map
    ViTPalmModel   — full fine-tuning model (encoder + decoder)
    MAEModel       — self-supervised pretraining model
"""

from lib2_vitpalm.model.patch_embed import PatchEmbed
from lib2_vitpalm.model.vit_encoder import ViTEncoder
from lib2_vitpalm.model.seg_decoder import SegDecoder
from lib2_vitpalm.model.vitpalm    import ViTPalmModel
from lib2_vitpalm.model.mae_model  import MAEModel, MAEDecoder, random_masking
from lib2_vitpalm.model.cloud_mae  import CloudMAEModel, CloudMAEEncoder, CloudMaskGenerator
from lib2_vitpalm.model.late_fusion_model import ViTLateFusionModel
from lib2_vitpalm.model.cross_attn_model  import ViTCrossAttnModel

__all__ = [
    "PatchEmbed",
    "ViTEncoder",
    "SegDecoder",
    "ViTPalmModel",
    "ViTLateFusionModel",
    "ViTCrossAttnModel",
    "MAEModel",
    "MAEDecoder",
    "random_masking",
    "CloudMAEModel",
    "CloudMAEEncoder",
    "CloudMaskGenerator",
]
