"""
lib3_geoai
===========
A High-level API for GEO + AI (Vision Transformers on Satellite Imagery).

Provides easy-to-use pipelines for:
  - Palm Oil Detection (GeoAI.palm)
  - Custom Satellite Segmentation Tasks (GeoAI.custom)
  - Automated Benchmarking (GeoAI.benchmark)
  - Self-Supervised Pretraining (GeoAI.pretrain)
"""

from lib3_geoai.config import GeoAIConfig
from lib3_geoai.builder import GeoAI

__all__ = [
    "GeoAIConfig",
    "GeoAI",
]
