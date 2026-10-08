"""
lib3_geoai.builder
===================
GeoAI factory class — จุดเข้าใช้งานหลัก (Entry Point) สำหรับ lib3_geoai

ใช้งานผ่าน:
    from lib3_geoai import GeoAI

    palm_model = GeoAI.palm(...)
    custom_model = GeoAI.custom(...)
    results = GeoAI.benchmark(...)
    mae = GeoAI.pretrain(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

from lib3_geoai.config import GeoAIConfig
from lib3_geoai.pipeline.palm import PalmPipeline
from lib3_geoai.pipeline.custom import CustomPipeline
from lib3_geoai.benchmark.runner import BenchmarkRunner, BenchmarkResults
from lib3_geoai.pretrain.cloud_mae import CloudMAE


class GeoAI:
    """
    Factory class สำหรับสร้าง GeoAI Pipelines แบบต่างๆ
    """

    @staticmethod
    def palm(
        data_dir:   Optional[str | Path] = None,
        fusion:     Optional[Literal["early", "late", "cross"]] = None,
        pretrained: Optional[str | Path] = None,
        config:     Optional[GeoAIConfig] = None,
        **kwargs
    ) -> PalmPipeline:
        """
        สร้าง Pipeline สำหรับตรวจจับสวนปาล์ม (Use Case 1)

        Args:
            data_dir   : path ไปยัง data split (train/ val/ test/)
            fusion     : รูปแบบการรวมข้อมูล ("early", "late", "cross")
            pretrained : path ไปยัง MAE checkpoint (optional)
            config     : GeoAIConfig object (ถ้ามี จะข้าม kwargs อื่นๆ)
            **kwargs   : ใช้ override ค่าต่างๆ ใน GeoAIConfig (เช่น epochs=50)

        Returns:
            PalmPipeline instance
        """
        if config is None:
            config = GeoAIConfig.palm_default()
            if data_dir: config.data_dir = Path(data_dir)
            if fusion:   config.fusion = fusion
            
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
                else:
                    raise ValueError(f"GeoAIConfig ไม่มี parameter ชื่อ '{k}'")

        return PalmPipeline(config=config, pretrained=pretrained)

    @staticmethod
    def custom(
        data_dir:    Optional[str | Path] = None,
        in_channels: Optional[int] = None,
        fusion:      Optional[Literal["early", "late", "cross"]] = None,
        pretrained:  Optional[str | Path] = None,
        config:      Optional[GeoAIConfig] = None,
        **kwargs
    ) -> CustomPipeline:
        """
        สร้าง Pipeline สำหรับข้อมูลรูปแบบอื่นที่กำหนดเอง (Use Case 2)

        Args:
            data_dir    : path ไปยัง data split
            in_channels : จำนวน bands ของ input
            fusion      : รูปแบบการรวมข้อมูล ("early", "late", "cross")
            pretrained  : path ไปยัง MAE checkpoint (optional)
            config      : GeoAIConfig object
            **kwargs    : ใช้ override ค่าต่างๆ ใน GeoAIConfig

        Returns:
            CustomPipeline instance
        """
        if config is None:
            config = GeoAIConfig()
            if data_dir:    config.data_dir = Path(data_dir)
            if in_channels: config.in_channels = in_channels
            if fusion:      config.fusion = fusion
            
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
                else:
                    raise ValueError(f"GeoAIConfig ไม่มี parameter ชื่อ '{k}'")

        return CustomPipeline(config=config, pretrained=pretrained)

    @staticmethod
    def benchmark(
        data_dir: str | Path,
        fusions:  list[str] = ["early", "late", "cross"],
        epochs:   int = 50,
        save_dir: str | Path = "results/benchmark",
        use_amp:  bool = True,
    ) -> BenchmarkResults:
        """
        รันการเปรียบเทียบ Fusion Strategies ทั้งหมดแบบอัตโนมัติ

        Args:
            data_dir : path ไปยัง data split
            fusions  : รูปแบบ fusion ที่ต้องการเทรนเปรียบเทียบ
            epochs   : จำนวน epoch สำหรับการเทรน
            save_dir : directory สำหรับเซฟผลลัพธ์
            use_amp  : ใช้ Mixed Precision หรือไม่
            
        Returns:
            BenchmarkResults object (สามารถเรียก .compare() ดูตารางผลลัพธ์ได้)
        """
        return BenchmarkRunner.run(
            data_dir=data_dir,
            fusions=fusions,
            epochs=epochs,
            save_dir=save_dir,
            use_amp=use_amp
        )

    @staticmethod
    def pretrain(
        data_dir:       Optional[str | Path] = None,
        mask_strategy:  Literal["random", "cloud", "band", "mixed"] = "random",
        cloud_mask_dir: Optional[str | Path] = None,
        config:         Optional[GeoAIConfig] = None,
        **kwargs
    ) -> CloudMAE:
        """
        สร้าง MAE Pretraining Pipeline (รองรับ Cloud Mask)

        Args:
            data_dir       : path ไปยัง tiles/ 
            mask_strategy  : "random" หรือ "cloud"
            cloud_mask_dir : path ไปยัง cloud masks (ถ้า strategy="cloud")
            config         : GeoAIConfig object
            **kwargs       : override config

        Returns:
            CloudMAE instance
        """
        if config is None:
            config = GeoAIConfig.mae_default()
            if data_dir:       config.data_dir = Path(data_dir)
            if mask_strategy:  config.mask_strategy = mask_strategy
            if cloud_mask_dir: config.cloud_mask_dir = Path(cloud_mask_dir)
            
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
                else:
                    raise ValueError(f"GeoAIConfig ไม่มี parameter ชื่อ '{k}'")

        return CloudMAE(config=config)
