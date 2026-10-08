"""
lib3_geoai.benchmark.runner
============================
BenchmarkRunner — รัน pipeline หลายๆ แบบเพื่อเปรียบเทียบผลลัพธ์โดยอัตโนมัติ

Usage:
    from lib3_geoai import GeoAI
    results = GeoAI.benchmark(fusions=["early", "late", "cross"])
    results.compare()
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

from lib3_geoai.config import GeoAIConfig
from lib3_geoai.pipeline.palm import PalmPipeline


class BenchmarkResults:
    """เก็บและแสดงผลลัพธ์จากการรัน Benchmark"""

    def __init__(self, metrics: dict[str, dict], save_dir: Path):
        self.metrics = metrics
        self.save_dir = save_dir
        self.df = pd.DataFrame.from_dict(metrics, orient="index")
        
        # Reorder columns
        cols = ["f1", "iou", "precision", "recall"]
        available_cols = [c for c in cols if c in self.df.columns]
        self.df = self.df[available_cols]

    def compare(self) -> None:
        """แสดงตารางเปรียบเทียบผลลัพธ์"""
        print("\n=== Benchmark Results ===")
        print(self.df.to_markdown(floatfmt=".4f"))
        
        if not self.df.empty and "f1" in self.df.columns:
            best_fusion = self.df["f1"].idxmax()
            print(f"\n⭐ Best Model by F1-Score: {best_fusion} ({self.df.loc[best_fusion, 'f1']:.4f})")
            
        csv_path = self.save_dir / "benchmark_results.csv"
        self.df.to_csv(csv_path)
        print(f"\nSaved results to: {csv_path}")

    def best_model(self) -> str:
        """คืนค่าชื่อ fusion ที่ดีที่สุด (ตาม F1-Score)"""
        if not self.df.empty and "f1" in self.df.columns:
            return self.df["f1"].idxmax()
        return list(self.metrics.keys())[0]


class BenchmarkRunner:
    """
    รัน PalmPipeline ด้วย fusion หลายๆ แบบเพื่อเทียบประสิทธิภาพ
    """

    @staticmethod
    def run(
        data_dir: str | Path,
        fusions: list[str] = ["early", "late", "cross"],
        epochs: int = 50,
        save_dir: str | Path = "results/benchmark",
        use_amp: bool = True,
    ) -> BenchmarkResults:
        """
        เริ่มการทำ Benchmark

        Args:
            data_dir : path ไปยัง data split (train/ val/ test/)
            fusions  : list ของ fusion strategies ที่ต้องการรัน
            epochs   : จำนวน epoch สำหรับแต่ละโมเดล
            save_dir : directory หลักสำหรับเซฟผลลัพธ์
            use_amp  : ใช้ Mixed Precision หรือไม่
            
        Returns:
            BenchmarkResults object
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        all_metrics = {}

        for fusion in fusions:
            print(f"\n{'-'*60}")
            print(f"🚀 Running Benchmark for: {fusion.upper()} FUSION")
            print(f"{'-'*60}")
            
            # ตั้ง config แยกสำหรับแต่ละ fusion
            config = GeoAIConfig.palm_default()
            config.data_dir = data_dir
            config.fusion = fusion
            config.epochs = epochs
            config.use_amp = use_amp
            config.save_dir = save_dir / fusion
            
            # รัน pipeline
            pipeline = PalmPipeline(config=config)
            
            # เทรน
            try:
                pipeline.fit()
                
                # ประเมินผล (โหลด best checkpoint อัตโนมัติ)
                best_ckpt = pipeline.best_checkpoint
                print(f"\nEvaluating {fusion} on test set...")
                metrics = pipeline.evaluate(ckpt_path=best_ckpt)
                
                all_metrics[fusion] = metrics
                
            except Exception as e:
                print(f"❌ Error running {fusion} fusion: {e}")
                all_metrics[fusion] = {"error": str(e)}

        return BenchmarkResults(all_metrics, save_dir)
