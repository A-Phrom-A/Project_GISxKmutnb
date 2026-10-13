"""
poc/run_full_pipeline.py
========================
Automated End-to-End Pipeline:
Stage 1: Cloud-MAE Pretraining (500 Epochs, Early Stop 20) on 5 Eastern Provinces
Stage 2: Master Benchmark & Downstream Fine-Tuning (EXP-01 to EXP-06) on Ban Bueng Labeled Data
"""

import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_stage(cmd: list[str], stage_name: str) -> None:
    print("=" * 70, flush=True)
    print(f"  [>] LAUNCHING {stage_name}", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[ERROR] {stage_name} failed with exit code {res.returncode}", flush=True)
        sys.exit(res.returncode)
    print(f"\n[DONE] {stage_name} completed in {elapsed/60:.1f} minutes.\n", flush=True)


def main() -> None:
    python_exe = sys.executable

    # Stage 1: Cloud-MAE Pretraining
    run_stage([python_exe, "-u", "poc/train_cloud_mae.py"], "STAGE 1: CLOUD-MAE PRETRAINING (500 EPOCHS)")

    # Stage 2: Master Benchmark Matrix (EXP-01 to EXP-06)
    run_stage([python_exe, "-u", "poc/benchmark_master.py"], "STAGE 2: MASTER BENCHMARK MATRIX & CLOUD ROBUSTNESS")

    print("=" * 70, flush=True)
    print("  [SUCCESS] FULL RESEARCH PIPELINE COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
