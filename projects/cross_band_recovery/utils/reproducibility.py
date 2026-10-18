"""
projects.cross_band_recovery.utils.reproducibility
===================================================
เครื่องมือจัดการความทำซ้ำได้ (Reproducibility) และการบันทึก Run Manifest
ตามข้อกำหนด P0 (CHK-G0.1, CHK-G0.4) ใน cross_band_recovery_development_plan.md
"""

import os
import sys
import random
import hashlib
import json
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    ตั้งค่า seed ครอบคลุม Python, NumPy, PyTorch (CPU + GPU) และ DataLoader worker
    เพื่อรับประกันความสามารถในการทำซ้ำได้ระดับ bitwise หรือเชิงสถิติ
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    ฟังก์ชันกำหนด seed สำหรับ PyTorch DataLoader worker
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_file_hash(file_path: str | Path, algorithm: str = "sha256") -> str:
    """คำนวณ Hash ของไฟล์ (default sha256)"""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    h = hashlib.new(algorithm)
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_hash() -> str:
    """ดึง Git commit hash ปัจจุบัน"""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return out
    except Exception:
        return "UNKNOWN_COMMIT"


def get_environment_info() -> Dict[str, Any]:
    """สรุปข้อมูลฮาร์ดแวร์ ซอฟต์แวร์ และเวอร์ชันระบบ"""
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    device_capability = torch.cuda.get_device_capability(0) if cuda_avail else None
    
    return {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "cuda_version": torch.version.cuda if cuda_avail else None,
        "gpu_device_name": device_name,
        "gpu_capability": device_capability,
        "os_platform": sys.platform,
    }


def create_run_manifest(
    config_path: str | Path,
    output_path: Optional[str | Path] = None,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    สร้าง Run Manifest เพื่อบันทึก:
    - Git commit
    - Config hash
    - Timestamp
    - Hardware / Environment specs
    - Extra metadata
    
    ตอบสนองเกณฑ์ CHK-G0.4
    """
    config_p = Path(config_path)
    config_hash = compute_file_hash(config_p) if config_p.exists() else "CONFIG_NOT_FOUND"
    
    manifest = {
        "run_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "config_path": str(config_p),
        "config_sha256": config_hash,
        "environment": get_environment_info(),
        "extra_metadata": extra_metadata or {}
    }
    
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            
    return manifest
