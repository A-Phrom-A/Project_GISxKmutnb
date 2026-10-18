"""
scripts/cloudmae_v2/s01_setup_and_verify.py
===========================================
Phase 0 - S0.1: Environment & Reproducibility Verification
Checks:
  - CHK 0.1a: Seed reproducibility (Forward/Backward pass gives exact same loss/grads)
  - CHK 0.1b: PyTorch BF16 AMP support on RTX 5060 Ti
Creates:
  - Required directories
  - reports/cloudmae_v2/s01_environment.json
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_reproducibility(seed: int = 42) -> dict:
    results = {}
    losses = []
    grad_norms = []

    for run_idx in range(2):
        set_seed(seed)
        model = nn.Sequential(
            nn.Linear(17, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )
        if torch.cuda.is_available():
            model = model.cuda()
            x = torch.randn(4, 17, device="cuda")
            y = torch.randn(4, 7, device="cuda")
        else:
            x = torch.randn(4, 17)
            y = torch.randn(4, 7)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.mse_loss(out, y)
        loss.backward()

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        losses.append(loss.item())
        grad_norms.append(total_norm)

    loss_diff = abs(losses[0] - losses[1])
    grad_diff = abs(grad_norms[0] - grad_norms[1])
    passed = (loss_diff < 1e-6) and (grad_diff < 1e-6)

    return {
        "run1_loss": losses[0],
        "run2_loss": losses[1],
        "loss_diff": loss_diff,
        "run1_grad_norm": grad_norms[0],
        "run2_grad_norm": grad_norms[1],
        "grad_diff": grad_diff,
        "passed": passed
    }


def main():
    config_path = ROOT_DIR / "configs" / "cloudmae_v2" / "base.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Compute config hash
    with open(config_path, "rb") as f:
        config_hash = hashlib.sha256(f.read()).hexdigest()

    # 1. Ensure required directories
    dirs_to_create = [
        ROOT_DIR / config["paths"]["raw_dir"],
        ROOT_DIR / config["paths"]["interim_dir"],
        ROOT_DIR / config["paths"]["processed_dir"],
        ROOT_DIR / config["paths"]["manifests_dir"],
        ROOT_DIR / config["paths"]["splits_dir"],
        ROOT_DIR / config["paths"]["reports_dir"],
        ROOT_DIR / config["paths"]["checkpoints_dir"],
        ROOT_DIR / "tests" / "cloudmae_v2",
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Hardware and BF16 check (CHK 0.1b)
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    total_vram = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if cuda_avail else 0.0
    bf16_supported = torch.cuda.is_bf16_supported() if cuda_avail else False

    # Test BF16 forward & backward
    bf16_test_passed = False
    if cuda_avail and bf16_supported:
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                m = nn.Linear(16, 16).cuda()
                inp = torch.randn(2, 16, device="cuda")
                out = m(inp)
                loss = out.sum()
            loss.backward()
            bf16_test_passed = True
        except Exception as e:
            bf16_test_passed = False

    chk_01b_passed = cuda_avail and bf16_supported and bf16_test_passed

    # 3. Reproducibility check (CHK 0.1a)
    repro = check_reproducibility(seed=config.get("seed", 42))
    chk_01a_passed = repro["passed"]

    report = {
        "phase": "Phase 0 - S0.1",
        "config_hash": config_hash,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "total_vram_gb": round(total_vram, 2),
        "bf16_supported": bf16_supported,
        "chk_01a_reproducibility": {
            "name": "CHK 0.1a: Seed reproducibility forward/backward",
            "status": "PASSED" if chk_01a_passed else "FAILED",
            "details": repro
        },
        "chk_01b_bf16_rtx5060ti": {
            "name": "CHK 0.1b: PyTorch BF16 on RTX 5060 Ti",
            "status": "PASSED" if chk_01b_passed else "FAILED",
            "details": {
                "device": gpu_name,
                "is_bf16_supported": bf16_supported,
                "autocast_test": bf16_test_passed
            }
        },
        "all_passed": chk_01a_passed and chk_01b_passed
    }

    report_file = ROOT_DIR / config["paths"]["reports_dir"] / "s01_environment.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== S0.1 ENVIRONMENT VERIFICATION SUMMARY ===")
    print(f"Device: {gpu_name} ({total_vram:.2f} GB VRAM)")
    print(f"CHK 0.1a (Reproducibility): {'PASSED' if chk_01a_passed else 'FAILED'}")
    print(f"CHK 0.1b (BF16 AMP): {'PASSED' if chk_01b_passed else 'FAILED'}")
    print(f"Report saved: {report_file}")
    
    if not (chk_01a_passed and chk_01b_passed):
        sys.exit(1)


if __name__ == "__main__":
    main()
