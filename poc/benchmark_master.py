"""
poc/benchmark_master.py
=======================
Master Benchmark & Evaluation Suite:
Executes the complete 6-experiment matrix:
  E1: Early Fusion (Scratch)
  E2: Early Fusion (+ Cloud-MAE Pretrained)
  E3: Late Fusion (Scratch)
  E4: Late Fusion (+ Cloud-MAE Pretrained)
  E5: Cross-Attention (Scratch)
  E6: Cross-Attention (+ Cloud-MAE Pretrained)

Also conducts the Cloud Robustness Stress-Test across 3 cloud levels:
  - Clean Sky (0% Cloud)
  - Light/Moderate Cloud (25% Cloud Mask)
  - Heavy Cloud (50% Cloud Mask)
"""

import sys
import time
import argparse
from pathlib import Path
import json
import yaml
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poc.dataset import PalmDataset
from poc.augmentation import get_training_augmentation, get_validation_augmentation
from lib2_vitpalm.training.trainer import Trainer, compute_metrics
from lib2_vitpalm.model.vitpalm import ViTPalmModel
from lib2_vitpalm.model.late_fusion_model import ViTLateFusionModel
from lib2_vitpalm.model.cross_attn_model import ViTCrossAttnModel
from lib2_vitpalm.model.cloud_mae import CloudMAEModel, CloudMaskGenerator


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(strategy: str, pretrained_path: Path | None, cfg: dict, device: torch.device) -> nn.Module:
    """Builds model and transfers Cloud-MAE weights if specified."""
    embed_dim = cfg["model"].get("embed_dim", 256)
    img_size = cfg["model"].get("img_size", 256)
    patch_size = cfg["model"].get("patch_size", 16)
    depth = cfg["model"].get("depth", 6)
    num_heads = cfg["model"].get("num_heads", 8)
    mlp_ratio = cfg["model"].get("mlp_ratio", 4.0)

    # 1. Instantiate downstream model
    if strategy == "EarlyFusion":
        model = ViTPalmModel(
            in_channels=12,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
    elif strategy == "LateFusion":
        model = ViTLateFusionModel(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
    elif strategy == "CrossAttn":
        model = ViTCrossAttnModel(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 2. Transfer pretrained weights if provided
    if pretrained_path is not None and pretrained_path.exists():
        print(f"  [Weight Transfer] Loading Cloud-MAE weights from {pretrained_path.name}...")
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        m_cfg = ckpt.get("config", {}).get("model", {})
        cloud_mae = CloudMAEModel(
            s1_channels=m_cfg.get("s1_channels", 2),
            s2_channels=m_cfg.get("s2_channels", 7),
            img_size=m_cfg.get("img_size", img_size),
            patch_size=m_cfg.get("patch_size", patch_size),
            embed_dim=m_cfg.get("embed_dim", embed_dim),
            depth=m_cfg.get("depth", depth),
            num_heads=m_cfg.get("num_heads", num_heads),
            decoder_dim=m_cfg.get("decoder_dim", 128),
            decoder_depth=m_cfg.get("decoder_depth", 3),
        )
        if "encoder_state_dict" in ckpt:
            cloud_mae.encoder.load_state_dict(ckpt["encoder_state_dict"])
        elif "model_state_dict" in ckpt:
            cloud_mae.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "state_dict" in ckpt:
            cloud_mae.load_state_dict(ckpt["state_dict"], strict=False)

        if strategy == "EarlyFusion":
            cloud_mae.transfer_to_early_fusion(model)
        elif strategy == "LateFusion":
            cloud_mae.transfer_to_late_fusion(model)
        elif strategy == "CrossAttn":
            cloud_mae.transfer_to_cross_attn(model)
    else:
        print(f"  [Scratch Init] Initializing {strategy} randomly from scratch...")

    return model.to(device)


def evaluate_cloud_robustness(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    cloud_ratios: list[float] = [0.0, 0.25, 0.50],
) -> dict:
    """Evaluates downstream model robustness under simulated optical cloud coverage."""
    model.eval()
    results = {}

    with torch.no_grad():
        for ratio in cloud_ratios:
            total_iou = 0.0
            total_f1 = 0.0
            total_prec = 0.0
            total_rec = 0.0
            count = 0

            for imgs, masks in test_loader:
                imgs = imgs.to(device)    # (B, 12, H, W)
                masks = masks.to(device)  # (B, H, W)
                B, _, H, W = imgs.shape

                # Apply cloud mask strictly to optical channels: S2 (ch 2..8) and GSat (ch 9..11)
                # SAR (ch 0..1) penetrates clouds and stays intact!
                if ratio > 0.0:
                    cloud_mask, _ = CloudMaskGenerator.generate_block_cloud_mask(
                        B, H, W, patch_size=16, target_ratio=ratio, device=device
                    )
                    # Mask out optical channels
                    imgs_cloudy = imgs.clone()
                    imgs_cloudy[:, 2:12, :, :] *= (1.0 - cloud_mask)
                else:
                    imgs_cloudy = imgs

                preds = model(imgs_cloudy)
                m = compute_metrics(preds, masks)
                total_iou += m["iou"]
                total_f1 += m["f1"]
                total_prec += m["precision"]
                total_rec += m["recall"]
                count += 1

            n = max(1, count)
            key = f"Cloud_{int(ratio*100)}pct"
            results[f"{key}_F1"] = total_f1 / n
            results[f"{key}_IoU"] = total_iou / n
            results[f"{key}_Precision"] = total_prec / n
            results[f"{key}_Recall"] = total_rec / n

    return results


def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per experiment")
    parser.add_argument("--skip_scratch", action="store_true", help="Skip scratch training")
    args = parser.parse_args()

    print("=" * 70)
    print("  [>] GISxKMUTNB - MASTER BENCHMARK & CLOUD ROBUSTNESS SUITE")
    print("=" * 70)

    cfg_path = PROJECT_ROOT / "configs" / "baseline_config.yaml"
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = PROJECT_ROOT / "data" / "processed" / "split" / "train"
    val_dir = PROJECT_ROOT / "data" / "processed" / "split" / "val"
    test_dir = PROJECT_ROOT / "data" / "processed" / "split" / "test"

    train_ds = PalmDataset(str(train_dir), augmentation=get_training_augmentation())
    val_ds = PalmDataset(str(val_dir), augmentation=get_validation_augmentation())
    test_ds = PalmDataset(str(test_dir), augmentation=None)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    mae_best = PROJECT_ROOT / "checkpoints" / "cloud_mae" / "cloud_mae_best.pth"

    experiments = [
        {"id": "E1", "name": "EarlyFusion_Scratch", "strategy": "EarlyFusion", "pretrained": None},
        {"id": "E2", "name": "EarlyFusion_CloudMAE", "strategy": "EarlyFusion", "pretrained": mae_best},
        {"id": "E3", "name": "LateFusion_Scratch", "strategy": "LateFusion", "pretrained": None},
        {"id": "E4", "name": "LateFusion_CloudMAE", "strategy": "LateFusion", "pretrained": mae_best},
        {"id": "E5", "name": "CrossAttn_Scratch", "strategy": "CrossAttn", "pretrained": None},
        {"id": "E6", "name": "CrossAttn_CloudMAE", "strategy": "CrossAttn", "pretrained": mae_best},
    ]

    benchmark_records = []
    robustness_records = []

    for exp in experiments:
        if args.skip_scratch and exp["pretrained"] is None:
            print(f"\nSkipping {exp['id']}: {exp['name']} (--skip_scratch)")
            continue

        print(f"\n" + "=" * 60)
        print(f"  Starting Experiment [{exp['id']}]: {exp['name']}")
        print("=" * 60)

        out_dir = PROJECT_ROOT / "checkpoints" / f"benchmark_{exp['name'].lower()}"
        model = build_model(exp["strategy"], exp["pretrained"], cfg, device)

        best_ckpt = out_dir / "best_model.pth"
        hist_file = out_dir / "train_history.json"

        if best_ckpt.exists() and hist_file.exists():
            print(f"  [Resume] Experiment {exp['name']} already completed! Loading existing checkpoint and history...")
            with open(hist_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            best_f1 = max(r.get("val_f1", 0.0) for r in history)
            best_record = next((r for r in history if r["val_f1"] == best_f1), history[-1])
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=out_dir,
                device=device,
                lr=float(cfg["training"]["lr"]),
                weight_decay=float(cfg["training"]["weight_decay"]),
                epochs=args.epochs,
                save_every=50,
                bce_weight=cfg["training"]["bce_weight"],
                dice_weight=cfg["training"]["dice_weight"],
                pos_weight=cfg["training"]["pos_weight"],
                threshold=cfg["training"]["threshold"],
                patience=cfg["training"]["patience"],
                use_amp=cfg["training"]["use_amp"],
                use_wandb=False,
            )

            history = trainer.train()
            best_f1 = trainer.best_f1
            best_record = next((r for r in history if r["val_f1"] == best_f1), history[-1])

            # Test Set Evaluation
            if best_ckpt.exists():
                ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model_state_dict"])

        print(f"\nEvaluating Cloud Robustness on Test Set for {exp['name']}...")
        rob_res = evaluate_cloud_robustness(model, test_loader, device)

        rec = {
            "Exp_ID": exp["id"],
            "Experiment": exp["name"],
            "Strategy": exp["strategy"],
            "Pretrained": "Cloud-MAE" if exp["pretrained"] else "Scratch",
            "Best_Val_F1": best_record.get("val_f1", 0.0),
            "Best_Val_IoU": best_record.get("val_iou", 0.0),
            "Test_Clean_F1": rob_res["Cloud_0pct_F1"],
            "Test_Clean_IoU": rob_res["Cloud_0pct_IoU"],
            "Test_Cloud25_F1": rob_res["Cloud_25pct_F1"],
            "Test_Cloud50_F1": rob_res["Cloud_50pct_F1"],
            "Best_Epoch": best_record.get("epoch", 0),
        }
        benchmark_records.append(rec)
        print(f"[{exp['id']} Results] Val F1: {rec['Best_Val_F1']:.4f} | Test Clean F1: {rec['Test_Clean_F1']:.4f} | Test Cloud 50% F1: {rec['Test_Cloud50_F1']:.4f}")

    df_bm = pd.DataFrame(benchmark_records)
    print("\n" + "=" * 70)
    print("  FINAL MASTER BENCHMARK & ROBUSTNESS RESULTS")
    print("=" * 70)
    print(df_bm.to_string(index=False))

    out_csv = PROJECT_ROOT / "results" / "metrics" / "master_benchmark_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_bm.to_csv(out_csv, index=False)
    print(f"\nSaved full benchmark results to {out_csv}")

    # Automatically generate updated comparison and pretraining curves
    try:
        from poc.plot_curves import plot_training_curves, plot_cloud_mae_pretrain_curve
        print("\nUpdating publication training curves comparison...")
        plot_training_curves()
        plot_cloud_mae_pretrain_curve()
    except Exception as e:
        print(f"[Warning] Could not generate plots automatically: {e}")


if __name__ == "__main__":
    run_benchmark()
