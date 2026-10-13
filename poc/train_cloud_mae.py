"""
poc/train_cloud_mae.py
======================
Training script for Cloud-MAE (Radar-Guided Optical Inpainting & Self-Supervised Pretraining).
Trained across Eastern Thailand (5 provinces) satellite tiles.
"""

import sys
import time
import json
import random
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib2_vitpalm.model.cloud_mae import CloudMAEModel
from lib2_vitpalm.utils import set_seed


class CloudPretrainDataset(Dataset):
    """Dataset loading compressed .npz tiles (s1: 2 bands, s2: 7 bands)."""

    def __init__(self, file_paths: list[Path], augment: bool = True):
        self.file_paths = file_paths
        self.augment = augment

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.file_paths[idx]
        data = np.load(path)
        s1 = data["s1"]  # (2, 256, 256)
        s2 = data["s2"]  # (7, 256, 256)

        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                s1 = np.flip(s1, axis=2).copy()
                s2 = np.flip(s2, axis=2).copy()
            # Random Vertical Flip
            if random.random() > 0.5:
                s1 = np.flip(s1, axis=1).copy()
                s2 = np.flip(s2, axis=1).copy()
            # Random 90-degree Rotation
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                s1 = np.rot90(s1, k=k, axes=(1, 2)).copy()
                s2 = np.rot90(s2, k=k, axes=(1, 2)).copy()

        t_s1 = torch.from_numpy(s1).float()
        t_s2 = torch.from_numpy(s2).float()
        return t_s1, t_s2


def save_reconstruction_sample(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    out_dir: Path,
) -> None:
    """Renders 4-panel figure: S1 Radar, S2 Cloudy Input, S2 Reconstructed, S2 Ground Truth."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        s1, s2 = next(iter(val_loader))
        s1, s2 = s1[:1].to(device), s2[:1].to(device)

        out = model(s1, s2)
        s2_true = s2[0].cpu().numpy()
        s2_masked = out["s2_masked"][0].cpu().numpy()
        s2_pred = out["pred_s2"][0].cpu().numpy()
        s1_sar = s1[0].cpu().numpy()

        mask_pixel = out["mask_pixel"][0].cpu().numpy()
        s2_composite = s2_true * (1.0 - mask_pixel) + s2_pred * mask_pixel

        # RGB bands: B4=idx 2, B3=idx 1, B2=idx 0
        def get_rgb(arr):
            rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1)
            return np.clip(rgb * 3.5, 0.0, 1.0)  # enhance brightness slightly for display

        fig, axs = plt.subplots(1, 5, figsize=(22, 5.0))
        title_fontsize = 11
        title_pad = 14

        # 1. SAR VV Backscatter
        axs[0].imshow(s1_sar[0], cmap="gray")
        axs[0].set_title("1. S1 SAR\n(100% Penetrating Radar)", fontsize=title_fontsize, fontweight="bold", pad=title_pad)
        axs[0].axis("off")

        # 2. Cloudy Optical
        axs[1].imshow(get_rgb(s2_masked))
        axs[1].set_title("2. S2 Optical Input\n(Cloud Masked 65%)", fontsize=title_fontsize, fontweight="bold", pad=title_pad)
        axs[1].axis("off")

        # 3. Cloud-MAE Raw Prediction
        axs[2].imshow(get_rgb(s2_pred))
        axs[2].set_title("3. Raw Decoder Prediction\n(Entire Latent Field)", fontsize=title_fontsize, fontweight="bold", pad=title_pad)
        axs[2].axis("off")

        # 4. Composite Inpainted (Target Result)
        axs[3].imshow(get_rgb(s2_composite))
        axs[3].set_title("4. Composite Inpainted\n(Original Clear + Inpainted Cloud)", fontsize=title_fontsize, fontweight="bold", color="darkgreen", pad=title_pad)
        axs[3].axis("off")

        # 5. Clear Sky Ground Truth
        axs[4].imshow(get_rgb(s2_true))
        axs[4].set_title("5. S2 True Clear-Sky\n(Ground Truth)", fontsize=title_fontsize, fontweight="bold", pad=title_pad)
        axs[4].axis("off")

        fig.suptitle(f"Cloud-MAE Self-Supervised Inpainting — Epoch {epoch}", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        save_path = out_dir / f"cloud_mae_recon_epoch_{epoch:03d}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [Saved Figure] {save_path.name}")


def train_cloud_mae():
    print("=" * 70)
    print("  [>] CLOUD-MAE PRETRAINING PIPELINE (5 PROVINCES)")
    print("=" * 70)

    cfg_path = PROJECT_ROOT / "configs" / "cloud_mae_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["data"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    use_amp = cfg["training"]["use_amp"] and (device.type == "cuda")
    print(f"AMP (BFloat16): {use_amp}")

    # 1. Gather all pretraining tiles
    tiles_dir = PROJECT_ROOT / cfg["data"]["tiles_dir"]
    all_tiles = sorted(tiles_dir.glob("*.npz"))
    if not all_tiles:
        raise RuntimeError(f"No .npz tiles found in {tiles_dir}! Run poc/prep_pretrain_tiles.py first.")

    random.shuffle(all_tiles)
    val_count = max(16, int(len(all_tiles) * cfg["data"]["val_split"]))
    val_files = all_tiles[:val_count]
    train_files = all_tiles[val_count:]

    print(f"Total Tiles: {len(all_tiles):,} | Train: {len(train_files):,} | Val: {len(val_files):,}")

    train_ds = CloudPretrainDataset(train_files, augment=True)
    val_ds = CloudPretrainDataset(val_files, augment=False)

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 0)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # 2. Build Model
    model = CloudMAEModel(**cfg["model"]).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Cloud-MAE Parameters: {total_params:,} ({total_params/1e6:.2f} M)")

    # 3. Optimizer & Scheduler
    epochs = cfg["training"]["epochs"]
    lr = float(cfg["training"]["lr"])
    min_lr = float(cfg["training"].get("min_lr", 1e-6))
    weight_decay = float(cfg["training"]["weight_decay"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    # 4. Checkpoints & Output Dirs
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "cloud_mae"
    figs_dir = PROJECT_ROOT / "results" / "figures"
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience = cfg["training"].get("patience", 20)
    patience_counter = 0
    history = []
    start_epoch = 1

    resume_path = ckpt_dir / "cloud_mae_best.pth"
    if resume_path.exists():
        try:
            ckpt_data = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt_data["model_state_dict"])
            best_val_loss = ckpt_data.get("val_loss", float("inf"))
            start_epoch = ckpt_data.get("epoch", 0) + 1
            print(f"  [Resume] Loaded previous best checkpoint: {resume_path.name}", flush=True)
            print(f"  [Resume] Resuming from Epoch {start_epoch}/{epochs} (Best Val Loss: {best_val_loss:.6f})", flush=True)
            if start_epoch > 1:
                for _ in range(start_epoch - 1):
                    scheduler.step()
                print(f"  [Scheduler] Advanced LR scheduler to Epoch {start_epoch} (LR: {scheduler.get_last_lr()[0]:.2e})", flush=True)
        except Exception as e:
            print(f"  [Warning] Could not resume checkpoint: {e}", flush=True)

    print("\nStarting Training Loop...", flush=True)
    print("-" * 70, flush=True)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^12} | {'LR':^10} | {'Time (s)':^8}", flush=True)
    print("-" * 70, flush=True)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss_acc = 0.0
        train_batches = 0

        for s1, s2 in train_loader:
            s1, s2 = s1.to(device), s2.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(s1, s2)
                    loss = out["loss"]
            else:
                out = model(s1, s2)
                loss = out["loss"]

            loss.backward()
            optimizer.step()

            train_loss_acc += loss.item()
            train_batches += 1

        train_loss = train_loss_acc / max(1, train_batches)

        # --- Val ---
        model.eval()
        val_loss_acc = 0.0
        val_batches = 0

        with torch.no_grad():
            for s1, s2 in val_loader:
                s1, s2 = s1.to(device), s2.to(device)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model(s1, s2)
                        loss = out["loss"]
                else:
                    out = model(s1, s2)
                    loss = out["loss"]

                val_loss_acc += loss.item()
                val_batches += 1

        val_loss = val_loss_acc / max(1, val_batches)
        curr_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        elapsed = time.time() - t0

        print(f"{epoch:^7d} | {train_loss:^12.6f} | {val_loss:^12.6f} | {curr_lr:^10.2e} | {elapsed:^8.1f}", flush=True)

        # Save History
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": curr_lr,
            "time_sec": elapsed,
        })

        # Check Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = ckpt_dir / "cloud_mae_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "config": cfg,
                "val_loss": val_loss,
            }, best_path)
            print(f"  --> Saved new best checkpoint: {best_path.name} (Val Loss: {val_loss:.6f})", flush=True)
        else:
            patience_counter += 1

        # Periodic Visualizations
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            save_reconstruction_sample(model, val_loader, device, epoch, figs_dir)

        # Early Stopping
        if patience_counter >= patience:
            print(f"\n[Early Stopping] No improvement in validation loss for {patience} epochs.", flush=True)
            break

    # Save training metrics json
    metrics_path = metrics_dir / "cloud_mae_history.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {metrics_path}")
    print(f"Best Cloud-MAE Validation Loss: {best_val_loss:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    train_cloud_mae()
