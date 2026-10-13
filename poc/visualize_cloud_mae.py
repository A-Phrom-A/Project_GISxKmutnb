"""
poc/visualize_cloud_mae.py
==========================
Standalone visualization tool for Cloud-MAE Inpainting.
Generates a comprehensive 5-panel comparison:
1. S1 SAR (100% Penetrating Radar)
2. S2 Optical (Cloud Masked)
3. Raw Model Reconstruction (Entire Field)
4. Composite Inpainted (Original Clear Sky + Repaired strictly under clouds)
5. S2 True Clear-Sky (Ground Truth)
"""

import sys
import random
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib2_vitpalm.model.cloud_mae import CloudMAEModel


def get_rgb(arr: np.ndarray) -> np.ndarray:
    """Extract and enhance RGB from 7-band Sentinel-2 array (B4=idx 2, B3=idx 1, B2=idx 0)."""
    rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1)
    return np.clip(rgb * 3.5, 0.0, 1.0)


def generate_visualization(sample_idx: int = 0, out_name: str = "cloud_mae_composite_demo.png"):
    ckpt_path = PROJECT_ROOT / "checkpoints" / "cloud_mae" / "cloud_mae_best.pth"
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {ckpt_path.name} on {device}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {
        "s1_channels": 2,
        "s2_channels": 7,
        "img_size": 256,
        "patch_size": 16,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 8,
        "decoder_dim": 128,
        "decoder_depth": 3,
        "mask_ratio": 0.65,
        "mask_type": "block",
    })

    model = CloudMAEModel(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load tile
    tiles_dir = PROJECT_ROOT / "data" / "pretrain" / "tiles"
    all_tiles = sorted(tiles_dir.glob("*.npz"))
    if not all_tiles:
        print("Error: No tiles found.")
        return

    # Deterministic sample selection
    tile_file = all_tiles[sample_idx % len(all_tiles)]
    print(f"Testing on Tile: {tile_file.name}")

    data = np.load(tile_file)
    s1_np = data["s1"]  # (2, 256, 256)
    s2_np = data["s2"]  # (7, 256, 256)

    t_s1 = torch.from_numpy(s1_np).unsqueeze(0).float().to(device)
    t_s2 = torch.from_numpy(s2_np).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model(t_s1, t_s2)

    s2_true = s2_np
    s2_masked = out["s2_masked"][0].cpu().numpy()
    s2_pred = out["pred_s2"][0].cpu().numpy()
    mask_pixel = out["mask_pixel"][0].cpu().numpy()  # (1, 256, 256)
    s1_sar = s1_np[0]

    # Composite: True Image where clear (1 - mask) + Model Inpainting where cloudy (mask)
    s2_composite = s2_true * (1.0 - mask_pixel) + s2_pred * mask_pixel

    # Render 5 Panels with comfortable vertical margins
    fig, axs = plt.subplots(1, 5, figsize=(22, 5.0))

    title_fontsize = 11
    title_pad = 14

    # 1. S1 SAR
    axs[0].imshow(s1_sar, cmap="gray")
    axs[0].set_title("1. S1 SAR\n(100% Penetrating Radar)", fontsize=title_fontsize, fontweight="bold", pad=title_pad)
    axs[0].axis("off")

    # 2. Cloudy Optical
    axs[1].imshow(get_rgb(s2_masked))
    axs[1].set_title("2. S2 Optical Input\n(Cloud Masked 65%)", fontsize=title_fontsize, fontweight="bold", pad=title_pad)
    axs[1].axis("off")

    # 3. Raw Model Prediction
    axs[2].imshow(get_rgb(s2_pred))
    axs[2].set_title("3. Raw Decoder Prediction\n(Entire Latent Field)", fontsize=title_fontsize, fontweight="bold", pad=title_pad)
    axs[2].axis("off")

    # 4. Composite Inpainted (Target result)
    axs[3].imshow(get_rgb(s2_composite))
    axs[3].set_title("4. Composite Inpainted\n(Original Clear + Inpainted Cloud)", fontsize=title_fontsize, fontweight="bold", color="darkgreen", pad=title_pad)
    axs[3].axis("off")

    # 5. Clear Sky Truth
    axs[4].imshow(get_rgb(s2_true))
    axs[4].set_title("5. S2 True Clear-Sky\n(Ground Truth)", fontsize=title_fontsize, fontweight="bold", pad=title_pad)
    axs[4].axis("off")

    epoch_info = ckpt.get("epoch", "?")
    val_loss_info = ckpt.get("val_loss", 0.0)
    fig.suptitle(
        f"Cloud-MAE Radar-Guided Inpainting — Epoch {epoch_info} (Val Loss: {val_loss_info:.6f})",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    out_dir = PROJECT_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / out_name
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Successfully saved 5-panel figure to: {save_path}")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    out_filename = sys.argv[2] if len(sys.argv) > 2 else "cloud_mae_composite_latest.png"
    generate_visualization(sample_idx=idx, out_name=out_filename)
