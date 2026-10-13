"""
poc/plot_curves.py
==================
Publication-grade plotting suite for GISxKMUTNB:
1. Downstream Master Benchmark training curves (Validation F1, IoU, Training Loss)
   comparing Scratch vs. Cloud-MAE across Early Fusion, Late Fusion, and Cross-Attention.
2. Cloud-MAE Self-Supervised Pretraining curves (Train vs. Val Loss + LR schedule).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def plot_training_curves():
    """
    Load train_history.json from benchmark runs and plot publication-quality comparisons.
    Supports both the 6-experiment Master Benchmark and legacy 3-model benchmark.
    """
    out_dir = PROJECT_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check for Master Benchmark (6 experiments)
    master_experiments = [
        ("earlyfusion_scratch", "Early Fusion (Scratch)", "#4A90E2", "--", 1.8),
        ("earlyfusion_cloudmae", "Early Fusion (+ Cloud-MAE)", "#0056B3", "-", 2.5),
        ("latefusion_scratch", "Late Fusion (Scratch)", "#50C878", "--", 1.8),
        ("latefusion_cloudmae", "Late Fusion (+ Cloud-MAE)", "#1E7E34", "-", 2.5),
        ("crossattn_scratch", "Cross-Attn (Scratch)", "#E74C3C", "--", 1.8),
        ("crossattn_cloudmae", "Cross-Attn (+ Cloud-MAE)", "#A93226", "-", 2.5),
    ]

    histories = {}
    is_master = False
    for key, label, color, style, width in master_experiments:
        hist_path = PROJECT_ROOT / "checkpoints" / f"benchmark_{key}" / "train_history.json"
        if hist_path.exists():
            with open(hist_path, "r", encoding="utf-8") as f:
                histories[key] = {
                    "data": json.load(f),
                    "label": label,
                    "color": color,
                    "style": style,
                    "width": width,
                }
                is_master = True

    # Fallback to legacy 3-model benchmark if master isn't populated yet
    if not is_master:
        legacy_models = [
            ("earlyfusion", "Early Fusion", "#0056B3", "-", 2.2),
            ("latefusion", "Late Fusion (Best)", "#1E7E34", "-", 2.5),
            ("crossattn", "Cross-Attention", "#A93226", "-", 2.2),
        ]
        for key, label, color, style, width in legacy_models:
            hist_path = PROJECT_ROOT / "checkpoints" / f"benchmark_{key}" / "train_history.json"
            if hist_path.exists():
                with open(hist_path, "r", encoding="utf-8") as f:
                    histories[key] = {
                        "data": json.load(f),
                        "label": label,
                        "color": color,
                        "style": style,
                        "width": width,
                    }

    if not histories:
        print("[Plot Curves] No downstream training histories found.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Validation F1 Score
    for key, item in histories.items():
        hist = item["data"]
        epochs = [h["epoch"] for h in hist]
        val_f1 = [h["val_f1"] for h in hist]
        axes[0].plot(
            epochs, val_f1,
            label=item["label"],
            color=item["color"],
            linestyle=item["style"],
            linewidth=item["width"],
        )
    axes[0].set_title("Validation F1-Score over Epochs", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("F1-Score", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9, loc="lower right" if is_master else "best", framealpha=0.9)

    # 2. Validation IoU
    for key, item in histories.items():
        hist = item["data"]
        epochs = [h["epoch"] for h in hist]
        val_iou = [h["val_iou"] for h in hist]
        axes[1].plot(
            epochs, val_iou,
            label=item["label"],
            color=item["color"],
            linestyle=item["style"],
            linewidth=item["width"],
        )
    axes[1].set_title("Validation IoU over Epochs", fontsize=14, fontweight="bold", pad=10)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("IoU", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9, loc="lower right" if is_master else "best", framealpha=0.9)

    # 3. Training Loss
    for key, item in histories.items():
        hist = item["data"]
        epochs = [h["epoch"] for h in hist]
        train_loss = [h["train_loss"] for h in hist]
        axes[2].plot(
            epochs, train_loss,
            label=item["label"],
            color=item["color"],
            linestyle=item["style"],
            linewidth=item["width"],
        )
    axes[2].set_title("Training Loss (BCE + Dice) over Epochs", fontsize=14, fontweight="bold", pad=10)
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("Loss", fontsize=12)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=9, loc="upper right", framealpha=0.9)

    plt.suptitle(
        "Downstream Multi-Modal Fusion Benchmark: Scratch vs. Cloud-MAE Pretraining" if is_master else "Downstream Architecture Benchmark",
        fontsize=16,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()

    save_path = out_dir / "training_curves_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Curves] Successfully saved updated {save_path}")
    return save_path


def plot_cloud_mae_pretrain_curve():
    """
    Plot Cloud-MAE self-supervised pretraining curve (Train Loss vs Val Loss over 500 epochs).
    """
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "cloud_mae_history.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        return None

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    lrs = [h.get("lr", 0.0) for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    line1 = ax1.plot(epochs, train_loss, label="Train Loss (Cloud Inpainting MSE)", color="#1F77B4", linewidth=2.0)
    line2 = ax1.plot(epochs, val_loss, label="Val Loss (Cloud Inpainting MSE)", color="#FF7F0E", linewidth=2.0)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MSE Reconstruction Loss", fontsize=12, color="#333333")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Best val loss annotation
    best_val = min(val_loss)
    best_ep = epochs[val_loss.index(best_val)]
    ax1.scatter([best_ep], [best_val], color="red", s=60, zorder=5)
    ax1.annotate(
        f"Best Val: {best_val:.6f}\n(Epoch {best_ep})",
        xy=(best_ep, best_val),
        xytext=(best_ep - 50 if best_ep > 60 else best_ep + 10, best_val + (max(val_loss) - min(val_loss)) * 0.15),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        fontweight="bold",
        color="red",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0F0", ec="red", lw=1),
    )

    # Learning rate on twin axis
    ax2 = ax1.twinx()
    line3 = ax2.plot(epochs, lrs, label="Learning Rate (Cosine Decay)", color="#2CA02C", linestyle=":", linewidth=1.8, alpha=0.8)
    ax2.set_ylabel("Learning Rate", fontsize=12, color="#2CA02C")
    ax2.tick_params(axis="y", labelcolor="#2CA02C")
    ax2.grid(False)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9, fontsize=10)

    plt.title("Cloud-MAE Self-Supervised Pretraining (5 Eastern Thailand Provinces, 8,996 Tiles)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    out_dir = PROJECT_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "cloud_mae_pretrain_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Curves] Successfully saved Cloud-MAE pretrain curves to {save_path}")
    return save_path


if __name__ == "__main__":
    plot_training_curves()
    plot_cloud_mae_pretrain_curve()
