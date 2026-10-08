import json
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def plot_training_curves():
    """
    Load train_history.json from all benchmark runs and plot comparisons.
    """
    models = ["earlyfusion", "latefusion", "crossattn"]
    colors = {"earlyfusion": "blue", "latefusion": "green", "crossattn": "red"}
    labels = {"earlyfusion": "Early Fusion", "latefusion": "Late Fusion (Best)", "crossattn": "Cross-Attention"}
    
    histories = {}
    for m in models:
        hist_path = PROJECT_ROOT / "checkpoints" / f"benchmark_{m}" / "train_history.json"
        if hist_path.exists():
            with open(hist_path, "r", encoding="utf-8") as f:
                histories[m] = json.load(f)
                
    if not histories:
        print("No training histories found.")
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Validation F1 Score
    for m, hist in histories.items():
        epochs = [h["epoch"] for h in hist]
        val_f1 = [h["val_f1"] for h in hist]
        axes[0].plot(epochs, val_f1, label=labels[m], color=colors[m], linewidth=2)
    axes[0].set_title("Validation F1-Score over Epochs", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("F1-Score", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()
    
    # 2. Validation IoU
    for m, hist in histories.items():
        epochs = [h["epoch"] for h in hist]
        val_iou = [h["val_iou"] for h in hist]
        axes[1].plot(epochs, val_iou, label=labels[m], color=colors[m], linewidth=2)
    axes[1].set_title("Validation IoU over Epochs", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("IoU", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()
    
    # 3. Training Loss
    for m, hist in histories.items():
        epochs = [h["epoch"] for h in hist]
        train_loss = [h["train_loss"] for h in hist]
        axes[2].plot(epochs, train_loss, label=labels[m], color=colors[m], linewidth=2)
    axes[2].set_title("Training Loss over Epochs", fontsize=14)
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("Loss (BCE + Dice)", fontsize=12)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()
    
    plt.tight_layout()
    
    out_dir = PROJECT_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "training_curves_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves comparison to {save_path}")

if __name__ == "__main__":
    plot_training_curves()
