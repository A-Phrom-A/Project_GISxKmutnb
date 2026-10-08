import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poc.dataset import PalmDataset
from poc.augmentation import get_validation_augmentation
from lib2_vitpalm.model.late_fusion_model import ViTLateFusionModel
from lib2_vitpalm.model.vitpalm import ViTPalmModel

def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def create_tp_fp_fn_overlay(pred_mask, true_mask):
    """
    Creates a 3-channel image indicating TP, FP, FN.
    TP (True Positive) -> Green
    FP (False Positive) -> Red
    FN (False Negative) -> Blue
    TN (True Negative) -> Black
    """
    overlay = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    
    tp = (pred_mask == 1) & (true_mask == 1)
    fp = (pred_mask == 1) & (true_mask == 0)
    fn = (pred_mask == 0) & (true_mask == 1)
    
    overlay[tp] = [0, 255, 0]   # Green
    overlay[fp] = [255, 0, 0]   # Red
    overlay[fn] = [0, 0, 255]   # Blue
    
    return overlay

def extract_rgb(image_tensor):
    """
    Extracts RGB bands from the (12, H, W) normalized tensor for visualization.
    We'll use S2 RGB (bands 2, 3, 4 of the 12-channel tensor).
    Actually, let's use GEO-Sat RGB (bands 9, 10, 11) as it is 50cm resolution,
    or S2 RGB (bands 2, 3, 4). Let's use S2.
    Indices: S1(0,1), S2(2,3,4=B2,B3,B4), GSat(9,10,11=R,G,B)
    S2 RGB corresponds to bands 4(Red), 3(Green), 2(Blue) in standard mapping.
    Our stack: S2_stack is B2, B3, B4, B8, B11, B12, RedEdge.
    So index 4 is B4(Red), index 3 is B3(Green), index 2 is B2(Blue).
    Let's extract them.
    """
    # Using S2 B4, B3, B2 for True Color
    r = image_tensor[4].numpy()
    g = image_tensor[3].numpy()
    b = image_tensor[2].numpy()
    
    rgb = np.stack([r, g, b], axis=-1)
    
    # Clip and normalize for display
    rgb = np.clip(rgb, 0, 1)
    # Brighten it slightly for better visibility
    rgb = np.clip(rgb * 2.5, 0, 1)
    
    return rgb

def visualize_predictions(model_path, model_type="late", threshold=0.5, num_samples=5):
    cfg_path = PROJECT_ROOT / "configs" / "baseline_config.yaml"
    cfg = load_config(cfg_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "late":
        model = ViTLateFusionModel(
            img_size=cfg['model']['img_size'],
            patch_size=cfg['model']['patch_size'],
            embed_dim=cfg['model']['embed_dim'],
            depth=cfg['model']['depth'],
            num_heads=cfg['model']['num_heads'],
            mlp_ratio=cfg['model']['mlp_ratio']
        )
    elif model_type == "early":
        model = ViTPalmModel(
            in_channels=12,
            img_size=cfg['model']['img_size'],
            patch_size=cfg['model']['patch_size'],
            embed_dim=cfg['model']['embed_dim'],
            depth=cfg['model']['depth'],
            num_heads=cfg['model']['num_heads'],
            mlp_ratio=cfg['model']['mlp_ratio']
        )
    else:
        raise ValueError("Unsupported model_type")
        
    checkpoint = torch.load(model_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    test_dir = PROJECT_ROOT / "data" / "processed" / "split" / "test"
    test_ds = PalmDataset(test_dir, augmentation=get_validation_augmentation())
    
    # Randomly sample from test set
    indices = np.random.choice(len(test_ds), num_samples, replace=False)
    
    out_dir = PROJECT_ROOT / "results" / "figures" / f"{model_type}_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for i, idx in enumerate(indices):
        img, mask = test_ds[idx]
        
        with torch.no_grad():
            img_batch = img.unsqueeze(0).to(device)
            
            if model_type == "early":
                pred_prob = model(img_batch)
            else:
                pred_prob = model(img_batch)
                
            pred_prob = pred_prob.squeeze().cpu().numpy()
            
        pred_mask = (pred_prob > threshold).astype(np.uint8)
        true_mask = mask.numpy().astype(np.uint8)
        
        rgb = extract_rgb(img)
        overlay = create_tp_fp_fn_overlay(pred_mask, true_mask)
        
        # Plotting 4-panel figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(rgb)
        axes[0].set_title("Sentinel-2 RGB (B4, B3, B2)")
        axes[0].axis('off')
        
        axes[1].imshow(true_mask, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f"Predicted Mask (prob > {threshold})")
        axes[2].axis('off')
        
        axes[3].imshow(overlay)
        axes[3].set_title("TP(Green), FP(Red), FN(Blue)")
        axes[3].axis('off')
        
        plt.tight_layout()
        save_path = out_dir / f"sample_{i+1}_idx_{idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to best checkpoint")
    parser.add_argument("--model_type", type=str, choices=["early", "late"], default="late")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    
    visualize_predictions(args.model_path, args.model_type, num_samples=args.samples)
