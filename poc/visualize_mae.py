import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from poc.dataset import PalmDataset
from poc.augmentation import get_validation_augmentation
from lib2_vitpalm.model.mae_model import MAEModel

def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def normalize_rgb(img_rgb):
    """Normalize RGB tensor (3, H, W) to [0, 1] for plotting."""
    img_np = img_rgb.cpu().numpy()
    p2, p98 = np.percentile(img_np, (2, 98))
    img_np = np.clip((img_np - p2) / (p98 - p2), 0, 1)
    return np.transpose(img_np, (1, 2, 0))

def unpatchify(x, patch_size, img_size, num_bands):
    """
    x: (B, N, patch_size**2 * num_bands)
    Returns: (B, num_bands, H, W)
    """
    B = x.shape[0]
    p = patch_size
    h = w = img_size // p
    
    x = x.reshape(B, h, w, p, p, num_bands)
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(B, num_bands, h * p, h * p)
    return imgs

def generate_mae_reconstruction():
    cfg_path = PROJECT_ROOT / "configs" / "mae_config.yaml"
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = MAEModel(
        in_channels=12,
        img_size=cfg['model']['img_size'],
        patch_size=cfg['model']['patch_size'],
        embed_dim=cfg['model']['embed_dim'],
        depth=cfg['model']['depth'],
        num_heads=cfg['model']['num_heads'],
        mlp_ratio=cfg['model'].get('mlp_ratio', 4.0),
        mask_ratio=cfg['model']['mask_ratio']
    )
    
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "mae" / "mae_best.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
        print(f"Loaded MAE checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: MAE checkpoint not found at {checkpoint_path}. Using random weights.")
        
    model.to(device)
    model.eval()
    
    # Load data
    test_dir = PROJECT_ROOT / "data" / "processed" / "split" / "test"
    dataset = PalmDataset(test_dir, augmentation=get_validation_augmentation())
    
    out_dir = PROJECT_ROOT / "results" / "figures" / "mae_reconstruction"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a few samples
    indices = [0, 5, 10, 15, 20]
    indices = [i for i in indices if i < len(dataset)]
    
    p = cfg['model']['patch_size']
    h = w = cfg['model']['img_size'] // p
    N = h * w
    
    for i, idx in enumerate(indices):
        img_12ch, _ = dataset[idx]
        img_tensor = img_12ch.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # MAE output: pred is (B, N_keep, P*P*C)
            pred, ids_keep, ids_masked = model(img_tensor)
            
            # Ground truth patch pixels for all patches
            target_patches = model.get_patch_targets(img_tensor, torch.arange(N).unsqueeze(0).to(device))
            
            # Create full reconstructed array (start with black)
            B, _, C_P_P = target_patches.shape
            recon_patches = torch.zeros(B, N, C_P_P, device=device)
            
            # Fill in the visible patches with predictions
            for b in range(B):
                recon_patches[b, ids_keep[b]] = pred[b]
                
            # Create masked input array (start with black, fill visible with ground truth)
            masked_in_patches = torch.zeros(B, N, C_P_P, device=device)
            for b in range(B):
                masked_in_patches[b, ids_keep[b]] = target_patches[b, ids_keep[b]]
        
        # Unpatchify
        orig_img = img_tensor[0] # (C, H, W)
        masked_img = unpatchify(masked_in_patches, p, cfg['model']['img_size'], 12)[0]
        recon_img = unpatchify(recon_patches, p, cfg['model']['img_size'], 12)[0]
        
        # Plot RGB (bands 0, 1, 2 = RGB for GSAT)
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(normalize_rgb(orig_img[:3]))
        plt.title("Original Image (RGB)")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(normalize_rgb(masked_img[:3]))
        plt.title(f"Masked Input (Ratio={cfg['model']['mask_ratio']})")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(normalize_rgb(recon_img[:3]))
        plt.title("MAE Reconstruction (Visible Patches)")
        plt.axis('off')
        
        plt.tight_layout()
        save_path = out_dir / f"mae_sample_{i+1}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

if __name__ == "__main__":
    generate_mae_reconstruction()
