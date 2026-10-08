import sys
from pathlib import Path
import yaml
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib2_vitpalm.model.late_fusion_model import ViTLateFusionModel

def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class PalmPredictor:
    def __init__(self, model_path: str, device_str: str = None):
        if device_str is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
            
        cfg_path = PROJECT_ROOT / "configs" / "baseline_config.yaml"
        cfg = load_config(cfg_path)
        
        self.model = ViTLateFusionModel(
            img_size=cfg['model']['img_size'],
            patch_size=cfg['model']['patch_size'],
            embed_dim=cfg['model']['embed_dim'],
            depth=cfg['model']['depth'],
            num_heads=cfg['model']['num_heads'],
            mlp_ratio=cfg['model']['mlp_ratio']
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, tensor_12ch: torch.Tensor, threshold=0.5) -> np.ndarray:
        """
        Inference on a single tile or batch of tiles.
        tensor_12ch: (12, H, W) or (B, 12, H, W)
        Returns:
            Binary mask: np.ndarray of shape (H, W) or (B, H, W)
        """
        with torch.no_grad():
            if tensor_12ch.ndim == 3:
                x = tensor_12ch.unsqueeze(0).to(self.device)
                single = True
            else:
                x = tensor_12ch.to(self.device)
                single = False
                
            prob = self.model(x).cpu().numpy()
            mask = (prob > threshold).astype(np.uint8)
            
            if single:
                return mask[0]
            return mask

if __name__ == "__main__":
    # Example usage
    best_model_path = PROJECT_ROOT / "checkpoints" / "benchmark_latefusion" / "vitpalm_best.pth"
    predictor = PalmPredictor(str(best_model_path))
    
    # Create dummy tensor matching the (12, 256, 256) format
    dummy_input = torch.randn(12, 256, 256)
    
    print(f"Running inference on {dummy_input.shape}...")
    pred_mask = predictor.predict(dummy_input)
    print(f"Prediction successful! Output mask shape: {pred_mask.shape}, Unique values: {np.unique(pred_mask)}")
