"""
scratch/test_modality_impact.py
================================
Empirical Test: Does Cloud-MAE and CrossAttn actually utilize Sentinel-1 SAR?
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from lib2_vitpalm.model.cloud_mae import CloudMAEModel, CloudMaskGenerator
from lib2_vitpalm.model.cross_attn_model import ViTCrossAttnModel


def test_cloud_mae_sar_impact():
    print("=" * 70)
    print("TEST 1: Cloud-MAE Sensitivity to Sentinel-1 SAR (Checkpoints Analysis)")
    print("=" * 70)

    checkpoint_path = ROOT_DIR / "checkpoints" / "cloud_mae" / "cloud_mae_best.pth"
    if not checkpoint_path.exists():
        print("Checkpoint not found:", checkpoint_path)
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    epoch = checkpoint.get("epoch", "N/A")
    val_loss = checkpoint.get("val_loss", "N/A")
    print(f"Loaded Checkpoint: {checkpoint_path.name} (Epoch: {epoch}, Val Loss: {val_loss:.6f})")

    model_state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    model = CloudMAEModel(
        s1_channels=2,
        s2_channels=7,
        img_size=256,
        patch_size=16,
        embed_dim=256,
        depth=6,
        decoder_dim=128,
        decoder_depth=3,
        num_heads=8
    )
    model.load_state_dict(model_state)
    model.eval()

    # Load 5 real pretrain tiles to get an average
    tiles = list((ROOT_DIR / "data" / "pretrain" / "tiles").glob("*.npz"))
    if not tiles:
        print("No npz tiles found!")
        return

    selected_tiles = tiles[:5]
    print(f"Testing on {len(selected_tiles)} real tiles from data/pretrain/tiles...")

    results = []
    for idx, tpath in enumerate(selected_tiles):
        tile_data = np.load(tpath)
        s1_raw = torch.from_numpy(tile_data["s1"]).unsqueeze(0)  # (1, 2, 256, 256)
        s2_raw = torch.from_numpy(tile_data["s2"]).unsqueeze(0)  # (1, 7, 256, 256)

        # Generate a cloud mask (20% coverage)
        torch.manual_seed(42 + idx)
        mask_pixel, mask_patch = CloudMaskGenerator.generate_random_patch_mask(
            B=1, H=256, W=256, patch_size=16, mask_ratio=0.20
        )

        with torch.no_grad():
            # A) Normal S1
            out_normal = model(s1_raw, s2_raw, mask_pixel=mask_pixel, mask_patch=mask_patch)["pred_s2"]
            
            # B) S1 zeroed out (Ablation: No Radar)
            s1_zero = torch.zeros_like(s1_raw)
            out_zero = model(s1_zero, s2_raw, mask_pixel=mask_pixel, mask_patch=mask_patch)["pred_s2"]
            
            # C) S1 random noise
            s1_noise = torch.randn_like(s1_raw)
            out_noise = model(s1_noise, s2_raw, mask_pixel=mask_pixel, mask_patch=mask_patch)["pred_s2"]

        mask_bool = (mask_pixel.squeeze(0).squeeze(0) > 0.5)
        
        # Pixels inside the cloud mask
        p_norm  = out_normal.squeeze(0)[:, mask_bool]
        p_zero  = out_zero.squeeze(0)[:, mask_bool]
        p_noise = out_noise.squeeze(0)[:, mask_bool]

        diff_zero  = torch.abs(p_norm - p_zero).mean().item()
        diff_noise = torch.abs(p_norm - p_noise).mean().item()
        base_val   = torch.abs(p_norm).mean().item() + 1e-7

        rel_zero  = (diff_zero / base_val) * 100
        rel_noise = (diff_noise / base_val) * 100
        
        results.append({
            "tile": tpath.name,
            "diff_zero": diff_zero,
            "rel_zero": rel_zero,
            "diff_noise": diff_noise,
            "rel_noise": rel_noise,
            "base_val": base_val
        })
        print(f"  [{idx+1}/5] {tpath.name}: diff_when_zero={diff_zero:.6f} ({rel_zero:.2f}%), diff_noise={diff_noise:.6f} ({rel_noise:.2f}%)")

    avg_rel_zero = np.mean([r["rel_zero"] for r in results])
    avg_rel_noise = np.mean([r["rel_noise"] for r in results])

    print("-" * 70)
    print(f"Summary over 5 tiles:")
    print(f"  Average Reconstruction Change when S1 is ZEROED: {avg_rel_zero:.2f}%")
    print(f"  Average Reconstruction Change when S1 is NOISE:  {avg_rel_noise:.2f}%")
    
    if avg_rel_zero < 1.0:
        print("  => VERDICT: Cloud-MAE ALMOST COMPLETELY IGNORED SAR! (Impact < 1%)")
    elif avg_rel_zero < 5.0:
        print(f"  => VERDICT: S1 has a WEAK/MINOR impact ({avg_rel_zero:.2f}%). Optical context dominates 95%+.")
    else:
        print(f"  => VERDICT: S1 has a SIGNIFICANT impact ({avg_rel_zero:.2f}%) on inpainting!")


def test_cross_attn_weights():
    print("\n" + "=" * 70)
    print("TEST 2: Cross-Modal Attention Weights (S2 Query -> S1 SAR vs GSat RGB Keys)")
    print("=" * 70)

    checkpoint_path = ROOT_DIR / "checkpoints" / "benchmark_crossattn_cloudmae" / "vitpalm_best.pth"
    if not checkpoint_path.exists():
        checkpoint_path = ROOT_DIR / "checkpoints" / "benchmark_crossattn" / "vitpalm_best.pth"

    if not checkpoint_path.exists():
        print("Cross-attn checkpoint not found.")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"Loaded Cross-Attn Checkpoint: {checkpoint_path.parent.name}/{checkpoint_path.name}")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            model_state = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        else:
            model_state = checkpoint
    else:
        model_state = checkpoint

    model = ViTCrossAttnModel(
        img_size=256,
        patch_size=16,
        embed_dim=256,
        depth=6,
        num_heads=8
    )
    model.load_state_dict(model_state)
    model.eval()

    # Load a real pretrain tile for S1 and S2
    tiles = list((ROOT_DIR / "data" / "pretrain" / "tiles").glob("*.npz"))
    tdata = np.load(tiles[0])
    s1_t = torch.from_numpy(tdata["s1"]).unsqueeze(0) # (1, 2, 256, 256)
    s2_t = torch.from_numpy(tdata["s2"]).unsqueeze(0) # (1, 7, 256, 256)
    gsat_t = torch.randn(1, 3, 256, 256)             # (1, 3, 256, 256)

    with torch.no_grad():
        feat_s1 = model.encoder_s1(s1_t)[:, 1:, :]       # (1, 256, 256)
        feat_s2 = model.encoder_s2(s2_t)[:, 1:, :]       # (1, 256, 256)
        feat_gsat = model.encoder_gsat(gsat_t)[:, 1:, :] # (1, 256, 256)

        kv = torch.cat([feat_s1, feat_gsat], dim=1) # (1, 512, 256) where 0..255 = S1, 256..511 = GSat

        attn_out, attn_weights = model.cross_attn(
            query=feat_s2,
            key=kv,
            value=kv,
            need_weights=True,
            average_attn_weights=True
        )

    # attn_weights shape: (1, 256, 512)
    weights = attn_weights.squeeze(0) # (256, 512)
    s1_weight_per_query = weights[:, 0:256].sum(dim=1)   # weight allocated to S1
    gsat_weight_per_query = weights[:, 256:512].sum(dim=1) # weight allocated to GSat

    mean_s1 = s1_weight_per_query.mean().item()
    mean_gsat = gsat_weight_per_query.mean().item()

    print(f"\nAverage Cross-Attention Allocation across all 256 patches:")
    print(f"  Attention to Sentinel-1 SAR (Radar):  {mean_s1*100:.2f}%")
    print(f"  Attention to GEO-Sat Optical (RGB):   {mean_gsat*100:.2f}%")

    print(f"\nDistribution of Attention to S1 across patches:")
    print(f"  Min S1 weight on any patch: {s1_weight_per_query.min().item()*100:.2f}%")
    print(f"  Max S1 weight on any patch: {s1_weight_per_query.max().item()*100:.2f}%")
    print(f"  Std Dev: {s1_weight_per_query.std().item()*100:.2f}%")

    if abs(mean_s1 - 0.5) < 0.05 and s1_weight_per_query.std().item() < 0.02:
        print("  => VERDICT: The attention is UNIFORM/FLAT (50/50). The network did NOT learn meaningful selective routing!")
    else:
        print(f"  => VERDICT: Network exhibits active selective routing between SAR ({mean_s1*100:.1f}%) and Optical ({mean_gsat*100:.1f}%).")


if __name__ == "__main__":
    test_cloud_mae_sar_impact()
    test_cross_attn_weights()
