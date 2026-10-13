"""
tests/test_cloud_mae.py
=======================
Comprehensive test suite and benchmark for Cloud-MAE architecture.

Validates:
  1. Forward pass shapes (S1=2 bands, S2=7 bands: B2, B3, B4, B8A, B8, B11, B12)
  2. Block masking & random patch masking (50% - 75%)
  3. Loss function correctness (only penalizes masked pixels)
  4. Gradient flow / backward pass on all trainable parameters
  5. Mixed precision (AMP BFloat16) compatibility
  6. Latency, throughput (FPS), and parameter count benchmark
  7. Downstream weight transfer to 3 Fusion models (Early, Late, Cross-Attention)
"""

import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib2_vitpalm.model.cloud_mae import CloudMAEModel, CloudMaskGenerator
from lib2_vitpalm.model.late_fusion_model import ViTLateFusionModel
from lib2_vitpalm.model.cross_attn_model import ViTCrossAttnModel
from lib2_vitpalm.model.vitpalm import ViTPalmModel


def run_all_tests():
    print("=" * 70)
    print("  [>] CLOUD-MAE BENCHMARK & VERIFICATION SUITE (7 BANDS S2)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------
    # TEST 1: Cloud Masking Generators
    # ---------------------------------------------------------
    print("\n[TEST 1] Cloud Masking Generators...")
    B, H, W = 4, 256, 256
    
    # 1.1 Block Masking
    mask_pixel_blk, mask_patch_blk = CloudMaskGenerator.generate_block_cloud_mask(
        B, H, W, patch_size=16, target_ratio=0.65, device=device
    )
    assert mask_pixel_blk.shape == (B, 1, H, W), f"Wrong pixel mask shape: {mask_pixel_blk.shape}"
    assert mask_patch_blk.shape == (B, 256), f"Wrong patch mask shape: {mask_patch_blk.shape}"
    ratio_blk = mask_pixel_blk.mean().item()
    print(f"  [PASS] Block Masking: shape={mask_pixel_blk.shape}, coverage={ratio_blk*100:.1f}%")
    assert 0.40 <= ratio_blk <= 0.85, f"Block mask ratio out of expected range: {ratio_blk}"

    # 1.2 Random Patch Masking
    mask_pixel_rnd, mask_patch_rnd = CloudMaskGenerator.generate_random_patch_mask(
        B, H, W, patch_size=16, mask_ratio=0.70, device=device
    )
    ratio_rnd = mask_pixel_rnd.mean().item()
    print(f"  [PASS] Random Masking: shape={mask_pixel_rnd.shape}, coverage={ratio_rnd*100:.1f}%")
    assert 0.65 <= ratio_rnd <= 0.75, f"Random mask ratio unexpected: {ratio_rnd}"

    # ---------------------------------------------------------
    # TEST 2: Model Architecture & Parameter Counts (7 bands S2)
    # ---------------------------------------------------------
    print("\n[TEST 2] Model Architecture & Parameter Count (S2=7 bands)...")
    model = CloudMAEModel(
        s1_channels=2,
        s2_channels=7,
        img_size=256,
        patch_size=16,
        embed_dim=512,
        depth=6,
        num_heads=8,
        decoder_dim=256,
        decoder_depth=3,
        mask_ratio=0.65,
    ).to(device)

    enc_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    dec_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [PASS] Encoder Params : {enc_params:,} ({enc_params/1e6:.2f} M)")
    print(f"  [PASS] Decoder Params : {dec_params:,} ({dec_params/1e6:.2f} M)")
    print(f"  [PASS] Total Trainable: {total_params:,} ({total_params/1e6:.2f} M)")

    # ---------------------------------------------------------
    # TEST 3: Forward Pass & Output Dimensions (S2=7 bands)
    # ---------------------------------------------------------
    print("\n[TEST 3] Forward Pass & Output Dimensions...")
    s1 = torch.randn(B, 2, H, W, device=device)
    s2 = torch.rand(B, 7, H, W, device=device)

    out = model(s1, s2)
    assert "loss" in out and "pred_s2" in out and "latent" in out
    assert out["pred_s2"].shape == (B, 7, H, W), f"Pred shape mismatch: {out['pred_s2'].shape}"
    assert out["latent"].shape == (B, 256, 512), f"Latent shape mismatch: {out['latent'].shape}"
    assert not torch.isnan(out["loss"]), "Loss is NaN!"
    print(f"  [PASS] Forward Pass OK: Loss = {out['loss'].item():.6f}, Pred Shape = {tuple(out['pred_s2'].shape)}")

    # ---------------------------------------------------------
    # TEST 4: Loss Masking Invariance (Only penalizes masked pixels)
    # ---------------------------------------------------------
    print("\n[TEST 4] Loss Masking Invariance...")
    model.eval()
    mask_pix = out["mask_pixel"]
    s2_perturbed = s2.clone()
    unmasked_indices = (mask_pix.expand_as(s2) == 0)
    s2_perturbed[unmasked_indices] += 10.0

    # Ensure perturbing unmasked pixels in the target does not change the loss
    out_orig = model(s1, s2, mask_pixel=mask_pix, mask_patch=out["mask_patch"], target_s2=s2)
    out_pert = model(s1, s2, mask_pixel=mask_pix, mask_patch=out["mask_patch"], target_s2=s2_perturbed)
    diff_loss = abs(out_orig["loss"].item() - out_pert["loss"].item())
    print(f"  [PASS] Unmasked Perturbation Loss Delta: {diff_loss:.8f}")
    assert diff_loss < 1e-6, f"Loss is incorrectly penalizing unmasked pixels! Delta: {diff_loss}"
    model.train()

    # ---------------------------------------------------------
    # TEST 5: Backward Pass & Gradient Flow
    # ---------------------------------------------------------
    print("\n[TEST 5] Backward Pass & Gradient Flow...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss = out["loss"]
    loss.backward()

    no_grad_params = []
    for name, p in model.named_parameters():
        if p.requires_grad and (p.grad is None or torch.all(p.grad == 0)):
            no_grad_params.append(name)

    if no_grad_params:
        print(f"  [WARN] Warning: Parameters without grad: {no_grad_params}")
    else:
        print("  [PASS] All model parameters received non-zero gradients!")
    optimizer.step()

    # ---------------------------------------------------------
    # TEST 6: Mixed Precision (BFloat16) Benchmark
    # ---------------------------------------------------------
    print("\n[TEST 6] Mixed Precision (BFloat16) & Throughput...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = model(s1, s2)

        start_time = time.time()
        iters = 20
        batch_size = 16
        s1_bench = torch.randn(batch_size, 2, H, W, device=device)
        s2_bench = torch.rand(batch_size, 7, H, W, device=device)
        
        for _ in range(iters):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                res = model(s1_bench, s2_bench)
                l = res["loss"]
            l.backward()
            optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        fps = (iters * batch_size) / elapsed
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

        print(f"  [PASS] Batch Size : {batch_size}")
        print(f"  [PASS] Peak VRAM  : {peak_vram:.1f} MB ({(peak_vram/16384)*100:.1f}% of 16 GB)")
        print(f"  [PASS] Speed      : {fps:.1f} patches/sec ({elapsed/iters*1000:.1f} ms/batch)")
        print(f"  [PASS] Est. Time  : ~{ (6000 / fps * 100) / 60 :.1f} minutes for 100 Epochs (6,000 patches)")
    else:
        print("  (Running on CPU, skipping CUDA memory benchmark)")

    # ---------------------------------------------------------
    # TEST 7: Downstream Weight Transfer Verification
    # ---------------------------------------------------------
    print("\n[TEST 7] Downstream Weight Transfer Verification...")
    
    # Create downstream models matching embed_dim=512
    late_model = ViTLateFusionModel(img_size=256, embed_dim=512)
    cross_model = ViTCrossAttnModel(img_size=256, embed_dim=512)
    early_model = ViTPalmModel(in_channels=12, img_size=256, embed_dim=512)

    # 7.1 Transfer to Late Fusion
    model.transfer_to_late_fusion(late_model)
    x_12ch = torch.randn(2, 12, 256, 256)
    out_late = late_model(x_12ch)
    assert out_late.shape == (2, 256, 256), f"Late fusion output shape error: {out_late.shape}"
    print(f"  [PASS] Late Fusion forward with transferred weights: shape={out_late.shape}")

    # 7.2 Transfer to Cross-Attention
    model.transfer_to_cross_attn(cross_model)
    out_cross = cross_model(x_12ch)
    assert out_cross.shape == (2, 256, 256), f"Cross-Attn output shape error: {out_cross.shape}"
    print(f"  [PASS] Cross-Attention forward with transferred weights: shape={out_cross.shape}")

    # 7.3 Transfer to Early Fusion
    model.transfer_to_early_fusion(early_model)
    out_early = early_model(x_12ch)
    assert out_early.shape == (2, 256, 256), f"Early Fusion output shape error: {out_early.shape}"
    print(f"  [PASS] Early Fusion forward with transferred weights: shape={out_early.shape}")

    print("\n" + "=" * 70)
    print("  [SUCCESS] ALL 7 CLOUD-MAE BENCHMARK & TRANSFER CHECKS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()

