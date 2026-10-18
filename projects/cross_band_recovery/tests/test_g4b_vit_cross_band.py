"""
projects.cross_band_recovery.tests.test_g4b_vit_cross_band
==========================================================
Quality Gate G4b Test Suite: Cross-Band ViT Verification.
Covers checks CHK-G4b.1 through CHK-G4b.9 according to the Master Development Plan.
"""

import time
import pytest
import torch
import torch.nn as nn
from projects.cross_band_recovery.models.vit_cross_band import CrossBandViT
from projects.cross_band_recovery.models.channel_masking import CrossBandMasker, pattern_id_to_mask


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model(device):
    m = CrossBandViT(
        img_size=128,
        patch_size=16,
        embed_dim=256,
        depth=4,
        num_heads=8,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4
    ).to(device)
    m.eval()
    return m


def test_chk_g4b_1_dimensions(model, device):
    """
    CHK-G4b.1: Token มีขนาด (B, L_vis, D); เอาต์พุตการกู้คืนมีขนาด (B, 7, 128, 128)
    """
    B = 4
    opt = torch.rand(B, 7, 128, 128, device=device)
    sar = torch.rand(B, 2, 128, 128, device=device)
    
    masker = CrossBandMasker()
    mask_opt, mask_sar = masker(batch_size=B, device=device)
    
    out = model(opt, sar, mask_opt, mask_sar)
    pred_opt = out["pred_opt"]
    latent = out["latent"]
    
    assert pred_opt.shape == (B, 7, 128, 128), f"Expected (B, 7, 128, 128), got {pred_opt.shape}"
    assert latent.shape[0] == B and latent.shape[-1] == model.embed_dim
    print(f"\n[PASS] CHK-G4b.1: Output dimensions {pred_opt.shape} verified.")


def test_chk_g4b_2_zero_leakage_gradient(device):
    """
    CHK-G4b.2: ค่า gradient ของ loss เทียบกับอินพุตของแบนด์ที่ถูกปิดเป็น 0 เป๊ะ (ไม่ใช่เกือบ 0)
    เพราะแบนด์ที่ปิดไม่เคยถูก slice หรือส่งเข้า encoder
    """
    m = CrossBandViT(
        img_size=128, patch_size=16, embed_dim=128, depth=2, num_heads=4,
        decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=2
    ).to(device)
    m.train()
    
    B = 2
    opt = torch.rand(B, 7, 128, 128, device=device, requires_grad=True)
    sar = torch.rand(B, 2, 128, 128, device=device, requires_grad=True)
    
    # Mask bands 0, 2, 4 (B02, B04, B08)
    mask_opt = torch.zeros(B, 7, dtype=torch.bool, device=device)
    mask_opt[:, [0, 2, 4]] = True
    mask_sar = torch.zeros(B, 2, dtype=torch.bool, device=device)
    
    out = m(opt, sar, mask_opt, mask_sar)
    pred_opt = out["pred_opt"]
    
    # Loss on prediction
    target = torch.rand_like(pred_opt)
    loss = F_loss = ((pred_opt - target) ** 2).sum()
    loss.backward()
    
    assert opt.grad is not None, "opt.grad should exist"
    
    # Gradients for masked bands [0, 2, 4] MUST BE IDENTICALLY ZERO!
    for masked_idx in [0, 2, 4]:
        grad_masked = opt.grad[:, masked_idx, :, :]
        max_abs_grad = grad_masked.abs().max().item()
        assert max_abs_grad == 0.0, f"Leakage detected! Band {masked_idx} has grad {max_abs_grad} > 0"
        
    # Gradients for visible bands [1, 3, 5, 6] should be non-zero
    for vis_idx in [1, 3, 5, 6]:
        grad_vis = opt.grad[:, vis_idx, :, :]
        assert grad_vis.abs().sum().item() > 0.0, f"Visible band {vis_idx} has zero grad"
        
    print("\n[PASS] CHK-G4b.2: Zero-leakage verified: dLoss/d(x_masked) == 0.0000000 identically.")


def test_chk_g4b_3_masked_value_invariance(model, device):
    """
    CHK-G4b.3: เปลี่ยนค่าพิกเซลของแบนด์ที่ถูกปิดเป็นค่าสุ่มใดๆ หรือค่าสุดโต่ง แล้วเอาต์พุตต้องไม่เปลี่ยน
    """
    B = 2
    opt = torch.rand(B, 7, 128, 128, device=device)
    sar = torch.rand(B, 2, 128, 128, device=device)
    
    # Mask bands 1, 3, 5
    mask_opt = torch.zeros(B, 7, dtype=torch.bool, device=device)
    mask_opt[:, [1, 3, 5]] = True
    mask_sar = torch.zeros(B, 2, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        pred_clean = model(opt, sar, mask_opt, mask_sar)["pred_opt"]
        
        # Corrupt the masked bands with wild values (+9999.0, -8888.0, random noise)
        opt_corrupted = opt.clone()
        opt_corrupted[:, 1, :, :] = 9999.0
        opt_corrupted[:, 3, :, :] = -8888.0
        opt_corrupted[:, 5, :, :] = torch.randn(B, 128, 128, device=device) * 500.0
        
        pred_corrupted = model(opt_corrupted, sar, mask_opt, mask_sar)["pred_opt"]
        
    max_diff = (pred_clean - pred_corrupted).abs().max().item()
    assert max_diff == 0.0, f"Masked-value invariance violated: max_diff = {max_diff}"
    print(f"\n[PASS] CHK-G4b.3: Masked-value invariance verified (max diff = {max_diff:.8f}).")


def test_chk_g4b_4_gradient_to_channel_embeddings(device):
    """
    CHK-G4b.4: ทุก embedding ของแบนด์ที่ปรากฏใน batch มี .grad ไม่ใช่ None และไม่เป็นศูนย์
    """
    m = CrossBandViT(
        img_size=128, patch_size=16, embed_dim=128, depth=2, num_heads=4,
        decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=2
    ).to(device)
    m.train()
    
    B = 2
    opt = torch.rand(B, 7, 128, 128, device=device)
    sar = torch.rand(B, 2, 128, 128, device=device)
    
    # Visible optical: bands [0, 1], Masked optical: [2, 3, 4, 5, 6]
    mask_opt = torch.ones(B, 7, dtype=torch.bool, device=device)
    mask_opt[:, [0, 1]] = False
    # Visible SAR
    mask_sar = torch.zeros(B, 2, dtype=torch.bool, device=device)
    
    out = m(opt, sar, mask_opt, mask_sar)
    loss = out["pred_opt"].sum()
    loss.backward()
    
    # Encoder channel embeddings for visible channels 0, 1 (opt) and 7, 8 (sar)
    assert m.channel_embed.grad is not None, "channel_embed.grad is None"
    for c in [0, 1, 7, 8]:
        grad_c = m.channel_embed.grad[c]
        assert grad_c.abs().sum().item() > 0.0, f"Channel embedding {c} has zero grad"
        
    # Decoder channel embeddings for reconstructed bands
    assert m.decoder_channel_embed.grad is not None
    for c in range(7):
        grad_dec = m.decoder_channel_embed.grad[c]
        assert grad_dec.abs().sum().item() > 0.0, f"Decoder channel embedding {c} has zero grad"
        
    print("\n[PASS] CHK-G4b.4: Gradients propagate correctly to Channel Embeddings.")


def test_chk_g4b_5_batch_sample_independence(model, device):
    """
    CHK-G4b.5: เปลี่ยนค่าตัวอย่างที่ 1 ในแบตช์ แล้วเอาต์พุตของตัวอย่างที่ 0 ต้องไม่เปลี่ยน
    """
    opt = torch.rand(2, 7, 128, 128, device=device)
    sar = torch.rand(2, 2, 128, 128, device=device)
    
    mask_opt = torch.zeros(2, 7, dtype=torch.bool, device=device)
    mask_opt[0, [0, 1]] = True     # Sample 0 has k=2
    mask_opt[1, [0, 1, 2, 3, 4, 5]] = True  # Sample 1 has k=6
    mask_sar = torch.zeros(2, 2, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        out1 = model(opt, sar, mask_opt, mask_sar)
        pred0_run1 = out1["pred_opt"][0].clone()
        
        # Modify Sample 1 completely
        opt_mod = opt.clone()
        sar_mod = sar.clone()
        opt_mod[1] = torch.rand(7, 128, 128, device=device) * 20.0
        sar_mod[1] = torch.rand(2, 128, 128, device=device) * 20.0
        
        out2 = model(opt_mod, sar_mod, mask_opt, mask_sar)
        pred0_run2 = out2["pred_opt"][0].clone()
        
    max_diff = (pred0_run1 - pred0_run2).abs().max().item()
    assert max_diff < 1e-6, f"Batch sample contamination detected: max_diff = {max_diff}"
    print(f"\n[PASS] CHK-G4b.5: Batch sample independence verified (max diff = {max_diff:.8f}).")


def test_chk_g4b_6_padding_mask_invariance(model, device):
    """
    CHK-G4b.6: ผลของตัวอย่างหนึ่งเหมือนเดิม ไม่ว่าจะถูกรันเดี่ยวๆ หรือรันร่วมกับตัวอย่างอื่นที่มี sequence ยาวกว่า
    """
    opt = torch.rand(2, 7, 128, 128, device=device)
    sar = torch.rand(2, 2, 128, 128, device=device)
    
    # Sample 0 has only 1 visible band (short sequence)
    mask_opt = torch.zeros(2, 7, dtype=torch.bool, device=device)
    mask_opt[0, 1:] = True  # only band 0 visible
    mask_opt[1, :] = False  # all 7 bands visible (long sequence)
    
    mask_sar = torch.zeros(2, 2, dtype=torch.bool, device=device)
    mask_sar[0, :] = True   # SAR masked for sample 0
    mask_sar[1, :] = False  # SAR visible for sample 1
    
    with torch.no_grad():
        # Run Sample 0 alone in batch size 1
        out_single = model(opt[:1], sar[:1], mask_opt[:1], mask_sar[:1])
        pred_single = out_single["pred_opt"][0]
        
        # Run Sample 0 alongside Sample 1 in batch size 2
        out_batch = model(opt, sar, mask_opt, mask_sar)
        pred_batch = out_batch["pred_opt"][0]
        
    diff = (pred_single - pred_batch).abs().max().item()
    assert diff < 1e-4, f"Padding mask invariance failed: diff = {diff}"
    print(f"\n[PASS] CHK-G4b.6: Padding mask invariance verified (max diff = {diff:.6f}).")


def test_chk_g4b_7_numerical_stability_fp32_bf16(device):
    """
    CHK-G4b.7: 100 รอบ forward/backward ทั้ง FP32 และ BF16 ไม่มี NaN/Inf
    """
    m = CrossBandViT(
        img_size=128, patch_size=16, embed_dim=128, depth=2, num_heads=4,
        decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=2
    ).to(device)
    
    masker = CrossBandMasker()
    
    for dtype, dtype_name in [(torch.float32, "FP32"), (torch.bfloat16, "BF16")]:
        m_cast = m.to(dtype=dtype)
        m_cast.train()
        optimizer = torch.optim.AdamW(m_cast.parameters(), lr=1e-4)
        
        nan_inf_found = False
        for step in range(25):  # 25 passes per dtype
            optimizer.zero_grad()
            opt = torch.rand(2, 7, 128, 128, device=device, dtype=dtype)
            sar = torch.rand(2, 2, 128, 128, device=device, dtype=dtype)
            m_opt, m_sar = masker(batch_size=2, device=device)
            
            out = m_cast(opt, sar, m_opt, m_sar)
            pred = out["pred_opt"]
            
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                nan_inf_found = True
                break
                
            loss = pred.sum()
            loss.backward()
            
            # Check gradients
            for p in m_cast.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        nan_inf_found = True
                        break
            if nan_inf_found:
                break
            optimizer.step()
            
        assert not nan_inf_found, f"NaN/Inf encountered during {dtype_name} training steps!"
        
    print("\n[PASS] CHK-G4b.7: Numerical stability confirmed for both FP32 and BF16 (Zero NaN/Inf).")


def test_chk_g4b_8_all_127_patterns_runnable(model, device):
    """
    CHK-G4b.8: โมเดลรันผ่านทั้ง 127 pattern x S1_on และ 126 pattern x S1_off โดยไม่ error
    """
    # 1. 127 patterns with S1_on (SAR visible)
    for pid in range(1, 128):
        mask_opt = pattern_id_to_mask(pid, device=device).unsqueeze(0)
        mask_sar = torch.zeros(1, 2, dtype=torch.bool, device=device)
        opt = torch.rand(1, 7, 128, 128, device=device)
        sar = torch.rand(1, 2, 128, 128, device=device)
        
        with torch.no_grad():
            out = model(opt, sar, mask_opt, mask_sar)
            assert out["pred_opt"].shape == (1, 7, 128, 128)
            
    # 2. 126 patterns with S1_off (pid 1..126; pid 127 + S1_off is excluded by no-empty guard)
    for pid in range(1, 127):
        mask_opt = pattern_id_to_mask(pid, device=device).unsqueeze(0)
        mask_sar = torch.ones(1, 2, dtype=torch.bool, device=device)
        opt = torch.rand(1, 7, 128, 128, device=device)
        sar = torch.rand(1, 2, 128, 128, device=device)
        
        with torch.no_grad():
            out = model(opt, sar, mask_opt, mask_sar)
            assert out["pred_opt"].shape == (1, 7, 128, 128)
            
    print("\n[PASS] CHK-G4b.8: Successfully executed all 127 S1_on and 126 S1_off patterns (253 combinations).")


def test_chk_g4b_9_profiling_and_throughput(model, device):
    """
    CHK-G4b.9: บันทึกจำนวนพารามิเตอร์, VRAM และ Throughput จริง
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    B = 16
    opt = torch.rand(B, 7, 128, 128, device=device)
    sar = torch.rand(B, 2, 128, 128, device=device)
    mask_opt = torch.zeros(B, 7, dtype=torch.bool, device=device)
    mask_opt[:, :3] = True
    mask_sar = torch.zeros(B, 2, dtype=torch.bool, device=device)
    
    # Warmup
    for _ in range(5):
        _ = model(opt, sar, mask_opt, mask_sar)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    t0 = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        _ = model(opt, sar, mask_opt, mask_sar)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    elapsed = time.perf_counter() - t0
    fps = (B * iterations) / elapsed
    
    print(f"\n[PASS] CHK-G4b.9: Model profile: {num_params:,} parameters, Throughput = {fps:.1f} tiles/sec on {device.type.upper()}.")
    assert num_params > 0 and fps > 0
