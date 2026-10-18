"""
projects.cross_band_recovery.models.channel_masking
===================================================
Phase 4 (P4) / Gate G4a: Cross-Band Channel Masking Module.

Features:
1. Vectorized PyTorch batch masking without per-sample Python loops (CHK-G4a.7).
2. Uniform k distribution over k in [1, 7] so every severity level is equally trained (CHK-G4a.3).
3. Independent SAR dropout with rate p_sar=0.20 (CHK-G4a.4).
4. Zero empty input guard: never drop optical k=7 and SAR simultaneously (CHK-G4a.5).
5. Deterministic 127 pattern mapping according to Appendix 11.2 (CHK-G4a.2).
6. Eval mode switches: 'train', 'eval_s1_on', 'eval_s1_off' (CHK-G4b.8 prep).
7. Seed / torch.Generator support for exact bitwise reproducibility (CHK-G4a.6).
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


OPTICAL_BAND_NAMES = ["B02", "B03", "B04", "B8A", "B08", "B11", "B12"]
SAR_BAND_NAMES = ["vv", "vh"]
NUM_OPTICAL_BANDS = len(OPTICAL_BAND_NAMES)  # 7
NUM_SAR_BANDS = len(SAR_BAND_NAMES)          # 2
NUM_TOTAL_BANDS = NUM_OPTICAL_BANDS + NUM_SAR_BANDS  # 9


def pattern_id_to_mask(pattern_id: Union[int, torch.Tensor], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    แปลง Pattern ID (1 ถึง 127) เป็น Optical Mask Tensor ขนาด (..., 7) ของ boolean
    ตามนิยามภาคผนวก 11.2:
      บิต i = 1 หมายถึง แบนด์ลำดับ i ถูกปิด (True = Masked)
      ลำดับบิต: [B02, B03, B04, B8A, B08, B11, B12]
    """
    if isinstance(pattern_id, int):
        if not (1 <= pattern_id <= 127):
            raise ValueError(f"pattern_id must be in [1, 127], got {pattern_id}")
        bits = [(pattern_id >> i) & 1 == 1 for i in range(NUM_OPTICAL_BANDS)]
        return torch.tensor(bits, dtype=torch.bool, device=device)
    
    # Tensor of pattern IDs
    pattern_tensor = pattern_id.to(device=device, dtype=torch.int64)
    shifts = torch.arange(NUM_OPTICAL_BANDS, device=device, dtype=torch.int64)
    # Shape: (..., 7)
    mask = ((pattern_tensor.unsqueeze(-1) >> shifts) & 1) == 1
    return mask


def mask_to_pattern_id(mask_opt: torch.Tensor) -> torch.Tensor:
    """
    แปลง Optical Mask Tensor ขนาด (..., 7) กลับเป็น Pattern ID (1 ถึง 127)
    """
    shifts = torch.arange(NUM_OPTICAL_BANDS, device=mask_opt.device, dtype=torch.int64)
    weights = 1 << shifts  # [1, 2, 4, 8, 16, 32, 64]
    pattern_ids = (mask_opt.to(torch.int64) * weights).sum(dim=-1)
    return pattern_ids


class CrossBandMasker(nn.Module):
    """
    Cross-Band Channel Masking Module for Multi-Modal Satellite Foundation Models.
    
    Generates optical masks (B, 7) and SAR masks (B, 2) where:
      True  = MASKED (band is dropped / missing and needs reconstruction)
      False = VISIBLE (band is observed and passed to encoder)
    """
    def __init__(
        self,
        p_sar: float = 0.20,
        k_min: int = 1,
        k_max: int = 7,
        num_opt_bands: int = 7,
        num_sar_bands: int = 2
    ):
        super().__init__()
        assert 1 <= k_min <= k_max <= num_opt_bands, f"Invalid k bounds: [{k_min}, {k_max}]"
        assert 0.0 <= p_sar <= 1.0, f"p_sar must be in [0, 1], got {p_sar}"
        
        self.p_sar = p_sar
        self.k_min = k_min
        self.k_max = k_max
        self.num_opt = num_opt_bands
        self.num_sar = num_sar_bands

    def forward(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
        eval_mode: str = "train",
        pattern_id: Optional[Union[int, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        สร้าง (mask_opt, mask_sar)
        
        Parameters:
        -----------
        batch_size: int
        device: torch.device
        generator: torch.Generator (สำหรับ seed determinism)
        eval_mode: 'train' | 'eval_s1_on' | 'eval_s1_off'
        pattern_id: Optional int or tensor in [1, 127] สำหรับ deterministic evaluation
        
        Returns:
        --------
        mask_opt: torch.Tensor of shape (B, 7), dtype=torch.bool
        mask_sar: torch.Tensor of shape (B, 2), dtype=torch.bool
        """
        # 1. Optical Masking
        if pattern_id is not None:
            # Deterministic pattern evaluation
            if isinstance(pattern_id, int):
                single_mask = pattern_id_to_mask(pattern_id, device=device)
                mask_opt = single_mask.unsqueeze(0).expand(batch_size, self.num_opt)
            else:
                p_tens = pattern_id.to(device=device)
                if p_tens.ndim == 0:
                    p_tens = p_tens.expand(batch_size)
                mask_opt = pattern_id_to_mask(p_tens, device=device)
        else:
            # Random masking with Uniform k in [k_min, k_max]
            # CHK-G4a.3: Uniform k distribution across batch
            k_choices = torch.randint(
                low=self.k_min,
                high=self.k_max + 1,
                size=(batch_size,),
                device=device,
                generator=generator
            )
            
            # Vectorized subset selection:
            # Generate random noise for all optical bands, sort to get random permutations
            noise = torch.rand(batch_size, self.num_opt, device=device, generator=generator)
            ranks = torch.argsort(noise, dim=-1)
            # Take the first k elements of permutation as masked
            mask_opt = ranks < k_choices.unsqueeze(-1)
            
        # 2. SAR Masking
        if eval_mode == "eval_s1_on":
            # SAR is strictly visible
            mask_sar = torch.zeros(batch_size, self.num_sar, dtype=torch.bool, device=device)
        elif eval_mode == "eval_s1_off":
            # SAR is strictly masked (disabled)
            mask_sar = torch.ones(batch_size, self.num_sar, dtype=torch.bool, device=device)
        else:
            # Training / Random mode: independent dropout with probability p_sar
            rand_sar = torch.rand(batch_size, device=device, generator=generator)
            drop_sar = rand_sar < self.p_sar
            mask_sar = drop_sar.unsqueeze(-1).expand(batch_size, self.num_sar).clone()
            
            # CHK-G4a.5: Zero Empty Input Guard
            # Never allow optical k=7 (all optical masked) AND SAR dropped simultaneously
            all_opt_masked = mask_opt.sum(dim=-1) == self.num_opt
            empty_conflict = all_opt_masked & mask_sar[:, 0]
            if empty_conflict.any():
                # Force SAR to be VISIBLE for samples with empty conflict
                mask_sar[empty_conflict] = False

        return mask_opt, mask_sar


# Convenient alias
RandomChannelMask = CrossBandMasker
