"""
projects.cross_band_recovery.eval.metrics
=========================================
Unified Evaluation Metrics for Cross-Band Satellite Reconstruction.
Implements hand-verifiable metrics:
1. MAE (Mean Absolute Error)
2. RMSE (Root Mean Squared Error)
3. PSNR (Peak Signal-to-Noise Ratio)
4. SSIM (Structural Similarity Index Measure)
5. Composite SAM (Spectral Angle Mapper) as defined in Appendix 11.1
"""

import math
from typing import Optional, Union, Dict
import numpy as np
import torch
import torch.nn.functional as F


def compute_mae(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """
    Mean Absolute Error on masked locations.
    y_true, y_pred: (..., H, W)
    mask: boolean tensor of same shape or broadcastable where True indicates evaluated pixels.
    """
    if isinstance(y_true, np.ndarray):
        diff = np.abs(y_true - y_pred)
        if mask is not None:
            return float(np.mean(diff[mask]))
        return float(np.mean(diff))
    else:
        diff = torch.abs(y_true - y_pred)
        if mask is not None:
            return float(diff[mask].mean().item())
        return float(diff.mean().item())


def compute_rmse(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """
    Root Mean Squared Error on masked locations.
    """
    if isinstance(y_true, np.ndarray):
        sq_diff = (y_true - y_pred) ** 2
        if mask is not None:
            return float(np.sqrt(np.mean(sq_diff[mask])))
        return float(np.sqrt(np.mean(sq_diff)))
    else:
        sq_diff = (y_true - y_pred) ** 2
        if mask is not None:
            return float(torch.sqrt(sq_diff[mask].mean()).item())
        return float(torch.sqrt(sq_diff.mean()).item())


def compute_psnr(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    PSNR = 10 * log10(data_range^2 / MSE)
    """
    rmse_val = compute_rmse(y_true, y_pred, mask=mask)
    if rmse_val <= 1e-10:
        return 99.0
    mse = rmse_val ** 2
    return float(10.0 * math.log10((data_range ** 2) / mse))


def compute_ssim_torch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0
) -> float:
    """
    Fast, differentiable 2D SSIM for PyTorch tensors of shape (B, C, H, W).
    """
    channel = y_true.size(1)
    # Gaussian window
    sigma = 1.5
    gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)], device=y_true.device, dtype=y_true.dtype)
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(y_true, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(y_pred, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(y_true * y_true, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(y_pred * y_pred, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(y_true * y_pred, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def compute_composite_sam(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask_opt: torch.Tensor,
    eps: float = 1e-7,
    eps2: float = 1e-5
) -> float:
    """
    Composite Spectral Angle Mapper (SAM) according to Appendix 11.1:
      V_composite[b] = X_pred[b] if b in M else X_true[b]
      SAM = arccos( clamp( (V_composite . X_true) / (||V_composite|| * ||X_true|| + eps), -1 + eps2, 1 - eps2 ) )
    
    y_true: (B, 7, H, W)
    y_pred: (B, 7, H, W)
    mask_opt: (B, 7) bool (True = MASKED)
    
    Returns:
        Mean SAM angle in radians.
    """
    # Expand mask_opt to (B, 7, H, W)
    mask_expanded = mask_opt.unsqueeze(-1).unsqueeze(-1).expand_as(y_true)
    
    # Form composite spectrum: take prediction on masked bands, ground truth on visible bands
    v_composite = torch.where(mask_expanded, y_pred, y_true)
    
    # Dot product along channel dimension (B, H, W)
    dot_product = (v_composite * y_true).sum(dim=1)
    
    # Norms along channel dimension
    norm_comp = torch.norm(v_composite, p=2, dim=1)
    norm_true = torch.norm(y_true, p=2, dim=1)
    
    cosine = dot_product / (norm_comp * norm_true + eps)
    clamped_cosine = torch.clamp(cosine, -1.0 + eps2, 1.0 - eps2)
    
    sam_map = torch.acos(clamped_cosine)
    return float(sam_map.mean().item())


def evaluate_all_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask_opt: torch.Tensor
) -> Dict[str, float]:
    """
    Computes all standard metrics between true and predicted optical imagery.
    """
    B, C, H, W = y_true.shape
    mask_expanded = mask_opt.unsqueeze(-1).unsqueeze(-1).expand_as(y_true)
    
    mae = compute_mae(y_true, y_pred, mask=mask_expanded)
    rmse = compute_rmse(y_true, y_pred, mask=mask_expanded)
    psnr = compute_psnr(y_true, y_pred, mask=mask_expanded)
    ssim = compute_ssim_torch(y_true, y_pred)
    sam = compute_composite_sam(y_true, y_pred, mask_opt)
    
    return {
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
        "sam_rad": round(sam, 5),
        "sam_deg": round(math.degrees(sam), 3)
    }
