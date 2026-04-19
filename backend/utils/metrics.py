"""
utils/metrics.py — Perceptual quality + perturbation-size metrics.
"""
import numpy as np
import torch
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as _ssim_fn
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False


def compute_psnr(orig: torch.Tensor, adv: torch.Tensor) -> float:
    """PSNR in dB.  Higher → more imperceptible perturbation."""
    mse = F.mse_loss(adv, orig).item()
    # Avoid log(0) if images are identical (PSNR would be infinite).
    mse = max(mse, 1e-12)
    return 10 * np.log10(1.0 / mse)


def compute_ssim(orig: torch.Tensor, adv: torch.Tensor) -> float:
    """SSIM ∈ (-1, 1]. Closer to 1 = near-identical images.
    Falls back to a fast approximation when pytorch-msssim is absent."""
    if HAS_MSSSIM:
        return float(_ssim_fn(orig, adv, data_range=1.0, size_average=True))
    orig_flat, adv_flat = orig.flatten(), adv.flatten()
    mu_orig, mu_adv = orig_flat.mean(), adv_flat.mean()
    sigma_cross = ((orig_flat - mu_orig) * (adv_flat - mu_adv)).mean()
    sigma_orig2 = ((orig_flat - mu_orig) ** 2).mean()
    sigma_adv2 = ((adv_flat - mu_adv) ** 2).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_val = (
        (2 * mu_orig * mu_adv + C1) * (2 * sigma_cross + C2)
    ) / (
        (mu_orig ** 2 + mu_adv ** 2 + C1) * (sigma_orig2 + sigma_adv2 + C2)
    )
    return float(ssim_val)


def perturbation_norms(orig: torch.Tensor, adv: torch.Tensor) -> dict:
    delta = adv - orig
    return {
        "linf":          float(delta.abs().max()),
        "l2":            float(delta.norm(p=2).item()),
        "l2_normalized": float(delta.norm(p=2).item() / orig.numel() ** 0.5),
    }


def full_quality_metrics(orig: torch.Tensor, adv: torch.Tensor) -> dict:
    """Single-call helper that returns all quality metrics."""
    return {
        "psnr_db": round(compute_psnr(orig, adv), 4),
        "ssim":    round(compute_ssim(orig, adv), 6),
        **{k: round(v, 6) for k, v in perturbation_norms(orig, adv).items()},
    }
