"""
attacks/art_attacks.py
======================
FGSM, MI-FGSM, PGD, and C&W L2 adversarial attacks for image classification.

All functions accept a PIL Image and return a full-resolution pixel tensor [1,3,H,W].
"""
import torch
import torch.nn.functional as F
from PIL import Image
from config import DEVICE
from models.art_models import PRIMARY_MODEL, ensemble_logits
from utils.image_utils import (
    normalize, preprocess_224, to_tensor, IMAGENET_STD,
)


# ─────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────

def _upsample_delta(delta_small: torch.Tensor, orig_size: tuple[int, int]) -> torch.Tensor:
    """Bilinear upsample a 224×224 perturbation to original (H, W)."""
    H, W = orig_size
    return F.interpolate(delta_small, (H, W), mode="bilinear", align_corners=False)


def _logits(x_norm: torch.Tensor, use_ensemble: bool) -> torch.Tensor:
    return ensemble_logits(x_norm) if use_ensemble else PRIMARY_MODEL(x_norm)


# ─────────────────────────────────────────────────────────────
#  FGSM  — single step
# ─────────────────────────────────────────────────────────────

def fgsm_attack(
    pil_img: Image.Image,
    target_idx: int,
    epsilon: float,
    targeted: bool,
    ensemble: bool = True,
) -> torch.Tensor:
    x = preprocess_224(pil_img).unsqueeze(0).to(DEVICE).requires_grad_(True)
    loss = F.cross_entropy(_logits(normalize(x), ensemble),
                           torch.tensor([target_idx], device=DEVICE))
    loss.backward()
    sign   = x.grad.sign()
    delta  = (-epsilon if targeted else epsilon) * sign * IMAGENET_STD
    delta_up = _upsample_delta(delta.detach(), (pil_img.height, pil_img.width))
    orig   = to_tensor(pil_img).unsqueeze(0).to(DEVICE)
    return torch.clamp(orig + delta_up, 0, 1).detach()


# ─────────────────────────────────────────────────────────────
#  MI-FGSM  — Momentum Iterative FGSM (Dong et al., 2018)
# ─────────────────────────────────────────────────────────────

def mi_fgsm_attack(
    pil_img: Image.Image,
    target_idx: int,
    epsilon: float,
    targeted: bool,
    steps: int = 10,
    mu: float = 1.0,
    ensemble: bool = True,
) -> torch.Tensor:
    alpha  = epsilon / steps
    x_orig = preprocess_224(pil_img).unsqueeze(0).to(DEVICE)
    x_adv  = x_orig.clone()
    g      = torch.zeros_like(x_adv)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss  = F.cross_entropy(_logits(normalize(x_adv), ensemble),
                                torch.tensor([target_idx], device=DEVICE))
        loss.backward()
        grad  = x_adv.grad / (x_adv.grad.abs().mean() + 1e-10)
        g     = mu * g + grad
        step  = (-alpha if targeted else alpha) * g.sign()
        x_adv = torch.clamp(
            torch.min(torch.max(x_adv + step, x_orig - epsilon), x_orig + epsilon),
            0, 1
        )

    delta_up = _upsample_delta(
        (x_adv - x_orig).detach() * IMAGENET_STD,
        (pil_img.height, pil_img.width),
    )
    orig = to_tensor(pil_img).unsqueeze(0).to(DEVICE)
    return torch.clamp(orig + delta_up, 0, 1).detach()


# ─────────────────────────────────────────────────────────────
#  PGD  — Projected Gradient Descent with random restart
# ─────────────────────────────────────────────────────────────

def pgd_attack(
    pil_img: Image.Image,
    target_idx: int,
    epsilon: float,
    targeted: bool,
    steps: int = 20,
    alpha_ratio: float = 2.5,
    ensemble: bool = True,
) -> torch.Tensor:
    alpha  = epsilon / alpha_ratio
    x_orig = preprocess_224(pil_img).unsqueeze(0).to(DEVICE)
    x_adv  = torch.clamp(
        x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon), 0, 1
    )

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss  = F.cross_entropy(_logits(normalize(x_adv), ensemble),
                                torch.tensor([target_idx], device=DEVICE))
        loss.backward()
        step  = (-alpha if targeted else alpha) * x_adv.grad.sign()
        x_adv = torch.clamp(
            torch.min(torch.max(x_adv + step, x_orig - epsilon), x_orig + epsilon),
            0, 1
        )

    delta_up = _upsample_delta(
        (x_adv - x_orig).detach() * IMAGENET_STD,
        (pil_img.height, pil_img.width),
    )
    orig = to_tensor(pil_img).unsqueeze(0).to(DEVICE)
    return torch.clamp(orig + delta_up, 0, 1).detach()


# ─────────────────────────────────────────────────────────────
#  C&W L2  — Carlini & Wagner minimum-distortion attack
# ─────────────────────────────────────────────────────────────

def cw_l2_attack(
    pil_img: Image.Image,
    target_idx: int,
    targeted: bool = True,
    c: float = 1.0,
    kappa: float = 0.0,
    steps: int = 200,
    lr: float = 5e-3,
) -> torch.Tensor:
    """
    Minimises  ||δ||₂  +  c·f(x+δ)   via change-of-variable
    w = atanh(2x − 1)  so  x_adv = 0.5·(tanh(w)+1) ∈ [0,1].
    Works in 224×224 space; delta is upsampled back to original resolution.
    """
    x_orig = preprocess_224(pil_img).unsqueeze(0).to(DEVICE)
    w = torch.atanh(torch.clamp(2 * x_orig - 1, -0.9999, 0.9999)).detach().requires_grad_(True)
    optim   = torch.optim.Adam([w], lr=lr)
    t_label = torch.tensor([target_idx], device=DEVICE)
    best_adv, best_l2 = x_orig.clone(), float("inf")

    for _ in range(steps):
        optim.zero_grad()
        x_adv  = 0.5 * (torch.tanh(w) + 1)
        logits = PRIMARY_MODEL(normalize(x_adv))
        l2     = ((x_adv - x_orig) ** 2).sum(dim=(1, 2, 3))
        one_hot = torch.zeros_like(logits).scatter_(1, t_label.unsqueeze(1), 1)
        Z_t     = (logits * one_hot).sum(1)
        Z_other = (logits * (1 - one_hot) - 1e9 * one_hot).max(1).values
        f_loss  = (
            torch.clamp(Z_other - Z_t + kappa, min=0) if targeted
            else torch.clamp(Z_t - Z_other + kappa, min=0)
        )
        (l2 + c * f_loss).mean().backward()
        optim.step()
        with torch.no_grad():
            cur_l2 = float(l2.mean())
            if cur_l2 < best_l2:
                best_l2  = cur_l2
                best_adv = x_adv.detach().clone()

    delta_up = _upsample_delta(
        (best_adv - x_orig).detach(),
        (pil_img.height, pil_img.width),
    )
    orig = to_tensor(pil_img).unsqueeze(0).to(DEVICE)
    return torch.clamp(orig + delta_up, 0, 1).detach()
