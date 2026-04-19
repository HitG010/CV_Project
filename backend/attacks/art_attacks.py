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
from utils.image_utils import normalize, preprocess_224, to_tensor, IMAGENET_STD


def _upsample_delta(delta_patch: torch.Tensor, original_size: tuple[int, int]) -> torch.Tensor:
    """Bilinear upsample a 224x224 perturbation to the original (H, W)."""
    height, width = original_size
    return F.interpolate(delta_patch, (height, width), mode="bilinear", align_corners=False)


def _logits(x_norm: torch.Tensor, use_ensemble: bool) -> torch.Tensor:
    return ensemble_logits(x_norm) if use_ensemble else PRIMARY_MODEL(x_norm)


def fgsm_attack(
    pil_img: Image.Image,
    target_idx: int,
    epsilon: float,
    targeted: bool,
    ensemble: bool = True,
) -> torch.Tensor:
    input_tensor = preprocess_224(pil_img).unsqueeze(0).to(DEVICE).requires_grad_(True)
    loss_val = F.cross_entropy(
        _logits(normalize(input_tensor), ensemble),
        torch.tensor([target_idx], device=DEVICE),
    )
    loss_val.backward()
    grad_sign = input_tensor.grad.sign()
    delta = (-epsilon if targeted else epsilon) * grad_sign * IMAGENET_STD
    delta_upsampled = _upsample_delta(delta.detach(), (pil_img.height, pil_img.width))
    orig_tensor = to_tensor(pil_img).unsqueeze(0).to(DEVICE)
    return torch.clamp(orig_tensor + delta_upsampled, 0, 1).detach()



def mi_fgsm_attack(
    pil_img: Image.Image,
    target_idx: int,
    epsilon: float,
    targeted: bool,
    steps: int = 10,
    mu: float = 1.0,
    ensemble: bool = True,
) -> torch.Tensor:
    step_size = epsilon / steps
    orig_tensor = preprocess_224(pil_img).unsqueeze(0).to(DEVICE)
    adv_tensor = orig_tensor.clone()
    momentum = torch.zeros_like(adv_tensor)

    for _ in range(steps):
        adv_tensor = adv_tensor.detach().requires_grad_(True)
        loss_val = F.cross_entropy(
            _logits(normalize(adv_tensor), ensemble),
            torch.tensor([target_idx], device=DEVICE),
        )
        loss_val.backward()
        grad = adv_tensor.grad / (adv_tensor.grad.abs().mean() + 1e-10)
        momentum = mu * momentum + grad
        step = (-step_size if targeted else step_size) * momentum.sign()
        adv_tensor = torch.clamp(
            torch.min(torch.max(adv_tensor + step, orig_tensor - epsilon), orig_tensor + epsilon),
            0, 1
        )

    delta_upsampled = _upsample_delta(
        (adv_tensor - orig_tensor).detach() * IMAGENET_STD,
        (pil_img.height, pil_img.width),
    )
    orig_full = to_tensor(pil_img).unsqueeze(0).to(DEVICE)
    return torch.clamp(orig_full + delta_upsampled, 0, 1).detach()



def pgd_attack(
    pil_img: Image.Image,
    target_idx: int,
    epsilon: float,
    targeted: bool,
    steps: int = 20,
    alpha_ratio: float = 2.5,
    ensemble: bool = True,
) -> torch.Tensor:
    step_size = epsilon / alpha_ratio
    orig_tensor = preprocess_224(pil_img).unsqueeze(0).to(DEVICE)
    adv_tensor = torch.clamp(
        orig_tensor + torch.empty_like(orig_tensor).uniform_(-epsilon, epsilon), 0, 1
    )

    for _ in range(steps):
        adv_tensor = adv_tensor.detach().requires_grad_(True)
        loss_val = F.cross_entropy(
            _logits(normalize(adv_tensor), ensemble),
            torch.tensor([target_idx], device=DEVICE),
        )
        loss_val.backward()
        step = (-step_size if targeted else step_size) * adv_tensor.grad.sign()
        adv_tensor = torch.clamp(
            torch.min(torch.max(adv_tensor + step, orig_tensor - epsilon), orig_tensor + epsilon),
            0, 1
        )

    delta_upsampled = _upsample_delta(
        (adv_tensor - orig_tensor).detach() * IMAGENET_STD,
        (pil_img.height, pil_img.width),
    )
    orig_full = to_tensor(pil_img).unsqueeze(0).to(DEVICE)
    return torch.clamp(orig_full + delta_upsampled, 0, 1).detach()


def cw_l2_attack(
    pil_img: Image.Image,
    target_idx: int,
    targeted: bool = True,
    c: float = 1.0,
    c_candidates: list[float] | None = None,
    kappa: float = 0.0,
    steps: int = 200,
    lr: float = 5e-3,
) -> tuple[torch.Tensor, dict]:
    """
    L2-norm Carlini-Wagner attack optimized in tanh-space with Adam.
    A small grid search over the confidence constant `c` picks the best result.
    """
    def _run_cw(c_val: float) -> tuple[torch.Tensor, float, float, bool]:
        small_orig = preprocess_224(pil_img).unsqueeze(0).to(DEVICE)
        tanh_w = torch.atanh(
            torch.clamp(2 * small_orig - 1, -0.9999, 0.9999)
        ).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([tanh_w], lr=lr)
        target_label = torch.tensor([target_idx], device=DEVICE)
        best_adv, best_l2 = None, float("inf")
        best_any, best_constraint, best_any_l2 = None, float("inf"), float("inf")

        for _ in range(steps):
            optimizer.zero_grad()
            adv_small  = 0.5 * (torch.tanh(tanh_w) + 1)
            logits = PRIMARY_MODEL(normalize(adv_small))
            l2 = ((adv_small - small_orig) ** 2).sum(dim=(1, 2, 3))
            one_hot = torch.zeros_like(logits).scatter_(1, target_label.unsqueeze(1), 1)
            logit_target = (logits * one_hot).sum(1)
            logit_other = (logits * (1 - one_hot) - 1e9 * one_hot).max(1).values
            constraint = (
                torch.clamp(logit_other - logit_target + kappa, min=0) if targeted
                else torch.clamp(logit_target - logit_other + kappa, min=0)
            )
            (l2 + c_val * constraint).mean().backward()
            optimizer.step()
            with torch.no_grad():
                cur_l2 = float(l2.mean())
                cur_constraint = float(constraint.mean())
                if cur_constraint < best_constraint:
                    best_constraint = cur_constraint
                    best_any = adv_small.detach().clone()
                    best_any_l2 = cur_l2
                if cur_constraint <= 1e-6 and cur_l2 < best_l2:
                    best_l2 = cur_l2
                    best_adv = adv_small.detach().clone()

        success = best_adv is not None
        if not success:
            best_adv = best_any if best_any is not None else adv_small.detach().clone()
            best_l2 = best_any_l2
        return best_adv, best_l2, best_constraint, success

    if c_candidates is None:
        base = max(float(c), 1e-4)
        c_candidates = [base / 10, base / 2, base, base * 2, base * 5]
    c_candidates = [float(v) for v in c_candidates if v > 0]
    if not c_candidates:
        c_candidates = [float(c)]
    c_candidates = sorted(set(c_candidates))

    best_success_adv, best_success_l2 = None, float("inf")
    best_success_c, best_success_constraint = None, float("inf")
    best_any_adv, best_any_f = None, float("inf")
    best_any_c, best_any_l2 = None, float("inf")

    for c_val in c_candidates:
        adv, l2, constraint_val, success = _run_cw(c_val)
        if constraint_val < best_any_f:
            best_any_f = constraint_val
            best_any_adv = adv
            best_any_c = c_val
            best_any_l2 = l2
        if success and l2 < best_success_l2:
            best_success_l2 = l2
            best_success_adv = adv
            best_success_c = c_val
            best_success_constraint = constraint_val

    if best_success_adv is not None:
        best_adv = best_success_adv
        best_c = best_success_c
        best_l2 = best_success_l2
        best_f = best_success_constraint
        success = True
    else:
        best_adv = best_any_adv
        best_c = best_any_c
        best_l2 = best_any_l2
        best_f = best_any_f
        success = False

    delta_upsampled = _upsample_delta(
        (best_adv - preprocess_224(pil_img).unsqueeze(0).to(DEVICE)).detach(),
        (pil_img.height, pil_img.width),
    )
    orig_full = to_tensor(pil_img).unsqueeze(0).to(DEVICE)
    return torch.clamp(orig_full + delta_upsampled, 0, 1).detach(), {
        "cw_best_c": best_c,
        "cw_best_f_loss": round(float(best_f), 6),
        "cw_best_l2": round(float(best_l2), 6),
        "cw_success": success,
        "cw_c_candidates": c_candidates,
    }
