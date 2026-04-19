
import torch
import torch.nn.functional as F
from PIL import Image
from config import DEVICE
from models.face_models import FACENET, MTCNN_DET
from utils.image_utils import face_preprocess, facenet_prewhiten, to_tensor
from utils.metrics import full_quality_metrics

def _face_loss(
    emb_adv: torch.Tensor,
    emb_orig: torch.Tensor,
    emb_target: torch.Tensor | None,
    targeted: bool,
) -> torch.Tensor:
    if targeted:
        return F.cosine_similarity(emb_adv, emb_target).mean()
    return -F.cosine_similarity(emb_adv, emb_orig).mean()


def _embed(face_t: torch.Tensor) -> torch.Tensor:
    embedding = FACENET(facenet_prewhiten(face_t))
    return F.normalize(embedding, p=2, dim=1)

def face_fgsm(
    face_t: torch.Tensor,
    emb_orig: torch.Tensor,
    emb_target: torch.Tensor | None,
    epsilon: float,
    targeted: bool,
) -> torch.Tensor:
    adv_input = face_t.clone().requires_grad_(True)
    _face_loss(_embed(adv_input), emb_orig, emb_target, targeted).backward()
    return torch.clamp(face_t + epsilon * adv_input.grad.sign(), 0, 1).detach()


def face_mi_fgsm(
    face_t: torch.Tensor,
    emb_orig: torch.Tensor,
    emb_target: torch.Tensor | None,
    epsilon: float,
    targeted: bool,
    steps: int = 10,
    mu: float = 1.0,
) -> torch.Tensor:
    step_size = epsilon / steps
    adv_tensor = face_t.clone()
    momentum = torch.zeros_like(adv_tensor)

    for _ in range(steps):
        adv_tensor = adv_tensor.detach().requires_grad_(True)
        _face_loss(_embed(adv_tensor), emb_orig, emb_target, targeted).backward()
        grad = adv_tensor.grad / (adv_tensor.grad.abs().mean() + 1e-10)
        momentum = mu * momentum + grad
        step = step_size * momentum.sign()
        adv_tensor = torch.clamp(
            torch.min(torch.max(adv_tensor + step, face_t - epsilon), face_t + epsilon),
            0, 1
        )
    return adv_tensor.detach()


def face_pgd(
    face_t: torch.Tensor,
    emb_orig: torch.Tensor,
    emb_target: torch.Tensor | None,
    epsilon: float,
    targeted: bool,
    steps: int = 20,
    alpha_ratio: float = 2.5,
) -> torch.Tensor:
    step_size = epsilon / alpha_ratio
    adv_tensor = torch.clamp(
        face_t + torch.empty_like(face_t).uniform_(-epsilon, epsilon), 0, 1
    )
    for _ in range(steps):
        adv_tensor = adv_tensor.detach().requires_grad_(True)
        _face_loss(_embed(adv_tensor), emb_orig, emb_target, targeted).backward()
        step = step_size * adv_tensor.grad.sign()
        adv_tensor = torch.clamp(
            torch.min(torch.max(adv_tensor + step, face_t - epsilon), face_t + epsilon),
            0, 1
        )
    return adv_tensor.detach()

def cloak_face(
    orig_img: Image.Image,
    intensity: float = 0.01,
    method: str = "mi_fgsm",
    targeted: bool = False,
    target_identity_img: Image.Image | None = None,
) -> tuple[torch.Tensor | None, dict]:

    def _crop_first_face(img: Image.Image) -> tuple[Image.Image | None, tuple[int, int, int, int] | None]:
        boxes, _ = MTCNN_DET.detect(img)
        if boxes is None:
            return None, None
        x1, y1, x2, y2 = map(int, boxes[0])
        w, h = img.size
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return img.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)

    face_crop, box = _crop_first_face(orig_img)
    if face_crop is None or box is None:
        return None, {"error": "No face detected"}

    x1, y1, x2, y2 = box
    face_tensor = face_preprocess(face_crop).unsqueeze(0).to(DEVICE)

    emb_orig   = _embed(face_tensor).detach()
    emb_target = None
    if targeted:
        if target_identity_img is None:
            return None, {"error": "Targeted attack requires a target identity image"}
        target_face, _ = _crop_first_face(target_identity_img)
        if target_face is None:
            return None, {"error": "No face detected in target identity image"}
        emb_target = _embed(
            face_preprocess(target_face).unsqueeze(0).to(DEVICE)
        ).detach()

    # Dispatch
    method = method.lower()
    if method == "fgsm":
        adv_face = face_fgsm(face_tensor, emb_orig, emb_target, intensity, targeted)
    elif method == "pgd":
        adv_face = face_pgd(face_tensor, emb_orig, emb_target, intensity, targeted)
    else:  # default: mi_fgsm
        adv_face = face_mi_fgsm(face_tensor, emb_orig, emb_target, intensity, targeted)

    # Upsample delta and paste back onto full image
    delta_small = adv_face - face_tensor
    face_H, face_W = y2 - y1, x2 - x1
    import torch.nn.functional as F2
    delta_big = F2.interpolate(
        delta_small, (face_H, face_W), mode="bilinear", align_corners=False
    )
    orig_tensor = to_tensor(orig_img).unsqueeze(0).to(DEVICE)
    perturbed   = orig_tensor.clone()
    perturbed[:, :, y1:y2, x1:x2] = torch.clamp(
        orig_tensor[:, :, y1:y2, x1:x2] + delta_big, 0, 1
    )

    # Post-attack embedding
    import torchvision.transforms as T
    adv_crop = T.ToPILImage()(perturbed[0, :, y1:y2, x1:x2].cpu())
    emb_adv = _embed(face_preprocess(adv_crop).unsqueeze(0).to(DEVICE)).detach()

    cos_after = float(F2.cosine_similarity(emb_orig, emb_adv))
    emb_dist = float((emb_orig - emb_adv).norm())
    quality = full_quality_metrics(orig_tensor, perturbed)

    metrics = {
        "method": method,
        "cosine_similarity_before": 1.0,
        "cosine_similarity_after":  round(cos_after, 6),
        "similarity_drop":          round(1.0 - cos_after, 6),
        "embedding_l2_distance":    round(emb_dist, 6),
        "attack_success":           cos_after < 0.85,
        "effective_cloaking_score": round(min(1.0, (1 - cos_after) * 1.3), 6),
        "quality_metrics": quality,
    }
    if targeted and emb_target is not None:
        metrics["target_similarity_before"] = round(
            float(F2.cosine_similarity(emb_orig, emb_target)), 6
        )
        metrics["target_similarity_after"] = round(
            float(F2.cosine_similarity(emb_adv, emb_target)), 6
        )
        metrics["push_toward_target"] = round(
            metrics["target_similarity_after"] - metrics["target_similarity_before"], 6
        )

    return perturbed, metrics
