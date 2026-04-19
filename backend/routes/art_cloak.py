"""
routes/art_cloak.py — /art-cloak and /compare-attacks endpoints.
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from flask import Blueprint, request, jsonify
from PIL import Image

from config import DEVICE, IDX_TO_CLASS
from models.art_models import PRIMARY_MODEL
from attacks import fgsm_attack, mi_fgsm_attack, pgd_attack, cw_l2_attack
from utils.image_utils import (
    normalize, preprocess_224, to_tensor,
    pil_to_b64, b64_to_pil, tensor_to_b64,
)
from utils.metrics import full_quality_metrics

art_bp = Blueprint("art_cloak", __name__)


# ─────────────────────────────────────────────────────────────
#  Core service function
# ─────────────────────────────────────────────────────────────

def run_art_cloak(
    image_b64: str,
    intensity: float = 0.01,
    mode: str = "untargeted",
    method: str = "mi_fgsm",
    target_class_name: str | None = None,
    ensemble: bool = True,
) -> tuple[str | None, dict]:

    orig_img = b64_to_pil(image_b64)
    orig_px  = to_tensor(orig_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        x_norm   = normalize(preprocess_224(orig_img).unsqueeze(0).to(DEVICE))
        probs_b  = F.softmax(PRIMARY_MODEL(x_norm), dim=1)[0]

    targeted = (mode == "targeted")

    if target_class_name is None:
        target_idx        = int(torch.argmax(probs_b))
        target_class_name = IDX_TO_CLASS[target_idx]
    else:
        try:
            target_idx = IDX_TO_CLASS.index(target_class_name)
        except ValueError:
            return None, {"error": f"Unknown class: {target_class_name}"}

    method = method.lower()
    if method == "fgsm":
        adv = fgsm_attack(orig_img, target_idx, intensity, targeted, ensemble)
    elif method == "pgd":
        adv = pgd_attack(orig_img, target_idx, intensity, targeted, ensemble=ensemble)
    elif method == "cw":
        adv = cw_l2_attack(orig_img, target_idx, targeted=targeted)
    else:
        adv = mi_fgsm_attack(orig_img, target_idx, intensity, targeted, ensemble=ensemble)

    with torch.no_grad():
        adv_pil = T.ToPILImage()(adv.squeeze(0).cpu())
        x_norm_a = normalize(preprocess_224(adv_pil).unsqueeze(0).to(DEVICE))
        probs_a  = F.softmax(PRIMARY_MODEL(x_norm_a), dim=1)[0]

    top_b = torch.topk(probs_b, 3)
    top_a = torch.topk(probs_a, 3)
    adv_full = (
        F.interpolate(adv, orig_px.shape[2:], mode="bilinear", align_corners=False)
        if adv.shape != orig_px.shape else adv
    )

    return tensor_to_b64(adv), {
        "method":   method,
        "mode":     mode,
        "ensemble": ensemble,
        "target_class": target_class_name,
        "original_top3": [
            {"class": IDX_TO_CLASS[top_b.indices[i]], "prob": round(float(top_b.values[i]), 4)}
            for i in range(3)
        ],
        "cloaked_top3": [
            {"class": IDX_TO_CLASS[top_a.indices[i]], "prob": round(float(top_a.values[i]), 4)}
            for i in range(3)
        ],
        "quality_metrics": full_quality_metrics(orig_px.to(DEVICE), adv_full.to(DEVICE)),
        "attack_fooled":   IDX_TO_CLASS[top_a.indices[0]] != IDX_TO_CLASS[top_b.indices[0]],
    }


# ─────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────

def _get_image_b64(data, files) -> str | None:
    b64 = data.get("image_base64")
    if b64:
        return b64
    f = files.get("image")
    return pil_to_b64(Image.open(f).convert("RGB")) if f else None


@art_bp.route("/art-cloak", methods=["POST"])
def art_cloak_api():
    data      = request.get_json() if request.is_json else request.form.to_dict()
    image_b64 = _get_image_b64(data, request.files)
    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    method    = data.get("method", "mi_fgsm").lower()
    intensity = float(data.get("intensity", 0.01))
    mode      = data.get("mode", "untargeted").lower()
    target    = data.get("target_class") or None
    ensemble  = str(data.get("ensemble", "true")).lower() != "false"

    cloaked_b64, resp = run_art_cloak(image_b64, intensity, mode, method, target, ensemble)
    if cloaked_b64 is None:
        return jsonify(resp), 400
    return jsonify({"cloaked_image": cloaked_b64, "response": resp}), 200


@art_bp.route("/compare-attacks", methods=["POST"])
def compare_attacks_api():
    data      = request.get_json() if request.is_json else request.form.to_dict()
    image_b64 = _get_image_b64(data, request.files)
    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    intensity = float(data.get("intensity", 0.01))
    mode      = data.get("mode", "untargeted").lower()
    target    = data.get("target_class") or None

    results = {}
    for m in ["fgsm", "mi_fgsm", "pgd"]:
        cloaked_b64, resp = run_art_cloak(image_b64, intensity, mode, m, target, ensemble=True)
        results[m] = {"cloaked_image": cloaked_b64, "response": resp}

    return jsonify({"comparison": results, "intensity": intensity}), 200
