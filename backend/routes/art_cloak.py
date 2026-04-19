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

    original_image = b64_to_pil(image_b64)
    original_tensor  = to_tensor(original_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        input_norm   = normalize(preprocess_224(original_image).unsqueeze(0).to(DEVICE))
        probs_before = F.softmax(PRIMARY_MODEL(input_norm), dim=1)[0]

    is_targeted = (mode == "targeted")

    if target_class_name is None:
        target_index      = int(torch.argmax(probs_before))
        target_class_name = IDX_TO_CLASS[target_index]
    else:
        try:
            target_index = IDX_TO_CLASS.index(target_class_name)
        except ValueError:
            return None, {"error": f"Unknown class: {target_class_name}"}

    method = method.lower()
    cw_info = None
    if method == "fgsm":
        adv_tensor = fgsm_attack(original_image, target_index, intensity, is_targeted, ensemble)
    elif method == "pgd":
        adv_tensor = pgd_attack(original_image, target_index, intensity, is_targeted, ensemble=ensemble)
    elif method == "cw":
        adv_tensor, cw_info = cw_l2_attack(original_image, target_index, targeted=is_targeted)
    else:
        adv_tensor = mi_fgsm_attack(original_image, target_index, intensity, is_targeted, ensemble=ensemble)

    with torch.no_grad():
        adv_pil = T.ToPILImage()(adv_tensor.squeeze(0).cpu())
        adv_norm = normalize(preprocess_224(adv_pil).unsqueeze(0).to(DEVICE))
        probs_after  = F.softmax(PRIMARY_MODEL(adv_norm), dim=1)[0]

    top_before = torch.topk(probs_before, 3)
    top_after = torch.topk(probs_after, 3)
    adv_full_res = (
        F.interpolate(adv_tensor, original_tensor.shape[2:], mode="bilinear", align_corners=False)
        if adv_tensor.shape != original_tensor.shape else adv_tensor
    )

    response = {
        "method":   method,
        "mode":     mode,
        "ensemble": ensemble,
        "target_class": target_class_name,
        "original_top3": [
            {"class": IDX_TO_CLASS[top_before.indices[i]], "prob": round(float(top_before.values[i]), 4)}
            for i in range(3)
        ],
        "cloaked_top3": [
            {"class": IDX_TO_CLASS[top_after.indices[i]], "prob": round(float(top_after.values[i]), 4)}
            for i in range(3)
        ],
        "quality_metrics": full_quality_metrics(original_tensor.to(DEVICE), adv_full_res.to(DEVICE)),
        "attack_fooled":   IDX_TO_CLASS[top_after.indices[0]] != IDX_TO_CLASS[top_before.indices[0]],
    }
    if cw_info is not None:
        response["cw_search"] = cw_info
    return tensor_to_b64(adv_tensor), response


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
    payload   = request.get_json() if request.is_json else request.form.to_dict()
    image_b64 = _get_image_b64(payload, request.files)
    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    method    = payload.get("method", "mi_fgsm").lower()
    intensity = float(payload.get("intensity", 0.01))
    mode      = payload.get("mode", "untargeted").lower()
    target    = payload.get("target_class") or None
    ensemble  = str(payload.get("ensemble", "true")).lower() != "false"

    cloaked_b64, resp = run_art_cloak(image_b64, intensity, mode, method, target, ensemble)
    if cloaked_b64 is None:
        return jsonify(resp), 400
    return jsonify({"cloaked_image": cloaked_b64, "response": resp}), 200


@art_bp.route("/compare-attacks", methods=["POST"])
def compare_attacks_api():
    payload   = request.get_json() if request.is_json else request.form.to_dict()
    image_b64 = _get_image_b64(payload, request.files)
    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    intensity = float(payload.get("intensity", 0.01))
    mode      = payload.get("mode", "untargeted").lower()
    target    = payload.get("target_class") or None

    results = {}
    for method_name in ["fgsm", "mi_fgsm", "pgd"]:
        cloaked_b64, resp = run_art_cloak(
            image_b64, intensity, mode, method_name, target, ensemble=True
        )
        results[method_name] = {"cloaked_image": cloaked_b64, "response": resp}

    return jsonify({"comparison": results, "intensity": intensity}), 200
