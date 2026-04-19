"""
routes/face_cloak.py — /face-cloak endpoint.
"""
from flask import Blueprint, request, jsonify
from PIL import Image

from attacks.face_attacks import cloak_face
from utils.image_utils import pil_to_b64, b64_to_pil, tensor_to_b64

face_bp = Blueprint("face_cloak", __name__)


def _get_b64(data, files, key_json, key_file) -> str | None:
    b64 = data.get(key_json)
    if b64:
        return b64
    f = files.get(key_file)
    return pil_to_b64(Image.open(f).convert("RGB")) if f else None


@face_bp.route("/face-cloak", methods=["POST"])
def face_cloak_api():
    payload = request.get_json() if request.is_json else request.form.to_dict()

    image_b64 = _get_b64(payload, request.files, "image_base64", "image")
    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    intensity  = float(payload.get("intensity", 0.01))
    method     = payload.get("method", "mi_fgsm").lower()
    targeted   = str(payload.get("targeted", "false")).lower() == "true"
    target_b64 = _get_b64(payload, request.files, "target_image_base64", "target_image")

    original_image = b64_to_pil(image_b64)
    target_image = b64_to_pil(target_b64) if (targeted and target_b64) else None

    perturbed, metrics = cloak_face(
        original_image, intensity, method, targeted, target_image
    )
    if perturbed is None:
        return jsonify(metrics), 400

    return jsonify({
        "cloaked_image": tensor_to_b64(perturbed),
        "response": metrics,
    }), 200
