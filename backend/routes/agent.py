"""
routes/agent.py — /agent endpoint (Gemini-powered intent router).
"""
import json, os, random, re, time, traceback
from flask import Blueprint, request, jsonify
from PIL import Image
import google.generativeai as genai

from routes.art_cloak import run_art_cloak
from attacks.face_attacks import cloak_face
from utils.image_utils import pil_to_b64, b64_to_pil, tensor_to_b64

agent_bp = Blueprint("agent", __name__)

genai.configure(api_key=os.getenv("GENAI_API_KEY", ""))

SYSTEM_PROMPT = """
You are Mirage-AI v2 assistant. Output ONLY valid JSON.
Fields when a tool was used:
  action_taken, tool_used, method_used, attack_summary,
  quality_metrics_summary, cloaked_image_provided, user_friendly_instructions.
Keep it under 120 words. Do not hallucinate metrics.
"""

_gemini = genai.GenerativeModel(
    "models/gemini-2.5-flash",
    generation_config={"max_output_tokens": 1024, "temperature": 0.2},
)


def _safe_generate(prompt: str) -> dict:
    for attempt in range(4):
        try:
            r    = _gemini.generate_content(prompt)
            text = (r.text or "").strip()
            m    = re.search(r"\{[\s\S]*\}", text)
            if m:
                return json.loads(m.group())
        except Exception as e:
            if "429" not in str(e) and "quota" not in str(e).lower():
                break
            time.sleep(min(20, 1.5 ** attempt) + random.uniform(0, 1))
    return {"action_taken": "none", "tool_used": None,
            "user_friendly_instructions": "Retry in a few seconds."}


@agent_bp.route("/agent", methods=["POST"])
def agent_api():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        user_message = (data.get("user_message") or "").strip()
        if not user_message:
            return jsonify({"error": "user_message required"}), 400

        image_b64 = data.get("image_base64")
        if not image_b64:
            f = request.files.get("file") or request.files.get("image")
            if f:
                image_b64 = pil_to_b64(Image.open(f).convert("RGB"))

        intensity    = float(data.get("intensity", 0.01))
        method       = data.get("method", "mi_fgsm").lower()
        mode         = data.get("mode", "untargeted").lower()
        targeted     = str(data.get("targeted", "false")).lower() == "true"
        target_class = data.get("target_class") or None
        target_b64   = data.get("target_image_base64")

        msg = user_message.lower()
        if any(k in msg for k in ["face", "facial", "identity"]):
            intent = "face_cloak"
        elif any(k in msg for k in ["art", "cloak", "image protect"]):
            intent = "art_cloak"
        else:
            intent = "chat"

        cloaked_b64, metrics = None, None
        if intent == "art_cloak" and image_b64:
            cloaked_b64, metrics = run_art_cloak(
                image_b64, intensity, mode, method, target_class
            )
        elif intent == "face_cloak" and image_b64:
            target_img = b64_to_pil(target_b64) if (targeted and target_b64) else None
            perturbed, metrics = cloak_face(
                b64_to_pil(image_b64), intensity, method, targeted, target_img
            )
            cloaked_b64 = tensor_to_b64(perturbed) if perturbed is not None else None

        expl = _safe_generate(
            f"{SYSTEM_PROMPT}\nUser: {user_message}\n"
            f"Tool: {intent}\nMetrics: {json.dumps(metrics or {})}"
        )

        return jsonify({
            "intent":               intent,
            "tool_used":            intent if cloaked_b64 else None,
            "metrics":              metrics,
            "explanation":          expl,
            "cloaked_image_base64": cloaked_b64,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
