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
            response = _gemini.generate_content(prompt)
            text = (response.text or "").strip()
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group())
        except Exception as e:
            if "429" not in str(e) and "quota" not in str(e).lower():
                break
            time.sleep(min(20, 1.5 ** attempt) + random.uniform(0, 1))
    return {"action_taken": "none", "tool_used": None,
            "user_friendly_instructions": "Retry in a few seconds."}


@agent_bp.route("/agent", methods=["POST"])
def agent_api():
    try:
        payload = request.get_json() if request.is_json else request.form.to_dict()
        user_message = (payload.get("user_message") or "").strip()
        if not user_message:
            return jsonify({"error": "user_message required"}), 400

        image_b64 = payload.get("image_base64")
        if not image_b64:
            uploaded_file = request.files.get("file") or request.files.get("image")
            if uploaded_file:
                image_b64 = pil_to_b64(Image.open(uploaded_file).convert("RGB"))

        intensity    = float(payload.get("intensity", 0.01))
        method       = payload.get("method", "mi_fgsm").lower()
        mode         = payload.get("mode", "untargeted").lower()
        targeted     = str(payload.get("targeted", "false")).lower() == "true"
        target_class = payload.get("target_class") or None
        target_b64   = payload.get("target_image_base64")

        message_lower = user_message.lower()
        # Basic intent routing based on keywords.
        if any(k in message_lower for k in ["face", "facial", "identity"]):
            intent = "face_cloak"
        elif any(k in message_lower for k in ["art", "cloak", "image protect"]):
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

        explanation = _safe_generate(
            f"{SYSTEM_PROMPT}\nUser: {user_message}\n"
            f"Tool: {intent}\nMetrics: {json.dumps(metrics or {})}"
        )

        return jsonify({
            "intent":               intent,
            "tool_used":            intent if cloaked_b64 else None,
            "metrics":              metrics,
            "explanation":          explanation,
            "cloaked_image_base64": cloaked_b64,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
