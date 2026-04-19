"""
app.py — Mirage-AI v2
Entry point for the main backend service (port 8080).

Run with:
    python app.py
or production:
    gunicorn -w 1 -b 0.0.0.0:8080 app:app
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context   # torchvision downloads

from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from routes import art_bp, face_bp, agent_bp

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(art_bp)
app.register_blueprint(face_bp)
app.register_blueprint(agent_bp)


@app.route("/")
def home():
    return jsonify({
        "service":   "Mirage-AI v2",
        "version":   "2.0.0",
        "attacks":   ["fgsm", "mi_fgsm", "pgd", "cw"],
        "endpoints": ["/art-cloak", "/face-cloak", "/compare-attacks", "/agent"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=8080, host="0.0.0.0")
