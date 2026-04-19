## 🔐 Mirage-AI: Adversarial Cloaking for Visual Privacy

In today’s digital ecosystem, images shared online are rapidly collected and analyzed by large-scale AI systems for tasks such as face recognition and object classification. These systems operate at the pixel level, extracting patterns that are often imperceptible to humans but highly informative for deep neural networks. As a result, individuals have little control over how their visual data is used once it is uploaded.

Mirage-AI addresses this problem by leveraging the concept of **adversarial perturbations**—carefully crafted, minimal modifications to an image that remain visually indistinguishable to humans but significantly disrupt the predictions of AI models. Unlike traditional privacy methods such as blurring or watermarking, which degrade image quality or can be easily removed, adversarial cloaking operates directly within the feature space learned by neural networks.

The system implements multiple state-of-the-art adversarial attack algorithms, including FGSM, PGD, MI-FGSM, and Carlini & Wagner (L2), to generate robust and transferable perturbations. These attacks are applied in two key scenarios:

* **Face Cloaking**, where identity recognition systems are misled by altering embedding representations
* **Art Cloaking**, where image classification models are forced to misclassify content

To improve real-world effectiveness, Mirage-AI also incorporates an ensemble attack strategy across multiple deep learning architectures, increasing the likelihood of fooling unseen models. The result is a practical, scalable tool that enables users to protect their visual data without compromising perceptual quality.

By bridging adversarial machine learning with privacy preservation, Mirage-AI highlights both the vulnerabilities of modern AI systems and the potential for defensive applications of these techniques.


# Mirage-AI v2 — Adversarial Image Cloaking

> MI-FGSM · FGSM · PGD · C&W L2 · Ensemble · Face Cloaking

---

## Project Structure

```
mirage-ai/
├── backend/
│   ├── app.py                   ← Flask entry point  (port 8080)
│   ├── config.py                ← Device selection + shared constants
│   ├── requirements.txt
│   ├── benchmark_runner.py      ← Separate benchmark service (port 8081)
│   │
│   ├── models/
│   │   ├── art_models.py        ← ResNet50 + VGG16 + DenseNet121 ensemble
│   │   └── face_models.py       ← MTCNN detector + FaceNet (VGGFace2)
│   │
│   ├── attacks/
│   │   ├── art_attacks.py       ← FGSM / MI-FGSM / PGD / C&W L2
│   │   └── face_attacks.py      ← Face embedding attacks
│   │
│   ├── utils/
│   │   ├── image_utils.py       ← PIL ↔ tensor ↔ base64
│   │   └── metrics.py           ← PSNR / SSIM / L2 / L∞
│   │
│   ├── routes/
│   │   ├── art_cloak.py         ← /art-cloak  /compare-attacks
│   │   ├── face_cloak.py        ← /face-cloak
│   │   └── agent.py             ← /agent  (Gemini NLP router)
│   │
│   └── data/
│       └── imagenet_classes.txt ← Add or auto download before running
│
└── frontend/
    ├── index.html
    ├── vite.config.js
    ├── package.json
    └── src/
        ├── main.jsx
        ├── index.css
        ├── App.jsx              ← Art Cloak · Face Cloak · Compare tabs
        └── components/
            ├── DropZone.jsx
            ├── MetricsPanel.jsx
            └── UI.jsx
```

---

## Step-by-Step Setup

### 0. Prerequisites

| Requirement | Version      |
| ----------- | ------------ |
| Python      | 3.10 or 3.11 |
| Node.js     | 18+          |
| pip         | latest       |

---

### 1. Download `imagenet_classes.txt` ← **Required before anything else**

```bash
mkdir -p backend/data
curl -o backend/data/imagenet_classes.txt \
  https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

Or manually: open the URL in a browser, save as `backend/data/imagenet_classes.txt`.

---

### 2. Set up the Python backend

```bash
cd backend

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**GPU users only** — replace the torch lines in requirements.txt with:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Optional: Gemini agent key

Create `backend/.env`:

```
GENAI_API_KEY=your_google_gemini_api_key_here
```

The `/agent` endpoint is skipped gracefully if the key is absent.

---

### 3. Start the backend

```bash
# From the backend/ folder with venv active
python app.py
```

You should see:

```
 * Running on http://0.0.0.0:8080
```

Test it:

```bash
curl http://localhost:8080/
```

---

### 4. Set up and start the React frontend

```bash
cd frontend

npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

---

### 5. (Optional) Start the benchmark runner

```bash
# In a separate terminal, from backend/
python benchmark_runner.py
# Runs on port 8081
```

---

## API Endpoints

| Method | Endpoint           | Description                              |
| ------ | ------------------ | ---------------------------------------- |
| POST   | `/art-cloak`       | Add adversarial perturbation to an image |
| POST   | `/face-cloak`      | Cloak face to defeat recognition         |
| POST   | `/compare-attacks` | Run FGSM + MI-FGSM + PGD side-by-side    |
| POST   | `/agent`           | Natural-language Gemini-powered router   |

### Quick test with curl

```bash
# Base64-encode a test image
B64=$(base64 -i /path/to/image.jpg)

# Art cloak
curl -X POST http://localhost:8080/art-cloak \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\":\"$B64\",\"method\":\"mi_fgsm\",\"intensity\":0.02}"
```

---

## Attack Methods

| Method  | Description                       | Transferability |
| ------- | --------------------------------- | --------------- |
| FGSM    | Single-step gradient sign         | Low             |
| MI-FGSM | Momentum iterative, 10 steps      | **High**        |
| PGD     | Random restart + projection       | High            |
| C&W L2  | Minimum L2 distortion (200 steps) | Medium          |

---

## Key External Dependency

| File                                | Source                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------- |
| `backend/data/imagenet_classes.txt` | https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt |

FaceNet model weights (`vggface2`) are downloaded automatically by `facenet-pytorch` on first run (~100 MB).

---

## Troubleshooting

**`FileNotFoundError: imagenet_classes.txt`**
→ Run the curl command in Step 1.

**`No face detected`**
→ Use a well-lit, front-facing photo. MTCNN requires a clearly visible face.

**Slow on CPU**
→ Reduce `steps` for MI-FGSM/PGD or use `method=fgsm`. C&W is the slowest (200 Adam steps).

**CORS errors in browser**
→ Make sure both `python app.py` (port 8080) and `npm run dev` (port 3000) are running.

**Port already in use**
→ `lsof -ti:8080 | xargs kill` or change port in `app.py`.
