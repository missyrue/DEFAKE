# DEFAKE — Anti-Deepfake Image Cloaking System

> **Hackathon Demo** | Adversarial image protection powered by PyTorch + CLIP + React

---

## 🏗️ Architecture Overview

```
defake/
├── backend/
│   ├── module1_vision.py   # Input handling + MediaPipe face detection
│   ├── module2_warp.py     # Local spatial distortions around landmarks
│   ├── module3_cloak.py    # PGD adversarial CLIP attack + EoT
│   ├── app.py              # Flask REST API (orchestrates modules 1→2→3)
│   └── requirements.txt
└── frontend/
    ├── index.html          # Standalone HTML demo (no build step needed)
    └── src/App.jsx         # React source (for Vite / CRA projects)
```

---

## ⚙️ Backend Setup

### 1. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **CUDA users**: install the CUDA-enabled PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Start the Flask server

```bash
python app.py
```

The API will start on **http://localhost:5000**.

First startup takes ~30–60 seconds as the CLIP model downloads (~600 MB).

---

## 🖥️ Frontend Setup

### Option A — Zero install (recommended for demo)

Just open `frontend/index.html` directly in your browser:

```bash
open frontend/index.html        # macOS
xdg-open frontend/index.html   # Linux
```

The page auto-detects the backend. If the Flask server isn't running, it falls back to a **2-second mock demo** showing a 14.2% similarity score.

### Option B — Vite / CRA project

```bash
npm create vite@latest defake-ui -- --template react
cd defake-ui
cp ../frontend/src/App.jsx src/App.jsx
npm install lucide-react
npm run dev
```

---

## 🔌 API Reference

### `POST /protect`

**Request** — multipart/form-data:
| Field | Type | Description |
|-------|------|-------------|
| `image` | file | JPEG / PNG / WEBP image |

**Response** — JSON:
```json
{
  "success": true,
  "face_detected": true,
  "protected_image": "<base64-encoded PNG>",
  "similarity_score": 0.142,
  "processing_time_s": 14.3,
  "message": "Image successfully protected."
}
```

**Identity Stability Score**: Cosine similarity between CLIP embeddings of the original and protected image.
- **0.0–0.3** → Strong protection (AI sees a different identity)
- **0.3–0.6** → Moderate protection
- **0.6–1.0** → Weak protection (original identity detectable)

### `GET /health`

Returns `{"status": "ok"}` — useful for smoke testing.

---

## 🔬 Technical Deep-Dive

### Module 1 — Vision (Face Detection)
- Uses **MediaPipe FaceMesh** (468 3D landmarks)
- Extracts bounding boxes for: left eye, right eye, nose, mouth
- If no face is detected → original image returned unchanged

### Module 2 — Warp (Local Distortions)
- Builds a **sparse random displacement grid** at each landmark region
- Upsamples to full resolution via bicubic interpolation (smooth, organic warp)
- Composites with **Gaussian-feathered alpha mask** (no hard seams)
- Max displacement: **2 pixels** (imperceptible to humans)

### Module 3 — Cloak (Adversarial Noise)
- **CLIP model**: `openai/clip-vit-base-patch32`
- **Attack**: PGD (Projected Gradient Descent), 40 steps
- **Objective**: Minimise cosine similarity between original and protected CLIP embeddings
- **EoT** (Expectation over Transformation): each step averages gradients across 4 random augmentations (resize + JPEG simulation) → noise survives screenshots
- **L∞ constraint**: ε = 8/255 (invisible to human eye)

---

## 🚀 Production Notes

For production deployment, use **Gunicorn** (single worker — PyTorch CUDA is not multi-process safe):

```bash
pip install gunicorn
gunicorn -w 1 --worker-class gthread --threads 2 -b 0.0.0.0:5000 app:app
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Tensor operations, PGD attack |
| `transformers` | CLIP model loading |
| `mediapipe` | Face landmark detection |
| `opencv-python` | Image processing |
| `kornia` | Differentiable JPEG simulation (EoT) |
| `flask`, `flask-cors` | REST API |
| `Pillow` | Image I/O |

---

*Built for a hackathon demo. Research use only.*
