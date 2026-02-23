"""
Module 4: API & Orchestration (The Flask Backend)
==================================================
Responsible for:
  - Exposing a /protect REST endpoint (multipart/form-data image upload)
  - Routing images through the pipeline: Vision → Warp → Cloak
  - Computing the Identity Stability Score (CLIP cosine similarity)
  - Returning the protected image as a base64 JSON payload with the score
  - Keeping all heavy models (CLIP, FaceMesh) loaded in memory between requests

Author : Principal ML Engineer
"""

import base64
import io
import logging
import os
import sys
import time

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(__file__))

from module1_vision import VisionModule
from module2_warp import WarpModule
from module3_cloak import CloakModule

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("defake.api")

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

# 1. Initialize Flask pointing to your frontend folder
app = Flask(__name__, 
            static_folder='../frontend', 
            static_url_path='')
CORS(app)   # Allow React frontend on localhost:3000 to call this API

# 2. Add a route to serve the landing page (index.html)
@app.route('/')
def index():
    return app.send_static_file('index.html')

# ---------------------------------------------------------------------------
# Module singletons (loaded once at startup — NOT per request)
# ---------------------------------------------------------------------------

logger.info("=" * 60)
logger.info("DEFAKE API: Initialising modules …")
logger.info("=" * 60)

_vision_module = VisionModule(padding_factor=0.25)
logger.info("✓ VisionModule ready.")

_warp_module = WarpModule(
    max_displacement_px=2,
    grid_spacing=8,
    blur_sigma=4.0,
)
logger.info("✓ WarpModule ready.")

_cloak_module = CloakModule(
    epsilon=8 / 255,
    pgd_steps=40,
    pgd_alpha=1 / 255,
    eot_samples=4,
)
logger.info("✓ CloakModule ready.")

logger.info("=" * 60)
logger.info("DEFAKE API: All modules loaded. Server starting …")
logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

def _allowed_file(filename: str) -> bool:
    """Check that the uploaded file has an acceptable extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _ndarray_to_b64(image_rgb: np.ndarray, fmt: str = "PNG") -> str:
    """
    Encode an H×W×3 uint8 numpy array to a base64 string.
    PNG is used by default to preserve every bit of the adversarial noise.
    """
    pil_img = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Simple liveness probe — useful for monitoring / smoke tests."""
    return jsonify({"status": "ok", "service": "DEFAKE API"}), 200


@app.route("/protect", methods=["POST"])
def protect():
    """
    POST /protect
    =============
    Accept an image via multipart/form-data (field name: 'image').

    Pipeline
    --------
    1. Validate and decode the uploaded image.
    2. Module 1 — Detect face and extract landmark bounding boxes.
       → If no face: return original image unchanged.
    3. Module 2 — Apply localised imperceptible spatial distortions.
    4. Module 3 — Embed adversarial CLIP noise (PGD + EoT).
    5. Compute Identity Stability Score (cosine similarity original ↔ protected).
    6. Return JSON with base64-encoded protected image and the score.

    Response Schema
    ---------------
    {
        "success":           bool,
        "face_detected":     bool,
        "protected_image":   str,   # base64-encoded PNG
        "similarity_score":  float, # in [0, 1]; lower = stronger protection
        "processing_time_s": float
    }
    """
    t_start = time.perf_counter()

    # ---- Validation --------------------------------------------------------
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No 'image' field in request."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}",
        }), 415

    # ---- Read raw bytes ----------------------------------------------------
    image_bytes = file.read()
    if len(image_bytes) == 0:
        return jsonify({"success": False, "error": "Uploaded file is empty."}), 400

    logger.info("Received image upload: '%s' (%d bytes)", file.filename, len(image_bytes))

    try:
        # ================================================================
        # Module 1 — Vision: decode + face detection + landmark extraction
        # ================================================================
        vision_result = _vision_module.process(image_bytes)
        original_rgb  = vision_result.image_rgb   # keep a clean copy for scoring

        if not vision_result.face_detected:
            # No face found → return the image unchanged with a note
            logger.info("No face detected — returning original image unchanged.")
            b64_image = _ndarray_to_b64(original_rgb)
            return jsonify({
                "success":           True,
                "face_detected":     False,
                "protected_image":   b64_image,
                "similarity_score":  1.0,          # identical to original
                "processing_time_s": round(time.perf_counter() - t_start, 3),
                "message":           "No face detected. Image returned unchanged.",
            }), 200

        # ================================================================
        # Module 2 — Warp: localised spatial distortions around landmarks
        # ================================================================
        warped_rgb = _warp_module.apply(vision_result.image_rgb, vision_result.regions)
        logger.info("Warp module applied.")

        # ================================================================
        # Module 3 — Cloak: adversarial CLIP noise (PGD + EoT)
        # ================================================================
        protected_rgb = _cloak_module.apply(warped_rgb)
        logger.info("Cloak module applied.")

        # ================================================================
        # Identity Stability Score
        # = cosine similarity between CLIP embeddings of original and protected
        # A score close to 0 means the AI sees a completely different "identity"
        # ================================================================
        similarity_score = _cloak_module.compute_similarity(original_rgb, protected_rgb)
        logger.info("Identity Stability Score: %.4f", similarity_score)

        # ---- Encode protected image as base64 PNG ----------------------
        b64_image = _ndarray_to_b64(protected_rgb, fmt="PNG")

        elapsed = round(time.perf_counter() - t_start, 3)
        logger.info("Request completed in %.3f seconds.", elapsed)

        return jsonify({
            "success":           True,
            "face_detected":     True,
            "protected_image":   b64_image,
            "similarity_score":  round(float(similarity_score), 4),
            "processing_time_s": elapsed,
            "message":           "Image successfully protected.",
        }), 200

    except Exception as exc:
        logger.exception("Unhandled error during /protect: %s", exc)
        return jsonify({
            "success": False,
            "error":   f"Internal server error: {str(exc)}",
        }), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("Starting DEFAKE Flask server on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=False)
    # threaded=False is IMPORTANT: PyTorch CUDA contexts are not thread-safe;
    # for production, use gunicorn with --workers=1 --worker-class=gthread
