"""
Module 1: Input Handling & Landmark Detection (The Vision Module)
=================================================================
Responsible for:
  - Accepting raw image bytes and normalizing them to RGB numpy arrays / PyTorch tensors
  - Running OpenCV's face-mesh detector to locate facial landmarks
  - Extracting bounding coordinates for eyes, nose, and mouth regions
  - Short-circuiting the pipeline if no face is detected (returns original image)

Author : Principal ML Engineer
"""

import io
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FacialRegions:
    """
    Bounding boxes for the three key facial regions used by the warp module.
    Each box is stored as (x_min, y_min, x_max, y_max) in pixel coordinates.
    """
    left_eye:  tuple[int, int, int, int]
    right_eye: tuple[int, int, int, int]
    nose:      tuple[int, int, int, int]
    mouth:     tuple[int, int, int, int]
    image_h:   int   # original image height
    image_w:   int   # original image width


@dataclass
class VisionResult:
    """
    Output contract of Module 1.
    `face_detected` is the primary flag used by the orchestrator
    to decide whether to continue the pipeline.
    """
    image_rgb: np.ndarray          # H×W×3 uint8 numpy array
    tensor:    torch.Tensor        # 1×3×H×W float32 tensor in [0,1]
    face_detected: bool
    regions:   Optional[FacialRegions] = None


# ---------------------------------------------------------------------------
# MediaPipe landmark indices for each facial region
# ---------------------------------------------------------------------------

# MediaPipe Face Mesh provides 468 3D landmarks.
# These index sets correspond to the approximate contour of each region.

_LEFT_EYE_LANDMARKS  = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                         173, 157, 158, 159, 160, 161, 246]

_RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                         466, 388, 387, 386, 385, 384, 398]

_NOSE_LANDMARKS      = [1, 2, 98, 327, 168, 6, 197, 195, 5,
                         4, 45, 275, 220, 440, 125, 354]

_MOUTH_LANDMARKS     = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                         291, 375, 321, 405, 314, 17, 84, 181, 91, 146]


# ---------------------------------------------------------------------------
# VisionModule class
# ---------------------------------------------------------------------------

class VisionModule:
    """
    Handles all image ingestion and facial landmark extraction.
    The MediaPipe FaceMesh model is loaded once at construction time
    and reused across requests for efficiency.
    """

    def __init__(self, padding_factor: float = 0.25):
        """
        Parameters
        ----------
        padding_factor : float
            Extra margin added around each landmark bounding box so that
            the warp module has enough context around each region.
            0.25 means ±25 % of the region's own width/height.
        """
        self.padding_factor = padding_factor

        # Initialise MediaPipe FaceMesh (static_image_mode for single frames)
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,   # enables iris landmarks too
            min_detection_confidence=0.5,
        )
        logger.info("VisionModule: MediaPipe FaceMesh loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image_bytes: bytes) -> VisionResult:
        """
        Main entry point called by the Flask orchestrator.

        Parameters
        ----------
        image_bytes : bytes
            Raw file bytes coming from the HTTP upload (JPEG, PNG, JPG).

        Returns
        -------
        VisionResult
            Contains the normalised image, a flag for face detection,
            and (optionally) the bounding boxes of facial regions.
        """
        # Step 1 — decode & normalise to RGB numpy array
        image_rgb = self._decode_to_rgb(image_bytes)

        # Step 2 — convert to float tensor for downstream modules
        tensor = self._to_tensor(image_rgb)

        # Step 3 — run MediaPipe FaceMesh
        regions = self._detect_landmarks(image_rgb)

        if regions is None:
            logger.warning("VisionModule: No face detected — skipping protection pipeline.")
            return VisionResult(
                image_rgb=image_rgb,
                tensor=tensor,
                face_detected=False,
                regions=None,
            )

        logger.info("VisionModule: Face detected. Regions extracted.")
        return VisionResult(
            image_rgb=image_rgb,
            tensor=tensor,
            face_detected=True,
            regions=regions,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decode_to_rgb(self, image_bytes: bytes) -> np.ndarray:
        """
        Convert raw bytes (any common image format) → uint8 H×W×3 RGB array.
        Using PIL for broad format support, then converting to numpy.
        """
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)

    def _to_tensor(self, image_rgb: np.ndarray) -> torch.Tensor:
        """
        Convert H×W×3 uint8 numpy array → 1×3×H×W float32 tensor in [0, 1].
        """
        t = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0)   # add batch dimension

    def _detect_landmarks(self, image_rgb: np.ndarray) -> Optional[FacialRegions]:
        """
        Run MediaPipe FaceMesh on the RGB image and extract bounding boxes
        for the left eye, right eye, nose, and mouth.

        Returns None when no face is detected.
        """
        h, w = image_rgb.shape[:2]

        # MediaPipe expects uint8 RGB; process() returns normalised coords in [0, 1]
        results = self._face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        # Use the first (and only) detected face
        landmarks = results.multi_face_landmarks[0].landmark

        def _bbox(indices: list[int]) -> tuple[int, int, int, int]:
            """
            Compute pixel bounding box for a set of landmark indices,
            with padding applied.
            """
            xs = [landmarks[i].x * w for i in indices]
            ys = [landmarks[i].y * h for i in indices]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Apply padding
            pad_x = (x_max - x_min) * self.padding_factor
            pad_y = (y_max - y_min) * self.padding_factor

            x_min = max(0, int(x_min - pad_x))
            y_min = max(0, int(y_min - pad_y))
            x_max = min(w, int(x_max + pad_x))
            y_max = min(h, int(y_max + pad_y))

            return (x_min, y_min, x_max, y_max)

        return FacialRegions(
            left_eye=_bbox(_LEFT_EYE_LANDMARKS),
            right_eye=_bbox(_RIGHT_EYE_LANDMARKS),
            nose=_bbox(_NOSE_LANDMARKS),
            mouth=_bbox(_MOUTH_LANDMARKS),
            image_h=h,
            image_w=w,
        )
