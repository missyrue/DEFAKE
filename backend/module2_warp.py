"""
Module 2: Local Distortion Logic (The Warp Module)
===================================================
Responsible for:
  - Receiving the facial landmark bounding boxes from Module 1
  - Applying subtle, localised spatial distortions around eyes, nose, and mouth
  - Ensuring all distortions remain VISUALLY IMPERCEPTIBLE to a human observer

Design Rationale
----------------
We use a thin-plate-spline (TPS) inspired grid warp confined to small
neighbourhood patches around each landmark region.  The key insight is:

  * Deep-fake feature-extractors encode fine spatial relationships between
    pixels (e.g., the exact distance between the inner canthi of the eyes).
  * A human perceives only the *coarse* shape of a face; microscopic pixel
    displacements (< 3 px) are invisible to the eye but confound AI geometry.
  * By restricting the warp to the bounding box of each landmark region (with
    a soft alpha mask to avoid hard edges) we avoid any noticeable blending
    artefact at the region boundary.

Author : Principal ML Engineer
"""

import logging
import math
from typing import List, Tuple, Optional

import cv2
import numpy as np

from module1_vision import FacialRegions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WarpModule class
# ---------------------------------------------------------------------------

class WarpModule:
    """
    Applies localised, imperceptible spatial distortions to a face image
    based on the landmark bounding boxes produced by VisionModule.

    All parameters default to values that keep distortions well below the
    human just-noticeable-difference (JND) threshold for spatial warping
    (empirically ~2–3 px shift at 512 px face size).
    """

    def __init__(
        self,
        max_displacement_px: int = 2,
        grid_spacing:        int = 8,
        blur_sigma:          float = 4.0,
        seed:                Optional[int] = None,
    ):
        """
        Parameters
        ----------
        max_displacement_px : int
            Maximum pixel displacement applied at any single grid node.
            Keep ≤ 3 to stay imperceptible. Default: 2.
        grid_spacing : int
            Number of pixels between displacement grid nodes within each
            region patch.  Smaller = smoother, finer warp. Default: 8.
        blur_sigma : float
            Gaussian sigma used to feather the alpha mask at region edges,
            preventing visible seams. Default: 4.0.
        seed : int or None
            Optional RNG seed for reproducible distortions.
        """
        self.max_displacement_px = max_displacement_px
        self.grid_spacing        = grid_spacing
        self.blur_sigma          = blur_sigma
        self._rng = np.random.default_rng(seed)

        logger.info(
            "WarpModule initialised — max_disp=%d px, grid_spacing=%d, blur_sigma=%.1f",
            max_displacement_px, grid_spacing, blur_sigma,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, image_rgb: np.ndarray, regions: FacialRegions) -> np.ndarray:
        """
        Apply localised warp distortions to the four facial regions.

        Parameters
        ----------
        image_rgb : np.ndarray
            H×W×3 uint8 source image (modified in place then returned).
        regions : FacialRegions
            Output of VisionModule._detect_landmarks().

        Returns
        -------
        np.ndarray
            H×W×3 uint8 image with imperceptible spatial distortions.
        """
        result = image_rgb.copy()

        for bbox in [
            regions.left_eye,
            regions.right_eye,
            regions.nose,
            regions.mouth,
        ]:
            result = self._warp_region(result, bbox)

        logger.debug("WarpModule: All four facial regions distorted.")
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _warp_region(
        self,
        image: np.ndarray,
        bbox:  Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Apply a smooth random displacement field to a single bounding-box patch.

        Strategy
        --------
        1. Crop the region with a small border for context.
        2. Build a sparse random displacement grid (node spacing = grid_spacing).
        3. Upsample the displacement grid to full patch resolution using bilinear
           interpolation → this gives a smooth, organic warp rather than a blocky one.
        4. Apply the displacement via cv2.remap() (bicubic, border reflect).
        5. Composite the warped patch back using a Gaussian-feathered alpha mask
           so no hard boundary is visible.
        """
        x1, y1, x2, y2 = bbox
        h_img, w_img = image.shape[:2]

        # Safety: skip degenerate boxes
        if x2 - x1 < 4 or y2 - y1 < 4:
            return image

        # ---- 1. Extract patch ------------------------------------------------
        patch_orig = image[y1:y2, x1:x2].copy()
        ph, pw = patch_orig.shape[:2]   # patch height, width

        # ---- 2. Build sparse displacement grid --------------------------------
        # Number of grid nodes in each dimension (at least 2 to avoid degenerate)
        gw = max(2, math.ceil(pw / self.grid_spacing) + 1)
        gh = max(2, math.ceil(ph / self.grid_spacing) + 1)

        # Random displacements at each node, clamped to max_displacement_px
        dx_sparse = self._rng.uniform(
            -self.max_displacement_px, self.max_displacement_px,
            size=(gh, gw),
        ).astype(np.float32)
        dy_sparse = self._rng.uniform(
            -self.max_displacement_px, self.max_displacement_px,
            size=(gh, gw),
        ).astype(np.float32)

        # Force displacements to zero on the border of the grid so the warp
        # fades away at the region edge (reduces seam visibility further)
        dx_sparse[[0, -1], :] = 0
        dx_sparse[:, [0, -1]] = 0
        dy_sparse[[0, -1], :] = 0
        dy_sparse[:, [0, -1]] = 0

        # ---- 3. Upsample sparse grid to full patch resolution -----------------
        dx_full = cv2.resize(dx_sparse, (pw, ph), interpolation=cv2.INTER_CUBIC)
        dy_full = cv2.resize(dy_sparse, (pw, ph), interpolation=cv2.INTER_CUBIC)

        # Build absolute remap arrays: each output pixel maps to a source pixel
        map_x = (np.tile(np.arange(pw, dtype=np.float32), (ph, 1)) + dx_full).astype(np.float32)
        map_y = (np.tile(np.arange(ph, dtype=np.float32).reshape(-1, 1), (1, pw)) + dy_full).astype(np.float32)

        # ---- 4. Remap patch ---------------------------------------------------
        patch_warped = cv2.remap(
            patch_orig,
            map_x, map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # ---- 5. Feathered alpha composite ------------------------------------
        # Create a white rectangle and blur its edges to produce a soft mask
        alpha = np.ones((ph, pw), dtype=np.float32)
        alpha = cv2.GaussianBlur(alpha, (0, 0), self.blur_sigma)
        alpha = (alpha / alpha.max()).clip(0, 1)           # normalise to [0, 1]
        alpha_3ch = alpha[:, :, np.newaxis]                # broadcast over channels

        blended = (
            patch_warped.astype(np.float32) * alpha_3ch
            + patch_orig.astype(np.float32) * (1.0 - alpha_3ch)
        ).clip(0, 255).astype(np.uint8)

        # Write blended patch back into result image
        result = image.copy()
        result[y1:y2, x1:x2] = blended
        return result


# Allow Optional import without circular dependency

