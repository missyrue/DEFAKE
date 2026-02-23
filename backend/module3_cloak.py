"""
Module 3: Adversarial CLIP Noise (The AI Cloak Module)
=======================================================
Responsible for:
  - Loading openai/clip-vit-base-patch32 (once, at construction time)
  - Running a PGD (Projected Gradient Descent) adversarial attack to embed
    invisible noise into the image
  - Objective: maximise cosine distance between CLIP embeddings of the original
    and the protected image (so AI tools see a completely different "identity")
  - Expectation over Transformation (EoT): each PGD step applies random
    differentiable augmentations (resize + JPEG-simulation) before the CLIP
    forward pass, making the noise robust to screenshot / resize attacks
  - L-infinity clamping: noise is strictly bounded to ε = 8/255 so the image
    remains visually identical

Theory
------
CLIP encodes semantic identity via a contrastive vision–language embedding.
Deepfake pipelines (like FaceSwap, SimSwap, etc.) typically use CLIP or
similar ViT encoders to align face identity.  By maximising:

    L = cosine_similarity(CLIP(x_orig), CLIP(x_adv))   →   minimise L

we push the adversarial image far from the original in embedding space,
causing the deepfake model to either fail face alignment or generate a
completely wrong identity.

PGD with sign gradient (FGSM-style step) is used for speed.
EoT (Expectation over Transformation) wraps each forward pass with random
resize + differentiable JPEG compression (using Kornia) so the noise survives
real-world transformations.

Author : Principal ML Engineer
"""

import io
import logging
import random
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import Kornia for differentiable JPEG compression simulation.
# Kornia is optional; if not installed, we fall back to a simple Gaussian
# noise approximation of JPEG artefacts.
# ---------------------------------------------------------------------------
try:
    import kornia
    import kornia.enhance as KE
    _KORNIA_AVAILABLE = True
    logger.info("Kornia available — using differentiable JPEG simulation.")
except ImportError:
    _KORNIA_AVAILABLE = False
    logger.warning(
        "Kornia not found. Falling back to Gaussian JPEG approximation. "
        "Install with: pip install kornia"
    )


# ---------------------------------------------------------------------------
# CloakModule class
# ---------------------------------------------------------------------------

class CloakModule:
    """
    Computes and applies an adversarial perturbation to an image using
    PGD + EoT against the CLIP vision encoder.

    The module is instantiated once per process and the CLIP model is kept
    in memory between requests — avoiding the costly reload on each API call.
    """

    # CLIP model identifier on Hugging Face Hub
    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        epsilon:        float = 8 / 255,     # L-inf bound on perturbation
        pgd_steps:      int   = 40,           # number of PGD iterations
        pgd_alpha:      float = 1 / 255,      # step size per PGD iteration
        eot_samples:    int   = 4,            # random transforms per PGD step
        min_resize:     float = 0.7,          # minimum resize factor for EoT
        max_resize:     float = 1.0,          # maximum resize factor for EoT
        jpeg_quality_range: tuple = (50, 90), # JPEG quality range for EoT
        device:         Optional[str] = None,
    ):
        """
        Parameters
        ----------
        epsilon : float
            Maximum L-inf norm of adversarial noise (default 8/255 ≈ invisible).
        pgd_steps : int
            Total number of gradient ascent iterations.  More steps = stronger
            cloak but slower.
        pgd_alpha : float
            Step size for each PGD update (should be ≈ epsilon / pgd_steps * 2).
        eot_samples : int
            How many random transformations to average per PGD step.  Higher =
            more robust against resizing but slower.
        min_resize, max_resize : float
            Range for random isotropic rescaling during EoT.
        jpeg_quality_range : tuple[int, int]
            (min, max) JPEG quality values simulated during EoT.
        device : str or None
            'cuda', 'cpu', or None (auto-detect).
        """
        self.epsilon             = epsilon
        self.pgd_steps           = pgd_steps
        self.pgd_alpha           = pgd_alpha
        self.eot_samples         = eot_samples
        self.min_resize          = min_resize
        self.max_resize          = max_resize
        self.jpeg_quality_range  = jpeg_quality_range

        # Auto-detect compute device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("CloakModule: using device=%s", self.device)

        # Load CLIP model and processor (heavy; done once at startup)
        self._load_clip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, image_rgb_uint8: "np.ndarray") -> "np.ndarray":
        """
        Generate and embed adversarial noise into the image.

        Parameters
        ----------
        image_rgb_uint8 : np.ndarray
            H×W×3 uint8 RGB image (output of WarpModule.apply()).

        Returns
        -------
        np.ndarray
            H×W×3 uint8 protected image (visually identical, adversarially perturbed).
        """
        import numpy as np

        # Convert to float tensor in [0, 1] — shape 1×3×H×W
        x_orig = (
            torch.from_numpy(image_rgb_uint8)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )

        # Compute original CLIP embedding (no grad needed; this is our target)
        with torch.no_grad():
            emb_orig = self._clip_embed(x_orig)  # shape: 1 × D

        # Run PGD attack
        x_adv = self._pgd_attack(x_orig, emb_orig)

        # Convert back to uint8 numpy
        result = (
            x_adv.squeeze(0)
            .permute(1, 2, 0)
            .mul(255.0)
            .clamp(0, 255)
            .byte()
            .cpu()
            .numpy()
        )
        return result

    def compute_similarity(
        self,
        image_a: "np.ndarray",
        image_b: "np.ndarray",
    ) -> float:
        """
        Compute cosine similarity between CLIP embeddings of two images.
        Used by the orchestrator to calculate the Identity Stability Score.

        Returns a float in [-1, 1]; lower = more cloaked.
        """
        import numpy as np

        def _to_tensor(img):
            return (
                torch.from_numpy(img)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(self.device)
            )

        with torch.no_grad():
            emb_a = self._clip_embed(_to_tensor(image_a))
            emb_b = self._clip_embed(_to_tensor(image_b))
            sim = F.cosine_similarity(emb_a, emb_b, dim=-1).item()

        return sim

    # ------------------------------------------------------------------
    # Private helpers — CLIP
    # ------------------------------------------------------------------

    def _load_clip(self):
        """Load CLIP model and processor from Hugging Face Hub."""
        logger.info("CloakModule: Loading CLIP model '%s' ...", self.CLIP_MODEL_ID)
        self._processor = CLIPProcessor.from_pretrained(self.CLIP_MODEL_ID)
        self._model = CLIPModel.from_pretrained(self.CLIP_MODEL_ID).to(self.device)
        self._model.eval()

        # Extract and freeze vision backbone — we only need the image encoder
        self._vision_model = self._model.vision_model
        self._visual_proj  = self._model.visual_projection

        # Freeze all parameters; we never update the model weights
        for p in self._model.parameters():
            p.requires_grad_(False)

        logger.info("CloakModule: CLIP loaded and frozen on device=%s.", self.device)

    def _clip_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute normalised CLIP image embedding.

        Parameters
        ----------
        x : torch.Tensor  — shape 1×3×H×W, values in [0, 1]

        Returns
        -------
        torch.Tensor — shape 1×D, L2-normalised
        """
        # CLIP expects inputs preprocessed to a specific size & normalisation.
        # We replicate the processor's normalisation here in a differentiable way
        # so gradients can flow back to the input image.

        # Resize to CLIP input size (224×224 for ViT-B/32)
        clip_size = 224
        x_resized = F.interpolate(x, size=(clip_size, clip_size), mode="bilinear", align_corners=False)

        # Normalise with CLIP's mean and std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        x_norm = (x_resized - mean) / std

        # Forward through CLIP vision backbone
        vision_out = self._vision_model(pixel_values=x_norm)
        pooled     = vision_out.pooler_output          # 1 × hidden_dim
        projected  = self._visual_proj(pooled)          # 1 × embed_dim

        # L2-normalise (standard CLIP behaviour)
        return F.normalize(projected, p=2, dim=-1)

    # ------------------------------------------------------------------
    # Private helpers — PGD attack with EoT
    # ------------------------------------------------------------------

    def _pgd_attack(
        self,
        x_orig: torch.Tensor,
        emb_orig: torch.Tensor,
    ) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) adversarial attack.

        Objective: MAXIMISE cosine distance between emb_orig and the embedding
        of the perturbed image, i.e. MINIMISE cosine similarity.

        With EoT: each PGD step averages gradients over `eot_samples` random
        augmentations of the current adversarial image to make noise robust
        to screenshot / resize.

        Parameters
        ----------
        x_orig   : 1×3×H×W tensor in [0, 1] — original clean image
        emb_orig : 1×D tensor — frozen CLIP embedding of x_orig

        Returns
        -------
        x_adv : 1×3×H×W tensor in [0, 1] — cloaked image
        """
        # Initialise perturbation delta in the feasible set (random start for PGD)
        delta = torch.empty_like(x_orig).uniform_(-self.epsilon, self.epsilon)
        delta = delta.clamp(
            -self.epsilon, self.epsilon
        ).clamp(
            -x_orig, 1.0 - x_orig  # keep pixel values in [0, 1]
        )
        delta.requires_grad_(True)

        for step in range(self.pgd_steps):
            # Zero out gradients from previous step
            if delta.grad is not None:
                delta.grad.zero_()

            # Accumulate gradient over EoT samples
            grad_accum = torch.zeros_like(delta)

            for _ in range(self.eot_samples):
                x_adv = x_orig + delta

                # Apply random differentiable augmentation (EoT)
                x_aug = self._eot_transform(x_adv)

                # Forward through CLIP
                emb_adv = self._clip_embed(x_aug)

                # Loss: we MINIMISE cosine similarity (= gradient ascent on distance)
                loss = F.cosine_similarity(emb_orig, emb_adv, dim=-1).mean()

                # Compute gradient w.r.t. delta
                loss.backward()

                grad_accum += delta.grad.data.clone()
                delta.grad.zero_()

            # Average gradient over EoT samples, then take sign step (FGSM-style)
            grad_sign = grad_accum.sign() / self.eot_samples

            # Gradient DESCENT on similarity = gradient ASCENT on distance
            # We subtract because we want to *minimise* similarity
            delta_data = delta.data - self.pgd_alpha * grad_sign

            # Project back onto L-inf ball and valid pixel range
            delta_data = delta_data.clamp(-self.epsilon, self.epsilon)
            delta_data = delta_data.clamp(-x_orig, 1.0 - x_orig)

            # Re-attach with grad
            delta = delta_data.detach().requires_grad_(True)

            if (step + 1) % 10 == 0:
                with torch.no_grad():
                    sim = F.cosine_similarity(
                        emb_orig, self._clip_embed(x_orig + delta), dim=-1
                    ).item()
                logger.debug("PGD step %d/%d — CLIP similarity: %.4f", step + 1, self.pgd_steps, sim)

        return (x_orig + delta.detach()).clamp(0.0, 1.0)

    def _eot_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a differentiable random augmentation to tensor x to simulate
        real-world image transformations (screenshot / resize / compression).

        Uses:
          1. Random isotropic resize (interpolation stays differentiable via F.interpolate)
          2. Differentiable JPEG approximation via Kornia (if available),
             or Gaussian blur as fallback compression simulation.

        Parameters
        ----------
        x : 1×3×H×W float tensor in [0, 1]

        Returns
        -------
        1×3×H×W float tensor, augmented but still in [0, 1]
        """
        _, _, H, W = x.shape

        # 1. Random resize
        scale     = random.uniform(self.min_resize, self.max_resize)
        new_h     = max(32, int(H * scale))
        new_w     = max(32, int(W * scale))
        x_resized = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # 2. JPEG-like compression simulation
        if _KORNIA_AVAILABLE:
            # Kornia's JPEG codec simulation operates in the frequency domain
            # and is fully differentiable.
            quality = random.randint(*self.jpeg_quality_range)
            x_compressed = kornia.enhance.jpeg_codec_differentiable(x_resized, jpeg_quality=torch.tensor([quality], dtype=torch.float32, device=x.device))
        else:
            # Fallback: slight Gaussian blur approximates low-frequency JPEG loss
            sigma = random.uniform(0.3, 0.8)
            kernel_size = 3  # must be odd
            x_compressed = TF.gaussian_blur(x_resized, kernel_size=[kernel_size, kernel_size], sigma=[sigma])

        # 3. Resize back to original dimensions so CLIP sees the right shape
        x_out = F.interpolate(x_compressed, size=(H, W), mode="bilinear", align_corners=False)
        return x_out
