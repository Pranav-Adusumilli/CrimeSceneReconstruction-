"""
Visual Realism Score
=====================

Measures generated image quality using no-reference metrics.

Mathematical formulation:

    S_visual(R) = δ · aesthetic_quality(R)
                + ε · noise_residual_score(R)
                + ζ · sharpness_score(R)

Components:
    1. Aesthetic quality    — CLIP aesthetic prediction via prompt comparison
    2. Noise residual      — low-frequency noise analysis (diffusion artifacts)
    3. Sharpness           — Laplacian variance (focus quality metric)

Range: [0,1]. Higher = more realistic.
"""

import logging
import numpy as np
import torch
from PIL import Image, ImageFilter
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class VisualRealismScorer:
    """
    Evaluates visual quality of generated images using
    no-reference image quality metrics and CLIP aesthetic scoring.

    Designed to detect diffusion artifacts:
    - Blurry regions (low Laplacian variance)
    - Noise patterns (high-frequency residuals)
    - Aesthetic quality (via CLIP prompt comparison)
    """

    def __init__(self, clip_model=None, clip_preprocess=None,
                 clip_tokenizer=None, device: str = "cpu",
                 w_aesthetic: float = 0.4, w_noise: float = 0.3,
                 w_sharpness: float = 0.3):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.device = device
        self.w_aesthetic = w_aesthetic
        self.w_noise = w_noise
        self.w_sharpness = w_sharpness

    def compute(self, image: Image.Image) -> Dict[str, float]:
        """
        Compute visual realism score.

        Returns:
            Dict with 'score' ∈ [0,1] and component breakdowns.
        """
        aesthetic = self._aesthetic_quality(image)
        noise = self._noise_residual_score(image)
        sharpness = self._sharpness_score(image)

        score = (self.w_aesthetic * aesthetic +
                 self.w_noise * noise +
                 self.w_sharpness * sharpness)

        return {
            "score": float(np.clip(score, 0, 1)),
            "aesthetic_quality": aesthetic,
            "noise_residual": noise,
            "sharpness": sharpness,
        }

    def _aesthetic_quality(self, image: Image.Image) -> float:
        """
        CLIP-based aesthetic scoring by comparing image to quality prompts.

        Computes similarity with positive quality descriptors vs negative ones
        and uses the gap as an aesthetic predictor.
        """
        if self.clip_model is None:
            return 0.5

        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            self.clip_model = self.clip_model.to(self.device)

            positive_prompts = [
                "a high quality, detailed, professional photograph",
                "a sharp, well-lit, realistic photo",
                "a professional forensic crime scene photograph",
                "a clear, detailed indoor photograph with natural lighting",
            ]
            negative_prompts = [
                "a blurry, distorted, low quality image",
                "an abstract, unrealistic, noisy image",
                "a cartoon, painting, or illustration",
            ]

            with torch.no_grad():
                img_feat = self.clip_model.encode_image(image_input)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                pos_scores = []
                for p in positive_prompts:
                    t = self.clip_tokenizer([p]).to(self.device)
                    tf = self.clip_model.encode_text(t)
                    tf = tf / tf.norm(dim=-1, keepdim=True)
                    pos_scores.append((img_feat @ tf.T).item())

                neg_scores = []
                for p in negative_prompts:
                    t = self.clip_tokenizer([p]).to(self.device)
                    tf = self.clip_model.encode_text(t)
                    tf = tf / tf.norm(dim=-1, keepdim=True)
                    neg_scores.append((img_feat @ tf.T).item())

            self.clip_model = self.clip_model.to("cpu")

            avg_pos = np.mean(pos_scores)
            avg_neg = np.mean(neg_scores)

            # Normalized score: how much more positive than negative
            # Typical range: -0.05 to 0.15 for SD outputs
            raw = avg_pos - avg_neg
            return float(np.clip(raw * 4.0 + 0.55, 0, 1))

        except Exception as e:
            logger.error(f"Aesthetic quality error: {e}")
            return 0.5

    def _noise_residual_score(self, image: Image.Image) -> float:
        """
        Detect diffusion noise artifacts via frequency analysis.

        Computes high-frequency energy ratio. Well-generated images
        have controlled high-freq content; noise-heavy images have more.

        Returns score ∈ [0,1] where 1 = clean, 0 = noisy.
        """
        try:
            # Convert to grayscale numpy
            gray = np.array(image.convert("L"), dtype=np.float32)

            # High-pass filter (Laplacian)
            from PIL import ImageFilter
            hp = np.array(
                image.convert("L").filter(ImageFilter.FIND_EDGES),
                dtype=np.float32
            )

            # Energy ratios
            total_energy = np.mean(gray ** 2) + 1e-8
            high_freq_energy = np.mean(hp ** 2)

            # Ratio: controlled images have ratio ~0.01-0.05
            # Realistic/textured images have ratio 0.05-0.15 (not noise)
            ratio = high_freq_energy / total_energy

            # Smooth sigmoid mapping instead of harsh buckets
            # Centered at 0.12: below → clean, above → noisy
            score = 1.0 / (1.0 + np.exp(15.0 * (ratio - 0.12)))
            return float(np.clip(score * 0.85 + 0.10, 0.1, 0.95))

        except Exception as e:
            logger.error(f"Noise analysis error: {e}")
            return 0.5

    def _sharpness_score(self, image: Image.Image) -> float:
        """
        Laplacian variance as a focus/sharpness metric.

        Higher variance = sharper image. Score normalized to [0,1].
        """
        try:
            gray = np.array(image.convert("L"), dtype=np.float64)

            # Laplacian kernel convolution (simple 3x3)
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                  dtype=np.float64)
            from scipy.signal import convolve2d
            lap = convolve2d(gray, laplacian, mode='same', boundary='symm')
            variance = lap.var()

            # Typical SD 512px output: variance 300-1200
            # Normalize: 400→0.5, 800→0.7, 1200→0.85
            return float(np.clip(variance / 1400.0, 0, 1))

        except ImportError:
            # Fallback without scipy
            gray = np.array(image.convert("L"), dtype=np.float64)
            # Simple gradient magnitude
            gx = np.diff(gray, axis=1)
            gy = np.diff(gray, axis=0)
            grad_mag = np.mean(gx[:, :-1]**2 + gy[:-1, :]**2)
            return float(np.clip(grad_mag / 3000.0, 0, 1))
        except Exception as e:
            logger.error(f"Sharpness analysis error: {e}")
            return 0.5
