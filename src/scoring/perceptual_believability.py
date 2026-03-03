"""
Human Perceptual Believability Score
======================================

Proxy model for human realism judgment.

Mathematical formulation:

    S_human(R) = λ · realism_probe(R)
              + μ · scene_coherence(R)
              + ν · uncanny_valley_penalty(R)

Components:
    1. Realism probe       — CLIP comparison: "real photo" vs "AI generated"
    2. Scene coherence     — CLIP similarity to scene-type description
    3. Uncanny penalty     — detects anomalous visual patterns

Range: [0,1]. Higher = more perceptually believable.
"""

import logging
import numpy as np
import torch
from PIL import Image
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PerceptualBelievabilityScorer:
    """
    Estimates human perceptual believability using CLIP-based probes.

    Acts as a proxy for human judgment by comparing the image against
    carefully designed positive (real) and negative (fake/AI) prompts.
    """

    def __init__(self, clip_model=None, clip_preprocess=None,
                 clip_tokenizer=None, device: str = "cpu",
                 w_realism: float = 0.45, w_coherence: float = 0.35,
                 w_uncanny: float = 0.20):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.device = device
        self.w_realism = w_realism
        self.w_coherence = w_coherence
        self.w_uncanny = w_uncanny

    def compute(self, image: Image.Image,
                scene_type: str = "room") -> Dict[str, float]:
        """
        Compute perceptual believability score.

        Returns:
            Dict with 'score' ∈ [0,1] and component breakdowns.
        """
        realism = self._realism_probe(image)
        coherence = self._scene_coherence(image, scene_type)
        uncanny = self._uncanny_penalty(image)

        score = (self.w_realism * realism +
                 self.w_coherence * coherence +
                 self.w_uncanny * (1.0 - uncanny))  # Penalty: lower is better

        return {
            "score": float(np.clip(score, 0, 1)),
            "realism_probe": realism,
            "scene_coherence": coherence,
            "uncanny_penalty": uncanny,
        }

    def _realism_probe(self, image: Image.Image) -> float:
        """
        Binary CLIP probe: "a real photograph" vs "an AI generated image".

        Computes P(real) = softmax(sim_real, sim_fake)[0].
        """
        if self.clip_model is None:
            return 0.5

        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            self.clip_model = self.clip_model.to(self.device)

            probes = [
                "a real photograph taken with a digital camera, photorealistic, natural",
                "a computer generated image, AI art, synthetic, diffusion model output",
            ]

            with torch.no_grad():
                img_feat = self.clip_model.encode_image(image_input)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                scores = []
                for p in probes:
                    t = self.clip_tokenizer([p]).to(self.device)
                    tf = self.clip_model.encode_text(t)
                    tf = tf / tf.norm(dim=-1, keepdim=True)
                    scores.append((img_feat @ tf.T).item())

            self.clip_model = self.clip_model.to("cpu")

            # Softmax to get P(real)
            scores = np.array(scores)
            exp_scores = np.exp(scores * 6)  # Temperature scaling (lower = less extreme)
            p_real = exp_scores[0] / exp_scores.sum()

            return float(np.clip(p_real, 0, 1))

        except Exception as e:
            logger.error(f"Realism probe error: {e}")
            return 0.5

    def _scene_coherence(self, image: Image.Image,
                          scene_type: str) -> float:
        """
        How well the image matches its expected scene type.

        Computes CLIP similarity with "a photograph of a {scene_type}".
        """
        if self.clip_model is None:
            return 0.5

        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            self.clip_model = self.clip_model.to(self.device)

            prompt = f"a realistic photograph of a {scene_type}, crime scene, forensic evidence, indoor photography"

            with torch.no_grad():
                img_feat = self.clip_model.encode_image(image_input)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                t = self.clip_tokenizer([prompt]).to(self.device)
                tf = self.clip_model.encode_text(t)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                sim = (img_feat @ tf.T).item()

            self.clip_model = self.clip_model.to("cpu")
            return float(np.clip((sim + 1) / 2, 0, 1))

        except Exception as e:
            logger.error(f"Scene coherence error: {e}")
            return 0.5

    def _uncanny_penalty(self, image: Image.Image) -> float:
        """
        Detect uncanny/anomalous visual patterns.

        Uses CLIP probe for common diffusion failure modes.
        Returns penalty ∈ [0,1] where 0 = no issues, 1 = severe.
        """
        if self.clip_model is None:
            return 0.0

        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            self.clip_model = self.clip_model.to(self.device)

            anomaly_probes = [
                "distorted objects with extra parts",
                "melting or warped shapes",
                "nonsensical text and symbols",
                "repeated patterns and glitches",
            ]

            with torch.no_grad():
                img_feat = self.clip_model.encode_image(image_input)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                scores = []
                for p in anomaly_probes:
                    t = self.clip_tokenizer([p]).to(self.device)
                    tf = self.clip_model.encode_text(t)
                    tf = tf / tf.norm(dim=-1, keepdim=True)
                    sim = (img_feat @ tf.T).item()
                    scores.append((sim + 1) / 2)

            self.clip_model = self.clip_model.to("cpu")

            # Max anomaly score across probes
            max_anomaly = max(scores) if scores else 0.0

            # Threshold: similarity > 0.6 with anomaly probes is concerning
            if max_anomaly > 0.65:
                return float(np.clip((max_anomaly - 0.55) * 3.0, 0, 1))
            return 0.0

        except Exception as e:
            logger.error(f"Uncanny penalty error: {e}")
            return 0.0
