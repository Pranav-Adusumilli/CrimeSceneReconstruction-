"""
STAGE 10 — Evaluation & Validation
====================================
Quantitative evaluation of reconstruction quality.

Metrics:
  1. CLIP Similarity: text-image alignment score
  2. Object Detection Verification: expected vs detected objects
  3. Spatial Consistency Score: relationship preservation
  4. Hypothesis Plausibility Ranking

All evaluation runs on CPU/GPU with minimal memory to avoid OOM.
"""

import logging
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from src.stages.stage1_text_understanding import SceneSemantics
from src.stages.stage3_scene_graph import SceneGraph
from src.stages.stage4_hypothesis_generation import SceneHypothesis
from src.utils.memory import flush_gpu_memory

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Holds all evaluation metrics for a single generation."""
    clip_similarity: float = 0.0
    object_detection_score: float = 0.0
    spatial_consistency: float = 0.0
    hypothesis_plausibility: float = 0.0
    overall_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "clip_similarity": round(self.clip_similarity, 4),
            "object_detection_score": round(self.object_detection_score, 4),
            "spatial_consistency": round(self.spatial_consistency, 4),
            "hypothesis_plausibility": round(self.hypothesis_plausibility, 4),
            "overall_score": round(self.overall_score, 4),
            "details": self.details,
        }


class Evaluator:
    """
    Evaluates generated crime scene reconstructions.
    """

    def __init__(self, clip_model=None, clip_preprocess=None,
                 clip_tokenizer=None, device: str = "cpu"):
        """
        Args:
            clip_model: OpenCLIP model for similarity scoring.
            clip_preprocess: CLIP image preprocessing transform.
            clip_tokenizer: CLIP text tokenizer.
            device: Evaluation device (prefer CPU to save VRAM).
        """
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.device = device

    def evaluate(self, image: Image.Image, prompt: str,
                  semantics: SceneSemantics,
                  scene_graph: SceneGraph,
                  hypothesis: SceneHypothesis) -> EvaluationResult:
        """
        Run full evaluation suite on a generated image.

        Args:
            image: Generated image to evaluate.
            prompt: Text prompt used for generation.
            semantics: Scene semantics (for object verification).
            scene_graph: Scene graph (for spatial consistency).
            hypothesis: The hypothesis used for generation.

        Returns:
            EvaluationResult with all metrics.
        """
        logger.info("=" * 50)
        logger.info("STAGE 10: Evaluation & Validation")
        logger.info("=" * 50)

        result = EvaluationResult()

        # 1. CLIP Similarity
        result.clip_similarity = self._compute_clip_similarity(image, prompt)
        logger.info(f"  CLIP similarity: {result.clip_similarity:.4f}")

        # 2. Object Detection Score (proxy via CLIP per-object matching)
        result.object_detection_score = self._compute_object_detection(
            image, semantics.objects
        )
        logger.info(f"  Object detection: {result.object_detection_score:.4f}")

        # 3. Spatial Consistency
        result.spatial_consistency = self._compute_spatial_consistency(
            scene_graph, hypothesis
        )
        logger.info(f"  Spatial consistency: {result.spatial_consistency:.4f}")

        # 4. Hypothesis Plausibility
        result.hypothesis_plausibility = hypothesis.confidence
        logger.info(f"  Hypothesis plausibility: {result.hypothesis_plausibility:.4f}")

        # 5. Overall Score
        result.overall_score = (
            0.35 * result.clip_similarity +
            0.25 * result.object_detection_score +
            0.20 * result.spatial_consistency +
            0.20 * result.hypothesis_plausibility
        )
        logger.info(f"  Overall score: {result.overall_score:.4f}")

        result.details = {
            "num_objects_expected": len(semantics.objects),
            "num_relationships": len(semantics.relationships),
            "scene_type": semantics.scene_type,
            "hypothesis_id": hypothesis.hypothesis_id,
        }

        return result

    def _compute_clip_similarity(self, image: Image.Image, text: str) -> float:
        """Compute CLIP cosine similarity between image and text."""
        if self.clip_model is None:
            logger.warning("CLIP model not loaded, returning default score")
            return 0.5

        try:
            # Preprocess image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            # Tokenize text
            text_input = self.clip_tokenizer([text]).to(self.device)

            self.clip_model = self.clip_model.to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)

                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features @ text_features.T).item()

            # Move back to CPU
            self.clip_model = self.clip_model.to("cpu")
            flush_gpu_memory()

            # CLIP similarity is typically in [-1, 1], normalize to [0, 1]
            return float(np.clip((similarity + 1) / 2, 0, 1))

        except Exception as e:
            logger.error(f"CLIP evaluation error: {e}")
            return 0.5

    def _compute_object_detection(self, image: Image.Image,
                                    expected_objects: List[str]) -> float:
        """
        Proxy object detection using per-object CLIP similarity.

        For each expected object, compute CLIP similarity with
        "a photo of a {object}" and average.
        """
        if self.clip_model is None or not expected_objects:
            return 0.5

        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            self.clip_model = self.clip_model.to(self.device)

            scores = []
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                for obj in expected_objects:
                    text = self.clip_tokenizer([f"a photo containing a {obj}"]).to(self.device)
                    text_features = self.clip_model.encode_text(text)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    sim = (image_features @ text_features.T).item()
                    scores.append(float(np.clip((sim + 1) / 2, 0, 1)))

            self.clip_model = self.clip_model.to("cpu")
            flush_gpu_memory()

            return float(np.mean(scores)) if scores else 0.5

        except Exception as e:
            logger.error(f"Object detection eval error: {e}")
            return 0.5

    def _compute_spatial_consistency(self, scene_graph: SceneGraph,
                                      hypothesis: SceneHypothesis) -> float:
        """
        Evaluate how well the hypothesis satisfies spatial constraints
        defined in the scene graph.
        """
        if not scene_graph.relationships:
            return 0.7  # Default for scenes without explicit relationships

        # Build position lookup
        positions = {p.name: (p.x, p.y, p.depth) for p in hypothesis.placements}

        satisfied = 0
        total = len(scene_graph.relationships)

        for subj, pred, obj in scene_graph.relationships:
            if subj not in positions or obj not in positions:
                continue

            sx, sy, sd = positions[subj]
            ox, oy, od = positions[obj]

            if pred in ("on", "on top of", "above", "over"):
                if sy < oy:  # subject above object
                    satisfied += 1
            elif pred in ("under", "underneath", "beneath", "below"):
                if sy > oy:
                    satisfied += 1
            elif pred in ("near", "next to", "beside", "close to"):
                dist = np.sqrt((sx - ox) ** 2 + (sy - oy) ** 2)
                if dist < 0.35:
                    satisfied += 1
            elif pred in ("behind",):
                if sd < od:
                    satisfied += 1
            elif pred in ("in front of",):
                if sd > od:
                    satisfied += 1
            elif pred in ("left", "on the left of"):
                if sx < ox:
                    satisfied += 1
            elif pred in ("right", "on the right of"):
                if sx > ox:
                    satisfied += 1
            else:
                # Generic: accept if reasonably close
                dist = np.sqrt((sx - ox) ** 2 + (sy - oy) ** 2)
                if dist < 0.5:
                    satisfied += 0.5

        return float(satisfied / max(total, 1))
