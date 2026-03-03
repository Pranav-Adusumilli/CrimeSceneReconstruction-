"""
Multi-View Consistency Score
==============================

Evaluates consistency across multiple generated viewpoints.

Mathematical formulation:

    S_multiview(R) = (1/3) · [persistence(R) + geometry(R) + depth_stability(R)]

Components:
    1. Object persistence      — same objects detected across views
    2. Geometry alignment      — CLIP similarity between view pairs
    3. Depth ordering stability — depth order is consistent across views

Range: [0,1]. Higher = more consistent across views.
"""

import logging
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class MultiViewConsistencyScorer:
    """
    Evaluates whether multiple viewpoints of the same scene are
    geometrically and semantically consistent.

    Uses CLIP cross-view similarity and depth ordering analysis.
    """

    def __init__(self, clip_model=None, clip_preprocess=None,
                 clip_tokenizer=None, device: str = "cpu",
                 w_persistence: float = 0.35,
                 w_geometry: float = 0.35,
                 w_depth: float = 0.30):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.device = device
        self.w_persistence = w_persistence
        self.w_geometry = w_geometry
        self.w_depth = w_depth

    def compute(self, view_images: List[Image.Image],
                view_depth_maps: List[Image.Image],
                expected_objects: List[str],
                base_prompt: str) -> Dict[str, float]:
        """
        Compute multi-view consistency score.

        Args:
            view_images: List of generated view images.
            view_depth_maps: List of corresponding depth maps.
            expected_objects: Objects that should appear in all views.
            base_prompt: The scene description prompt.

        Returns:
            Dict with 'score' ∈ [0,1] and component breakdowns.
        """
        if not view_images or len(view_images) < 2:
            return {"score": 0.5, "persistence": 0.5,
                    "geometry": 0.5, "depth_stability": 0.5,
                    "note": "insufficient_views"}

        persistence = self._object_persistence(view_images, expected_objects)
        geometry = self._geometry_alignment(view_images)
        depth_stab = self._depth_ordering_stability(view_depth_maps)

        score = (self.w_persistence * persistence +
                 self.w_geometry * geometry +
                 self.w_depth * depth_stab)

        return {
            "score": float(np.clip(score, 0, 1)),
            "persistence": persistence,
            "geometry": geometry,
            "depth_stability": depth_stab,
        }

    def _object_persistence(self, views: List[Image.Image],
                              expected_objects: List[str]) -> float:
        """
        Check that the same objects are detected (via CLIP) across all views.

        persistence = min_v |detected_v| / |expected|
        """
        if self.clip_model is None or not expected_objects:
            return 0.5

        try:
            self.clip_model = self.clip_model.to(self.device)
            min_recall = 1.0

            for view in views:
                img_input = self.clip_preprocess(view).unsqueeze(0).to(self.device)
                detected = 0

                with torch.no_grad():
                    img_feat = self.clip_model.encode_image(img_input)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                    for obj in expected_objects:
                        t = self.clip_tokenizer(
                            [f"a photo containing a {obj}"]
                        ).to(self.device)
                        tf = self.clip_model.encode_text(t)
                        tf = tf / tf.norm(dim=-1, keepdim=True)
                        sim = (img_feat @ tf.T).item()
                        if (sim + 1) / 2 > 0.55:
                            detected += 1

                recall = detected / max(len(expected_objects), 1)
                min_recall = min(min_recall, recall)

            self.clip_model = self.clip_model.to("cpu")
            return min_recall

        except Exception as e:
            logger.error(f"Object persistence error: {e}")
            return 0.5

    def _geometry_alignment(self, views: List[Image.Image]) -> float:
        """
        Cross-view CLIP image-image similarity.

        Views of the same scene should have high mutual similarity.
        Computes mean pairwise CLIP similarity.
        """
        if self.clip_model is None or len(views) < 2:
            return 0.5

        try:
            self.clip_model = self.clip_model.to(self.device)
            features = []

            with torch.no_grad():
                for v in views:
                    img_input = self.clip_preprocess(v).unsqueeze(0).to(self.device)
                    feat = self.clip_model.encode_image(img_input)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    features.append(feat)

            # Pairwise similarities
            sims = []
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    sim = (features[i] @ features[j].T).item()
                    sims.append((sim + 1) / 2)  # Normalize to [0,1]

            self.clip_model = self.clip_model.to("cpu")
            return float(np.mean(sims))

        except Exception as e:
            logger.error(f"Geometry alignment error: {e}")
            return 0.5

    def _depth_ordering_stability(self,
                                    depth_maps: List[Image.Image]) -> float:
        """
        Check that depth ordering is consistent across views.

        Computes correlation between depth maps — views of the same scene
        should have correlated overall depth structure.
        """
        if not depth_maps or len(depth_maps) < 2:
            return 0.5

        try:
            arrays = []
            for dm in depth_maps:
                arr = np.array(dm.convert("L").resize((64, 64)),
                               dtype=np.float32).flatten()
                arrays.append(arr)

            correlations = []
            for i in range(len(arrays)):
                for j in range(i + 1, len(arrays)):
                    corr = np.corrcoef(arrays[i], arrays[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

            if not correlations:
                return 0.5

            # Map correlation (typically 0.3-0.9) to [0,1]
            mean_corr = np.mean(correlations)
            return float(np.clip(mean_corr, 0, 1))

        except Exception as e:
            logger.error(f"Depth stability error: {e}")
            return 0.5
