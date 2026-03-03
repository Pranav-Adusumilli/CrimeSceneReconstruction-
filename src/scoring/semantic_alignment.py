"""
Semantic Alignment Score
=========================

Measures how well the generated reconstruction matches the input text.

Mathematical formulation:

    S_semantic(R) = α · CLIP_sim(image, prompt)
                  + β · object_recall(R)
                  + γ · relationship_satisfaction(R)

Where:
    CLIP_sim ∈ [0,1]  — cosine similarity normalized to [0,1]
    object_recall     — |detected ∩ expected| / |expected|
    rel_satisfaction  — fraction of satisfied spatial relationships

    α + β + γ = 1  (default: 0.5, 0.3, 0.2)
"""

import logging
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional

from src.stages.stage1_text_understanding import SceneSemantics
from src.stages.stage3_scene_graph import SceneGraph
from src.stages.stage4_hypothesis_generation import SceneHypothesis

logger = logging.getLogger(__name__)


class SemanticAlignmentScorer:
    """
    Computes semantic alignment between generated image and input description.

    Components:
        1. CLIP text-image similarity (global semantic match)
        2. Object presence recall (per-object CLIP detection)
        3. Relationship satisfaction rate (spatial constraint checking)
    """

    def __init__(self, clip_model=None, clip_preprocess=None,
                 clip_tokenizer=None, device: str = "cpu",
                 alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        """
        Args:
            clip_model: OpenCLIP model instance.
            clip_preprocess: CLIP image preprocessor.
            clip_tokenizer: CLIP text tokenizer.
            device: Computation device.
            alpha: Weight for CLIP similarity.
            beta: Weight for object recall.
            gamma: Weight for relationship satisfaction.
        """
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute(self, image: Image.Image, prompt: str,
                semantics: SceneSemantics, scene_graph: SceneGraph,
                hypothesis: SceneHypothesis) -> Dict[str, float]:
        """
        Compute the full semantic alignment score.

        Returns:
            Dict with 'score' ∈ [0,1] and component breakdowns.
        """
        clip_sim = self._clip_similarity(image, prompt)
        obj_recall = self._object_recall(image, semantics.objects)
        rel_sat = self._relationship_satisfaction(scene_graph, hypothesis)

        score = (self.alpha * clip_sim +
                 self.beta * obj_recall +
                 self.gamma * rel_sat)

        return {
            "score": float(np.clip(score, 0, 1)),
            "clip_similarity": clip_sim,
            "object_recall": obj_recall,
            "relationship_satisfaction": rel_sat,
        }

    def _clip_similarity(self, image: Image.Image, text: str) -> float:
        """Compute CLIP cosine similarity, normalized to [0,1]."""
        if self.clip_model is None:
            return 0.5

        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_input = self.clip_tokenizer([text]).to(self.device)
            self.clip_model = self.clip_model.to(self.device)

            with torch.no_grad():
                img_feat = self.clip_model.encode_image(image_input)
                txt_feat = self.clip_model.encode_text(text_input)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                sim = (img_feat @ txt_feat.T).item()

            self.clip_model = self.clip_model.to("cpu")
            return float(np.clip((sim + 1) / 2, 0, 1))
        except Exception as e:
            logger.error(f"CLIP similarity error: {e}")
            return 0.5

    def _object_recall(self, image: Image.Image,
                       expected_objects: List[str]) -> float:
        """
        Proxy object detection via per-object CLIP matching.

        For each object, compute CLIP(image, "a photo containing a {obj}")
        and threshold to determine presence.
        """
        if self.clip_model is None or not expected_objects:
            return 0.5

        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            self.clip_model = self.clip_model.to(self.device)

            detected = 0
            with torch.no_grad():
                img_feat = self.clip_model.encode_image(image_input)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                for obj in expected_objects:
                    text = self.clip_tokenizer(
                        [f"a photo containing a {obj}"]
                    ).to(self.device)
                    txt_feat = self.clip_model.encode_text(text)
                    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                    sim = (img_feat @ txt_feat.T).item()
                    # Threshold: normalized similarity > 0.55
                    if (sim + 1) / 2 > 0.55:
                        detected += 1

            self.clip_model = self.clip_model.to("cpu")
            return detected / max(len(expected_objects), 1)
        except Exception as e:
            logger.error(f"Object recall error: {e}")
            return 0.5

    def _relationship_satisfaction(self, scene_graph: SceneGraph,
                                    hypothesis: SceneHypothesis) -> float:
        """
        Check what fraction of scene graph relationships are satisfied
        by the hypothesis spatial layout.
        """
        if not scene_graph.relationships:
            return 0.7  # Default for no explicit relationships

        positions = {p.name: (p.x, p.y, p.depth)
                     for p in hypothesis.placements}

        satisfied = 0
        total = len(scene_graph.relationships)

        for subj, pred, obj in scene_graph.relationships:
            if subj not in positions or obj not in positions:
                continue

            sx, sy, sd = positions[subj]
            ox, oy, od = positions[obj]

            if pred in ("on", "on top of", "above", "over"):
                satisfied += 1.0 if sy < oy else 0.0
            elif pred in ("under", "underneath", "beneath", "below"):
                satisfied += 1.0 if sy > oy else 0.0
            elif pred in ("near", "next to", "beside", "close to"):
                dist = np.sqrt((sx - ox)**2 + (sy - oy)**2)
                satisfied += 1.0 if dist < 0.35 else 0.0
            elif pred in ("behind",):
                satisfied += 1.0 if sd < od else 0.0
            elif pred in ("in front of",):
                satisfied += 1.0 if sd > od else 0.0
            elif pred in ("left", "on the left of"):
                satisfied += 1.0 if sx < ox else 0.0
            elif pred in ("right", "on the right of"):
                satisfied += 1.0 if sx > ox else 0.0
            else:
                dist = np.sqrt((sx - ox)**2 + (sy - oy)**2)
                satisfied += 0.5 if dist < 0.5 else 0.0

        return float(satisfied / max(total, 1))
