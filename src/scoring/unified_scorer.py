"""
Unified Multi-Objective Reconstruction Scorer
================================================

Computes the global reconstruction objective function:

    S(R) = w_semantic      · semantic_alignment(R)
         + w_spatial       · spatial_consistency(R)
         + w_physical      · physical_plausibility(R)
         + w_visual        · visual_realism(R)
         + w_probabilistic · prior_likelihood(R)
         + w_multiview     · multi_view_consistency(R)
         + w_human         · perceptual_believability(R)

All weights are configurable. The energy for optimization is E(R) = -S(R).
"""

import logging
import time
import numpy as np
from PIL import Image
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

from src.stages.stage1_text_understanding import SceneSemantics
from src.stages.stage3_scene_graph import SceneGraph
from src.stages.stage4_hypothesis_generation import SceneHypothesis
from src.stages.stage5_spatial_layout import SpatialLayout

from src.scoring.semantic_alignment import SemanticAlignmentScorer
from src.scoring.spatial_consistency import SpatialConsistencyScorer
from src.scoring.physical_plausibility import PhysicalPlausibilityScorer
from src.scoring.visual_realism import VisualRealismScorer
from src.scoring.probabilistic_prior import ProbabilisticPriorScorer
from src.scoring.multiview_consistency import MultiViewConsistencyScorer
from src.scoring.perceptual_believability import PerceptualBelievabilityScorer

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Configurable weights for the unified objective function."""
    w_semantic: float = 0.20
    w_spatial: float = 0.15
    w_physical: float = 0.10
    w_visual: float = 0.15
    w_probabilistic: float = 0.10
    w_multiview: float = 0.10
    w_human: float = 0.20

    def validate(self):
        """Verify weights are non-negative. They need not sum to 1."""
        for name, val in asdict(self).items():
            assert val >= 0, f"Weight {name} must be non-negative, got {val}"

    def normalized(self) -> "ScoringWeights":
        """Return a copy with weights normalized to sum to 1."""
        d = asdict(self)
        total = sum(d.values())
        if total == 0:
            return ScoringWeights()
        return ScoringWeights(**{k: v / total for k, v in d.items()})


@dataclass
class ScoreBreakdown:
    """Complete scoring breakdown for one reconstruction."""
    # Final scores
    total_score: float = 0.0
    energy: float = 0.0  # E(R) = -S(R)

    # Component scores (each ∈ [0,1])
    semantic_score: float = 0.0
    spatial_score: float = 0.0
    physical_score: float = 0.0
    visual_score: float = 0.0
    probabilistic_score: float = 0.0
    multiview_score: float = 0.0
    human_score: float = 0.0

    # Detailed sub-component dicts
    semantic_details: Dict = field(default_factory=dict)
    spatial_details: Dict = field(default_factory=dict)
    physical_details: Dict = field(default_factory=dict)
    visual_details: Dict = field(default_factory=dict)
    probabilistic_details: Dict = field(default_factory=dict)
    multiview_details: Dict = field(default_factory=dict)
    human_details: Dict = field(default_factory=dict)

    # Metadata
    weights_used: Dict = field(default_factory=dict)
    computation_time_s: float = 0.0
    hypothesis_id: int = 0

    def to_dict(self) -> dict:
        return {
            "total_score": round(self.total_score, 4),
            "energy": round(self.energy, 4),
            "components": {
                "semantic": round(self.semantic_score, 4),
                "spatial": round(self.spatial_score, 4),
                "physical": round(self.physical_score, 4),
                "visual": round(self.visual_score, 4),
                "probabilistic": round(self.probabilistic_score, 4),
                "multiview": round(self.multiview_score, 4),
                "human": round(self.human_score, 4),
            },
            "details": {
                "semantic": self.semantic_details,
                "spatial": self.spatial_details,
                "physical": self.physical_details,
                "visual": self.visual_details,
                "probabilistic": self.probabilistic_details,
                "multiview": self.multiview_details,
                "human": self.human_details,
            },
            "weights_used": self.weights_used,
            "computation_time_s": round(self.computation_time_s, 2),
            "hypothesis_id": self.hypothesis_id,
        }


class UnifiedScorer:
    """
    Computes the unified multi-objective reconstruction score S(R).

    Orchestrates all 7 scoring components and produces a weighted sum.
    The energy function E(R) = -S(R) is used by the optimization engine.
    """

    def __init__(self, weights: ScoringWeights = None,
                 clip_model=None, clip_preprocess=None,
                 clip_tokenizer=None, device: str = "cpu",
                 vg_relationships_path: str = None,
                 relationship_aliases: Dict = None):
        """
        Args:
            weights: ScoringWeights for objective function.
            clip_model: Shared OpenCLIP model (used by multiple scorers).
            clip_preprocess: CLIP image preprocessor.
            clip_tokenizer: CLIP text tokenizer.
            device: Computation device.
            vg_relationships_path: Path to VG relationships.json.
            relationship_aliases: VG alias map for normalization.
        """
        self.weights = weights or ScoringWeights()
        self.weights.validate()

        # Initialize sub-scorers with shared CLIP model
        self.semantic_scorer = SemanticAlignmentScorer(
            clip_model=clip_model, clip_preprocess=clip_preprocess,
            clip_tokenizer=clip_tokenizer, device=device,
        )
        self.spatial_scorer = SpatialConsistencyScorer()
        self.physical_scorer = PhysicalPlausibilityScorer()
        self.visual_scorer = VisualRealismScorer(
            clip_model=clip_model, clip_preprocess=clip_preprocess,
            clip_tokenizer=clip_tokenizer, device=device,
        )
        self.prior_scorer = ProbabilisticPriorScorer(
            vg_relationships_path=vg_relationships_path,
            relationship_aliases=relationship_aliases,
        )
        self.multiview_scorer = MultiViewConsistencyScorer(
            clip_model=clip_model, clip_preprocess=clip_preprocess,
            clip_tokenizer=clip_tokenizer, device=device,
        )
        self.human_scorer = PerceptualBelievabilityScorer(
            clip_model=clip_model, clip_preprocess=clip_preprocess,
            clip_tokenizer=clip_tokenizer, device=device,
        )

    def score(self, image: Image.Image, prompt: str,
              semantics: SceneSemantics, scene_graph: SceneGraph,
              hypothesis: SceneHypothesis, layout: SpatialLayout,
              view_images: List[Image.Image] = None,
              view_depth_maps: List[Image.Image] = None,
              skip_multiview: bool = False,
              skip_visual: bool = False) -> ScoreBreakdown:
        """
        Compute the full unified score S(R).

        Args:
            image: Generated reconstruction image.
            prompt: Generation prompt.
            semantics: Parsed scene semantics.
            scene_graph: Scene graph.
            hypothesis: The hypothesis used.
            layout: Spatial layout.
            view_images: Multi-view images (optional).
            view_depth_maps: Multi-view depth maps (optional).
            skip_multiview: Skip multi-view scoring.
            skip_visual: Skip visual quality (faster for optimization).

        Returns:
            ScoreBreakdown with all component scores and energy.
        """
        t_start = time.time()
        w = self.weights
        bd = ScoreBreakdown()
        bd.hypothesis_id = hypothesis.hypothesis_id
        bd.weights_used = {k: round(v, 3) for k, v in asdict(w).items()}

        # 1. Semantic Alignment
        sem_result = self.semantic_scorer.compute(
            image, prompt, semantics, scene_graph, hypothesis
        )
        bd.semantic_score = sem_result["score"]
        bd.semantic_details = sem_result

        # 2. Spatial Consistency
        sp_result = self.spatial_scorer.compute(
            scene_graph, hypothesis, layout
        )
        bd.spatial_score = sp_result["score"]
        bd.spatial_details = sp_result

        # 3. Physical Plausibility
        ph_result = self.physical_scorer.compute(
            scene_graph, hypothesis, layout
        )
        bd.physical_score = ph_result["score"]
        bd.physical_details = ph_result

        # 4. Visual Realism
        if not skip_visual:
            vis_result = self.visual_scorer.compute(image)
            bd.visual_score = vis_result["score"]
            bd.visual_details = vis_result
        else:
            bd.visual_score = 0.5
            bd.visual_details = {"note": "skipped"}

        # 5. Probabilistic Prior
        pr_result = self.prior_scorer.compute(scene_graph, hypothesis)
        bd.probabilistic_score = pr_result["score"]
        bd.probabilistic_details = pr_result

        # 6. Multi-View Consistency
        if not skip_multiview and view_images and len(view_images) >= 2:
            mv_result = self.multiview_scorer.compute(
                view_images, view_depth_maps or [],
                semantics.objects, prompt,
            )
            bd.multiview_score = mv_result["score"]
            bd.multiview_details = mv_result
        else:
            bd.multiview_score = 0.5
            bd.multiview_details = {"note": "skipped_or_insufficient_views"}

        # 7. Human Perceptual Believability
        hm_result = self.human_scorer.compute(image, semantics.scene_type)
        bd.human_score = hm_result["score"]
        bd.human_details = hm_result

        # Weighted sum
        bd.total_score = (
            w.w_semantic * bd.semantic_score +
            w.w_spatial * bd.spatial_score +
            w.w_physical * bd.physical_score +
            w.w_visual * bd.visual_score +
            w.w_probabilistic * bd.probabilistic_score +
            w.w_multiview * bd.multiview_score +
            w.w_human * bd.human_score
        )

        # Energy (for optimization)
        bd.energy = -bd.total_score

        bd.computation_time_s = time.time() - t_start

        logger.info(f"  Unified Score: {bd.total_score:.4f} "
                     f"(E={bd.energy:.4f}) [{bd.computation_time_s:.1f}s]")

        return bd

    def score_layout_only(self, scene_graph: SceneGraph,
                           hypothesis: SceneHypothesis,
                           layout: SpatialLayout) -> float:
        """
        Fast layout-only scoring (no image generation needed).

        Uses spatial + physical + probabilistic components only.
        Suitable for optimization inner loops.

        Returns:
            Float score ∈ [0, ~0.35] (subset of full score).
        """
        sp = self.spatial_scorer.compute(scene_graph, hypothesis, layout)
        ph = self.physical_scorer.compute(scene_graph, hypothesis, layout)
        pr = self.prior_scorer.compute(scene_graph, hypothesis)

        return (self.weights.w_spatial * sp["score"] +
                self.weights.w_physical * ph["score"] +
                self.weights.w_probabilistic * pr["score"])
