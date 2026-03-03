"""
Spatial Consistency Score
==========================

Measures structural correctness of the spatial layout.

Mathematical formulation:

    S_spatial(R) = 1 − (1/N) Σ_i  violation_i(R)

Violation types:
    1. Relationship violations    — v_rel : predicate not satisfied
    2. Relative position errors   — v_pos : incorrect ordering
    3. Depth ordering errors      — v_depth : depth contradicts spatial semantics
    4. Object interpenetration    — v_overlap : IoU between bounding regions

Each violation ∈ [0,1]. Final score ∈ [0,1].
"""

import logging
import numpy as np
from typing import List, Dict, Tuple

from src.stages.stage3_scene_graph import SceneGraph
from src.stages.stage4_hypothesis_generation import SceneHypothesis, ObjectPlacement
from src.stages.stage5_spatial_layout import SpatialLayout, BoundingRegion

logger = logging.getLogger(__name__)


class SpatialConsistencyScorer:
    """
    Evaluates spatial layout correctness via constraint violation analysis.

    Computes four violation sub-scores and aggregates into [0,1] consistency.
    """

    def __init__(self, w_rel: float = 0.35, w_pos: float = 0.25,
                 w_depth: float = 0.20, w_overlap: float = 0.20):
        """
        Args:
            w_rel: Weight for relationship violations.
            w_pos: Weight for position ordering errors.
            w_depth: Weight for depth ordering errors.
            w_overlap: Weight for interpenetration violations.
        """
        self.w_rel = w_rel
        self.w_pos = w_pos
        self.w_depth = w_depth
        self.w_overlap = w_overlap

    def compute(self, scene_graph: SceneGraph,
                hypothesis: SceneHypothesis,
                layout: SpatialLayout) -> Dict[str, float]:
        """
        Compute spatial consistency score.

        Returns:
            Dict with 'score' ∈ [0,1] and violation breakdowns.
        """
        v_rel = self._relationship_violations(scene_graph, hypothesis)
        v_pos = self._position_ordering_errors(scene_graph, hypothesis)
        v_depth = self._depth_ordering_errors(scene_graph, hypothesis)
        v_overlap = self._interpenetration_score(layout)

        # Score = 1 - weighted violation rate
        total_violation = (self.w_rel * v_rel +
                           self.w_pos * v_pos +
                           self.w_depth * v_depth +
                           self.w_overlap * v_overlap)

        score = float(np.clip(1.0 - total_violation, 0, 1))

        return {
            "score": score,
            "relationship_violations": v_rel,
            "position_ordering_errors": v_pos,
            "depth_ordering_errors": v_depth,
            "interpenetration_score": v_overlap,
        }

    def _relationship_violations(self, scene_graph: SceneGraph,
                                   hypothesis: SceneHypothesis) -> float:
        """Fraction of scene graph relationships that are violated."""
        if not scene_graph.relationships:
            return 0.0

        positions = {p.name: (p.x, p.y, p.depth)
                     for p in hypothesis.placements}

        violations = 0
        total = len(scene_graph.relationships)

        for subj, pred, obj in scene_graph.relationships:
            if subj not in positions or obj not in positions:
                violations += 0.5
                continue

            sx, sy, sd = positions[subj]
            ox, oy, od = positions[obj]

            violated = False
            if pred in ("on", "on top of", "above", "over"):
                violated = sy >= oy
            elif pred in ("under", "underneath", "beneath", "below"):
                violated = sy <= oy
            elif pred in ("near", "next to", "beside", "close to"):
                dist = np.sqrt((sx - ox)**2 + (sy - oy)**2)
                violated = dist >= 0.40
            elif pred in ("behind",):
                violated = sd >= od
            elif pred in ("in front of",):
                violated = sd <= od
            elif pred in ("left", "on the left of"):
                violated = sx >= ox
            elif pred in ("right", "on the right of"):
                violated = sx <= ox

            if violated:
                violations += 1

        return violations / max(total, 1)

    def _position_ordering_errors(self, scene_graph: SceneGraph,
                                    hypothesis: SceneHypothesis) -> float:
        """
        Check if the relative position ordering matches expected layout.
        Objects that should be above/below/left/right must have correct
        coordinate ordering.
        """
        positions = {p.name: (p.x, p.y, p.depth)
                     for p in hypothesis.placements}

        errors = 0
        checks = 0

        for subj, pred, obj in scene_graph.relationships:
            if subj not in positions or obj not in positions:
                continue

            sx, sy, _ = positions[subj]
            ox, oy, _ = positions[obj]

            if pred in ("on", "on top of", "above"):
                checks += 1
                if sy > oy + 0.05:  # Should be above (lower y)
                    errors += 1
            elif pred in ("under", "underneath", "beneath"):
                checks += 1
                if sy < oy - 0.05:
                    errors += 1
            elif pred in ("left", "on the left of"):
                checks += 1
                if sx > ox + 0.05:
                    errors += 1
            elif pred in ("right", "on the right of"):
                checks += 1
                if sx < ox - 0.05:
                    errors += 1

        return errors / max(checks, 1)

    def _depth_ordering_errors(self, scene_graph: SceneGraph,
                                 hypothesis: SceneHypothesis) -> float:
        """
        Check depth ordering consistency with spatial relationships.
        'behind' → smaller depth, 'in front of' → larger depth.
        """
        positions = {p.name: (p.x, p.y, p.depth)
                     for p in hypothesis.placements}

        errors = 0
        checks = 0

        for subj, pred, obj in scene_graph.relationships:
            if subj not in positions or obj not in positions:
                continue

            sd = positions[subj][2]
            od = positions[obj][2]

            if pred in ("behind",):
                checks += 1
                if sd >= od:
                    errors += 1
            elif pred in ("in front of",):
                checks += 1
                if sd <= od:
                    errors += 1
            elif pred in ("on", "on top of"):
                # Objects on top should have similar or slightly closer depth
                checks += 1
                if abs(sd - od) > 0.3:
                    errors += 1

        return errors / max(checks, 1)

    def _interpenetration_score(self, layout: SpatialLayout) -> float:
        """
        Compute average IoU between all pairs of bounding regions.
        High IoU indicates interpenetration (violation).
        """
        regions = layout.regions
        if len(regions) < 2:
            return 0.0

        total_iou = 0.0
        n_pairs = 0

        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                iou = self._compute_iou(regions[i], regions[j])
                total_iou += iou
                n_pairs += 1

        avg_iou = total_iou / max(n_pairs, 1)
        # Penalize: IoU > 0.1 is concerning, > 0.3 is severe
        return float(np.clip(avg_iou * 3.0, 0, 1))

    @staticmethod
    def _compute_iou(r1: BoundingRegion, r2: BoundingRegion) -> float:
        """Compute Intersection over Union between two bounding regions."""
        x_overlap = max(0, min(r1.x2, r2.x2) - max(r1.x1, r2.x1))
        y_overlap = max(0, min(r1.y2, r2.y2) - max(r1.y1, r2.y1))
        intersection = x_overlap * y_overlap

        area1 = r1.width * r1.height
        area2 = r2.width * r2.height
        union = area1 + area2 - intersection

        return intersection / max(union, 1)
