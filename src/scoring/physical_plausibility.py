"""
Physical Plausibility Score
=============================

Measures real-world geometric validity of the scene layout.

Mathematical formulation:

    S_physical(R) = (1/4) · [gravity(R) + support(R) + floor(R) + scale(R)]

Components:
    1. Gravity alignment   — objects are not floating (y position anchored)
    2. Support relations   — 'on' objects are above their support surfaces
    3. Floor contact       — non-wall objects are within floor plane
    4. Scale realism       — object sizes are within plausible bounds

Each sub-score ∈ [0,1]. Final score ∈ [0,1].
"""

import logging
import numpy as np
from typing import List, Dict

from src.stages.stage3_scene_graph import SceneGraph
from src.stages.stage4_hypothesis_generation import SceneHypothesis, ObjectPlacement
from src.stages.stage5_spatial_layout import SpatialLayout

logger = logging.getLogger(__name__)

# Typical relative sizes of common crime scene objects (fraction of scene)
OBJECT_SCALE_PRIORS = {
    "bed": (0.15, 0.40),
    "sofa": (0.12, 0.35),
    "table": (0.10, 0.30),
    "chair": (0.08, 0.25),
    "desk": (0.10, 0.30),
    "window": (0.08, 0.25),
    "door": (0.06, 0.20),
    "knife": (0.02, 0.10),
    "gun": (0.02, 0.08),
    "blood": (0.02, 0.25),
    "glass": (0.02, 0.15),
    "bottle": (0.02, 0.08),
    "body": (0.15, 0.45),
    "drawer": (0.05, 0.20),
    "counter": (0.10, 0.35),
    "refrigerator": (0.08, 0.25),
    "stove": (0.08, 0.25),
    "handprint": (0.01, 0.06),
}

# Objects that typically rest on surfaces (not on walls/ceiling)
FLOOR_OBJECTS = {
    "table", "chair", "sofa", "bed", "desk", "body", "blood",
    "knife", "gun", "bottle", "glass", "drawer", "counter",
    "refrigerator", "stove", "rug", "carpet",
}

# Objects that can be on walls (not necessarily floor-level)
WALL_OBJECTS = {
    "window", "handprint", "mirror", "clock", "painting", "shelf",
}


class PhysicalPlausibilityScorer:
    """
    Evaluates physical plausibility of object placements.

    Checks gravity, support, floor contact, and scale realism
    against common-sense physical constraints.
    """

    def __init__(self, w_gravity: float = 0.25, w_support: float = 0.25,
                 w_floor: float = 0.25, w_scale: float = 0.25):
        self.w_gravity = w_gravity
        self.w_support = w_support
        self.w_floor = w_floor
        self.w_scale = w_scale

    def compute(self, scene_graph: SceneGraph,
                hypothesis: SceneHypothesis,
                layout: SpatialLayout) -> Dict[str, float]:
        """
        Compute physical plausibility score.

        Returns:
            Dict with 'score' ∈ [0,1] and component breakdowns.
        """
        gravity = self._gravity_score(hypothesis)
        support = self._support_score(scene_graph, hypothesis)
        floor_contact = self._floor_contact_score(hypothesis)
        scale = self._scale_realism_score(hypothesis)

        score = (self.w_gravity * gravity +
                 self.w_support * support +
                 self.w_floor * floor_contact +
                 self.w_scale * scale)

        return {
            "score": float(np.clip(score, 0, 1)),
            "gravity_alignment": gravity,
            "support_relations": support,
            "floor_contact": floor_contact,
            "scale_realism": scale,
        }

    def _gravity_score(self, hypothesis: SceneHypothesis) -> float:
        """
        Check that objects are not 'floating' in the upper portion of
        the scene without support. Floor-level objects should have
        y > 0.3 (below the ceiling line).

        Score: fraction of non-wall objects that are in plausible y range.
        """
        if not hypothesis.placements:
            return 0.5

        valid = 0
        checked = 0

        for p in hypothesis.placements:
            name_lower = p.name.lower()
            if name_lower in WALL_OBJECTS:
                continue  # Wall objects can be at any height

            checked += 1
            # Non-wall objects should not float near the ceiling (y < 0.15)
            if p.y >= 0.15:
                valid += 1
            # Partial credit for borderline positions
            elif p.y >= 0.08:
                valid += 0.5

        return valid / max(checked, 1)

    def _support_score(self, scene_graph: SceneGraph,
                        hypothesis: SceneHypothesis) -> float:
        """
        For each 'on' relationship, verify the subject is above
        (smaller y) its support surface but not too far away.

        Score: fraction of 'on' relationships with valid support geometry.
        """
        positions = {p.name: (p.x, p.y, p.depth)
                     for p in hypothesis.placements}

        valid = 0
        checks = 0

        for subj, pred, obj in scene_graph.relationships:
            if pred not in ("on", "on top of"):
                continue
            if subj not in positions or obj not in positions:
                continue

            checks += 1
            sy = positions[subj][1]
            oy = positions[obj][1]

            # Subject should be above (smaller y) to slightly equal with object
            if sy <= oy + 0.05:
                # Check they're close vertically (support contact)
                if abs(sy - oy) < 0.25:
                    valid += 1
                else:
                    valid += 0.3  # Too far but correct direction

        return valid / max(checks, 1) if checks > 0 else 0.8

    def _floor_contact_score(self, hypothesis: SceneHypothesis) -> float:
        """
        Floor-level objects should be in the lower portion of the scene
        (y > 0.35 in normalized coords where 0=top, 1=bottom).

        Wall-mounted objects are excluded from this check.
        """
        if not hypothesis.placements:
            return 0.5

        valid = 0
        checked = 0

        for p in hypothesis.placements:
            name_lower = p.name.lower()
            if name_lower in WALL_OBJECTS:
                continue
            # Skip abstract concepts
            if name_lower in ("night", "dark", "scene"):
                continue

            checked += 1
            if name_lower in FLOOR_OBJECTS:
                # Should be in lower half of scene
                if p.y >= 0.30:
                    valid += 1
                elif p.y >= 0.20:
                    valid += 0.5
            else:
                # Unknown objects: lenient
                valid += 0.7

        return valid / max(checked, 1)

    def _scale_realism_score(self, hypothesis: SceneHypothesis) -> float:
        """
        Check if object scales are within plausible bounds from priors.

        Uses OBJECT_SCALE_PRIORS lookup table.
        """
        if not hypothesis.placements:
            return 0.5

        valid = 0
        checked = 0

        for p in hypothesis.placements:
            name_lower = p.name.lower()
            if name_lower in OBJECT_SCALE_PRIORS:
                checked += 1
                min_s, max_s = OBJECT_SCALE_PRIORS[name_lower]
                # Scale maps to approximate fraction of scene
                approx_frac = p.scale * 0.12  # base_size factor
                if min_s <= approx_frac <= max_s:
                    valid += 1
                else:
                    # Partial credit for close to bounds
                    dist = min(abs(approx_frac - min_s),
                               abs(approx_frac - max_s))
                    valid += max(0, 1.0 - dist * 5)

        return valid / max(checked, 1) if checked > 0 else 0.7
