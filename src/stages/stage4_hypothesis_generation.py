"""
STAGE 4 — Multi-Hypothesis Generation
=======================================
Generates multiple plausible spatial configurations from the scene graph.

Crime scenes are inherently uncertain — this stage produces ranked hypotheses
with confidence scores using probabilistic spatial sampling.
"""

import logging
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from src.stages.stage3_scene_graph import SceneGraph

logger = logging.getLogger(__name__)


@dataclass
class ObjectPlacement:
    """Spatial placement of a single object."""
    name: str
    x: float  # normalized [0, 1]
    y: float  # normalized [0, 1]
    depth: float  # relative depth [0=far, 1=near]
    scale: float  # relative size [0.5-1.5]
    attributes: List[str] = field(default_factory=list)


@dataclass
class SceneHypothesis:
    """A single plausible scene configuration."""
    hypothesis_id: int
    confidence: float
    placements: List[ObjectPlacement] = field(default_factory=list)
    description: str = ""
    prompt_modifier: str = ""

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "confidence": self.confidence,
            "description": self.description,
            "prompt_modifier": self.prompt_modifier,
            "placements": [
                {
                    "name": p.name,
                    "x": round(p.x, 3),
                    "y": round(p.y, 3),
                    "depth": round(p.depth, 3),
                    "scale": round(p.scale, 2),
                    "attributes": p.attributes,
                }
                for p in self.placements
            ],
        }


# ── Spatial constraint templates ────────────────────────────────

# Maps relationship predicates to relative position offsets
RELATION_SPATIAL_MAP = {
    "on": {"dy": -0.1, "dz": 0.0},       # subject is on top of object
    "on top of": {"dy": -0.15, "dz": 0.0},
    "under": {"dy": 0.15, "dz": 0.0},
    "underneath": {"dy": 0.15, "dz": 0.0},
    "beneath": {"dy": 0.15, "dz": 0.0},
    "near": {"dx_range": (-0.15, 0.15), "dy_range": (-0.1, 0.1)},
    "next to": {"dx_range": (-0.15, 0.15)},
    "beside": {"dx_range": (-0.15, 0.15)},
    "close to": {"dx_range": (-0.12, 0.12), "dy_range": (-0.08, 0.08)},
    "behind": {"dz": -0.15, "dy": -0.05},
    "in front of": {"dz": 0.15, "dy": 0.05},
    "above": {"dy": -0.2, "dz": 0.0},
    "over": {"dy": -0.2, "dz": 0.0},
    "against": {"dx_range": (-0.05, 0.05)},
    "leaning against": {"dx_range": (-0.05, 0.05), "dy": 0.05},
    "inside": {"dx_range": (-0.05, 0.05), "dy_range": (-0.05, 0.05)},
    "left": {"dx": -0.2},
    "right": {"dx": 0.2},
    "between": {"dx": 0.0},
}


class HypothesisGenerator:
    """
    Generates multiple spatial layout hypotheses from a scene graph.

    Each hypothesis represents a plausible arrangement of objects in 2D+depth space
    satisfying the relationship constraints with probabilistic variation.
    """

    def __init__(self, num_hypotheses: int = 3, seed: int = 42):
        self.num_hypotheses = num_hypotheses
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate(self, scene_graph: SceneGraph) -> List[SceneHypothesis]:
        """
        Generate multiple spatial hypotheses for the given scene graph.

        Each hypothesis varies:
          - Base positions of unanchored objects
          - Jitter on relationship-constrained positions
          - Scale variation

        Returns:
            Sorted list of hypotheses (highest confidence first).
        """
        logger.info("=" * 50)
        logger.info("STAGE 4: Multi-Hypothesis Generation")
        logger.info("=" * 50)

        G = scene_graph.graph
        objects = list(G.nodes())
        relationships = [
            (u, data["relation"], v) for u, v, data in G.edges(data=True)
        ]

        hypotheses = []

        for h_idx in range(self.num_hypotheses):
            logger.info(f"  Generating hypothesis {h_idx + 1}/{self.num_hypotheses}")

            # Assign base positions
            base_positions = self._sample_base_positions(objects, h_idx)

            # Apply relationship constraints with jitter
            constrained_positions = self._apply_constraints(
                base_positions, relationships, G, jitter_scale=0.05 * (h_idx + 1)
            )

            # Build placements
            placements = []
            for obj in objects:
                pos = constrained_positions[obj]
                attrs = G.nodes[obj].get("attributes", [])
                placements.append(ObjectPlacement(
                    name=obj,
                    x=np.clip(pos["x"], 0.05, 0.95),
                    y=np.clip(pos["y"], 0.05, 0.95),
                    depth=np.clip(pos["depth"], 0.0, 1.0),
                    scale=np.clip(pos.get("scale", 1.0), 0.5, 1.5),
                    attributes=attrs,
                ))

            # Compute confidence based on constraint satisfaction
            confidence = self._compute_confidence(constrained_positions, relationships)

            # Descriptive text for this hypothesis
            desc = self._describe_hypothesis(placements, h_idx)
            prompt_mod = self._make_prompt_modifier(placements, scene_graph.scene_type)

            hypothesis = SceneHypothesis(
                hypothesis_id=h_idx + 1,
                confidence=confidence,
                placements=placements,
                description=desc,
                prompt_modifier=prompt_mod,
            )
            hypotheses.append(hypothesis)
            logger.info(f"    Confidence: {confidence:.3f}")

        # Sort by confidence descending
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        logger.info(f"  Generated {len(hypotheses)} hypotheses")
        return hypotheses

    def _sample_base_positions(self, objects: List[str],
                                variant: int) -> Dict[str, Dict]:
        """Generate base spatial positions for all objects."""
        positions = {}
        n = len(objects)

        for i, obj in enumerate(objects):
            # Distribute objects across the scene with variation per hypothesis
            angle = (2 * np.pi * i / max(n, 1)) + variant * 0.3
            radius = 0.25 + self.np_rng.uniform(-0.1, 0.1)

            positions[obj] = {
                "x": 0.5 + radius * np.cos(angle) + self.np_rng.normal(0, 0.05),
                "y": 0.5 + radius * np.sin(angle) * 0.6 + self.np_rng.normal(0, 0.03),
                "depth": 0.3 + self.np_rng.uniform(0, 0.4),
                "scale": 1.0 + self.np_rng.normal(0, 0.1),
            }

        return positions

    def _apply_constraints(self, positions: Dict[str, Dict],
                            relationships: List[Tuple[str, str, str]],
                            G, jitter_scale: float = 0.05) -> Dict[str, Dict]:
        """Apply spatial relationship constraints to positions."""
        # Iterative constraint satisfaction (2 passes)
        for _ in range(2):
            for subj, pred, obj in relationships:
                if subj not in positions or obj not in positions:
                    continue

                spatial = RELATION_SPATIAL_MAP.get(pred, {})

                # Apply fixed offsets
                if "dx" in spatial:
                    target_x = positions[obj]["x"] + spatial["dx"]
                    positions[subj]["x"] = target_x + self.np_rng.normal(0, jitter_scale)
                if "dy" in spatial:
                    target_y = positions[obj]["y"] + spatial["dy"]
                    positions[subj]["y"] = target_y + self.np_rng.normal(0, jitter_scale)
                if "dz" in spatial:
                    positions[subj]["depth"] = positions[obj]["depth"] + spatial["dz"]

                # Apply range-based offsets
                if "dx_range" in spatial:
                    lo, hi = spatial["dx_range"]
                    dx = self.np_rng.uniform(lo, hi)
                    positions[subj]["x"] = positions[obj]["x"] + dx
                if "dy_range" in spatial:
                    lo, hi = spatial["dy_range"]
                    dy = self.np_rng.uniform(lo, hi)
                    positions[subj]["y"] = positions[obj]["y"] + dy

        return positions

    def _compute_confidence(self, positions: Dict[str, Dict],
                             relationships: List[Tuple[str, str, str]]) -> float:
        """
        Score hypothesis confidence based on:
          - Constraint satisfaction (are relationships spatially consistent?)
          - Object distribution (no extreme overlaps)
        """
        if not relationships:
            return 0.7  # Default when no constraints

        satisfied = 0
        total = len(relationships)

        for subj, pred, obj in relationships:
            if subj not in positions or obj not in positions:
                continue

            dx = positions[subj]["x"] - positions[obj]["x"]
            dy = positions[subj]["y"] - positions[obj]["y"]
            dist = np.sqrt(dx ** 2 + dy ** 2)

            # Check if spatial constraint is roughly satisfied
            if pred in ("on", "on top of") and dy < 0:
                satisfied += 1
            elif pred in ("under", "underneath", "beneath") and dy > 0:
                satisfied += 1
            elif pred in ("near", "next to", "beside", "close to") and dist < 0.3:
                satisfied += 1
            elif pred in ("behind",) and positions[subj]["depth"] < positions[obj]["depth"]:
                satisfied += 1
            elif pred in ("in front of",) and positions[subj]["depth"] > positions[obj]["depth"]:
                satisfied += 1
            elif pred in ("above", "over") and dy < 0:
                satisfied += 1
            else:
                # Generous: close objects likely satisfy generic relations
                if dist < 0.4:
                    satisfied += 0.5

        score = satisfied / max(total, 1)
        # Add small noise to differentiate hypotheses
        score = np.clip(score + self.np_rng.uniform(-0.05, 0.05), 0.1, 1.0)
        return round(float(score), 3)

    def _describe_hypothesis(self, placements: List[ObjectPlacement],
                              h_idx: int) -> str:
        """Generate a human-readable description of the hypothesis."""
        desc_parts = [f"Hypothesis {h_idx + 1}:"]
        for p in placements:
            loc = "center"
            if p.x < 0.35:
                loc = "left"
            elif p.x > 0.65:
                loc = "right"

            depth_desc = "foreground" if p.depth > 0.6 else ("midground" if p.depth > 0.3 else "background")
            attr_str = f" ({', '.join(p.attributes)})" if p.attributes else ""
            desc_parts.append(f"  {p.name}{attr_str}: {loc}, {depth_desc}")

        return "\n".join(desc_parts)

    def _make_prompt_modifier(self, placements: List[ObjectPlacement],
                               scene_type: str) -> str:
        """Create a prompt modifier string that describes the layout."""
        parts = []
        for p in placements:
            attr_str = " ".join(p.attributes) if p.attributes else ""
            if attr_str:
                parts.append(f"{attr_str} {p.name}")
            else:
                parts.append(p.name)

        layout_desc = ", ".join(parts)
        return f"a {scene_type} scene with {layout_desc}"
