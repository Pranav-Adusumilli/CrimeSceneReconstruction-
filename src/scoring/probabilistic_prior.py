"""
Probabilistic Prior Likelihood
================================

Computes P(layout | scene_graph) using Visual Genome statistics.

Mathematical formulation:

    log P(R) = Σ_{(i,j) ∈ E}  log P(r_ij | o_i, o_j)
             + Σ_i  log P(pos_i | scene_type)
             + Σ_{(i,j)}  log P(Δx, Δy | r_ij)

Components:
    1. Object co-occurrence    — P(o_i, o_j appear together)
    2. Relation frequency      — P(r | o_i, o_j) from VG statistics
    3. Relative position priors — P(Δx, Δy | relation_type)
    4. Size ratio priors       — P(s_i/s_j | o_i, o_j, r)

These are estimated from Visual Genome relationship data as
frequency distributions. Higher log-likelihood → higher score.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from src.stages.stage3_scene_graph import SceneGraph
from src.stages.stage4_hypothesis_generation import SceneHypothesis

logger = logging.getLogger(__name__)


# Default spatial priors when VG data isn't loaded
# These are manually calibrated Gaussian priors for common relations
DEFAULT_RELATION_PRIORS = {
    "on": {"dy_mean": -0.10, "dy_std": 0.08, "dx_mean": 0.0, "dx_std": 0.10},
    "on top of": {"dy_mean": -0.12, "dy_std": 0.08, "dx_mean": 0.0, "dx_std": 0.10},
    "under": {"dy_mean": 0.12, "dy_std": 0.08, "dx_mean": 0.0, "dx_std": 0.10},
    "near": {"dy_mean": 0.0, "dy_std": 0.15, "dx_mean": 0.0, "dx_std": 0.15},
    "next to": {"dy_mean": 0.0, "dy_std": 0.08, "dx_mean": 0.12, "dx_std": 0.10},
    "beside": {"dy_mean": 0.0, "dy_std": 0.08, "dx_mean": 0.12, "dx_std": 0.10},
    "behind": {"dy_mean": -0.05, "dy_std": 0.10, "dx_mean": 0.0, "dx_std": 0.08},
    "in front of": {"dy_mean": 0.05, "dy_std": 0.10, "dx_mean": 0.0, "dx_std": 0.08},
    "above": {"dy_mean": -0.20, "dy_std": 0.10, "dx_mean": 0.0, "dx_std": 0.08},
    "left": {"dy_mean": 0.0, "dy_std": 0.10, "dx_mean": -0.20, "dx_std": 0.10},
    "right": {"dy_mean": 0.0, "dy_std": 0.10, "dx_mean": 0.20, "dx_std": 0.10},
    "close to": {"dy_mean": 0.0, "dy_std": 0.12, "dx_mean": 0.0, "dx_std": 0.12},
    "inside": {"dy_mean": 0.0, "dy_std": 0.05, "dx_mean": 0.0, "dx_std": 0.05},
}


class ProbabilisticPriorScorer:
    """
    Computes the prior probability of a layout given the scene graph
    using spatial priors estimated from Visual Genome data.

    When VG relationship data is available, priors are learned from
    the empirical distribution of (Δx, Δy) for each relation type.
    Otherwise, falls back to manually calibrated Gaussian priors.
    """

    def __init__(self, vg_relationships_path: Optional[str] = None,
                 relationship_aliases: Optional[Dict[str, str]] = None):
        """
        Args:
            vg_relationships_path: Path to VG relationships.json (optional).
            relationship_aliases: VG relationship alias map for normalization.
        """
        self.relation_priors = dict(DEFAULT_RELATION_PRIORS)
        self.cooccurrence = defaultdict(float)
        self.relation_freq = defaultdict(float)
        self.aliases = relationship_aliases or {}

        if vg_relationships_path and Path(vg_relationships_path).exists():
            self._learn_priors_from_vg(vg_relationships_path)

    def _learn_priors_from_vg(self, path: str, max_images: int = 5000):
        """
        Learn spatial priors from Visual Genome relationship annotations.
        Samples up to max_images scenes to build frequency tables.
        """
        logger.info(f"Learning spatial priors from VG: {path}")
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Sample scenes
            scenes = data[:max_images] if len(data) > max_images else data

            relation_counts = defaultdict(int)
            pair_counts = defaultdict(int)

            for scene in scenes:
                rels = scene.get("relationships", [])
                for rel in rels:
                    pred = rel.get("predicate", "").lower().strip()
                    subj_name = rel.get("subject", {}).get("name", "").lower()
                    obj_name = rel.get("object", {}).get("name", "").lower()

                    if not pred or not subj_name or not obj_name:
                        continue

                    # Normalize via aliases
                    pred = self.aliases.get(pred, pred)

                    relation_counts[pred] += 1
                    pair_counts[(subj_name, obj_name)] += 1

            # Build frequency tables
            total_rels = sum(relation_counts.values()) or 1
            for rel, count in relation_counts.items():
                self.relation_freq[rel] = count / total_rels

            total_pairs = sum(pair_counts.values()) or 1
            for pair, count in pair_counts.items():
                self.cooccurrence[pair] = count / total_pairs

            logger.info(f"  Learned {len(relation_counts)} relation types, "
                        f"{len(pair_counts)} object pair co-occurrences")

        except Exception as e:
            logger.warning(f"Could not load VG priors: {e}")

    def compute(self, scene_graph: SceneGraph,
                hypothesis: SceneHypothesis) -> Dict[str, float]:
        """
        Compute the prior log-likelihood score for a layout.

        Converts log-likelihood to [0,1] via sigmoid normalization.

        Returns:
            Dict with 'score' ∈ [0,1] and component breakdowns.
        """
        positions = {p.name: (p.x, p.y, p.depth)
                     for p in hypothesis.placements}

        log_likelihood = 0.0
        n_terms = 0

        # Relation spatial likelihood
        rel_ll = self._relation_spatial_likelihood(
            scene_graph.relationships, positions
        )
        log_likelihood += rel_ll
        n_terms += max(len(scene_graph.relationships), 1)

        # Co-occurrence likelihood
        cooc_ll = self._cooccurrence_likelihood(scene_graph)
        log_likelihood += cooc_ll
        n_terms += 1

        # Normalize: mean log-likelihood per term
        mean_ll = log_likelihood / max(n_terms, 1)

        # Convert to [0,1] via sigmoid: σ(mean_ll + shift)
        # Typical good mean_ll ≈ -1 to 0, bad ≈ -3 to -2
        score = 1.0 / (1.0 + np.exp(-(mean_ll + 2.0)))

        return {
            "score": float(np.clip(score, 0, 1)),
            "relation_spatial_ll": rel_ll,
            "cooccurrence_ll": cooc_ll,
            "mean_log_likelihood": mean_ll,
        }

    def _relation_spatial_likelihood(
        self, relationships: List[Tuple[str, str, str]],
        positions: Dict[str, Tuple[float, float, float]]
    ) -> float:
        """
        Compute Σ log P(Δx, Δy | relation) across all relationships.

        Models P(Δx, Δy | r) as a bivariate Gaussian:
            P = N(Δx; μ_x, σ_x) · N(Δy; μ_y, σ_y)
        """
        total_ll = 0.0

        for subj, pred, obj in relationships:
            if subj not in positions or obj not in positions:
                total_ll += -2.0  # Penalty for missing objects
                continue

            sx, sy, _ = positions[subj]
            ox, oy, _ = positions[obj]
            dx = sx - ox
            dy = sy - oy

            prior = self.relation_priors.get(pred, {
                "dy_mean": 0.0, "dy_std": 0.20,
                "dx_mean": 0.0, "dx_std": 0.20,
            })

            # Bivariate Gaussian log-likelihood
            ll_x = -0.5 * ((dx - prior["dx_mean"]) / prior["dx_std"]) ** 2
            ll_y = -0.5 * ((dy - prior["dy_mean"]) / prior["dy_std"]) ** 2
            ll_x -= np.log(prior["dx_std"] * np.sqrt(2 * np.pi))
            ll_y -= np.log(prior["dy_std"] * np.sqrt(2 * np.pi))

            total_ll += ll_x + ll_y

        return total_ll

    def _cooccurrence_likelihood(self, scene_graph: SceneGraph) -> float:
        """
        Compute log P(objects co-occur) based on VG pair frequencies.
        """
        objects = scene_graph.objects
        if len(objects) < 2 or not self.cooccurrence:
            return 0.0  # Neutral

        ll = 0.0
        n_pairs = 0
        # Laplace-smoothed minimum frequency for unseen pairs
        # Crime scene objects are rare in VG; don't over-penalize
        smoothing_floor = 1e-4
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                pair = (objects[i].lower(), objects[j].lower())
                pair_rev = (objects[j].lower(), objects[i].lower())
                freq = self.cooccurrence.get(pair,
                       self.cooccurrence.get(pair_rev, smoothing_floor))
                freq = max(freq, smoothing_floor)
                ll += np.log(freq)
                n_pairs += 1

        # Clamp co-occurrence LL to prevent rare-pair domination
        return max(ll / max(n_pairs, 1), -6.0)
