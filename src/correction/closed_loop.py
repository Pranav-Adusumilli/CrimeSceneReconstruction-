"""
Closed-Loop Self-Correction
==============================

Iterative refinement pipeline that detects discrepancies between the
generated reconstruction and the input specifications, then applies
targeted corrections to improve the final output.

Algorithm:
  1. **Detect**: Evaluate S(R) breakdown to identify weakest components.
  2. **Diagnose**: Map low-scoring components to corrective actions.
  3. **Correct**: Apply corrections (layout updates, prompt refinement,
     seed variation, conditioning strength adjustment).
  4. **Regenerate**: Produce a new reconstruction with corrections.
  5. **Accept/Reject**: Keep the corrected version only if S(R) improves.
  6. **Iterate**: Repeat until convergence or max iterations.

Self-correction targets:
  - Object presence: Missing objects → add to prompt emphasis
  - Spatial accuracy: Wrong positions → adjust layout + depth map
  - Physical violations: Floating/overlapping → fix y-positions, scales
  - Visual quality: Artifacts → vary seed, adjust guidance scale
  - Realism: Uncanny artifacts → strengthen negative prompt

Designed for RTX 3060 6GB: each correction iteration requires one
image generation cycle (~15-30s), so max 3-5 iterations recommended.
"""

import copy
import logging
import time
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from src.stages.stage1_text_understanding import SceneSemantics
from src.stages.stage3_scene_graph import SceneGraph
from src.stages.stage4_hypothesis_generation import SceneHypothesis, ObjectPlacement
from src.stages.stage5_spatial_layout import SpatialLayout, SpatialLayoutEstimator
from src.scoring.unified_scorer import UnifiedScorer, ScoreBreakdown, ScoringWeights

logger = logging.getLogger(__name__)


# ── Correction Actions ──────────────────────────────────────────────

@dataclass
class CorrectionAction:
    """A single corrective action applied to the reconstruction."""
    action_type: str  # e.g., "boost_object", "fix_layout", "vary_seed"
    target: str  # What the action targets (e.g., object name, component name)
    description: str  # Human-readable description
    parameters: Dict = field(default_factory=dict)


@dataclass
class CorrectionIteration:
    """Record of one correction iteration."""
    iteration: int
    score_before: float
    score_after: float
    actions_applied: List[CorrectionAction] = field(default_factory=list)
    accepted: bool = False
    breakdown_before: Dict = field(default_factory=dict)
    breakdown_after: Dict = field(default_factory=dict)


@dataclass
class CorrectionResult:
    """Complete result of the closed-loop correction process."""
    initial_score: float = 0.0
    final_score: float = 0.0
    improvement: float = 0.0
    iterations: List[CorrectionIteration] = field(default_factory=list)
    total_corrections: int = 0
    total_regenerations: int = 0
    converged: bool = False
    total_time_s: float = 0.0
    final_hypothesis: SceneHypothesis = None
    final_layout: SpatialLayout = None
    final_prompt_modifications: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "initial_score": round(self.initial_score, 4),
            "final_score": round(self.final_score, 4),
            "improvement": round(self.improvement, 4),
            "improvement_pct": round(self.improvement / max(self.initial_score, 1e-6) * 100, 1),
            "total_iterations": len(self.iterations),
            "total_corrections": self.total_corrections,
            "total_regenerations": self.total_regenerations,
            "converged": self.converged,
            "total_time_s": round(self.total_time_s, 2),
        }


# ── Correction Strategies ───────────────────────────────────────────

class CorrectionStrategies:
    """Library of correction strategies mapped to scoring components."""

    @staticmethod
    def diagnose(breakdown: ScoreBreakdown,
                 threshold: float = 0.4) -> List[CorrectionAction]:
        """
        Analyze score breakdown and propose corrective actions.

        A component is flagged when its score falls below threshold.
        Priorities: most impactful corrections first.
        """
        actions = []

        # Check each component
        if breakdown.semantic_score < threshold:
            actions.extend(
                CorrectionStrategies._semantic_corrections(breakdown))

        if breakdown.spatial_score < threshold:
            actions.extend(
                CorrectionStrategies._spatial_corrections(breakdown))

        if breakdown.physical_score < threshold:
            actions.extend(
                CorrectionStrategies._physical_corrections(breakdown))

        if breakdown.visual_score < threshold:
            actions.extend(
                CorrectionStrategies._visual_corrections(breakdown))

        if breakdown.human_score < threshold:
            actions.extend(
                CorrectionStrategies._realism_corrections(breakdown))

        # If no specific issues, try general quality improvements
        if not actions:
            actions.append(CorrectionAction(
                action_type="vary_seed",
                target="generation",
                description="No specific issues detected; trying seed variation",
                parameters={"seed_delta": 1},
            ))

        return actions

    @staticmethod
    def _semantic_corrections(breakdown: ScoreBreakdown) -> List[CorrectionAction]:
        """Corrections for low semantic alignment."""
        actions = []
        details = breakdown.semantic_details

        # Missing objects
        missing = details.get("missing_objects", [])
        for obj in missing[:3]:  # Max 3 boost actions
            actions.append(CorrectionAction(
                action_type="boost_object_prompt",
                target=obj,
                description=f"Boost '{obj}' weight in prompt (missing from generation)",
                parameters={"weight_increase": 0.3, "object_name": obj},
            ))

        # Low CLIP similarity
        clip_sim = details.get("clip_similarity", 1.0)
        if clip_sim < 0.25:
            actions.append(CorrectionAction(
                action_type="simplify_prompt",
                target="prompt",
                description="Simplify prompt to improve CLIP alignment",
                parameters={"max_objects": 5},
            ))

        return actions

    @staticmethod
    def _spatial_corrections(breakdown: ScoreBreakdown) -> List[CorrectionAction]:
        """Corrections for spatial consistency issues."""
        actions = []
        details = breakdown.spatial_details

        # Relationship violations
        violations = details.get("relationship_violations", [])
        for v in violations[:2]:
            actions.append(CorrectionAction(
                action_type="fix_spatial_relation",
                target=str(v),
                description=f"Adjust positions to satisfy spatial relation: {v}",
                parameters={"violation": v},
            ))

        # Interpenetration
        overlap_score = details.get("overlap_violation", 0)
        if overlap_score > 0.3:
            actions.append(CorrectionAction(
                action_type="resolve_overlap",
                target="layout",
                description="Spread overlapping objects apart",
                parameters={"spread_factor": 1.3},
            ))

        return actions

    @staticmethod
    def _physical_corrections(breakdown: ScoreBreakdown) -> List[CorrectionAction]:
        """Corrections for physical plausibility issues."""
        actions = []
        details = breakdown.physical_details

        # Gravity violations (floating objects)
        gravity = details.get("gravity_score", 1.0)
        if gravity < 0.5:
            actions.append(CorrectionAction(
                action_type="fix_gravity",
                target="layout",
                description="Move floating objects toward floor/support surfaces",
                parameters={"gravity_pull": 0.1},
            ))

        # Scale issues
        scale = details.get("scale_score", 1.0)
        if scale < 0.5:
            actions.append(CorrectionAction(
                action_type="normalize_scales",
                target="layout",
                description="Normalize object scales to realistic proportions",
                parameters={},
            ))

        return actions

    @staticmethod
    def _visual_corrections(breakdown: ScoreBreakdown) -> List[CorrectionAction]:
        """Corrections for visual quality issues."""
        actions = []

        actions.append(CorrectionAction(
            action_type="increase_steps",
            target="generation",
            description="Increase inference steps for higher quality",
            parameters={"step_increase": 5},
        ))

        actions.append(CorrectionAction(
            action_type="vary_seed",
            target="generation",
            description="Try different random seed for visual variation",
            parameters={"seed_delta": 7},
        ))

        return actions

    @staticmethod
    def _realism_corrections(breakdown: ScoreBreakdown) -> List[CorrectionAction]:
        """Corrections for low realism/believability."""
        actions = []

        actions.append(CorrectionAction(
            action_type="strengthen_negative_prompt",
            target="generation",
            description="Add stronger terms to negative prompt",
            parameters={"additions": [
                "artificial", "fake", "computer generated",
                "uncanny", "plastic", "mannequin",
            ]},
        ))

        actions.append(CorrectionAction(
            action_type="adjust_guidance",
            target="generation",
            description="Fine-tune guidance scale for realism",
            parameters={"guidance_delta": -0.5},
        ))

        return actions


# ── Closed-Loop Corrector ───────────────────────────────────────────

class ClosedLoopCorrector:
    """
    Iterative closed-loop self-correction engine.

    Each iteration:
      1. Evaluate current reconstruction with full S(R)
      2. Diagnose weaknesses and propose corrections
      3. Apply corrections to hypothesis/layout/generation params
      4. Regenerate image
      5. Accept only if score improves (greedy hill-climbing)
    """

    def __init__(self,
                 unified_scorer: UnifiedScorer = None,
                 layout_estimator: SpatialLayoutEstimator = None,
                 max_iterations: int = 3,
                 improvement_threshold: float = 0.01,
                 component_threshold: float = 0.4,
                 resolution: int = 512):
        """
        Args:
            unified_scorer: UnifiedScorer for evaluation.
            layout_estimator: For converting updated hypotheses to layouts.
            max_iterations: Maximum correction iterations.
            improvement_threshold: Minimum score improvement to continue.
            component_threshold: Score below which a component triggers correction.
            resolution: Image resolution.
        """
        self.scorer = unified_scorer
        self.layout_estimator = layout_estimator or SpatialLayoutEstimator(resolution)
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.component_threshold = component_threshold

    def correct(self,
                image: Image.Image,
                prompt: str,
                semantics: SceneSemantics,
                scene_graph: SceneGraph,
                hypothesis: SceneHypothesis,
                layout: SpatialLayout,
                generate_fn=None,
                initial_breakdown: ScoreBreakdown = None) -> CorrectionResult:
        """
        Run the closed-loop correction process.

        Args:
            image: Current reconstruction image.
            prompt: Current generation prompt.
            semantics: Scene semantics.
            scene_graph: Scene graph.
            hypothesis: Current hypothesis.
            layout: Current layout.
            generate_fn: Callable(prompt, hypothesis, layout) → Image.
                         If None, only layout corrections are applied (no regeneration).
            initial_breakdown: Pre-computed score breakdown (optional).

        Returns:
            CorrectionResult with improvement history.
        """
        logger.info("=" * 50)
        logger.info("CLOSED-LOOP SELF-CORRECTION")
        logger.info(f"  Max iterations: {self.max_iterations}")
        logger.info("=" * 50)

        t_start = time.time()
        result = CorrectionResult()

        # Get initial score
        if initial_breakdown is not None:
            current_breakdown = initial_breakdown
        else:
            current_breakdown = self.scorer.score(
                image, prompt, semantics, scene_graph, hypothesis, layout,
                skip_multiview=True)

        result.initial_score = current_breakdown.total_score
        current_score = current_breakdown.total_score
        current_hypothesis = copy.deepcopy(hypothesis)
        current_layout = layout
        current_prompt = prompt
        current_image = image

        # Track generation parameter modifications
        gen_modifications = {
            "seed_offset": 0,
            "extra_steps": 0,
            "guidance_delta": 0.0,
            "negative_prompt_additions": [],
            "prompt_weight_boosts": {},
        }

        logger.info(f"  Initial score: {current_score:.4f}")

        for iteration in range(self.max_iterations):
            logger.info(f"\n  --- Correction Iteration {iteration + 1} ---")

            # 1. Diagnose
            actions = CorrectionStrategies.diagnose(
                current_breakdown, threshold=self.component_threshold)

            if not actions:
                logger.info("  No corrections needed, stopping")
                result.converged = True
                break

            logger.info(f"  Proposed {len(actions)} corrections:")
            for a in actions:
                logger.info(f"    - [{a.action_type}] {a.description}")

            # 2. Apply corrections
            corrected_hypothesis, prompt_mods = self._apply_corrections(
                current_hypothesis, actions, gen_modifications)

            # Update generation modifications
            gen_modifications.update(prompt_mods)

            # 3. Re-estimate layout with corrected hypothesis
            corrected_layout = self.layout_estimator.estimate(
                corrected_hypothesis, semantics.scene_type)

            # 4. Regenerate if generation function provided
            if generate_fn is not None:
                try:
                    corrected_prompt = self._build_corrected_prompt(
                        current_prompt, gen_modifications)
                    corrected_image = generate_fn(
                        corrected_prompt, corrected_hypothesis, corrected_layout)
                    result.total_regenerations += 1
                except Exception as e:
                    logger.warning(f"  Regeneration failed: {e}")
                    corrected_image = current_image
                    corrected_prompt = current_prompt
            else:
                corrected_image = current_image
                corrected_prompt = current_prompt

            # 5. Evaluate corrected reconstruction
            new_breakdown = self.scorer.score(
                corrected_image, corrected_prompt, semantics, scene_graph,
                corrected_hypothesis, corrected_layout,
                skip_multiview=True)

            new_score = new_breakdown.total_score

            # 6. Accept/Reject
            improvement = new_score - current_score
            accepted = improvement > 0

            iter_record = CorrectionIteration(
                iteration=iteration + 1,
                score_before=current_score,
                score_after=new_score,
                actions_applied=actions,
                accepted=accepted,
                breakdown_before=current_breakdown.to_dict(),
                breakdown_after=new_breakdown.to_dict(),
            )
            result.iterations.append(iter_record)
            result.total_corrections += len(actions)

            if accepted:
                logger.info(f"  ACCEPTED: {current_score:.4f} → {new_score:.4f} "
                             f"(+{improvement:.4f})")
                current_hypothesis = corrected_hypothesis
                current_layout = corrected_layout
                current_breakdown = new_breakdown
                current_score = new_score
                current_image = corrected_image
                current_prompt = corrected_prompt
            else:
                logger.info(f"  REJECTED: {current_score:.4f} → {new_score:.4f} "
                             f"({improvement:+.4f})")

            # Early stopping if improvement is negligible
            if abs(improvement) < self.improvement_threshold:
                logger.info(f"  Converged (improvement {improvement:.4f} < "
                             f"threshold {self.improvement_threshold})")
                result.converged = True
                break

        result.final_score = current_score
        result.improvement = current_score - result.initial_score
        result.final_hypothesis = current_hypothesis
        result.final_layout = current_layout
        result.final_prompt_modifications = gen_modifications
        result.total_time_s = time.time() - t_start

        logger.info(f"\n  Correction complete: {result.initial_score:.4f} → "
                     f"{result.final_score:.4f} "
                     f"(+{result.improvement:.4f}, "
                     f"{len(result.iterations)} iterations, "
                     f"{result.total_time_s:.1f}s)")

        return result

    def _apply_corrections(self, hypothesis: SceneHypothesis,
                            actions: List[CorrectionAction],
                            gen_mods: Dict) -> Tuple[SceneHypothesis, Dict]:
        """
        Apply correction actions to hypothesis and generation parameters.

        Returns:
            Tuple of (corrected_hypothesis, generation_param_updates).
        """
        corrected = copy.deepcopy(hypothesis)
        param_updates = {}

        for action in actions:
            if action.action_type == "boost_object_prompt":
                obj_name = action.parameters.get("object_name", "")
                boost = action.parameters.get("weight_increase", 0.2)
                current = gen_mods.get("prompt_weight_boosts", {})
                current[obj_name] = current.get(obj_name, 0) + boost
                param_updates["prompt_weight_boosts"] = current

            elif action.action_type == "fix_gravity":
                pull = action.parameters.get("gravity_pull", 0.1)
                for p in corrected.placements:
                    # Pull objects slightly toward the bottom (higher y)
                    if p.y < 0.6:
                        p.y = min(0.95, p.y + pull)

            elif action.action_type == "resolve_overlap":
                spread = action.parameters.get("spread_factor", 1.2)
                self._spread_objects(corrected.placements, spread)

            elif action.action_type == "normalize_scales":
                for p in corrected.placements:
                    p.scale = np.clip(p.scale, 0.7, 1.3)

            elif action.action_type == "vary_seed":
                delta = action.parameters.get("seed_delta", 1)
                param_updates["seed_offset"] = gen_mods.get("seed_offset", 0) + delta

            elif action.action_type == "increase_steps":
                increase = action.parameters.get("step_increase", 5)
                param_updates["extra_steps"] = gen_mods.get("extra_steps", 0) + increase

            elif action.action_type == "adjust_guidance":
                delta = action.parameters.get("guidance_delta", 0)
                param_updates["guidance_delta"] = gen_mods.get("guidance_delta", 0) + delta

            elif action.action_type == "strengthen_negative_prompt":
                additions = action.parameters.get("additions", [])
                current_adds = gen_mods.get("negative_prompt_additions", [])
                current_adds.extend(additions)
                param_updates["negative_prompt_additions"] = current_adds

            elif action.action_type == "simplify_prompt":
                param_updates["simplify_prompt"] = True
                param_updates["max_objects"] = action.parameters.get("max_objects", 5)

            elif action.action_type == "fix_spatial_relation":
                # Try to fix by adjusting positions based on the violation
                violation = action.parameters.get("violation", {})
                if isinstance(violation, dict):
                    self._fix_relation(corrected.placements, violation)

        return corrected, param_updates

    def _spread_objects(self, placements: List[ObjectPlacement],
                        factor: float):
        """Spread overlapping objects apart from centroid."""
        if len(placements) < 2:
            return

        # Compute centroid
        cx = np.mean([p.x for p in placements])
        cy = np.mean([p.y for p in placements])

        for p in placements:
            dx = p.x - cx
            dy = p.y - cy
            p.x = float(np.clip(cx + dx * factor, 0.05, 0.95))
            p.y = float(np.clip(cy + dy * factor, 0.05, 0.95))

    def _fix_relation(self, placements: List[ObjectPlacement],
                      violation: dict):
        """Attempt to fix a spatial relation violation."""
        subj = violation.get("subject", "")
        obj = violation.get("object", "")
        relation = violation.get("relation", "")

        subj_p = next((p for p in placements if p.name == subj), None)
        obj_p = next((p for p in placements if p.name == obj), None)

        if subj_p is None or obj_p is None:
            return

        # Apply simple spatial fixes
        if relation in ("on", "on top of", "above", "over"):
            subj_p.y = obj_p.y - 0.1
        elif relation in ("under", "underneath", "beneath", "below"):
            subj_p.y = obj_p.y + 0.1
        elif relation in ("left", "left of"):
            subj_p.x = obj_p.x - 0.15
        elif relation in ("right", "right of"):
            subj_p.x = obj_p.x + 0.15
        elif relation in ("near", "next to", "beside", "close to"):
            # Move closer
            dx = obj_p.x - subj_p.x
            dy = obj_p.y - subj_p.y
            dist = max(np.sqrt(dx**2 + dy**2), 0.01)
            if dist > 0.25:
                subj_p.x = float(np.clip(obj_p.x - 0.12 * dx / dist, 0.05, 0.95))
                subj_p.y = float(np.clip(obj_p.y - 0.12 * dy / dist, 0.05, 0.95))

    def _build_corrected_prompt(self, base_prompt: str,
                                 gen_mods: Dict) -> str:
        """Apply prompt modifications from corrections."""
        prompt = base_prompt

        # Apply weight boosts
        boosts = gen_mods.get("prompt_weight_boosts", {})
        for obj_name, boost in boosts.items():
            # Increase weight of mentioned objects
            new_weight = min(1.8, 1.3 + boost)
            old_pattern = f"({obj_name}:1.3)"
            new_pattern = f"({obj_name}:{new_weight:.1f})"
            prompt = prompt.replace(old_pattern, new_pattern)

            # If object not in prompt at all, add it
            if obj_name not in prompt:
                prompt = prompt.rstrip(", ") + f", ({obj_name}:{new_weight:.1f})"

        return prompt
