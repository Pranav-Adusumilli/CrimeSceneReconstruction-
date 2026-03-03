"""
Weighted Energy Optimization Engine
======================================

Optimizes the reconstruction objective S(R) over the combinatorial
layout configuration space using two complementary search strategies:

  1. **Simulated Annealing (SA)**: Metropolis-Hastings MCMC over layout
     perturbations with exponential cooling schedule.
  2. **Evolutionary Search (ES)**: μ+λ evolution strategy with crossover
     and mutation operators on object placements.

The energy function is E(R) = -S(R) where S(R) is the unified
multi-objective score from UnifiedScorer.

For the inner optimization loop, only layout-dependent components
(spatial, physical, probabilistic) are evaluated via score_layout_only()
to amortize generation cost. Full image-based scoring is reserved
for selecting among the top-K final candidates.

Designed for RTX 3060 6GB: no GPU required during layout optimization;
image generation occurs only for top candidates at the end.
"""

import copy
import logging
import math
import time
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable

from src.stages.stage3_scene_graph import SceneGraph
from src.stages.stage4_hypothesis_generation import (
    SceneHypothesis, ObjectPlacement, HypothesisGenerator,
)
from src.stages.stage5_spatial_layout import SpatialLayout, SpatialLayoutEstimator

logger = logging.getLogger(__name__)


# ── Optimization Result ─────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Complete record of an optimization run."""
    best_hypothesis: SceneHypothesis = None
    best_layout: SpatialLayout = None
    best_energy: float = float("inf")
    best_score: float = 0.0
    energy_trajectory: List[float] = field(default_factory=list)
    score_trajectory: List[float] = field(default_factory=list)
    acceptance_trajectory: List[bool] = field(default_factory=list)
    temperature_trajectory: List[float] = field(default_factory=list)
    total_iterations: int = 0
    total_time_s: float = 0.0
    method: str = ""
    top_k_candidates: List[Tuple[float, SceneHypothesis]] = field(default_factory=list)
    convergence_iteration: int = -1  # iteration where best was found

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "best_energy": round(self.best_energy, 6),
            "best_score": round(self.best_score, 6),
            "total_iterations": self.total_iterations,
            "total_time_s": round(self.total_time_s, 2),
            "convergence_iteration": self.convergence_iteration,
            "num_top_k": len(self.top_k_candidates),
            "trajectory_length": len(self.energy_trajectory),
        }


# ── Perturbation Operators ──────────────────────────────────────────

def _perturb_position(placement: ObjectPlacement, sigma: float,
                      rng: np.random.RandomState) -> ObjectPlacement:
    """Perturb x, y position with Gaussian noise."""
    return ObjectPlacement(
        name=placement.name,
        x=float(np.clip(placement.x + rng.normal(0, sigma), 0.05, 0.95)),
        y=float(np.clip(placement.y + rng.normal(0, sigma * 0.7), 0.05, 0.95)),
        depth=placement.depth,
        scale=placement.scale,
        attributes=placement.attributes,
    )


def _perturb_depth(placement: ObjectPlacement, sigma: float,
                   rng: np.random.RandomState) -> ObjectPlacement:
    """Perturb depth value."""
    return ObjectPlacement(
        name=placement.name,
        x=placement.x,
        y=placement.y,
        depth=float(np.clip(placement.depth + rng.normal(0, sigma), 0.0, 1.0)),
        scale=placement.scale,
        attributes=placement.attributes,
    )


def _perturb_scale(placement: ObjectPlacement, sigma: float,
                   rng: np.random.RandomState) -> ObjectPlacement:
    """Perturb scale factor."""
    return ObjectPlacement(
        name=placement.name,
        x=placement.x,
        y=placement.y,
        depth=placement.depth,
        scale=float(np.clip(placement.scale + rng.normal(0, sigma * 0.5), 0.5, 1.5)),
        attributes=placement.attributes,
    )


def _swap_positions(placements: List[ObjectPlacement],
                    rng: np.random.RandomState) -> List[ObjectPlacement]:
    """Swap positions of two randomly selected objects."""
    if len(placements) < 2:
        return placements
    result = [copy.deepcopy(p) for p in placements]
    i, j = rng.choice(len(result), size=2, replace=False)
    result[i].x, result[j].x = result[j].x, result[i].x
    result[i].y, result[j].y = result[j].y, result[i].y
    return result


# ── Simulated Annealing ─────────────────────────────────────────────

class SimulatedAnnealing:
    """
    Simulated annealing optimizer for layout configuration.

    Temperature schedule: T(t) = T_0 * α^t  (geometric cooling)

    Proposal distribution: Gaussian perturbations on (x, y, depth, scale)
    with occasional swap moves.

    Acceptance: Metropolis criterion P(accept) = min(1, exp(-ΔE / T))
    """

    def __init__(self,
                 T_initial: float = 1.0,
                 T_final: float = 0.01,
                 cooling_rate: float = 0.995,
                 max_iterations: int = 500,
                 position_sigma: float = 0.05,
                 depth_sigma: float = 0.08,
                 scale_sigma: float = 0.06,
                 swap_probability: float = 0.15,
                 seed: int = 42):
        self.T_initial = T_initial
        self.T_final = T_final
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.position_sigma = position_sigma
        self.depth_sigma = depth_sigma
        self.scale_sigma = scale_sigma
        self.swap_probability = swap_probability
        self.rng = np.random.RandomState(seed)

    def propose(self, hypothesis: SceneHypothesis,
                iteration: int) -> SceneHypothesis:
        """Generate a neighbor hypothesis via perturbation."""
        new_placements = [copy.deepcopy(p) for p in hypothesis.placements]

        # Adaptive sigma: shrink as we cool
        adaptive_factor = max(0.3, 1.0 - iteration / self.max_iterations)

        if self.rng.random() < self.swap_probability and len(new_placements) >= 2:
            # Swap move
            new_placements = _swap_positions(new_placements, self.rng)
        else:
            # Perturbation move: randomly select one object
            idx = self.rng.randint(len(new_placements))
            move_type = self.rng.choice(["position", "depth", "scale"],
                                        p=[0.5, 0.3, 0.2])

            sigma = adaptive_factor
            if move_type == "position":
                new_placements[idx] = _perturb_position(
                    new_placements[idx], self.position_sigma * sigma, self.rng)
            elif move_type == "depth":
                new_placements[idx] = _perturb_depth(
                    new_placements[idx], self.depth_sigma * sigma, self.rng)
            else:
                new_placements[idx] = _perturb_scale(
                    new_placements[idx], self.scale_sigma * sigma, self.rng)

        return SceneHypothesis(
            hypothesis_id=hypothesis.hypothesis_id,
            confidence=hypothesis.confidence,
            placements=new_placements,
            description=f"SA-optimized (iter {iteration})",
            prompt_modifier=hypothesis.prompt_modifier,
        )

    def accept(self, delta_energy: float, temperature: float) -> bool:
        """Metropolis acceptance criterion."""
        if delta_energy < 0:
            return True  # Always accept improvements
        if temperature <= 0:
            return False
        prob = math.exp(-delta_energy / max(temperature, 1e-10))
        return self.rng.random() < prob


# ── Evolutionary Search ──────────────────────────────────────────────

class EvolutionarySearch:
    """
    (μ+λ) Evolution Strategy for layout optimization.

    Population of μ parents produces λ offspring via crossover + mutation.
    Best μ from combined (parents ∪ offspring) survive.
    """

    def __init__(self,
                 population_size: int = 12,
                 offspring_count: int = 18,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.5,
                 tournament_size: int = 3,
                 max_generations: int = 50,
                 position_sigma: float = 0.05,
                 seed: int = 42):
        self.mu = population_size
        self.lam = offspring_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        self.position_sigma = position_sigma
        self.rng = np.random.RandomState(seed)

    def initialize_population(self, base_hypothesis: SceneHypothesis,
                               scene_graph: SceneGraph) -> List[SceneHypothesis]:
        """Create initial population from base hypothesis + random variants."""
        population = [copy.deepcopy(base_hypothesis)]

        # Generate diverse initial hypotheses
        hyp_gen = HypothesisGenerator(
            num_hypotheses=min(self.mu - 1, 5),
            seed=self.rng.randint(100000),
        )
        variants = hyp_gen.generate(scene_graph)
        for v in variants:
            population.append(v)

        # Fill remaining with random perturbations of base
        while len(population) < self.mu:
            perturbed = self._mutate(copy.deepcopy(base_hypothesis))
            population.append(perturbed)

        return population[:self.mu]

    def _mutate(self, hypothesis: SceneHypothesis) -> SceneHypothesis:
        """Apply random mutation to a hypothesis."""
        new_placements = []
        for p in hypothesis.placements:
            if self.rng.random() < self.mutation_rate:
                move = self.rng.choice(["position", "depth", "scale"])
                if move == "position":
                    p = _perturb_position(p, self.position_sigma, self.rng)
                elif move == "depth":
                    p = _perturb_depth(p, self.position_sigma, self.rng)
                else:
                    p = _perturb_scale(p, self.position_sigma, self.rng)
            new_placements.append(p)

        return SceneHypothesis(
            hypothesis_id=hypothesis.hypothesis_id,
            confidence=hypothesis.confidence,
            placements=new_placements,
            description="ES-mutated",
            prompt_modifier=hypothesis.prompt_modifier,
        )

    def _crossover(self, parent_a: SceneHypothesis,
                   parent_b: SceneHypothesis) -> SceneHypothesis:
        """Uniform crossover: each object placement selected from either parent."""
        if len(parent_a.placements) != len(parent_b.placements):
            return copy.deepcopy(parent_a)

        new_placements = []
        for pa, pb in zip(parent_a.placements, parent_b.placements):
            chosen = copy.deepcopy(pa if self.rng.random() < 0.5 else pb)
            # Ensure names match (crossover by position in list)
            chosen.name = pa.name
            chosen.attributes = pa.attributes
            new_placements.append(chosen)

        return SceneHypothesis(
            hypothesis_id=parent_a.hypothesis_id,
            confidence=(parent_a.confidence + parent_b.confidence) / 2,
            placements=new_placements,
            description="ES-crossover",
            prompt_modifier=parent_a.prompt_modifier,
        )

    def tournament_select(self, population: List[SceneHypothesis],
                          fitness: List[float]) -> SceneHypothesis:
        """Tournament selection."""
        indices = self.rng.choice(
            len(population), size=min(self.tournament_size, len(population)),
            replace=False)
        best_idx = indices[np.argmax([fitness[i] for i in indices])]
        return population[best_idx]

    def produce_offspring(self, population: List[SceneHypothesis],
                          fitness: List[float]) -> List[SceneHypothesis]:
        """Generate λ offspring via crossover + mutation."""
        offspring = []
        for _ in range(self.lam):
            if self.rng.random() < self.crossover_rate:
                p1 = self.tournament_select(population, fitness)
                p2 = self.tournament_select(population, fitness)
                child = self._crossover(p1, p2)
            else:
                parent = self.tournament_select(population, fitness)
                child = copy.deepcopy(parent)
            child = self._mutate(child)
            offspring.append(child)
        return offspring


# ── Energy Optimizer (Main Interface) ────────────────────────────────

class EnergyOptimizer:
    """
    Unified energy optimization engine.

    Supports:
      - Simulated annealing over layout perturbations
      - Evolutionary (μ+λ) search
      - Hybrid: SA warmup → ES refinement

    Uses score_layout_only() for fast inner-loop scoring (no image generation).
    Image-based scoring only for final top-K candidate selection.
    """

    def __init__(self,
                 unified_scorer=None,
                 layout_estimator: SpatialLayoutEstimator = None,
                 method: str = "simulated_annealing",
                 sa_config: dict = None,
                 es_config: dict = None,
                 top_k: int = 3,
                 resolution: int = 512,
                 seed: int = 42):
        """
        Args:
            unified_scorer: UnifiedScorer instance (for score_layout_only).
            layout_estimator: SpatialLayoutEstimator for hypothesis→layout.
            method: "simulated_annealing", "evolutionary", or "hybrid".
            sa_config: Dict of SimulatedAnnealing kwargs overrides.
            es_config: Dict of EvolutionarySearch kwargs overrides.
            top_k: Number of top candidates to preserve.
            resolution: Image resolution for layout estimation.
            seed: Random seed.
        """
        self.scorer = unified_scorer
        self.layout_estimator = layout_estimator or SpatialLayoutEstimator(resolution)
        self.method = method
        self.top_k = top_k
        self.seed = seed

        sa_kwargs = {"seed": seed}
        if sa_config:
            sa_kwargs.update(sa_config)
        self.sa = SimulatedAnnealing(**sa_kwargs)

        es_kwargs = {"seed": seed}
        if es_config:
            es_kwargs.update(es_config)
        self.es = EvolutionarySearch(**es_kwargs)

    def optimize(self, initial_hypothesis: SceneHypothesis,
                 scene_graph: SceneGraph,
                 scene_type: str = "unknown") -> OptimizationResult:
        """
        Run energy optimization to find the best layout configuration.

        Args:
            initial_hypothesis: Starting hypothesis from Stage 4.
            scene_graph: Scene graph for scoring context.
            scene_type: Scene type label for layout estimation.

        Returns:
            OptimizationResult with best hypothesis, trajectory, etc.
        """
        logger.info("=" * 50)
        logger.info("OPTIMIZATION ENGINE")
        logger.info(f"  Method: {self.method}")
        logger.info("=" * 50)

        t_start = time.time()

        if self.method == "simulated_annealing":
            result = self._run_sa(initial_hypothesis, scene_graph, scene_type)
        elif self.method == "evolutionary":
            result = self._run_es(initial_hypothesis, scene_graph, scene_type)
        elif self.method == "hybrid":
            result = self._run_hybrid(initial_hypothesis, scene_graph, scene_type)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        result.total_time_s = time.time() - t_start
        result.method = self.method

        logger.info(f"  Optimization complete: best_score={result.best_score:.4f} "
                     f"({result.total_iterations} iters, {result.total_time_s:.1f}s)")

        return result

    def _evaluate(self, hypothesis: SceneHypothesis,
                  scene_graph: SceneGraph,
                  scene_type: str) -> Tuple[float, SpatialLayout]:
        """Evaluate a hypothesis using fast layout-only scoring."""
        layout = self.layout_estimator.estimate(hypothesis, scene_type)
        score = self.scorer.score_layout_only(scene_graph, hypothesis, layout)
        return score, layout

    def _run_sa(self, initial: SceneHypothesis,
                scene_graph: SceneGraph,
                scene_type: str) -> OptimizationResult:
        """Run simulated annealing optimization."""
        result = OptimizationResult()

        current = copy.deepcopy(initial)
        current_score, current_layout = self._evaluate(current, scene_graph, scene_type)
        current_energy = -current_score

        best = copy.deepcopy(current)
        best_layout = current_layout
        best_energy = current_energy
        best_score = current_score

        # Top-K tracker (min-heap by score: keep highest scores)
        top_k: List[Tuple[float, SceneHypothesis]] = [(current_score, copy.deepcopy(current))]

        T = self.sa.T_initial
        no_improve_count = 0
        max_no_improve = 100  # Early stopping patience

        for iteration in range(self.sa.max_iterations):
            if T < self.sa.T_final:
                break

            # Propose neighbor
            candidate = self.sa.propose(current, iteration)
            cand_score, cand_layout = self._evaluate(candidate, scene_graph, scene_type)
            cand_energy = -cand_score

            delta_energy = cand_energy - current_energy
            accepted = self.sa.accept(delta_energy, T)

            # Record trajectory
            result.energy_trajectory.append(current_energy)
            result.score_trajectory.append(current_score)
            result.acceptance_trajectory.append(accepted)
            result.temperature_trajectory.append(T)

            if accepted:
                current = candidate
                current_score = cand_score
                current_layout = cand_layout
                current_energy = cand_energy

            # Update global best
            if cand_score > best_score:
                best = copy.deepcopy(candidate)
                best_layout = cand_layout
                best_energy = cand_energy
                best_score = cand_score
                result.convergence_iteration = iteration
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Update top-K
            self._update_top_k(top_k, cand_score, candidate)

            # Cool
            T *= self.sa.cooling_rate

            # Progress logging
            if iteration % 50 == 0:
                logger.info(
                    f"  SA iter {iteration}: T={T:.4f} "
                    f"score={current_score:.4f} best={best_score:.4f} "
                    f"accepted={sum(result.acceptance_trajectory[-50:])}/50"
                )

            # Early stopping
            if no_improve_count >= max_no_improve:
                logger.info(f"  SA early stop at iter {iteration} (no improvement for {max_no_improve} iters)")
                break

        result.best_hypothesis = best
        result.best_layout = best_layout
        result.best_energy = best_energy
        result.best_score = best_score
        result.total_iterations = len(result.energy_trajectory)
        result.top_k_candidates = sorted(top_k, key=lambda x: x[0], reverse=True)[:self.top_k]

        return result

    def _run_es(self, initial: SceneHypothesis,
                scene_graph: SceneGraph,
                scene_type: str) -> OptimizationResult:
        """Run evolutionary search optimization."""
        result = OptimizationResult()

        # Initialize population
        population = self.es.initialize_population(initial, scene_graph)
        fitness = []
        layouts = []
        for h in population:
            s, l = self._evaluate(h, scene_graph, scene_type)
            fitness.append(s)
            layouts.append(l)

        best_idx = np.argmax(fitness)
        best = copy.deepcopy(population[best_idx])
        best_layout = layouts[best_idx]
        best_score = fitness[best_idx]
        best_energy = -best_score

        top_k: List[Tuple[float, SceneHypothesis]] = []
        for s, h in zip(fitness, population):
            self._update_top_k(top_k, s, h)

        for generation in range(self.es.max_generations):
            # Produce offspring
            offspring = self.es.produce_offspring(population, fitness)

            # Evaluate offspring
            off_fitness = []
            off_layouts = []
            for h in offspring:
                s, l = self._evaluate(h, scene_graph, scene_type)
                off_fitness.append(s)
                off_layouts.append(l)

            # Combine parents + offspring
            combined = population + offspring
            combined_fitness = fitness + off_fitness
            combined_layouts = layouts + off_layouts

            # Select top-μ
            sorted_indices = np.argsort(combined_fitness)[::-1][:self.es.mu]
            population = [combined[i] for i in sorted_indices]
            fitness = [combined_fitness[i] for i in sorted_indices]
            layouts = [combined_layouts[i] for i in sorted_indices]

            # Update global best
            gen_best_idx = sorted_indices[0]
            gen_best_score = combined_fitness[gen_best_idx]
            if gen_best_score > best_score:
                best = copy.deepcopy(combined[gen_best_idx])
                best_layout = combined_layouts[gen_best_idx]
                best_score = gen_best_score
                best_energy = -best_score
                result.convergence_iteration = generation

            # Update top-K
            for s, h in zip(off_fitness, offspring):
                self._update_top_k(top_k, s, h)

            # Trajectory
            result.energy_trajectory.append(-fitness[0])
            result.score_trajectory.append(fitness[0])

            if generation % 10 == 0:
                logger.info(
                    f"  ES gen {generation}: best={fitness[0]:.4f} "
                    f"mean={np.mean(fitness):.4f} global_best={best_score:.4f}"
                )

        result.best_hypothesis = best
        result.best_layout = best_layout
        result.best_energy = best_energy
        result.best_score = best_score
        result.total_iterations = self.es.max_generations * (self.es.mu + self.es.lam)
        result.top_k_candidates = sorted(top_k, key=lambda x: x[0], reverse=True)[:self.top_k]

        return result

    def _run_hybrid(self, initial: SceneHypothesis,
                    scene_graph: SceneGraph,
                    scene_type: str) -> OptimizationResult:
        """
        Hybrid optimization: SA warmup → ES refinement.

        Phase 1 (SA): Broad exploration with ~60% of iteration budget.
        Phase 2 (ES): Focused exploitation seeded from SA top candidates.
        """
        logger.info("  Hybrid Phase 1: Simulated Annealing warmup")

        # Phase 1: SA with reduced iterations
        orig_sa_iters = self.sa.max_iterations
        self.sa.max_iterations = max(int(orig_sa_iters * 0.6), 100)
        sa_result = self._run_sa(initial, scene_graph, scene_type)
        self.sa.max_iterations = orig_sa_iters

        logger.info(f"  SA phase complete: best={sa_result.best_score:.4f}")
        logger.info("  Hybrid Phase 2: Evolutionary refinement")

        # Phase 2: ES seeded from SA top candidates
        orig_es_gens = self.es.max_generations
        self.es.max_generations = max(int(orig_es_gens * 0.4), 10)

        # Seed ES population from SA top-K
        es_result = self._run_es(
            sa_result.best_hypothesis, scene_graph, scene_type)
        self.es.max_generations = orig_es_gens

        # Merge results
        merged = OptimizationResult()
        merged.energy_trajectory = sa_result.energy_trajectory + es_result.energy_trajectory
        merged.score_trajectory = sa_result.score_trajectory + es_result.score_trajectory
        merged.acceptance_trajectory = sa_result.acceptance_trajectory
        merged.temperature_trajectory = sa_result.temperature_trajectory
        merged.total_iterations = sa_result.total_iterations + es_result.total_iterations

        # Take the overall best
        if es_result.best_score >= sa_result.best_score:
            merged.best_hypothesis = es_result.best_hypothesis
            merged.best_layout = es_result.best_layout
            merged.best_energy = es_result.best_energy
            merged.best_score = es_result.best_score
            merged.convergence_iteration = (
                sa_result.total_iterations + es_result.convergence_iteration)
        else:
            merged.best_hypothesis = sa_result.best_hypothesis
            merged.best_layout = sa_result.best_layout
            merged.best_energy = sa_result.best_energy
            merged.best_score = sa_result.best_score
            merged.convergence_iteration = sa_result.convergence_iteration

        # Merge top-K
        all_candidates = sa_result.top_k_candidates + es_result.top_k_candidates
        merged.top_k_candidates = sorted(
            all_candidates, key=lambda x: x[0], reverse=True)[:self.top_k]

        return merged

    def _update_top_k(self, top_k: List[Tuple[float, SceneHypothesis]],
                      score: float, hypothesis: SceneHypothesis):
        """Add candidate to top-K list if it qualifies."""
        if len(top_k) < self.top_k:
            top_k.append((score, copy.deepcopy(hypothesis)))
            top_k.sort(key=lambda x: x[0])
        elif score > top_k[0][0]:
            top_k[0] = (score, copy.deepcopy(hypothesis))
            top_k.sort(key=lambda x: x[0])
