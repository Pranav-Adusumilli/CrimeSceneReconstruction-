"""
Automatic Weight Calibration
==============================

Tunes the scoring weights w = [w₁, ..., w₇] of the unified objective
function S(R) to maximize reconstruction quality.

Two calibration strategies:

  1. **Grid Search**: Exhaustive search over a discretized weight simplex.
     Suitable for coarse initial calibration with few effective dimensions.

  2. **Bayesian Optimization (Tree-Parzen Estimator)**: Uses surrogate
     modeling to efficiently explore the continuous weight space.
     More sample-efficient for fine-tuning.

The calibration objective evaluates each weight configuration on a set
of calibration scenes, scoring the resulting reconstructions with the
full unified scorer.

For computational tractability on RTX 3060 6GB:
  - Layout-only scoring for fast inner loop
  - Optional full image-based scoring for final candidates
  - Configurable scene budget (number of calibration texts)
"""

import logging
import itertools
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable

from src.scoring.unified_scorer import UnifiedScorer, ScoringWeights, ScoreBreakdown

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of a weight calibration run."""
    best_weights: ScoringWeights = None
    best_objective: float = 0.0
    all_evaluations: List[Tuple[Dict, float]] = field(default_factory=list)
    total_evaluations: int = 0
    total_time_s: float = 0.0
    method: str = ""
    convergence_curve: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return {
            "method": self.method,
            "best_objective": round(self.best_objective, 6),
            "best_weights": asdict(self.best_weights) if self.best_weights else {},
            "total_evaluations": self.total_evaluations,
            "total_time_s": round(self.total_time_s, 2),
        }


class WeightCalibrator:
    """
    Calibrates the 7 scoring weights to maximize reconstruction quality.

    Usage:
        calibrator = WeightCalibrator(evaluation_fn=my_eval_fn)
        result = calibrator.calibrate(method="grid_search")
        best_weights = result.best_weights
    """

    # Default calibration grid: each weight sampled from {0.05, 0.10, 0.15, 0.20, 0.25}
    DEFAULT_GRID_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]

    # Weight names in order
    WEIGHT_NAMES = [
        "w_semantic", "w_spatial", "w_physical", "w_visual",
        "w_probabilistic", "w_multiview", "w_human",
    ]

    def __init__(self,
                 evaluation_fn: Callable[[ScoringWeights], float] = None,
                 seed: int = 42):
        """
        Args:
            evaluation_fn: Function that takes ScoringWeights and returns
                           a scalar quality metric (higher is better).
                           Must handle running the pipeline or layout-only scoring.
            seed: Random seed for reproducibility.
        """
        self.evaluation_fn = evaluation_fn
        self.rng = np.random.RandomState(seed)

    def calibrate(self, method: str = "grid_search",
                  grid_values: List[float] = None,
                  bayesian_iterations: int = 50,
                  max_grid_configs: int = 500) -> CalibrationResult:
        """
        Run weight calibration.

        Args:
            method: "grid_search" or "bayesian".
            grid_values: Custom grid values for grid search.
            bayesian_iterations: Number of Bayesian optimization iterations.
            max_grid_configs: Maximum number of grid configurations to evaluate.

        Returns:
            CalibrationResult with best weights and evaluation history.
        """
        logger.info("=" * 50)
        logger.info(f"WEIGHT CALIBRATION ({method})")
        logger.info("=" * 50)

        t_start = time.time()

        if method == "grid_search":
            result = self._grid_search(grid_values or self.DEFAULT_GRID_VALUES,
                                        max_configs=max_grid_configs)
        elif method == "bayesian":
            result = self._bayesian_optimization(bayesian_iterations)
        elif method == "random_search":
            result = self._random_search(bayesian_iterations)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        result.total_time_s = time.time() - t_start
        result.method = method

        logger.info(f"  Calibration complete: {result.total_evaluations} evaluations "
                     f"in {result.total_time_s:.1f}s")
        if result.best_weights:
            from dataclasses import asdict
            logger.info(f"  Best weights: {asdict(result.best_weights)}")
            logger.info(f"  Best objective: {result.best_objective:.4f}")

        return result

    def _grid_search(self, grid_values: List[float],
                     max_configs: int = 500) -> CalibrationResult:
        """
        Grid search over the weight simplex.

        Generates all combinations and evaluates each. For tractability,
        randomly samples down to max_configs if the full grid is too large.
        """
        result = CalibrationResult()

        # Generate all weight combinations
        all_combos = list(itertools.product(grid_values, repeat=7))
        logger.info(f"  Full grid: {len(all_combos)} configurations")

        # Subsample if too large
        if len(all_combos) > max_configs:
            indices = self.rng.choice(len(all_combos), size=max_configs, replace=False)
            combos = [all_combos[i] for i in indices]
            logger.info(f"  Subsampled to {max_configs} configurations")
        else:
            combos = all_combos

        best_score = -float("inf")
        best_weights = None

        for i, combo in enumerate(combos):
            weights = ScoringWeights(
                w_semantic=combo[0], w_spatial=combo[1],
                w_physical=combo[2], w_visual=combo[3],
                w_probabilistic=combo[4], w_multiview=combo[5],
                w_human=combo[6],
            )

            try:
                score = self.evaluation_fn(weights)
            except Exception as e:
                logger.warning(f"  Evaluation failed for config {i}: {e}")
                score = 0.0

            weight_dict = {n: combo[j] for j, n in enumerate(self.WEIGHT_NAMES)}
            result.all_evaluations.append((weight_dict, score))
            result.convergence_curve.append(max(best_score, score) if best_score > -float("inf") else score)

            if score > best_score:
                best_score = score
                best_weights = weights

            if (i + 1) % 50 == 0:
                logger.info(f"  Grid search: {i + 1}/{len(combos)} "
                             f"best={best_score:.4f}")

        result.best_weights = best_weights
        result.best_objective = best_score
        result.total_evaluations = len(combos)

        return result

    def _random_search(self, num_iterations: int = 50) -> CalibrationResult:
        """
        Random search over the weight space with Dirichlet sampling.

        Samples weights from a Dirichlet distribution to ensure they
        represent meaningful proportions.
        """
        result = CalibrationResult()

        best_score = -float("inf")
        best_weights = None

        for i in range(num_iterations):
            # Dirichlet sampling for diverse weight configurations
            raw_weights = self.rng.dirichlet(np.ones(7) * 2.0)

            weights = ScoringWeights(
                w_semantic=float(raw_weights[0]),
                w_spatial=float(raw_weights[1]),
                w_physical=float(raw_weights[2]),
                w_visual=float(raw_weights[3]),
                w_probabilistic=float(raw_weights[4]),
                w_multiview=float(raw_weights[5]),
                w_human=float(raw_weights[6]),
            )

            try:
                score = self.evaluation_fn(weights)
            except Exception as e:
                logger.warning(f"  Random search iter {i} failed: {e}")
                score = 0.0

            weight_dict = {n: float(raw_weights[j])
                           for j, n in enumerate(self.WEIGHT_NAMES)}
            result.all_evaluations.append((weight_dict, score))
            result.convergence_curve.append(
                max(best_score, score) if best_score > -float("inf") else score)

            if score > best_score:
                best_score = score
                best_weights = weights

            if (i + 1) % 10 == 0:
                logger.info(f"  Random search: {i + 1}/{num_iterations} "
                             f"best={best_score:.4f}")

        result.best_weights = best_weights
        result.best_objective = best_score
        result.total_evaluations = num_iterations

        return result

    def _bayesian_optimization(self, num_iterations: int = 50) -> CalibrationResult:
        """
        Tree-Parzen Estimator (TPE) style Bayesian optimization.

        Implements a simplified TPE:
          1. Initial random exploration phase (20% of budget).
          2. Split evaluations into good/bad using quantile γ.
          3. Sample from good distribution with local perturbation.
        """
        result = CalibrationResult()

        # Phase 1: Random exploration
        n_explore = max(int(num_iterations * 0.2), 5)
        n_exploit = num_iterations - n_explore

        evaluations: List[Tuple[np.ndarray, float]] = []
        best_score = -float("inf")
        best_weights = None

        # Exploration phase
        logger.info(f"  Bayesian Phase 1: {n_explore} random explorations")
        for i in range(n_explore):
            raw = self.rng.dirichlet(np.ones(7) * 2.0)
            weights = self._array_to_weights(raw)
            score = self._safe_evaluate(weights)
            evaluations.append((raw, score))

            if score > best_score:
                best_score = score
                best_weights = weights

            result.convergence_curve.append(best_score)

        # Exploitation phase using TPE-like approach
        logger.info(f"  Bayesian Phase 2: {n_exploit} TPE exploitations")
        gamma = 0.25  # Top 25% are "good"

        for i in range(n_exploit):
            if len(evaluations) < 3:
                raw = self.rng.dirichlet(np.ones(7) * 2.0)
            else:
                # Split into good/bad
                scores = np.array([e[1] for e in evaluations])
                threshold = np.percentile(scores, (1 - gamma) * 100)
                good = [e[0] for e in evaluations if e[1] >= threshold]

                if len(good) == 0:
                    raw = self.rng.dirichlet(np.ones(7) * 2.0)
                else:
                    # Sample from good distribution with perturbation
                    base_idx = self.rng.randint(len(good))
                    base = good[base_idx].copy()

                    # Local Gaussian perturbation (adaptive bandwidth)
                    bandwidth = max(0.02, 0.1 * (1 - i / max(n_exploit, 1)))
                    perturbation = self.rng.normal(0, bandwidth, size=7)
                    raw = np.clip(base + perturbation, 0.01, 1.0)
                    raw = raw / raw.sum()  # Re-normalize

            weights = self._array_to_weights(raw)
            score = self._safe_evaluate(weights)
            evaluations.append((raw, score))

            if score > best_score:
                best_score = score
                best_weights = weights

            result.convergence_curve.append(best_score)

            if (n_explore + i + 1) % 10 == 0:
                logger.info(f"  Bayesian: {n_explore + i + 1}/{num_iterations} "
                             f"best={best_score:.4f}")

        # Record all evaluations
        for raw, score in evaluations:
            weight_dict = {n: float(raw[j]) for j, n in enumerate(self.WEIGHT_NAMES)}
            result.all_evaluations.append((weight_dict, score))

        result.best_weights = best_weights
        result.best_objective = best_score
        result.total_evaluations = num_iterations

        return result

    def _array_to_weights(self, arr: np.ndarray) -> ScoringWeights:
        """Convert numpy array to ScoringWeights."""
        return ScoringWeights(
            w_semantic=float(arr[0]),
            w_spatial=float(arr[1]),
            w_physical=float(arr[2]),
            w_visual=float(arr[3]),
            w_probabilistic=float(arr[4]),
            w_multiview=float(arr[5]),
            w_human=float(arr[6]),
        )

    def _safe_evaluate(self, weights: ScoringWeights) -> float:
        """Evaluate with error handling."""
        try:
            return self.evaluation_fn(weights)
        except Exception as e:
            logger.warning(f"  Evaluation error: {e}")
            return 0.0
