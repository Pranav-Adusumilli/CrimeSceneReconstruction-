"""
Research-Grade Structured Logger
==================================

Provides comprehensive structured logging for reproducible research:

  - Score breakdowns per reconstruction (component-level S(R))
  - Scoring weight configurations
  - Optimization trajectories (energy convergence curves)
  - Correction iteration histories
  - Hypothesis likelihoods and rankings
  - Timing and resource utilization
  - Experiment-level aggregation

All data is stored in structured JSON format for:
  - Programmatic analysis
  - Plot generation (matplotlib / seaborn)
  - Cross-experiment comparison
  - Paper figure reproduction

Storage layout:
  logs/
    {experiment_id}/
      config.json          # Full configuration snapshot
      scores.jsonl         # Per-reconstruction score records
      optimization.jsonl   # Optimization trajectory records
      corrections.jsonl    # Correction iteration records
      summary.json         # Aggregate statistics
      plots/               # Generated visualizations
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScoreRecord:
    """A single score recording."""
    timestamp: str = ""
    scene_id: str = ""
    hypothesis_id: int = 0
    config_name: str = ""
    total_score: float = 0.0
    component_scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    energy: float = 0.0
    computation_time_s: float = 0.0


@dataclass
class OptimizationRecord:
    """A single optimization run recording."""
    timestamp: str = ""
    scene_id: str = ""
    method: str = ""
    initial_score: float = 0.0
    final_score: float = 0.0
    improvement: float = 0.0
    total_iterations: int = 0
    convergence_iteration: int = -1
    energy_trajectory: List[float] = field(default_factory=list)
    score_trajectory: List[float] = field(default_factory=list)
    temperature_trajectory: List[float] = field(default_factory=list)
    acceptance_rate: float = 0.0
    total_time_s: float = 0.0


@dataclass
class CorrectionRecord:
    """A single correction run recording."""
    timestamp: str = ""
    scene_id: str = ""
    initial_score: float = 0.0
    final_score: float = 0.0
    improvement: float = 0.0
    iterations: int = 0
    corrections_applied: int = 0
    regenerations: int = 0
    converged: bool = False
    total_time_s: float = 0.0
    iteration_details: List[Dict] = field(default_factory=list)


class ResearchLogger:
    """
    Structured logger for research-grade reproducible experiments.

    Stores all experimental data in machine-readable JSON/JSONL format
    with automatic experiment ID generation and aggregate statistics.

    Usage:
        rlog = ResearchLogger(base_dir="outputs/logs")
        rlog.start_experiment("exp_001", config={...})
        rlog.log_score(scene_id, breakdown)
        rlog.log_optimization(scene_id, opt_result)
        rlog.log_correction(scene_id, corr_result)
        rlog.finalize()
    """

    def __init__(self, base_dir: str = "outputs/research_logs"):
        self.base_dir = Path(base_dir)
        self.experiment_id = None
        self.experiment_dir = None
        self.config = {}
        self.start_time = None

        # In-memory buffers
        self._score_records: List[ScoreRecord] = []
        self._optimization_records: List[OptimizationRecord] = []
        self._correction_records: List[CorrectionRecord] = []
        self._custom_events: List[Dict] = []

    def start_experiment(self, experiment_id: str = None,
                          config: Dict = None):
        """
        Initialize a new experiment session.

        Args:
            experiment_id: Unique experiment identifier (auto-generated if None).
            config: Full configuration snapshot to store.
        """
        if experiment_id is None:
            experiment_id = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

        self.experiment_id = experiment_id
        self.config = config or {}
        self.start_time = time.time()

        # Create experiment directory
        self.experiment_dir = self.base_dir / experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "plots").mkdir(exist_ok=True)

        # Save config
        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "experiment_id": experiment_id,
                "start_time": datetime.now().isoformat(),
                "config": self._serialize(config),
            }, f, indent=2)

        logger.info(f"  Research logger started: {experiment_id}")
        logger.info(f"  Log directory: {self.experiment_dir}")

    def log_score(self, scene_id: str, breakdown,
                  hypothesis_id: int = 0,
                  config_name: str = ""):
        """
        Log a score breakdown for one reconstruction.

        Args:
            scene_id: Scene identifier.
            breakdown: ScoreBreakdown instance.
            hypothesis_id: Hypothesis number.
            config_name: Configuration name.
        """
        record = ScoreRecord(
            timestamp=datetime.now().isoformat(),
            scene_id=scene_id,
            hypothesis_id=hypothesis_id,
            config_name=config_name,
            total_score=breakdown.total_score,
            component_scores={
                "semantic": breakdown.semantic_score,
                "spatial": breakdown.spatial_score,
                "physical": breakdown.physical_score,
                "visual": breakdown.visual_score,
                "probabilistic": breakdown.probabilistic_score,
                "multiview": breakdown.multiview_score,
                "human": breakdown.human_score,
            },
            energy=breakdown.energy,
            computation_time_s=breakdown.computation_time_s,
        )
        self._score_records.append(record)

        # Append to JSONL file
        self._append_jsonl("scores.jsonl", asdict(record))

    def log_optimization(self, scene_id: str, opt_result):
        """
        Log an optimization run result.

        Args:
            scene_id: Scene identifier.
            opt_result: OptimizationResult instance.
        """
        # Compute acceptance rate from trajectory
        acceptances = opt_result.acceptance_trajectory
        acceptance_rate = (sum(acceptances) / max(len(acceptances), 1)
                          if acceptances else 0.0)

        record = OptimizationRecord(
            timestamp=datetime.now().isoformat(),
            scene_id=scene_id,
            method=opt_result.method,
            initial_score=(opt_result.score_trajectory[0]
                           if opt_result.score_trajectory else 0),
            final_score=opt_result.best_score,
            improvement=opt_result.best_score - (
                opt_result.score_trajectory[0]
                if opt_result.score_trajectory else 0),
            total_iterations=opt_result.total_iterations,
            convergence_iteration=opt_result.convergence_iteration,
            energy_trajectory=opt_result.energy_trajectory,
            score_trajectory=opt_result.score_trajectory,
            temperature_trajectory=opt_result.temperature_trajectory,
            acceptance_rate=acceptance_rate,
            total_time_s=opt_result.total_time_s,
        )
        self._optimization_records.append(record)

        # Save trajectory (can be large, so separate file)
        self._append_jsonl("optimization.jsonl", {
            "timestamp": record.timestamp,
            "scene_id": scene_id,
            "method": record.method,
            "initial_score": record.initial_score,
            "final_score": record.final_score,
            "improvement": record.improvement,
            "total_iterations": record.total_iterations,
            "convergence_iteration": record.convergence_iteration,
            "acceptance_rate": round(record.acceptance_rate, 4),
            "total_time_s": round(record.total_time_s, 2),
            "trajectory_length": len(record.energy_trajectory),
        })

        # Save full trajectory to separate file
        traj_path = self.experiment_dir / f"trajectory_{scene_id}.json"
        with open(traj_path, "w") as f:
            json.dump({
                "scene_id": scene_id,
                "method": record.method,
                "energy_trajectory": [round(e, 6) for e in record.energy_trajectory],
                "score_trajectory": [round(s, 6) for s in record.score_trajectory],
                "temperature_trajectory": [round(t, 6) for t in record.temperature_trajectory],
            }, f, indent=1)

    def log_correction(self, scene_id: str, corr_result):
        """
        Log a closed-loop correction result.

        Args:
            scene_id: Scene identifier.
            corr_result: CorrectionResult instance.
        """
        iteration_details = []
        for it in corr_result.iterations:
            iteration_details.append({
                "iteration": it.iteration,
                "score_before": round(it.score_before, 4),
                "score_after": round(it.score_after, 4),
                "accepted": it.accepted,
                "num_actions": len(it.actions_applied),
                "actions": [a.action_type for a in it.actions_applied],
            })

        record = CorrectionRecord(
            timestamp=datetime.now().isoformat(),
            scene_id=scene_id,
            initial_score=corr_result.initial_score,
            final_score=corr_result.final_score,
            improvement=corr_result.improvement,
            iterations=len(corr_result.iterations),
            corrections_applied=corr_result.total_corrections,
            regenerations=corr_result.total_regenerations,
            converged=corr_result.converged,
            total_time_s=corr_result.total_time_s,
            iteration_details=iteration_details,
        )
        self._correction_records.append(record)

        self._append_jsonl("corrections.jsonl", asdict(record))

    def log_event(self, event_type: str, data: Dict):
        """Log a custom event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **data,
        }
        self._custom_events.append(event)
        self._append_jsonl("events.jsonl", event)

    def log_weights(self, weights, context: str = ""):
        """Log a weight configuration."""
        from dataclasses import asdict as da
        self.log_event("weight_config", {
            "context": context,
            "weights": da(weights),
        })

    def finalize(self) -> str:
        """
        Finalize the experiment: compute aggregates, save summary, generate plots.

        Returns:
            Path to experiment directory.
        """
        total_time = time.time() - self.start_time if self.start_time else 0

        summary = {
            "experiment_id": self.experiment_id,
            "total_time_s": round(total_time, 2),
            "num_score_records": len(self._score_records),
            "num_optimization_records": len(self._optimization_records),
            "num_correction_records": len(self._correction_records),
            "num_custom_events": len(self._custom_events),
        }

        # Score statistics
        if self._score_records:
            scores = [r.total_score for r in self._score_records]
            summary["score_stats"] = {
                "mean": round(float(np.mean(scores)), 4),
                "std": round(float(np.std(scores)), 4),
                "min": round(float(np.min(scores)), 4),
                "max": round(float(np.max(scores)), 4),
                "median": round(float(np.median(scores)), 4),
            }

            # Per-component statistics
            component_names = ["semantic", "spatial", "physical",
                               "visual", "probabilistic", "multiview", "human"]
            summary["component_stats"] = {}
            for comp in component_names:
                vals = [r.component_scores.get(comp, 0) for r in self._score_records]
                summary["component_stats"][comp] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                }

        # Optimization statistics
        if self._optimization_records:
            improvements = [r.improvement for r in self._optimization_records]
            summary["optimization_stats"] = {
                "mean_improvement": round(float(np.mean(improvements)), 4),
                "mean_iterations": round(float(np.mean(
                    [r.total_iterations for r in self._optimization_records])), 1),
                "mean_acceptance_rate": round(float(np.mean(
                    [r.acceptance_rate for r in self._optimization_records])), 4),
                "mean_time_s": round(float(np.mean(
                    [r.total_time_s for r in self._optimization_records])), 2),
            }

        # Correction statistics
        if self._correction_records:
            improvements = [r.improvement for r in self._correction_records]
            summary["correction_stats"] = {
                "mean_improvement": round(float(np.mean(improvements)), 4),
                "convergence_rate": round(
                    sum(1 for r in self._correction_records if r.converged) /
                    len(self._correction_records), 3),
                "mean_iterations": round(float(np.mean(
                    [r.iterations for r in self._correction_records])), 2),
            }

        # Save summary
        summary_path = self.experiment_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Generate plots
        self._generate_plots()

        logger.info(f"  Research log finalized: {self.experiment_dir}")
        logger.info(f"  Total records: {summary['num_score_records']} scores, "
                     f"{summary['num_optimization_records']} optimizations, "
                     f"{summary['num_correction_records']} corrections")

        return str(self.experiment_dir)

    def _generate_plots(self):
        """Generate matplotlib visualization plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("  matplotlib not available, skipping plot generation")
            return

        plots_dir = self.experiment_dir / "plots"

        # 1. Score distribution
        if self._score_records:
            scores = [r.total_score for r in self._score_records]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(scores, bins=20, edgecolor="black", alpha=0.7)
            ax.set_xlabel("S(R)")
            ax.set_ylabel("Count")
            ax.set_title("Score Distribution")
            ax.axvline(np.mean(scores), color="red", linestyle="--",
                        label=f"Mean: {np.mean(scores):.3f}")
            ax.legend()
            fig.savefig(plots_dir / "score_distribution.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 2. Component score breakdown (radar-like bar chart)
        if self._score_records:
            components = ["semantic", "spatial", "physical",
                          "visual", "probabilistic", "multiview", "human"]
            means = [np.mean([r.component_scores.get(c, 0)
                              for r in self._score_records]) for c in components]

            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(components))
            ax.bar(x, means, color=plt.cm.Set2(np.linspace(0, 1, len(components))))
            ax.set_xticks(x)
            ax.set_xticklabels(components, rotation=45, ha="right")
            ax.set_ylabel("Mean Score")
            ax.set_title("Component Score Breakdown")
            ax.set_ylim(0, 1)
            fig.savefig(plots_dir / "component_breakdown.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 3. Optimization convergence curves
        if self._optimization_records:
            fig, ax = plt.subplots(figsize=(10, 5))
            for rec in self._optimization_records:
                if rec.score_trajectory:
                    ax.plot(rec.score_trajectory, alpha=0.6,
                            label=f"{rec.scene_id}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Score")
            ax.set_title("Optimization Convergence")
            if len(self._optimization_records) <= 10:
                ax.legend(fontsize=8)
            fig.savefig(plots_dir / "optimization_convergence.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 4. Correction improvement
        if self._correction_records:
            fig, ax = plt.subplots(figsize=(8, 5))
            scene_ids = [r.scene_id for r in self._correction_records]
            improvements = [r.improvement for r in self._correction_records]
            x = np.arange(len(scene_ids))
            colors = ["green" if imp > 0 else "red" for imp in improvements]
            ax.bar(x, improvements, color=colors, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(scene_ids, rotation=45, ha="right")
            ax.set_ylabel("Score Improvement")
            ax.set_title("Closed-Loop Correction Impact")
            ax.axhline(0, color="black", linewidth=0.5)
            fig.savefig(plots_dir / "correction_improvement.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        logger.info(f"  Plots saved to {plots_dir}")

    def _append_jsonl(self, filename: str, record: Dict):
        """Append a record to a JSONL file."""
        if self.experiment_dir is None:
            return
        filepath = self.experiment_dir / filename
        with open(filepath, "a") as f:
            f.write(json.dumps(self._serialize(record)) + "\n")

    def _serialize(self, obj) -> Any:
        """Recursively serialize an object for JSON."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._serialize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (int, float, str, bool)):
            return obj
        # Try dataclass
        try:
            from dataclasses import asdict
            return self._serialize(asdict(obj))
        except (TypeError, AttributeError):
            return str(obj)
