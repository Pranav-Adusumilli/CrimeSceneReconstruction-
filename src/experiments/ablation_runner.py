"""
Ablation Study Runner
======================

Systematically evaluates the contribution of each pipeline component
by removing one at a time and measuring the impact on S(R).

Ablation configurations:
  1. Full system (reference)
  2. –Spatial priors (remove spatial consistency scoring)
  3. –Energy optimization (remove SA/ES optimizer)
  4. –Segmentation conditioning (remove segmentation maps)
  5. –Multi-view (remove multi-view generation & scoring)
  6. –Closed-loop correction (remove self-correction)
  7. –Probabilistic priors (remove VG learned priors)
  8. –Physical plausibility (remove gravity/support checks)

Each ablation preserves all other components to isolate the effect
of the removed component. Reports per-component score deltas and
overall S(R) degradation.
"""

import json
import logging
import time
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from pathlib import Path

from src.scoring.unified_scorer import ScoringWeights, ScoreBreakdown
from src.experiments.experiment_runner import (
    ExperimentConfig, ExperimentResult, ExperimentRunner, DEFAULT_TEST_SCENES,
)

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for one ablation condition."""
    name: str
    description: str
    removed_component: str  # Human-readable name of removed component
    experiment_config: ExperimentConfig = None


@dataclass
class AblationResult:
    """Results of an ablation study."""
    reference_result: ExperimentResult = None
    ablation_results: List[ExperimentResult] = field(default_factory=list)
    ablation_configs: List[AblationConfig] = field(default_factory=list)
    deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    total_time_s: float = 0.0

    def compute_deltas(self):
        """Compute score deltas relative to reference."""
        if not self.reference_result:
            return

        ref_means = self.reference_result.mean_scores
        components = list(ref_means.keys())

        for ablation_cfg, result in zip(self.ablation_configs, self.ablation_results):
            name = ablation_cfg.name
            self.deltas[name] = {}
            for comp in components:
                ref_val = ref_means.get(comp, 0)
                abl_val = result.mean_scores.get(comp, 0)
                self.deltas[name][comp] = abl_val - ref_val

    def to_dict(self) -> dict:
        self.compute_deltas()
        return {
            "reference": self.reference_result.to_dict() if self.reference_result else {},
            "ablations": [
                {
                    "config": ac.name,
                    "removed": ac.removed_component,
                    "result": ar.to_dict(),
                    "deltas": self.deltas.get(ac.name, {}),
                }
                for ac, ar in zip(self.ablation_configs, self.ablation_results)
            ],
            "total_time_s": round(self.total_time_s, 2),
        }


# ── Predefined Ablation Configurations ──────────────────────────────

def _build_ablation_configs() -> List[AblationConfig]:
    """Build the standard ablation configuration set."""

    full_config = ExperimentConfig(
        name="Full System",
        description="Complete pipeline (reference)",
        use_scene_graph=True,
        use_depth_conditioning=True,
        use_optimization=True,
        use_correction=True,
        use_segmentation=True,
    )

    ablations = [
        AblationConfig(
            name="–Spatial Priors",
            description="Remove spatial consistency scoring component",
            removed_component="Spatial Consistency",
            experiment_config=ExperimentConfig(
                name="–Spatial",
                description="Spatial scoring weight set to 0",
                use_scene_graph=True, use_depth_conditioning=True,
                use_optimization=True, use_correction=True, use_segmentation=True,
                scoring_weights=ScoringWeights(
                    w_semantic=0.25, w_spatial=0.0, w_physical=0.12,
                    w_visual=0.18, w_probabilistic=0.12, w_multiview=0.12,
                    w_human=0.21),
            ),
        ),
        AblationConfig(
            name="–Optimization",
            description="Remove energy-based layout optimization",
            removed_component="Energy Optimizer",
            experiment_config=ExperimentConfig(
                name="–Optimization",
                description="No SA/ES optimization",
                use_scene_graph=True, use_depth_conditioning=True,
                use_optimization=False, use_correction=True, use_segmentation=True,
            ),
        ),
        AblationConfig(
            name="–Segmentation",
            description="Remove segmentation conditioning maps",
            removed_component="Segmentation Layout",
            experiment_config=ExperimentConfig(
                name="–Segmentation",
                description="No segmentation conditioning",
                use_scene_graph=True, use_depth_conditioning=True,
                use_optimization=True, use_correction=True, use_segmentation=False,
            ),
        ),
        AblationConfig(
            name="–Multi-View",
            description="Remove multi-view generation and scoring",
            removed_component="Multi-View Consistency",
            experiment_config=ExperimentConfig(
                name="–Multi-View",
                description="No multi-view generation or scoring",
                use_scene_graph=True, use_depth_conditioning=True,
                use_optimization=True, use_correction=True, use_segmentation=True,
                use_multiview=False,
                scoring_weights=ScoringWeights(
                    w_semantic=0.22, w_spatial=0.17, w_physical=0.11,
                    w_visual=0.17, w_probabilistic=0.11, w_multiview=0.0,
                    w_human=0.22),
            ),
        ),
        AblationConfig(
            name="–Closed-Loop",
            description="Remove self-correction feedback loop",
            removed_component="Closed-Loop Correction",
            experiment_config=ExperimentConfig(
                name="–Closed-Loop",
                description="No self-correction",
                use_scene_graph=True, use_depth_conditioning=True,
                use_optimization=True, use_correction=False, use_segmentation=True,
            ),
        ),
        AblationConfig(
            name="–Probabilistic Prior",
            description="Remove VG learned relationship priors",
            removed_component="Probabilistic Prior",
            experiment_config=ExperimentConfig(
                name="–Prior",
                description="Probabilistic prior weight set to 0",
                use_scene_graph=True, use_depth_conditioning=True,
                use_optimization=True, use_correction=True, use_segmentation=True,
                scoring_weights=ScoringWeights(
                    w_semantic=0.22, w_spatial=0.17, w_physical=0.11,
                    w_visual=0.17, w_probabilistic=0.0, w_multiview=0.11,
                    w_human=0.22),
            ),
        ),
        AblationConfig(
            name="–Physical Plausibility",
            description="Remove gravity and support checks",
            removed_component="Physical Plausibility",
            experiment_config=ExperimentConfig(
                name="–Physical",
                description="Physical plausibility weight set to 0",
                use_scene_graph=True, use_depth_conditioning=True,
                use_optimization=True, use_correction=True, use_segmentation=True,
                scoring_weights=ScoringWeights(
                    w_semantic=0.22, w_spatial=0.17, w_physical=0.0,
                    w_visual=0.17, w_probabilistic=0.11, w_multiview=0.11,
                    w_human=0.22),
            ),
        ),
    ]

    return full_config, ablations


class AblationRunner:
    """
    Runs systematic ablation studies for the reconstruction pipeline.

    Each ablation removes one component and evaluates the impact
    on the unified score S(R) across test scenes.

    Usage:
        runner = AblationRunner(pipeline_fn=my_fn)
        result = runner.run()
        runner.generate_report(result, "outputs/ablation")
    """

    def __init__(self,
                 pipeline_fn: Callable = None,
                 test_scenes: List[Dict] = None,
                 output_dir: str = "outputs/ablation"):
        """
        Args:
            pipeline_fn: Function(scene_text, config) → (ScoreBreakdown, time, path).
            test_scenes: List of test scene dicts.
            output_dir: Output directory.
        """
        self.pipeline_fn = pipeline_fn
        self.test_scenes = test_scenes or DEFAULT_TEST_SCENES
        self.output_dir = Path(output_dir)
        self.experiment_runner = ExperimentRunner(
            pipeline_fn=pipeline_fn, output_dir=output_dir)

    def run(self, custom_ablations: List[AblationConfig] = None) -> AblationResult:
        """
        Run the full ablation study.

        Args:
            custom_ablations: Optional custom ablation configs.
                              Defaults to the standard 7-ablation set.

        Returns:
            AblationResult with deltas and full breakdowns.
        """
        logger.info("=" * 60)
        logger.info("ABLATION STUDY")
        logger.info("=" * 60)

        t_start = time.time()

        full_config, ablation_configs = _build_ablation_configs()
        if custom_ablations:
            ablation_configs = custom_ablations

        result = AblationResult()
        result.ablation_configs = ablation_configs

        # 1. Run reference (full system)
        logger.info("\n  Running Reference (Full System)...")
        ref_results = self.experiment_runner.run_comparison(
            configs=[full_config], test_scenes=self.test_scenes)
        result.reference_result = ref_results[0] if ref_results else ExperimentResult()

        logger.info(f"  Reference mean S(R): "
                     f"{result.reference_result.mean_scores.get('total_score', 0):.4f}")

        # 2. Run each ablation
        for ablation in ablation_configs:
            logger.info(f"\n  Running ablation: {ablation.name}")
            logger.info(f"  Removed: {ablation.removed_component}")

            abl_results = self.experiment_runner.run_comparison(
                configs=[ablation.experiment_config],
                test_scenes=self.test_scenes)

            if abl_results:
                result.ablation_results.append(abl_results[0])
                abl_score = abl_results[0].mean_scores.get("total_score", 0)
                ref_score = result.reference_result.mean_scores.get("total_score", 0)
                delta = abl_score - ref_score
                logger.info(f"  {ablation.name}: S(R)={abl_score:.4f} "
                             f"(Δ={delta:+.4f})")
            else:
                result.ablation_results.append(ExperimentResult())

        result.total_time_s = time.time() - t_start
        result.compute_deltas()

        logger.info(f"\n  Ablation study complete ({result.total_time_s:.1f}s)")

        return result

    def generate_report(self, result: AblationResult,
                        output_dir: str = None) -> str:
        """Generate ablation study report."""
        out = Path(output_dir or self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save full results JSON
        json_path = out / "ablation_results.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Generate text table
        table = self._format_ablation_table(result)
        table_path = out / "ablation_table.txt"
        with open(table_path, "w") as f:
            f.write(table)

        # Generate LaTeX table
        latex = self._format_latex_ablation(result)
        latex_path = out / "ablation_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex)

        logger.info(f"  Ablation report saved to {out}")
        return str(out)

    def _format_ablation_table(self, result: AblationResult) -> str:
        """Format ablation results as text table."""
        lines = []
        lines.append("=" * 70)
        lines.append("ABLATION STUDY RESULTS")
        lines.append("=" * 70)

        ref_score = result.reference_result.mean_scores.get("total_score", 0)
        lines.append(f"Reference (Full System): S(R) = {ref_score:.4f}")
        lines.append("-" * 70)
        lines.append(f"{'Ablation':<30} {'S(R)':>8} {'ΔS(R)':>10} {'%Δ':>8}")
        lines.append("-" * 70)

        for cfg, res in zip(result.ablation_configs, result.ablation_results):
            abl_score = res.mean_scores.get("total_score", 0)
            delta = abl_score - ref_score
            pct = (delta / max(ref_score, 1e-6)) * 100
            lines.append(f"{cfg.name:<30} {abl_score:>8.4f} {delta:>+10.4f} {pct:>+7.1f}%")

        lines.append("=" * 70)
        return "\n".join(lines)

    def _format_latex_ablation(self, result: AblationResult) -> str:
        """Format ablation results as LaTeX table."""
        ref_score = result.reference_result.mean_scores.get("total_score", 0)

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Ablation study: impact of removing individual components.}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Configuration & $S(R)$ & $\Delta S(R)$ \\",
            r"\midrule",
            f"Full System (Reference) & \\textbf{{{ref_score:.4f}}} & --- \\\\",
            r"\midrule",
        ]

        for cfg, res in zip(result.ablation_configs, result.ablation_results):
            abl_score = res.mean_scores.get("total_score", 0)
            delta = abl_score - ref_score
            name = cfg.name.replace("–", "$-$")
            lines.append(f"{name} & {abl_score:.4f} & {delta:+.4f} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)
