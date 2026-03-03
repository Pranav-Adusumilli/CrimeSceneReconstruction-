"""
Experimental Evaluation Framework
====================================

Systematic comparison of reconstruction configurations to produce
research-grade evaluation tables suitable for conference submission.

Configurations:
  A. **Baseline-Text**: Text-to-image only (no scene graph, no depth)
  B. **Baseline-Depth**: Depth-conditioned ControlNet (no optimization)
  C. **SceneGraph-NoOpt**: Full scene graph pipeline without energy optimization
  D. **Proposed (Full)**: Complete pipeline with weighted optimization, correction, segmentation

Each configuration is evaluated on a set of test scenes using the
full unified scoring function S(R), producing per-component breakdowns.

Output: comparison tables (LaTeX-ready), score distributions,
statistical significance tests (paired t-test, Wilcoxon signed-rank).
"""

import json
import logging
import time
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path

from src.scoring.unified_scorer import ScoringWeights, ScoreBreakdown

logger = logging.getLogger(__name__)


# ── Test Scenes ─────────────────────────────────────────────────────

# Default test suite — diverse crime scene descriptions for evaluation
DEFAULT_TEST_SCENES = [
    {
        "id": "bedroom_01",
        "text": "Small bedroom. Broken window. Knife on the nightstand. Blood on the floor near the bed.",
        "scene_type": "bedroom",
        "expected_objects": ["bed", "window", "knife", "nightstand", "blood"],
    },
    {
        "id": "kitchen_02",
        "text": "Kitchen with overturned chair. Open drawer. Shattered glass on the floor.",
        "scene_type": "kitchen",
        "expected_objects": ["chair", "drawer", "glass"],
    },
    {
        "id": "living_room_03",
        "text": "Living room in disarray. Sofa overturned. Lamp on the floor. Blood stain on the carpet.",
        "scene_type": "living_room",
        "expected_objects": ["sofa", "lamp", "blood", "carpet"],
    },
    {
        "id": "bathroom_04",
        "text": "Bathroom. Mirror cracked. Towel on the floor. Water on the tiles.",
        "scene_type": "bathroom",
        "expected_objects": ["mirror", "towel"],
    },
    {
        "id": "office_05",
        "text": "Office with desk. Computer monitor smashed. Papers scattered. Overturned chair.",
        "scene_type": "office",
        "expected_objects": ["desk", "monitor", "papers", "chair"],
    },
]


@dataclass
class ExperimentConfig:
    """Configuration for one experimental condition."""
    name: str
    description: str
    use_scene_graph: bool = True
    use_depth_conditioning: bool = True
    use_optimization: bool = True
    use_correction: bool = True
    use_segmentation: bool = True
    use_multiview: bool = False
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    optimization_method: str = "simulated_annealing"
    optimization_iterations: int = 300
    correction_iterations: int = 3
    generation_steps: int = 35
    guidance_scale: float = 9.0
    seed: int = 42

    def to_dict(self) -> dict:
        from dataclasses import asdict
        d = asdict(self)
        return d


# ── Predefined Experimental Configurations ──────────────────────────

BASELINE_TEXT_ONLY = ExperimentConfig(
    name="Baseline-A (Text-Only)",
    description="Text-to-image only, no scene graph, no depth conditioning",
    use_scene_graph=False,
    use_depth_conditioning=False,
    use_optimization=False,
    use_correction=False,
    use_segmentation=False,
)

BASELINE_DEPTH = ExperimentConfig(
    name="Baseline-B (Depth-Conditioned)",
    description="Depth-conditioned ControlNet, no scene graph optimization",
    use_scene_graph=False,
    use_depth_conditioning=True,
    use_optimization=False,
    use_correction=False,
    use_segmentation=False,
)

SCENE_GRAPH_NO_OPT = ExperimentConfig(
    name="Baseline-C (SceneGraph-NoOpt)",
    description="Full scene graph pipeline without energy optimization or correction",
    use_scene_graph=True,
    use_depth_conditioning=True,
    use_optimization=False,
    use_correction=False,
    use_segmentation=False,
)

PROPOSED_FULL = ExperimentConfig(
    name="Proposed (Full Pipeline)",
    description="Complete pipeline with weighted optimization, correction, and segmentation",
    use_scene_graph=True,
    use_depth_conditioning=True,
    use_optimization=True,
    use_correction=True,
    use_segmentation=True,
)


@dataclass
class SceneResult:
    """Results for one scene under one configuration."""
    scene_id: str
    config_name: str
    breakdown: ScoreBreakdown = None
    total_score: float = 0.0
    generation_time_s: float = 0.0
    output_path: str = ""
    error: str = ""


@dataclass
class ExperimentResult:
    """Complete results for one experimental configuration across all scenes."""
    config: ExperimentConfig = None
    scene_results: List[SceneResult] = field(default_factory=list)
    mean_scores: Dict[str, float] = field(default_factory=dict)
    std_scores: Dict[str, float] = field(default_factory=dict)
    total_time_s: float = 0.0

    def compute_statistics(self):
        """Compute mean and std for each scoring component."""
        if not self.scene_results:
            return

        component_names = [
            "total_score", "semantic_score", "spatial_score",
            "physical_score", "visual_score", "probabilistic_score",
            "multiview_score", "human_score",
        ]

        for name in component_names:
            values = []
            for sr in self.scene_results:
                if sr.breakdown:
                    values.append(getattr(sr.breakdown, name, 0.0))
                else:
                    values.append(sr.total_score if name == "total_score" else 0.0)

            self.mean_scores[name] = float(np.mean(values)) if values else 0.0
            self.std_scores[name] = float(np.std(values)) if values else 0.0

    def to_dict(self) -> dict:
        self.compute_statistics()
        return {
            "config": self.config.to_dict() if self.config else {},
            "num_scenes": len(self.scene_results),
            "mean_scores": {k: round(v, 4) for k, v in self.mean_scores.items()},
            "std_scores": {k: round(v, 4) for k, v in self.std_scores.items()},
            "total_time_s": round(self.total_time_s, 2),
            "scene_results": [
                {
                    "scene_id": sr.scene_id,
                    "total_score": round(sr.total_score, 4),
                    "generation_time_s": round(sr.generation_time_s, 2),
                    "error": sr.error,
                }
                for sr in self.scene_results
            ],
        }


class ExperimentRunner:
    """
    Runs systematic experiments comparing reconstruction configurations.

    Usage:
        runner = ExperimentRunner(pipeline_fn=my_pipeline_fn)
        results = runner.run_comparison(configs, test_scenes)
        runner.generate_report(results, output_dir)
    """

    def __init__(self,
                 pipeline_fn: Callable = None,
                 output_dir: str = "outputs/experiments"):
        """
        Args:
            pipeline_fn: Function(scene_text, config) → (ScoreBreakdown, generation_time, output_path).
                         The main pipeline entry point parameterized by config.
            output_dir: Base output directory for experiment results.
        """
        self.pipeline_fn = pipeline_fn
        self.output_dir = Path(output_dir)

    def run_comparison(self,
                       configs: List[ExperimentConfig] = None,
                       test_scenes: List[Dict] = None) -> List[ExperimentResult]:
        """
        Run all configurations on all test scenes.

        Args:
            configs: List of experimental configurations.
                     Defaults to the 4 standard baselines.
            test_scenes: List of test scene dicts.
                         Defaults to DEFAULT_TEST_SCENES.

        Returns:
            List of ExperimentResult, one per configuration.
        """
        if configs is None:
            configs = [BASELINE_TEXT_ONLY, BASELINE_DEPTH,
                       SCENE_GRAPH_NO_OPT, PROPOSED_FULL]
        if test_scenes is None:
            test_scenes = DEFAULT_TEST_SCENES

        logger.info("=" * 60)
        logger.info("EXPERIMENTAL EVALUATION")
        logger.info(f"  Configurations: {len(configs)}")
        logger.info(f"  Test scenes: {len(test_scenes)}")
        logger.info("=" * 60)

        all_results = []

        for config in configs:
            logger.info(f"\n{'='*50}")
            logger.info(f"  Configuration: {config.name}")
            logger.info(f"  {config.description}")
            logger.info(f"{'='*50}")

            exp_result = ExperimentResult(config=config)
            t_config_start = time.time()

            for scene in test_scenes:
                logger.info(f"  Scene: {scene['id']} — {scene['text'][:60]}...")
                t_scene_start = time.time()

                try:
                    if self.pipeline_fn:
                        breakdown, gen_time, out_path = self.pipeline_fn(
                            scene["text"], config)
                    else:
                        # Dry run mode: generate placeholder scores
                        breakdown = self._placeholder_breakdown(config)
                        gen_time = 0.0
                        out_path = ""

                    sr = SceneResult(
                        scene_id=scene["id"],
                        config_name=config.name,
                        breakdown=breakdown,
                        total_score=breakdown.total_score if breakdown else 0.0,
                        generation_time_s=gen_time,
                        output_path=out_path,
                    )
                except Exception as e:
                    logger.error(f"  Scene {scene['id']} failed: {e}")
                    sr = SceneResult(
                        scene_id=scene["id"],
                        config_name=config.name,
                        total_score=0.0,
                        generation_time_s=time.time() - t_scene_start,
                        error=str(e),
                    )

                exp_result.scene_results.append(sr)
                logger.info(f"    Score: {sr.total_score:.4f} ({sr.generation_time_s:.1f}s)")

            exp_result.total_time_s = time.time() - t_config_start
            exp_result.compute_statistics()
            all_results.append(exp_result)

            logger.info(f"  Config mean score: "
                         f"{exp_result.mean_scores.get('total_score', 0):.4f} ± "
                         f"{exp_result.std_scores.get('total_score', 0):.4f}")

        return all_results

    def generate_report(self, results: List[ExperimentResult],
                        output_dir: str = None) -> str:
        """
        Generate a comprehensive experiment report with comparison tables.

        Args:
            results: List of ExperimentResult from run_comparison.
            output_dir: Output directory.

        Returns:
            Path to the generated report.
        """
        out = Path(output_dir or self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save raw JSON results
        json_path = out / "experiment_results.json"
        with open(json_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        # Generate comparison table (text format)
        table = self._format_comparison_table(results)
        table_path = out / "comparison_table.txt"
        with open(table_path, "w") as f:
            f.write(table)

        # Generate LaTeX table
        latex = self._format_latex_table(results)
        latex_path = out / "comparison_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex)

        # Statistical tests
        stats = self._compute_statistical_tests(results)
        stats_path = out / "statistical_tests.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"  Report saved to {out}")
        return str(out)

    def _format_comparison_table(self, results: List[ExperimentResult]) -> str:
        """Format a text comparison table."""
        components = ["total_score", "semantic_score", "spatial_score",
                       "physical_score", "visual_score", "probabilistic_score",
                       "human_score"]
        headers = ["Config", "S(R)", "Sem", "Spa", "Phy", "Vis", "Prior", "Hum"]

        lines = []
        lines.append("=" * 90)
        lines.append("EXPERIMENTAL COMPARISON TABLE")
        lines.append("=" * 90)

        # Header row
        header = f"{'Config':<35} " + " ".join(f"{h:>7}" for h in headers[1:])
        lines.append(header)
        lines.append("-" * 90)

        for r in results:
            name = r.config.name[:35] if r.config else "unknown"
            values = [f"{r.mean_scores.get(c, 0):.3f}" for c in components]
            row = f"{name:<35} " + " ".join(f"{v:>7}" for v in values)
            lines.append(row)

            # Std row
            stds = [f"±{r.std_scores.get(c, 0):.3f}" for c in components]
            std_row = f"{'':35} " + " ".join(f"{s:>7}" for s in stds)
            lines.append(std_row)

        lines.append("=" * 90)
        return "\n".join(lines)

    def _format_latex_table(self, results: List[ExperimentResult]) -> str:
        """Format a LaTeX comparison table for paper inclusion."""
        components = ["total_score", "semantic_score", "spatial_score",
                       "physical_score", "visual_score", "probabilistic_score",
                       "human_score"]

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Comparative evaluation of reconstruction configurations.}",
            r"\label{tab:comparison}",
            r"\begin{tabular}{l" + "c" * len(components) + "}",
            r"\toprule",
            r"Configuration & $S(R)$ & Sem. & Spa. & Phy. & Vis. & Prior & Hum. \\",
            r"\midrule",
        ]

        # Find best scores for bolding
        best_per_component = {}
        for c in components:
            values = [r.mean_scores.get(c, 0) for r in results]
            best_per_component[c] = max(values) if values else 0

        for r in results:
            name = r.config.name if r.config else "Unknown"
            # Escape LaTeX special characters
            name = name.replace("_", r"\_")
            cells = []
            for c in components:
                mean = r.mean_scores.get(c, 0)
                std = r.std_scores.get(c, 0)
                val_str = f"{mean:.3f}$\\pm${std:.3f}"
                if abs(mean - best_per_component[c]) < 1e-6 and mean > 0:
                    val_str = r"\textbf{" + val_str + "}"
                cells.append(val_str)
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def _compute_statistical_tests(self, results: List[ExperimentResult]) -> Dict:
        """Compute pairwise statistical significance tests."""
        from scipy import stats as scipy_stats

        tests = {}
        proposed = None
        baselines = []

        for r in results:
            if r.config and "Proposed" in r.config.name:
                proposed = r
            else:
                baselines.append(r)

        if proposed is None or not baselines:
            return {"note": "Could not identify proposed vs baseline for testing"}

        proposed_scores = [
            sr.total_score for sr in proposed.scene_results if not sr.error]

        for baseline in baselines:
            baseline_scores = [
                sr.total_score for sr in baseline.scene_results if not sr.error]

            if len(proposed_scores) < 2 or len(baseline_scores) < 2:
                continue

            # Pad shorter to match length (for paired tests)
            min_len = min(len(proposed_scores), len(baseline_scores))
            p_scores = proposed_scores[:min_len]
            b_scores = baseline_scores[:min_len]

            # Paired t-test
            try:
                t_stat, t_pval = scipy_stats.ttest_rel(p_scores, b_scores)
            except Exception:
                t_stat, t_pval = 0, 1

            # Wilcoxon signed-rank test
            try:
                w_stat, w_pval = scipy_stats.wilcoxon(
                    p_scores, b_scores, alternative="greater")
            except Exception:
                w_stat, w_pval = 0, 1

            name = baseline.config.name if baseline.config else "unknown"
            tests[name] = {
                "paired_t_test": {"t_statistic": float(t_stat), "p_value": float(t_pval)},
                "wilcoxon": {"w_statistic": float(w_stat), "p_value": float(w_pval)},
                "proposed_mean": float(np.mean(p_scores)),
                "baseline_mean": float(np.mean(b_scores)),
                "improvement": float(np.mean(p_scores) - np.mean(b_scores)),
            }

        return tests

    def _placeholder_breakdown(self, config: ExperimentConfig) -> ScoreBreakdown:
        """Generate placeholder scores for dry-run mode."""
        bd = ScoreBreakdown()

        # Generate scores that reflect the config capabilities
        base = 0.3
        if config.use_scene_graph:
            base += 0.1
        if config.use_depth_conditioning:
            base += 0.1
        if config.use_optimization:
            base += 0.1
        if config.use_correction:
            base += 0.05
        if config.use_segmentation:
            base += 0.05

        noise = np.random.normal(0, 0.03)

        bd.semantic_score = min(1.0, base + 0.05 + noise)
        bd.spatial_score = min(1.0, base + noise if config.use_scene_graph else base * 0.7)
        bd.physical_score = min(1.0, base + noise)
        bd.visual_score = min(1.0, base + 0.1 + noise if config.use_depth_conditioning else base * 0.8)
        bd.probabilistic_score = min(1.0, base + noise)
        bd.multiview_score = 0.5
        bd.human_score = min(1.0, base + 0.05 + noise)

        w = config.scoring_weights
        bd.total_score = (
            w.w_semantic * bd.semantic_score +
            w.w_spatial * bd.spatial_score +
            w.w_physical * bd.physical_score +
            w.w_visual * bd.visual_score +
            w.w_probabilistic * bd.probabilistic_score +
            w.w_multiview * bd.multiview_score +
            w.w_human * bd.human_score
        )
        bd.energy = -bd.total_score

        return bd
