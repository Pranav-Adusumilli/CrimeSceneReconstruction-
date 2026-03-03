"""
Research Pipeline — Unified Multi-Objective Reconstruction
============================================================

Orchestrates the complete research-grade pipeline:

  Text → NLP → Scene Graph → Hypotheses → Layout
    → [Energy Optimization over S(R)]
    → Depth Map + [Segmentation Conditioning]
    → ControlNet Image Generation
    → [Closed-Loop Self-Correction]
    → [Multi-View Generation]
    → Unified Scoring S(R) with 7-component breakdown
    → [Research-Grade Logging]

New vs Original Pipeline:
  + Unified multi-objective scoring S(R)
  + Energy-based layout optimization (SA / ES / Hybrid)
  + Segmentation-conditioned generation
  + Closed-loop self-correction
  + Structured research logging with plots
  + Experiment and ablation runner integration

Usage:
    python -m src.research_pipeline --input "Small bedroom. Knife on table. Blood on floor."
    python -m src.research_pipeline --config configs/research_config.yaml --input "..."
    python -m src.research_pipeline --experiments  # Run comparative experiments
    python -m src.research_pipeline --ablation     # Run ablation studies
    python -m src.research_pipeline --calibrate    # Run weight calibration
"""

import argparse
import logging
import sys
import os
import time
import yaml
import torch
from pathlib import Path
from typing import Optional, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, PROJECT_ROOT as CFG_ROOT
from src.utils.logging_utils import setup_logging
from src.utils.memory import flush_gpu_memory, log_memory

# Stage imports
from src.stages.stage0_init import initialize_environment, RuntimeEnvironment
from src.stages.stage1_text_understanding import TextUnderstanding
from src.stages.stage2_vocabulary_normalization import VocabularyNormalizer
from src.stages.stage3_scene_graph import SceneGraphBuilder
from src.stages.stage4_hypothesis_generation import HypothesisGenerator
from src.stages.stage5_spatial_layout import SpatialLayoutEstimator
from src.stages.stage6_depth_map import DepthMapGenerator
from src.stages.stage7_image_generation import ImageGenerator
from src.stages.stage8_multiview import MultiViewGenerator
from src.stages.stage9_explainability import ExplainabilityReport
from src.stages.stage10_evaluation import Evaluator
from src.stages.stage11_packaging import ResultPackager

# Research framework imports
from src.scoring.unified_scorer import UnifiedScorer, ScoringWeights, ScoreBreakdown
from src.optimization.energy_optimizer import EnergyOptimizer, OptimizationResult
from src.optimization.weight_calibration import WeightCalibrator
from src.conditioning.segmentation_layout import SegmentationLayoutGenerator
from src.correction.closed_loop import ClosedLoopCorrector
from src.experiments.experiment_runner import (
    ExperimentRunner, ExperimentConfig, PROPOSED_FULL,
    BASELINE_TEXT_ONLY, BASELINE_DEPTH, SCENE_GRAPH_NO_OPT,
)
from src.experiments.ablation_runner import AblationRunner
from src.experiments.research_logger import ResearchLogger

logger = logging.getLogger(__name__)


# ── Research Config Loader ──────────────────────────────────────────

def load_research_config(config_path: str = None) -> dict:
    """Load the full research config YAML and return raw dict."""
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "research_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_scoring_weights(raw_config: dict) -> ScoringWeights:
    """Build ScoringWeights from config dict."""
    scoring = raw_config.get("scoring", {})
    weights = scoring.get("weights", {})
    return ScoringWeights(
        w_semantic=weights.get("w_semantic", 0.20),
        w_spatial=weights.get("w_spatial", 0.15),
        w_physical=weights.get("w_physical", 0.10),
        w_visual=weights.get("w_visual", 0.15),
        w_probabilistic=weights.get("w_probabilistic", 0.10),
        w_multiview=weights.get("w_multiview", 0.10),
        w_human=weights.get("w_human", 0.20),
    )


# ── Main Research Pipeline ──────────────────────────────────────────

def run_research_pipeline(
    input_text: str,
    config_path: str = None,
    skip_multiview: bool = False,
    skip_optimization: bool = False,
    skip_correction: bool = False,
    skip_segmentation: bool = False,
):
    """
    Execute the full research-grade reconstruction pipeline.

    Integrates energy optimization, segmentation conditioning,
    closed-loop correction, and unified scoring.

    Args:
        input_text: Natural language crime scene description.
        config_path: Path to research_config.yaml.
        skip_multiview: Skip multi-view generation.
        skip_optimization: Skip energy optimization.
        skip_correction: Skip closed-loop correction.
        skip_segmentation: Skip segmentation conditioning.

    Returns:
        Dict with output_dir, score_breakdown, optimization_result, correction_result.
    """
    t_pipeline_start = time.time()

    # ── Setup ──────────────────────────────────────────────────
    setup_logging()
    config = load_config(config_path)
    raw_config = load_research_config(config_path)
    scoring_weights = build_scoring_weights(raw_config)

    opt_config = raw_config.get("optimization", {})
    corr_config = raw_config.get("correction", {})
    seg_config = raw_config.get("segmentation", {})
    log_config = raw_config.get("research_logging", {})

    logger.info("=" * 60)
    logger.info("  RESEARCH-GRADE CRIME SCENE RECONSTRUCTION")
    logger.info("=" * 60)
    logger.info(f"  Input: {input_text[:100]}...")
    logger.info(f"  Optimization: {not skip_optimization and opt_config.get('enabled', True)}")
    logger.info(f"  Correction: {not skip_correction and corr_config.get('enabled', True)}")
    logger.info(f"  Segmentation: {not skip_segmentation and seg_config.get('enabled', True)}")
    logger.info(f"  Multi-view: {not skip_multiview}")

    # ── Research Logger ──────────────────────────────────────
    rlog = ResearchLogger(base_dir=log_config.get("log_dir", "outputs/research_logs"))
    if log_config.get("enabled", True):
        rlog.start_experiment(config=raw_config)

    # ── STAGE 0: Environment Init ──────────────────────────────
    env = initialize_environment(config_path)

    # ── STAGE 1: Text Understanding ────────────────────────────
    text_engine = TextUnderstanding(
        nlp=env.nlp,
        scene_types=config.scene_types,
    )
    semantics = text_engine.parse(input_text)

    # ── Explainability ─────────────────────────────────────────
    report = ExplainabilityReport()
    report.add_text_understanding(semantics)

    # ── STAGE 2: Vocabulary Normalization ──────────────────────
    normalizer = VocabularyNormalizer(
        object_aliases=env.object_aliases,
        relationship_aliases=env.relationship_aliases,
    )
    semantics = normalizer.normalize_semantics(semantics)

    # ── STAGE 3: Scene Graph Construction ──────────────────────
    graph_builder = SceneGraphBuilder()
    scene_graph = graph_builder.build(semantics)
    report.add_scene_graph(scene_graph)

    # ── STAGE 4: Multi-Hypothesis Generation ───────────────────
    hyp_generator = HypothesisGenerator(
        num_hypotheses=config.generation.num_hypotheses,
        seed=config.generation.seed,
    )
    hypotheses = hyp_generator.generate(scene_graph)
    report.add_hypotheses(hypotheses)

    # ── Initialize Unified Scorer ──────────────────────────────
    # Load CLIP for scoring (CPU to save VRAM)
    vg_rel_path = raw_config.get("data", {}).get(
        "vg_relationships_file",
        "Data/VisualGenome/relationships.json/relationships.json")

    unified_scorer = UnifiedScorer(
        weights=scoring_weights,
        clip_model=env.clip_model,
        clip_preprocess=env.clip_preprocess,
        clip_tokenizer=env.clip_tokenizer,
        device="cpu",
        vg_relationships_path=str(CFG_ROOT / vg_rel_path),
        relationship_aliases=env.relationship_aliases,
    )

    if log_config.get("enabled", True):
        rlog.log_weights(scoring_weights, context="initial")

    # ── Result Packaging ───────────────────────────────────────
    packager = ResultPackager(base_output_dir=str(CFG_ROOT / config.output_dir))
    packager.initialize()

    # Save scene graph
    sg_path = str(packager.run_dir / "scene_graphs" / "scene_graph.png")
    graph_builder.visualize(scene_graph, sg_path)

    # ── ENERGY OPTIMIZATION ────────────────────────────────────
    do_optimization = (not skip_optimization and
                       opt_config.get("enabled", True))

    best_hypothesis = hypotheses[0]
    layout_estimator = SpatialLayoutEstimator(resolution=config.hardware.resolution)
    opt_result = None

    if do_optimization:
        logger.info("\n" + "=" * 50)
        logger.info("  ENERGY OPTIMIZATION PHASE")
        logger.info("=" * 50)

        sa_config = opt_config.get("simulated_annealing", {})
        es_config = opt_config.get("evolutionary", {})

        optimizer = EnergyOptimizer(
            unified_scorer=unified_scorer,
            layout_estimator=layout_estimator,
            method=opt_config.get("method", "simulated_annealing"),
            sa_config=sa_config,
            es_config=es_config,
            top_k=opt_config.get("top_k_candidates", 3),
            resolution=config.hardware.resolution,
            seed=config.generation.seed,
        )

        opt_result = optimizer.optimize(
            initial_hypothesis=best_hypothesis,
            scene_graph=scene_graph,
            scene_type=semantics.scene_type,
        )

        best_hypothesis = opt_result.best_hypothesis
        logger.info(f"  Optimization: score {opt_result.best_score:.4f} "
                     f"({opt_result.total_iterations} iters, "
                     f"{opt_result.total_time_s:.1f}s)")

        if log_config.get("enabled", True) and log_config.get("log_optimization", True):
            rlog.log_optimization("main", opt_result)

    # ── STAGE 5: Spatial Layout ────────────────────────────────
    layout = layout_estimator.estimate(best_hypothesis, scene_type=semantics.scene_type)
    report.add_spatial_layout(layout)

    # Save layout preview
    layout_preview_path = str(packager.run_dir / "images" / "layout_preview.png")
    layout_estimator.render_layout_preview(layout, layout_preview_path)

    # ── STAGE 6: Depth Map Generation ──────────────────────────
    depth_gen = DepthMapGenerator(
        midas_model=env.midas_model,
        midas_feature_extractor=env.midas_feature_extractor,
    )
    depth_map = depth_gen.from_layout(layout)

    depth_path = str(packager.run_dir / "depth_maps" / "depth_map_h1.png")
    depth_map.save(depth_path)

    # ── SEGMENTATION CONDITIONING ──────────────────────────────
    do_segmentation = (not skip_segmentation and
                       seg_config.get("enabled", True))
    seg_image = None

    if do_segmentation:
        logger.info("\n  SEGMENTATION CONDITIONING")
        seg_gen = SegmentationLayoutGenerator(resolution=config.hardware.resolution)
        seg_image, seg_info = seg_gen.generate(layout, semantics.scene_type)

        # Save segmentation map
        seg_path = str(packager.run_dir / "images" / "segmentation_map.png")
        Path(seg_path).parent.mkdir(parents=True, exist_ok=True)
        seg_image.save(seg_path)

        # Optionally create composite conditioning
        composite = seg_gen.generate_composite_conditioning(
            layout, depth_map, semantics.scene_type,
            depth_weight=seg_config.get("depth_weight", 0.6),
            seg_weight=seg_config.get("seg_weight", 0.4),
        )
        composite_path = str(packager.run_dir / "images" / "composite_conditioning.png")
        composite.save(composite_path)

        # Use composite as conditioning if segmentation is enabled
        conditioning_image = composite
    else:
        conditioning_image = depth_map

    # ── STAGE 7: Image Generation ──────────────────────────────
    cn_pipe = env.load_controlnet_pipeline()

    img_gen = ImageGenerator(
        controlnet_pipeline=cn_pipe,
        num_inference_steps=config.generation.num_inference_steps,
        guidance_scale=config.generation.guidance_scale,
        negative_prompt=config.generation.negative_prompt,
        seed=config.generation.seed,
    )

    prompt = img_gen.build_scene_prompt(
        scene_type=semantics.scene_type,
        hypothesis=best_hypothesis,
        base_description=input_text,
    )

    use_two_pass = getattr(config.generation, 'two_pass', True)

    if use_two_pass:
        logger.info("  Using TWO-PASS generation")
        image, midas_depth = img_gen.two_pass_generate(
            prompt=prompt,
            depth_generator=depth_gen,
            midas_device=torch.device("cpu"),
            output_dir=str(packager.run_dir / "images"),
        )
        midas_depth_path = str(packager.run_dir / "depth_maps" / "midas_depth_h1.png")
        Path(midas_depth_path).parent.mkdir(parents=True, exist_ok=True)
        midas_depth.save(midas_depth_path)
        depth_map = midas_depth
    else:
        image = img_gen.generate_with_controlnet(
            prompt=prompt,
            depth_image=conditioning_image,
            output_path=str(packager.run_dir / "images" / "reconstruction_h1.png"),
        )

    report.add_generation_params(
        prompt=prompt,
        negative_prompt=config.generation.negative_prompt,
        steps=config.generation.num_inference_steps,
        guidance_scale=config.generation.guidance_scale,
        seed=config.generation.seed,
        resolution=config.hardware.resolution,
    )

    # ── CLOSED-LOOP SELF-CORRECTION ────────────────────────────
    do_correction = (not skip_correction and
                     corr_config.get("enabled", True))
    corr_result = None

    if do_correction:
        logger.info("\n" + "=" * 50)
        logger.info("  CLOSED-LOOP SELF-CORRECTION PHASE")
        logger.info("=" * 50)

        corrector = ClosedLoopCorrector(
            unified_scorer=unified_scorer,
            layout_estimator=layout_estimator,
            max_iterations=corr_config.get("max_iterations", 3),
            improvement_threshold=corr_config.get("improvement_threshold", 0.01),
            component_threshold=corr_config.get("component_threshold", 0.4),
            resolution=config.hardware.resolution,
        )

        # Define regeneration function for the corrector
        def regenerate_fn(corrected_prompt, corrected_hyp, corrected_layout):
            new_depth = depth_gen.from_layout(corrected_layout)
            if do_segmentation:
                seg_gen_inner = SegmentationLayoutGenerator(
                    resolution=config.hardware.resolution)
                cond = seg_gen_inner.generate_composite_conditioning(
                    corrected_layout, new_depth, semantics.scene_type,
                    depth_weight=seg_config.get("depth_weight", 0.6),
                    seg_weight=seg_config.get("seg_weight", 0.4),
                )
            else:
                cond = new_depth

            return img_gen.generate_with_controlnet(
                prompt=corrected_prompt, depth_image=cond)

        corr_result = corrector.correct(
            image=image,
            prompt=prompt,
            semantics=semantics,
            scene_graph=scene_graph,
            hypothesis=best_hypothesis,
            layout=layout,
            generate_fn=regenerate_fn,
        )

        if corr_result.improvement > 0:
            logger.info(f"  Correction improved score: +{corr_result.improvement:.4f}")
            best_hypothesis = corr_result.final_hypothesis
            layout = corr_result.final_layout

        if log_config.get("enabled", True) and log_config.get("log_corrections", True):
            rlog.log_correction("main", corr_result)

    # ── STAGE 8: Multi-View Generation ─────────────────────────
    view_images = []
    view_depth_maps = []

    if not skip_multiview and config.views:
        mv_gen = MultiViewGenerator(
            image_generator=img_gen,
            depth_generator=depth_gen,
            views=config.views,
        )
        view_results = mv_gen.generate_views(
            base_prompt=prompt,
            base_depth_map=depth_map,
            output_dir=str(packager.run_dir / "multiview"),
            hypothesis_id=best_hypothesis.hypothesis_id,
        )
        view_images = [vr.image for vr in view_results if hasattr(vr, 'image') and vr.image]
        report.add_multiview_info([v.view_name for v in view_results])

    # ── Unload Generation Models ───────────────────────────────
    env.unload_controlnet_pipeline()
    env.unload_sd_pipeline()
    flush_gpu_memory()

    # ── UNIFIED SCORING ────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("  UNIFIED SCORING S(R)")
    logger.info("=" * 50)

    breakdown = unified_scorer.score(
        image=image,
        prompt=prompt,
        semantics=semantics,
        scene_graph=scene_graph,
        hypothesis=best_hypothesis,
        layout=layout,
        view_images=view_images if view_images else None,
        view_depth_maps=view_depth_maps if view_depth_maps else None,
        skip_multiview=skip_multiview or not view_images,
    )

    logger.info(f"  S(R) = {breakdown.total_score:.4f}  E(R) = {breakdown.energy:.4f}")
    logger.info(f"  Components:")
    logger.info(f"    Semantic:  {breakdown.semantic_score:.4f}")
    logger.info(f"    Spatial:   {breakdown.spatial_score:.4f}")
    logger.info(f"    Physical:  {breakdown.physical_score:.4f}")
    logger.info(f"    Visual:    {breakdown.visual_score:.4f}")
    logger.info(f"    Prior:     {breakdown.probabilistic_score:.4f}")
    logger.info(f"    MultiView: {breakdown.multiview_score:.4f}")
    logger.info(f"    Human:     {breakdown.human_score:.4f}")

    if log_config.get("enabled", True) and log_config.get("log_scores", True):
        rlog.log_score("main", breakdown, hypothesis_id=best_hypothesis.hypothesis_id)

    # ── STAGE 9: Explainability Report ─────────────────────────
    report.add_evaluation(breakdown.to_dict())
    report.print_summary()
    report.save(str(packager.run_dir / "reports" / "explainability_report.json"))

    # ── STAGE 11: Packaging ────────────────────────────────────
    packager.create_summary(
        input_text=input_text,
        scene_type=semantics.scene_type,
        num_objects=len(semantics.objects),
        num_hypotheses=len(hypotheses),
        best_score=breakdown.total_score,
    )

    # Save research-specific outputs
    import json
    research_output_path = packager.run_dir / "reports" / "research_output.json"
    Path(research_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(research_output_path, "w") as f:
        json.dump({
            "unified_score": breakdown.to_dict(),
            "optimization": opt_result.to_dict() if opt_result else None,
            "correction": corr_result.to_dict() if corr_result else None,
            "scoring_weights": {k: round(v, 3) for k, v in
                                breakdown.weights_used.items()},
            "pipeline_time_s": round(time.time() - t_pipeline_start, 2),
        }, f, indent=2)

    output_dir = packager.package_complete()

    # ── Finalize Research Logger ───────────────────────────────
    if log_config.get("enabled", True):
        rlog.finalize()

    # ── Cleanup ────────────────────────────────────────────────
    env.cleanup_all()

    total_time = time.time() - t_pipeline_start

    logger.info("\n" + "=" * 60)
    logger.info("  RESEARCH PIPELINE COMPLETE")
    logger.info(f"  S(R) = {breakdown.total_score:.4f}")
    logger.info(f"  Time: {total_time:.1f}s")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    return {
        "output_dir": output_dir,
        "breakdown": breakdown,
        "optimization_result": opt_result,
        "correction_result": corr_result,
    }


# ── Experiment Runner Entry Point ───────────────────────────────────

def run_experiments(config_path: str = None):
    """Run the comparative experiment suite."""
    setup_logging()
    raw_config = load_research_config(config_path)

    logger.info("=" * 60)
    logger.info("  RUNNING EXPERIMENTAL EVALUATION")
    logger.info("=" * 60)

    # Use dry-run mode (placeholder scores) if no pipeline_fn
    runner = ExperimentRunner(
        pipeline_fn=None,  # Set to real pipeline_fn for full evaluation
        output_dir=str(CFG_ROOT / "outputs" / "experiments"),
    )

    test_scenes = raw_config.get("experiments", {}).get("test_scenes", None)
    configs = [BASELINE_TEXT_ONLY, BASELINE_DEPTH, SCENE_GRAPH_NO_OPT, PROPOSED_FULL]

    results = runner.run_comparison(configs=configs, test_scenes=test_scenes)
    report_dir = runner.generate_report(results)

    logger.info(f"  Experiment report: {report_dir}")
    return results


def run_ablation(config_path: str = None):
    """Run the ablation study suite."""
    setup_logging()
    raw_config = load_research_config(config_path)

    logger.info("=" * 60)
    logger.info("  RUNNING ABLATION STUDY")
    logger.info("=" * 60)

    runner = AblationRunner(
        pipeline_fn=None,  # Set to real pipeline_fn for full evaluation
        output_dir=str(CFG_ROOT / "outputs" / "ablation"),
    )

    result = runner.run()
    report_dir = runner.generate_report(result)

    logger.info(f"  Ablation report: {report_dir}")
    return result


def run_calibration(config_path: str = None):
    """Run weight calibration."""
    setup_logging()
    raw_config = load_research_config(config_path)
    cal_config = raw_config.get("calibration", {})

    logger.info("=" * 60)
    logger.info("  RUNNING WEIGHT CALIBRATION")
    logger.info("=" * 60)

    # Placeholder evaluation function (replace with real scoring loop)
    def eval_fn(weights: ScoringWeights) -> float:
        """Evaluate a weight configuration (placeholder)."""
        import numpy as np
        # In real usage, this would run the pipeline with these weights
        # and return the combined score across calibration scenes
        score = (weights.w_semantic * 0.6 + weights.w_spatial * 0.5 +
                 weights.w_physical * 0.4 + weights.w_visual * 0.7 +
                 weights.w_probabilistic * 0.3 + weights.w_multiview * 0.5 +
                 weights.w_human * 0.65)
        return score + np.random.normal(0, 0.02)

    calibrator = WeightCalibrator(evaluation_fn=eval_fn, seed=42)
    result = calibrator.calibrate(
        method=cal_config.get("method", "bayesian"),
        bayesian_iterations=cal_config.get("bayesian_iterations", 50),
    )

    logger.info(f"  Best weights: {result.best_weights}")
    logger.info(f"  Best objective: {result.best_objective:.4f}")

    # Save calibration results
    import json
    out_dir = CFG_ROOT / "outputs" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "calibration_result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    return result


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Research-Grade Crime Scene Reconstruction Pipeline"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Natural language crime scene description",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to research_config.yaml",
    )
    parser.add_argument(
        "--no-multiview", action="store_true",
        help="Skip multi-view generation",
    )
    parser.add_argument(
        "--no-optimization", action="store_true",
        help="Skip energy optimization",
    )
    parser.add_argument(
        "--no-correction", action="store_true",
        help="Skip closed-loop correction",
    )
    parser.add_argument(
        "--no-segmentation", action="store_true",
        help="Skip segmentation conditioning",
    )
    parser.add_argument(
        "--experiments", action="store_true",
        help="Run comparative experiment suite",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run ablation studies",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Run weight calibration",
    )

    args = parser.parse_args()

    if args.experiments:
        run_experiments(args.config)
    elif args.ablation:
        run_ablation(args.config)
    elif args.calibrate:
        run_calibration(args.config)
    elif args.input:
        run_research_pipeline(
            input_text=args.input,
            config_path=args.config,
            skip_multiview=args.no_multiview,
            skip_optimization=args.no_optimization,
            skip_correction=args.no_correction,
            skip_segmentation=args.no_segmentation,
        )
    else:
        parser.print_help()
        print("\nExample:")
        print('  python -m src.research_pipeline -i "Small bedroom. Knife on table. Blood on floor."')
        print('  python -m src.research_pipeline --experiments')
        print('  python -m src.research_pipeline --ablation')
        print('  python -m src.research_pipeline --calibrate')


if __name__ == "__main__":
    main()
