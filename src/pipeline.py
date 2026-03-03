"""
Main Pipeline Orchestrator
===========================
Orchestrates the full 12-stage crime scene reconstruction pipeline.

Text → NLP → Normalization → Scene Graph → Hypotheses → Layout
    → Depth Map → ControlNet Image → Multi-View → Evaluation → Packaging

Usage:
    python -m src.pipeline --input "Small bedroom. Broken window. Knife on table."
"""

import argparse
import logging
import sys
import os
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, PROJECT_ROOT as CFG_ROOT
from src.utils.logging_utils import setup_logging
from src.utils.memory import flush_gpu_memory, get_gpu_memory_info, log_memory

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

logger = logging.getLogger(__name__)


def run_pipeline(input_text: str, config_path: str = None,
                  skip_multiview: bool = False,
                  skip_evaluation: bool = False,
                  controlnet_mode: bool = True):
    """
    Execute the full crime scene reconstruction pipeline.

    Args:
        input_text: Natural language crime scene description.
        config_path: Path to YAML config (None for default).
        skip_multiview: Skip multi-view generation to save time/memory.
        skip_evaluation: Skip CLIP evaluation to save memory.
        controlnet_mode: Use ControlNet (True) or plain SD (False).

    Returns:
        Path to the output directory with all results.
    """

    # ── Setup ────────────────────────────────────────────────
    setup_logging()
    config = load_config(config_path)

    logger.info("=" * 60)
    logger.info("  CRIME SCENE RECONSTRUCTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"  Input: {input_text[:100]}...")
    logger.info(f"  ControlNet: {controlnet_mode}")
    logger.info(f"  Multi-view: {not skip_multiview}")
    logger.info(f"  Evaluation: {not skip_evaluation}")

    # ── STAGE 0: Environment Init ────────────────────────────
    env = initialize_environment(config_path)

    # ── STAGE 1: Text Understanding ──────────────────────────
    text_engine = TextUnderstanding(
        nlp=env.nlp,
        scene_types=config.scene_types,
    )
    semantics = text_engine.parse(input_text)

    # ── Explainability: record ───────────────────────────────
    report = ExplainabilityReport()
    report.add_text_understanding(semantics)

    # ── STAGE 2: Vocabulary Normalization ────────────────────
    normalizer = VocabularyNormalizer(
        object_aliases=env.object_aliases,
        relationship_aliases=env.relationship_aliases,
    )
    semantics_before = semantics
    semantics = normalizer.normalize_semantics(semantics)
    report.add_normalization(semantics_before, semantics)

    # ── STAGE 3: Scene Graph Construction ────────────────────
    graph_builder = SceneGraphBuilder()
    scene_graph = graph_builder.build(semantics)
    report.add_scene_graph(scene_graph)

    # ── STAGE 4: Multi-Hypothesis Generation ─────────────────
    hyp_generator = HypothesisGenerator(
        num_hypotheses=config.generation.num_hypotheses,
        seed=config.generation.seed,
    )
    hypotheses = hyp_generator.generate(scene_graph)
    report.add_hypotheses(hypotheses)

    # ── Result Packaging (initialize) ────────────────────────
    packager = ResultPackager(base_output_dir=str(CFG_ROOT / config.output_dir))
    packager.initialize()

    # Save scene graph visualization
    sg_path = str(packager.run_dir / "scene_graphs" / "scene_graph.png")
    graph_builder.visualize(scene_graph, sg_path)

    # ── Process best hypothesis ──────────────────────────────
    best_hypothesis = hypotheses[0]
    logger.info(f"\n  Processing best hypothesis: H{best_hypothesis.hypothesis_id} "
                f"(confidence={best_hypothesis.confidence:.3f})")

    # ── STAGE 5: Spatial Layout Estimation ───────────────────
    layout_estimator = SpatialLayoutEstimator(resolution=config.hardware.resolution)
    layout = layout_estimator.estimate(best_hypothesis, scene_type=semantics.scene_type)
    report.add_spatial_layout(layout)

    # Save layout preview
    layout_preview_path = str(packager.run_dir / "images" / "layout_preview.png")
    layout_estimator.render_layout_preview(layout, layout_preview_path)

    # ── STAGE 6: Depth Map Generation ────────────────────────
    depth_gen = DepthMapGenerator(
        midas_model=env.midas_model if controlnet_mode else None,
        midas_feature_extractor=env.midas_feature_extractor if controlnet_mode else None,
    )
    depth_map = depth_gen.from_layout(layout)

    # Save depth map
    depth_path = str(packager.run_dir / "depth_maps" / "depth_map_h1.png")
    depth_map.save(depth_path)

    # ── STAGE 7: Image Generation ────────────────────────────
    use_two_pass = getattr(config.generation, 'two_pass', True)

    if controlnet_mode:
        # Load ControlNet pipeline
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

        if use_two_pass:
            # ── Two-pass generation ──────────────────────────
            # Pass 1: Low ControlNet influence with synthetic depth → base image
            # Pass 2: MiDaS depth from base image
            # Pass 3: High ControlNet influence with MiDaS depth → final
            logger.info("  Using TWO-PASS generation for higher quality")
            image, midas_depth = img_gen.two_pass_generate(
                prompt=prompt,
                depth_generator=depth_gen,
                midas_device=torch.device("cpu"),
                output_dir=str(packager.run_dir / "images"),
            )

            # Save the MiDaS depth map (overwrites synthetic one)
            midas_depth_path = str(packager.run_dir / "depth_maps" / "midas_depth_h1.png")
            Path(midas_depth_path).parent.mkdir(parents=True, exist_ok=True)
            midas_depth.save(midas_depth_path)
            # Update depth_map reference for multi-view
            depth_map = midas_depth
        else:
            # Single-pass: synthetic depth → ControlNet
            image = img_gen.generate_with_controlnet(
                prompt=prompt,
                depth_image=depth_map,
                output_path=str(packager.run_dir / "images" / "reconstruction_h1.png"),
            )
    else:
        # Fallback: text-to-image only
        sd_pipe = env.load_sd_pipeline()

        img_gen = ImageGenerator(
            sd_pipeline=sd_pipe,
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

        image = img_gen.generate_text_to_image(
            prompt=prompt,
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

    # ── STAGE 8: Multi-View Generation ───────────────────────
    if not skip_multiview and controlnet_mode and config.views:
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
        report.add_multiview_info([v.view_name for v in view_results])
    else:
        logger.info("  Multi-view generation skipped")

    # ── Unload generation models to free VRAM ────────────────
    env.unload_controlnet_pipeline()
    env.unload_sd_pipeline()
    flush_gpu_memory()

    # ── STAGE 10: Evaluation ─────────────────────────────────
    eval_results = {}
    if not skip_evaluation:
        evaluator = Evaluator(
            clip_model=env.clip_model,
            clip_preprocess=env.clip_preprocess,
            clip_tokenizer=env.clip_tokenizer,
            device="cpu",  # Evaluate on CPU to save VRAM
        )
        eval_result = evaluator.evaluate(
            image=image,
            prompt=prompt,
            semantics=semantics,
            scene_graph=scene_graph,
            hypothesis=best_hypothesis,
        )
        eval_results[best_hypothesis.hypothesis_id] = eval_result
        report.add_evaluation(eval_result.to_dict())
    else:
        logger.info("  Evaluation skipped")

    # ── STAGE 9: Explainability Report ───────────────────────
    report.print_summary()
    report.save(str(packager.run_dir / "reports" / "explainability_report.json"))

    # ── STAGE 11: Final Packaging ────────────────────────────
    if eval_results:
        packager.save_evaluation(eval_results)
        packager.save_hypothesis_ranking(hypotheses, eval_results)

    best_score = 0.0
    if eval_results:
        best_score = max(r.overall_score for r in eval_results.values())

    packager.create_summary(
        input_text=input_text,
        scene_type=semantics.scene_type,
        num_objects=len(semantics.objects),
        num_hypotheses=len(hypotheses),
        best_score=best_score,
    )

    output_dir = packager.package_complete()

    # ── Cleanup ──────────────────────────────────────────────
    env.cleanup_all()

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Crime Scene Reconstruction Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Natural language crime scene description",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--no-multiview",
        action="store_true",
        help="Skip multi-view generation",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation metrics",
    )
    parser.add_argument(
        "--no-controlnet",
        action="store_true",
        help="Use plain SD instead of ControlNet",
    )

    args = parser.parse_args()

    run_pipeline(
        input_text=args.input,
        config_path=args.config,
        skip_multiview=args.no_multiview,
        skip_evaluation=args.no_eval,
        controlnet_mode=not args.no_controlnet,
    )


if __name__ == "__main__":
    main()
