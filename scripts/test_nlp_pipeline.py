"""
Test NLP Pipeline — Stages 1-5 (no GPU required)
==================================================
Validates text understanding, normalization, graph construction,
hypothesis generation, and spatial layout estimation.

Run: python scripts/test_nlp_pipeline.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.stages.stage0_init import initialize_environment
from src.stages.stage1_text_understanding import TextUnderstanding
from src.stages.stage2_vocabulary_normalization import VocabularyNormalizer
from src.stages.stage3_scene_graph import SceneGraphBuilder
from src.stages.stage4_hypothesis_generation import HypothesisGenerator
from src.stages.stage5_spatial_layout import SpatialLayoutEstimator
from src.stages.stage6_depth_map import DepthMapGenerator
from src.utils.logging_utils import setup_logging


def test_nlp_pipeline():
    setup_logging()

    print("=" * 60)
    print("  Test NLP Pipeline (Stages 1-6, CPU only)")
    print("=" * 60)

    # Test input
    test_input = "Small bedroom. Broken window. Knife on table. Blood near sofa."

    print(f"\n  Input: {test_input}")

    # Stage 0: Init (lightweight)
    env = initialize_environment()

    # Stage 1: Text Understanding
    text_engine = TextUnderstanding(nlp=env.nlp, scene_types=env.config.scene_types)
    semantics = text_engine.parse(test_input)

    print(f"\n  Scene type: {semantics.scene_type}")
    print(f"  Objects: {semantics.objects}")
    print(f"  Attributes: {semantics.attributes}")
    print(f"  Relationships: {semantics.relationships}")

    # Stage 2: Vocabulary Normalization
    normalizer = VocabularyNormalizer(
        object_aliases=env.object_aliases,
        relationship_aliases=env.relationship_aliases,
    )
    semantics = normalizer.normalize_semantics(semantics)

    print(f"\n  After normalization:")
    print(f"  Objects: {semantics.objects}")
    print(f"  Relationships: {semantics.relationships}")

    # Stage 3: Scene Graph
    graph_builder = SceneGraphBuilder()
    scene_graph = graph_builder.build(semantics)

    print(f"\n  Scene Graph: {scene_graph.graph.number_of_nodes()} nodes, "
          f"{scene_graph.graph.number_of_edges()} edges")
    print(f"  Graph JSON:\n{scene_graph.to_json()}")

    # Visualize
    os.makedirs("outputs/scene_graphs", exist_ok=True)
    graph_builder.visualize(scene_graph, "outputs/scene_graphs/test_scene_graph.png")

    # Stage 4: Multi-Hypothesis
    hyp_gen = HypothesisGenerator(num_hypotheses=3, seed=42)
    hypotheses = hyp_gen.generate(scene_graph)

    print(f"\n  Generated {len(hypotheses)} hypotheses:")
    for h in hypotheses:
        print(f"    H{h.hypothesis_id}: confidence={h.confidence:.3f}")
        for p in h.placements:
            print(f"      {p.name}: x={p.x:.2f}, y={p.y:.2f}, depth={p.depth:.2f}")

    # Stage 5: Spatial Layout
    layout_est = SpatialLayoutEstimator(resolution=512)
    layout = layout_est.estimate(hypotheses[0], scene_type=semantics.scene_type)

    print(f"\n  Layout for best hypothesis:")
    print(f"  Depth ordering: {layout.depth_ordering}")
    for r in layout.regions:
        print(f"    {r.name}: ({r.x1},{r.y1})-({r.x2},{r.y2}) depth={r.depth_order}")

    layout_est.render_layout_preview(layout, "outputs/images/test_layout_preview.png")

    # Stage 6: Synthetic Depth Map
    depth_gen = DepthMapGenerator()
    depth_map = depth_gen.from_layout(layout)
    os.makedirs("outputs/depth_maps", exist_ok=True)
    depth_map.save("outputs/depth_maps/test_depth_map.png")

    print(f"\n  Depth map saved: outputs/depth_maps/test_depth_map.png")

    print("\n" + "=" * 60)
    print("  [PASS] NLP Pipeline test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_nlp_pipeline()
