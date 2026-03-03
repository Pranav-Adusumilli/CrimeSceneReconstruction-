"""Integration tests for all new research framework modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import networkx as nx
import tempfile
import os

print("=== Component Integration Tests ===")
print()

# 1. Test ScoringWeights
print("1. ScoringWeights...")
from src.scoring.unified_scorer import ScoringWeights
from dataclasses import asdict
w = ScoringWeights()
w.validate()
wn = w.normalized()
total = sum(asdict(wn).values())
print(f"   Weights sum (normalized): {total:.4f}")
assert abs(total - 1.0) < 1e-6, "Normalization failed"
print("   OK")

# 2. Test Spatial Consistency Scorer
print("2. SpatialConsistencyScorer...")
from src.scoring.spatial_consistency import SpatialConsistencyScorer
from src.stages.stage3_scene_graph import SceneGraph
from src.stages.stage4_hypothesis_generation import SceneHypothesis, ObjectPlacement
from src.stages.stage5_spatial_layout import SpatialLayout, BoundingRegion

G = nx.DiGraph()
G.add_node("bed", attributes=["large"])
G.add_node("knife", attributes=["sharp"])
G.add_edge("knife", "bed", relation="on")

sg = SceneGraph(scene_type="bedroom", graph=G, objects=["bed", "knife"],
                relationships=[("knife", "on", "bed")])

hyp = SceneHypothesis(
    hypothesis_id=1, confidence=0.8,
    placements=[
        ObjectPlacement(name="bed", x=0.5, y=0.6, depth=0.3, scale=1.2),
        ObjectPlacement(name="knife", x=0.5, y=0.5, depth=0.3, scale=0.6),
    ])

layout = SpatialLayout(
    resolution=512,
    regions=[
        BoundingRegion(name="bed", x_center=256, y_center=307, width=100, height=80, depth_order=0),
        BoundingRegion(name="knife", x_center=256, y_center=256, width=40, height=30, depth_order=1),
    ],
    depth_ordering=["bed", "knife"],
    scene_type="bedroom")

scorer = SpatialConsistencyScorer()
result = scorer.compute(sg, hyp, layout)
score_val = result["score"]
print(f"   Spatial score: {score_val:.4f}")
assert 0 <= score_val <= 1, "Score out of range"
print("   OK")

# 3. Test Physical Plausibility
print("3. PhysicalPlausibilityScorer...")
from src.scoring.physical_plausibility import PhysicalPlausibilityScorer
phys = PhysicalPlausibilityScorer()
result = phys.compute(sg, hyp, layout)
score_val = result["score"]
print(f"   Physical score: {score_val:.4f}")
assert 0 <= score_val <= 1
print("   OK")

# 4. Test Probabilistic Prior
print("4. ProbabilisticPriorScorer...")
from src.scoring.probabilistic_prior import ProbabilisticPriorScorer
prior = ProbabilisticPriorScorer()
result = prior.compute(sg, hyp)
score_val = result["score"]
print(f"   Prior score: {score_val:.4f}")
assert 0 <= score_val <= 1
print("   OK")

# 5. Test Energy Optimizer (fast layout-only)
print("5. EnergyOptimizer...")
from src.optimization.energy_optimizer import EnergyOptimizer
from src.stages.stage5_spatial_layout import SpatialLayoutEstimator

# Minimal scorer for testing (layout-only components)
class MockScorer:
    def score_layout_only(self, sg, h, l):
        return (
            scorer.compute(sg, h, l)["score"] * 0.15 +
            phys.compute(sg, h, l)["score"] * 0.10 +
            prior.compute(sg, h)["score"] * 0.10
        )

mock_scorer = MockScorer()
optimizer = EnergyOptimizer(
    unified_scorer=mock_scorer,
    method="simulated_annealing",
    sa_config={"max_iterations": 50, "T_initial": 0.5, "cooling_rate": 0.95},
    seed=42,
)
opt_result = optimizer.optimize(hyp, sg, "bedroom")
print(f"   Best score: {opt_result.best_score:.4f} ({opt_result.total_iterations} iters, {opt_result.total_time_s:.1f}s)")
assert opt_result.best_hypothesis is not None
print("   OK")

# 6. Test Segmentation Layout
print("6. SegmentationLayoutGenerator...")
from src.conditioning.segmentation_layout import SegmentationLayoutGenerator
seg_gen = SegmentationLayoutGenerator(resolution=512)
seg_img, seg_info = seg_gen.generate(layout, "bedroom")
print(f"   Segmentation: {seg_img.size}, {seg_info.num_object_classes} classes")
assert seg_img.size == (512, 512)
print("   OK")

# 7. Test Research Logger
print("7. ResearchLogger...")
from src.experiments.research_logger import ResearchLogger
with tempfile.TemporaryDirectory() as tmpdir:
    rlog = ResearchLogger(base_dir=tmpdir)
    rlog.start_experiment("test_exp", config={"test": True})
    rlog.log_event("test", {"value": 42})
    rlog.finalize()
    assert os.path.exists(os.path.join(tmpdir, "test_exp", "summary.json"))
print("   OK")

# 8. Test Experiment Runner (dry run)
print("8. ExperimentRunner (dry run)...")
from src.experiments.experiment_runner import ExperimentRunner, BASELINE_TEXT_ONLY
runner = ExperimentRunner()
results = runner.run_comparison(
    configs=[BASELINE_TEXT_ONLY],
    test_scenes=[{"id": "test", "text": "Test scene", "scene_type": "bedroom", "expected_objects": []}],
)
assert len(results) == 1
mean_score = results[0].mean_scores.get("total_score", 0)
print(f"   Mean score: {mean_score:.4f}")
print("   OK")

# 9. Test Weight Calibrator
print("9. WeightCalibrator...")
from src.optimization.weight_calibration import WeightCalibrator
def dummy_eval(weights):
    return weights.w_semantic * 0.6 + weights.w_visual * 0.7
cal = WeightCalibrator(evaluation_fn=dummy_eval, seed=42)
cal_result = cal.calibrate(method="random_search", bayesian_iterations=10)
print(f"   Best objective: {cal_result.best_objective:.4f}")
assert cal_result.best_weights is not None
print("   OK")

# 10. Test Ablation Runner (dry run)
print("10. AblationRunner (dry run)...")
from src.experiments.ablation_runner import AblationRunner
abl_runner = AblationRunner()
abl_result = abl_runner.run()
abl_result.compute_deltas()
print(f"   Ablation configs: {len(abl_result.ablation_configs)}")
print("   OK")

print()
print("=== All 10 integration tests PASSED ===")
