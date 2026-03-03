"""
STAGE 9 — Explainability Module
=================================
Provides reasoning transparency for the reconstruction process.

Outputs:
  - Extracted objects and their attributes
  - Relationships used for graph construction
  - Hypothesis ranking rationale
  - Spatial assumptions made
  - Generation parameters used
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.stages.stage1_text_understanding import SceneSemantics
from src.stages.stage3_scene_graph import SceneGraph
from src.stages.stage4_hypothesis_generation import SceneHypothesis
from src.stages.stage5_spatial_layout import SpatialLayout

logger = logging.getLogger(__name__)


class ExplainabilityReport:
    """
    Captures and formats the full reasoning chain for a single pipeline run.
    """

    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.sections: Dict[str, Any] = {}

    def add_text_understanding(self, semantics: SceneSemantics):
        """Record Stage 1 outputs."""
        self.sections["text_understanding"] = {
            "raw_input": semantics.raw_text,
            "scene_type_detected": semantics.scene_type,
            "objects_extracted": semantics.objects,
            "attributes_extracted": semantics.attributes,
            "relationships_extracted": [
                {"subject": s, "predicate": p, "object": o}
                for s, p, o in semantics.relationships
            ],
            "num_sentences": len(semantics.sentences),
        }

    def add_normalization(self, before: SceneSemantics, after: SceneSemantics):
        """Record Stage 2 vocabulary normalization changes."""
        changes = []
        for old_obj, new_obj in zip(before.objects, after.objects):
            if old_obj != new_obj:
                changes.append({"from": old_obj, "to": new_obj})

        self.sections["normalization"] = {
            "objects_before": before.objects,
            "objects_after": after.objects,
            "changes": changes,
        }

    def add_scene_graph(self, scene_graph: SceneGraph):
        """Record Stage 3 graph structure."""
        self.sections["scene_graph"] = scene_graph.to_dict()

    def add_hypotheses(self, hypotheses: List[SceneHypothesis]):
        """Record Stage 4 hypothesis generation."""
        self.sections["hypotheses"] = {
            "count": len(hypotheses),
            "ranking": [
                {
                    "id": h.hypothesis_id,
                    "confidence": h.confidence,
                    "description": h.description,
                }
                for h in hypotheses
            ],
        }

    def add_spatial_layout(self, layout: SpatialLayout):
        """Record Stage 5 spatial layout."""
        self.sections["spatial_layout"] = layout.to_dict()

    def add_generation_params(self, prompt: str, negative_prompt: str,
                               steps: int, guidance_scale: float,
                               seed: int, resolution: int):
        """Record Stage 7 generation parameters."""
        self.sections["generation_params"] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "resolution": f"{resolution}x{resolution}",
        }

    def add_multiview_info(self, views: List[str]):
        """Record Stage 8 multi-view information."""
        self.sections["multiview"] = {
            "views_generated": views,
            "count": len(views),
        }

    def add_evaluation(self, eval_results: Dict[str, Any]):
        """Record Stage 10 evaluation results."""
        self.sections["evaluation"] = eval_results

    def add_custom(self, key: str, data: Any):
        """Add any custom section."""
        self.sections[key] = data

    def to_dict(self) -> dict:
        """Convert full report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "pipeline": "CrimeSceneReconstruction v1.0",
            "stages": self.sections,
        }

    def to_json(self) -> str:
        """Convert to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, output_path: str):
        """Save report to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        logger.info(f"  Explainability report saved: {output_path}")

    def print_summary(self):
        """Print a concise text summary to the logger."""
        logger.info("=" * 50)
        logger.info("STAGE 9: Explainability Summary")
        logger.info("=" * 50)

        tu = self.sections.get("text_understanding", {})
        logger.info(f"  Input: {tu.get('raw_input', 'N/A')[:80]}...")
        logger.info(f"  Scene type: {tu.get('scene_type_detected', 'N/A')}")
        logger.info(f"  Objects: {tu.get('objects_extracted', [])}")
        logger.info(f"  Relationships: {len(tu.get('relationships_extracted', []))}")

        hyp = self.sections.get("hypotheses", {})
        logger.info(f"  Hypotheses: {hyp.get('count', 0)}")
        for h in hyp.get("ranking", []):
            logger.info(f"    H{h['id']}: confidence={h['confidence']:.3f}")

        gen = self.sections.get("generation_params", {})
        logger.info(f"  Generation: steps={gen.get('num_inference_steps')}, "
                     f"cfg={gen.get('guidance_scale')}, seed={gen.get('seed')}")

        ev = self.sections.get("evaluation", {})
        if ev:
            logger.info(f"  CLIP score: {ev.get('clip_similarity', 'N/A')}")
            logger.info(f"  Spatial consistency: {ev.get('spatial_consistency', 'N/A')}")
