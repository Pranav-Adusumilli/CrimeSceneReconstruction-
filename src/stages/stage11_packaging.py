"""
STAGE 11 — Result Packaging
==============================
Assembles all outputs into a structured final deliverable.

Final output includes:
  - Generated images (per hypothesis)
  - Multi-view images
  - Scene graph visualizations
  - Depth maps
  - Reasoning/explainability log (JSON)
  - Evaluation scores
  - Ranked hypotheses
"""

import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image

from src.stages.stage4_hypothesis_generation import SceneHypothesis
from src.stages.stage9_explainability import ExplainabilityReport
from src.stages.stage10_evaluation import EvaluationResult

logger = logging.getLogger(__name__)


class ResultPackager:
    """
    Packages all pipeline outputs into a structured directory.
    """

    def __init__(self, base_output_dir: str = "outputs"):
        self.base_dir = Path(base_output_dir)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / f"run_{self.run_id}"

    def initialize(self):
        """Create the output directory structure."""
        dirs = [
            self.run_dir,
            self.run_dir / "images",
            self.run_dir / "depth_maps",
            self.run_dir / "multiview",
            self.run_dir / "scene_graphs",
            self.run_dir / "reports",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"  Output directory: {self.run_dir}")

    def save_image(self, image: Image.Image, name: str,
                    subfolder: str = "images") -> str:
        """Save an image to the run directory."""
        path = self.run_dir / subfolder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(path))
        return str(path)

    def save_depth_map(self, depth_image: Image.Image, name: str) -> str:
        """Save a depth map."""
        return self.save_image(depth_image, name, subfolder="depth_maps")

    def save_scene_graph(self, graph_image_path: str, name: str = None) -> str:
        """Copy scene graph visualization to output directory."""
        src = Path(graph_image_path)
        if not src.exists():
            return ""
        dst = self.run_dir / "scene_graphs" / (name or src.name)
        shutil.copy2(str(src), str(dst))
        return str(dst)

    def save_report(self, report: ExplainabilityReport) -> str:
        """Save the explainability report."""
        path = self.run_dir / "reports" / "explainability_report.json"
        report.save(str(path))
        return str(path)

    def save_evaluation(self, results: Dict[int, EvaluationResult]) -> str:
        """Save evaluation results for all hypotheses."""
        path = self.run_dir / "reports" / "evaluation_results.json"
        data = {
            f"hypothesis_{h_id}": result.to_dict()
            for h_id, result in results.items()
        }
        with open(str(path), "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"  Evaluation results saved: {path}")
        return str(path)

    def save_hypothesis_ranking(self, hypotheses: List[SceneHypothesis],
                                  eval_results: Dict[int, EvaluationResult]) -> str:
        """Save final ranked hypothesis list."""
        path = self.run_dir / "reports" / "hypothesis_ranking.json"

        ranking = []
        for h in hypotheses:
            entry = {
                "hypothesis_id": h.hypothesis_id,
                "confidence": h.confidence,
                "description": h.description,
            }
            if h.hypothesis_id in eval_results:
                entry["evaluation"] = eval_results[h.hypothesis_id].to_dict()
            ranking.append(entry)

        # Sort by overall score if available
        ranking.sort(
            key=lambda x: x.get("evaluation", {}).get("overall_score", x["confidence"]),
            reverse=True,
        )

        with open(str(path), "w") as f:
            json.dump({"ranking": ranking}, f, indent=2)

        logger.info(f"  Hypothesis ranking saved: {path}")
        return str(path)

    def create_summary(self, input_text: str, scene_type: str,
                        num_objects: int, num_hypotheses: int,
                        best_score: float) -> str:
        """Create a concise run summary."""
        path = self.run_dir / "reports" / "summary.json"

        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "input_text": input_text,
            "scene_type": scene_type,
            "num_objects_detected": num_objects,
            "num_hypotheses": num_hypotheses,
            "best_overall_score": round(best_score, 4),
            "output_directory": str(self.run_dir),
            "contents": {
                "images": list(str(p.name) for p in (self.run_dir / "images").glob("*.png")),
                "depth_maps": list(str(p.name) for p in (self.run_dir / "depth_maps").glob("*.png")),
                "multiview": list(str(p.name) for p in (self.run_dir / "multiview").glob("*.png")),
                "scene_graphs": list(str(p.name) for p in (self.run_dir / "scene_graphs").glob("*.png")),
            },
        }

        with open(str(path), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"  Run summary saved: {path}")
        return str(path)

    def package_complete(self) -> str:
        """
        Print final packaging summary.

        Returns:
            Path to the run directory.
        """
        logger.info("=" * 50)
        logger.info("STAGE 11: Result Packaging Complete")
        logger.info("=" * 50)
        logger.info(f"  Run ID: {self.run_id}")
        logger.info(f"  Output: {self.run_dir}")

        # Count files
        total = sum(1 for _ in self.run_dir.rglob("*") if _.is_file())
        logger.info(f"  Total files: {total}")

        return str(self.run_dir)
