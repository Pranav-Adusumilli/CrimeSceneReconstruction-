"""
STAGE 8 — Multi-View Generation
=================================
Generates multiple camera viewpoints of the same crime scene.

Method:
  - Perturb the depth map for each view angle
  - Modify the text prompt with view-specific suffixes
  - Re-run ControlNet generation for each view

Views: front, left, right, overhead (configurable)
"""

import logging
from PIL import Image
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field

from src.stages.stage6_depth_map import DepthMapGenerator
from src.stages.stage7_image_generation import ImageGenerator
from src.utils.config import ViewConfig
from src.utils.memory import flush_gpu_memory

logger = logging.getLogger(__name__)


@dataclass
class ViewResult:
    """Result for a single viewpoint."""
    view_name: str
    image: Image.Image
    depth_map: Image.Image
    prompt: str
    output_path: str = ""


class MultiViewGenerator:
    """
    Generates multiple viewpoints of a scene by varying depth maps
    and prompts, then re-running ControlNet generation.
    """

    def __init__(self, image_generator: ImageGenerator,
                 depth_generator: DepthMapGenerator,
                 views: List[ViewConfig]):
        """
        Args:
            image_generator: ImageGenerator with loaded ControlNet pipeline.
            depth_generator: DepthMapGenerator for depth manipulation.
            views: List of ViewConfig entries defining each view.
        """
        self.image_generator = image_generator
        self.depth_generator = depth_generator
        self.views = views

    def generate_views(self, base_prompt: str, base_depth_map: Image.Image,
                        output_dir: str,
                        hypothesis_id: int = 1) -> List[ViewResult]:
        """
        Generate multi-view images from a base prompt and depth map.

        Args:
            base_prompt: The text prompt for the scene.
            base_depth_map: The base depth map (from Stage 6).
            output_dir: Directory to save generated views.
            hypothesis_id: ID of the hypothesis being rendered.

        Returns:
            List of ViewResult objects.
        """
        logger.info("=" * 50)
        logger.info("STAGE 8: Multi-View Generation")
        logger.info("=" * 50)
        logger.info(f"  Views to generate: {[v.name for v in self.views]}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = []

        for view_cfg in self.views:
            logger.info(f"\n  Generating view: {view_cfg.name}")

            # Shift depth map for this view
            shifted_depth = self.depth_generator.shift_depth_map(
                base_depth_map,
                shift_x=view_cfg.depth_shift,
                shift_y=0.0 if "overhead" not in view_cfg.name else -0.2,
                scale=1.0 if "zoom" not in view_cfg.name else 1.2,
            )

            # Modify prompt for this view
            view_prompt = base_prompt + view_cfg.prompt_suffix

            # Generate image
            out_path = str(Path(output_dir) / f"h{hypothesis_id}_{view_cfg.name}.png")
            depth_path = str(Path(output_dir) / f"h{hypothesis_id}_{view_cfg.name}_depth.png")

            # Save shifted depth
            shifted_depth.save(depth_path)

            image = self.image_generator.generate_with_controlnet(
                prompt=view_prompt,
                depth_image=shifted_depth,
                output_path=out_path,
            )

            results.append(ViewResult(
                view_name=view_cfg.name,
                image=image,
                depth_map=shifted_depth,
                prompt=view_prompt,
                output_path=out_path,
            ))

            # Free memory between views
            flush_gpu_memory()
            logger.info(f"  View '{view_cfg.name}' complete: {out_path}")

        logger.info(f"\n  Multi-view generation complete: {len(results)} views")
        return results
