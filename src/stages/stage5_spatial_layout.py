"""
STAGE 5 — Spatial Layout Estimation
=====================================
Converts symbolic scene graph + hypothesis into a geometric layout.

Computes:
  - Object bounding regions in 2D image space
  - Relative depth ordering
  - Camera framing parameters
  - Occlusion ordering

Output: SpatialLayout used by depth map and image generation stages.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from src.stages.stage4_hypothesis_generation import SceneHypothesis, ObjectPlacement

logger = logging.getLogger(__name__)


@dataclass
class BoundingRegion:
    """2D bounding region for an object in image coordinates."""
    name: str
    x_center: int  # pixel
    y_center: int  # pixel
    width: int
    height: int
    depth_order: int  # 0 = farthest
    attributes: List[str] = field(default_factory=list)

    @property
    def x1(self) -> int:
        return self.x_center - self.width // 2

    @property
    def y1(self) -> int:
        return self.y_center - self.height // 2

    @property
    def x2(self) -> int:
        return self.x_center + self.width // 2

    @property
    def y2(self) -> int:
        return self.y_center + self.height // 2


@dataclass
class SpatialLayout:
    """Complete spatial configuration for image generation."""
    resolution: int = 512
    regions: List[BoundingRegion] = field(default_factory=list)
    depth_ordering: List[str] = field(default_factory=list)  # far to near
    scene_type: str = "unknown"
    hypothesis_id: int = 0

    def to_dict(self) -> dict:
        return {
            "resolution": self.resolution,
            "scene_type": self.scene_type,
            "hypothesis_id": self.hypothesis_id,
            "depth_ordering": self.depth_ordering,
            "regions": [
                {
                    "name": r.name,
                    "bbox": [r.x1, r.y1, r.x2, r.y2],
                    "depth_order": r.depth_order,
                    "attributes": r.attributes,
                }
                for r in self.regions
            ],
        }


class SpatialLayoutEstimator:
    """
    Converts a SceneHypothesis into pixel-space SpatialLayout.

    Maps normalized positions [0,1] → pixel coordinates with
    depth-aware sizing (farther objects appear smaller).
    """

    def __init__(self, resolution: int = 512):
        self.resolution = resolution
        # Base object size as fraction of image
        self.base_size = 0.12
        # Depth-to-size scaling
        self.depth_size_factor = 0.6

    def estimate(self, hypothesis: SceneHypothesis,
                  scene_type: str = "unknown") -> SpatialLayout:
        """
        Compute spatial layout from a hypothesis.

        Args:
            hypothesis: A SceneHypothesis with object placements.
            scene_type: Label for the scene type.

        Returns:
            SpatialLayout with bounding regions and depth ordering.
        """
        logger.info("=" * 50)
        logger.info("STAGE 5: Spatial Layout Estimation")
        logger.info("=" * 50)
        logger.info(f"  Hypothesis: {hypothesis.hypothesis_id} | Objects: {len(hypothesis.placements)}")

        layout = SpatialLayout(
            resolution=self.resolution,
            scene_type=scene_type,
            hypothesis_id=hypothesis.hypothesis_id,
        )

        # Sort placements by depth (farthest first)
        sorted_placements = sorted(hypothesis.placements, key=lambda p: p.depth)

        regions = []
        for depth_idx, placement in enumerate(sorted_placements):
            region = self._placement_to_region(placement, depth_idx)
            regions.append(region)
            logger.info(
                f"  {placement.name}: pos=({region.x_center},{region.y_center}) "
                f"size=({region.width}x{region.height}) depth_order={depth_idx}"
            )

        layout.regions = regions
        layout.depth_ordering = [p.name for p in sorted_placements]

        return layout

    def _placement_to_region(self, placement: ObjectPlacement,
                              depth_idx: int) -> BoundingRegion:
        """Convert normalized placement to pixel-space bounding region."""
        res = self.resolution

        # Position
        x_center = int(placement.x * res)
        y_center = int(placement.y * res)

        # Depth-aware sizing: farther objects are smaller
        depth_scale = 1.0 - (1.0 - placement.depth) * self.depth_size_factor
        obj_size = int(self.base_size * res * placement.scale * depth_scale)
        obj_size = max(obj_size, 20)  # Minimum 20px

        # Slight aspect ratio variation
        w = obj_size
        h = int(obj_size * (0.8 + 0.4 * (hash(placement.name) % 100) / 100))

        return BoundingRegion(
            name=placement.name,
            x_center=np.clip(x_center, w // 2, res - w // 2),
            y_center=np.clip(y_center, h // 2, res - h // 2),
            width=w,
            height=h,
            depth_order=depth_idx,
            attributes=placement.attributes,
        )

    def render_layout_preview(self, layout: SpatialLayout,
                               output_path: str) -> str:
        """
        Render a debug visualization of the spatial layout.

        Args:
            layout: The SpatialLayout to visualize.
            output_path: Path to save the preview image.

        Returns:
            Path to saved image.
        """
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        img = Image.new("RGB", (layout.resolution, layout.resolution), color=(30, 30, 40))
        draw = ImageDraw.Draw(img)

        colors = [
            (78, 205, 196), (255, 107, 107), (255, 230, 109),
            (85, 98, 112), (196, 77, 88), (130, 204, 221),
            (255, 165, 0), (144, 238, 144), (221, 160, 221),
        ]

        for i, region in enumerate(layout.regions):
            color = colors[i % len(colors)]
            # Draw bounding box
            draw.rectangle(
                [region.x1, region.y1, region.x2, region.y2],
                outline=color,
                width=2,
            )
            # Label
            label = region.name
            if region.attributes:
                label = f"{', '.join(region.attributes)} {label}"
            draw.text((region.x1, region.y1 - 12), label, fill=color)
            # Depth indicator
            draw.text(
                (region.x1, region.y2 + 2),
                f"d={region.depth_order}",
                fill=(150, 150, 150),
            )

        img.save(output_path)
        logger.info(f"  Layout preview saved: {output_path}")
        return output_path
