"""
Segmentation + Depth Conditioned Generation
=============================================

Generates semantic segmentation maps from spatial layouts to provide
additional conditioning alongside depth maps.

Semantic Segmentation Map:
  - Each pixel is assigned a class label based on the layout's
    bounding regions and scene structure (floor, wall, ceiling).
  - Color-coded using ADE20K-consistent palette for compatibility
    with ControlNet-seg models.
  - Can be combined with depth maps for dual-conditioned generation
    or used independently.

The segmentation map encodes:
  1. Room structural elements (floor, wall, ceiling, window, door)
  2. Object regions from the spatial layout
  3. Background fill based on scene type

This provides the diffusion model with explicit spatial guidance
about WHAT should appear WHERE, complementing the depth map's
guidance about geometric structure.
"""

import logging
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from src.stages.stage5_spatial_layout import SpatialLayout, BoundingRegion

logger = logging.getLogger(__name__)


# ── Semantic Class Definitions ──────────────────────────────────────

# ADE20K-style class palette (subset relevant to crime scenes)
# Format: class_name → (R, G, B)
SEMANTIC_PALETTE = {
    "wall": (120, 120, 120),
    "floor": (180, 120, 120),
    "ceiling": (6, 230, 230),
    "bed": (4, 200, 3),
    "table": (204, 5, 255),
    "chair": (235, 255, 7),
    "sofa": (150, 5, 61),
    "door": (8, 255, 51),
    "window": (255, 6, 82),
    "cabinet": (143, 255, 140),
    "shelf": (204, 255, 4),
    "lamp": (255, 51, 7),
    "rug": (204, 70, 3),
    "curtain": (0, 102, 200),
    "desk": (61, 230, 250),
    "counter": (255, 6, 51),
    "sink": (11, 102, 255),
    "stove": (255, 7, 71),
    "refrigerator": (255, 9, 224),
    "toilet": (9, 7, 230),
    "bathtub": (220, 220, 220),
    "mirror": (255, 255, 0),
    "person": (220, 20, 60),
    "body": (220, 20, 60),
    "blood": (128, 0, 0),
    "knife": (192, 192, 128),
    "weapon": (192, 192, 128),
    "gun": (192, 192, 128),
    "car": (0, 0, 142),
    "generic_object": (128, 128, 0),
    "background": (0, 0, 0),
}

# Object-to-semantic-class mapping (fuzzy matching)
OBJECT_CLASS_MAP = {
    "bed": "bed",
    "mattress": "bed",
    "pillow": "bed",
    "blanket": "bed",
    "table": "table",
    "desk": "desk",
    "nightstand": "table",
    "end table": "table",
    "coffee table": "table",
    "dining table": "table",
    "chair": "chair",
    "sofa": "sofa",
    "couch": "sofa",
    "armchair": "sofa",
    "door": "door",
    "window": "window",
    "cabinet": "cabinet",
    "dresser": "cabinet",
    "wardrobe": "cabinet",
    "closet": "cabinet",
    "shelf": "shelf",
    "bookshelf": "shelf",
    "lamp": "lamp",
    "light": "lamp",
    "rug": "rug",
    "carpet": "rug",
    "curtain": "curtain",
    "counter": "counter",
    "sink": "sink",
    "stove": "stove",
    "oven": "stove",
    "refrigerator": "refrigerator",
    "fridge": "refrigerator",
    "toilet": "toilet",
    "bathtub": "bathtub",
    "shower": "bathtub",
    "mirror": "mirror",
    "person": "person",
    "body": "body",
    "victim": "body",
    "blood": "blood",
    "bloodstain": "blood",
    "blood pool": "blood",
    "knife": "knife",
    "weapon": "weapon",
    "gun": "gun",
    "pistol": "gun",
    "car": "car",
    "vehicle": "car",
}


@dataclass
class SegmentationInfo:
    """Metadata about a generated segmentation map."""
    class_counts: Dict[str, int] = field(default_factory=dict)
    total_pixels: int = 0
    num_object_classes: int = 0
    has_room_structure: bool = True


class SegmentationLayoutGenerator:
    """
    Generates semantic segmentation maps from spatial layouts.

    Creates color-coded segmentation images where each pixel's color
    indicates its semantic class. These maps can be used as:
      1. Additional ControlNet conditioning (ControlNet-seg)
      2. Compositing masks for object-level generation
      3. Evaluation reference for spatial accuracy

    Room structure is inferred from scene_type:
      - Indoor scenes: wall (top), floor (bottom), ceiling (top strip)
      - Outdoor scenes: ground (bottom), sky (top)
    """

    def __init__(self, resolution: int = 512):
        self.resolution = resolution

    def generate(self, layout: SpatialLayout,
                 scene_type: str = "unknown") -> Tuple[Image.Image, SegmentationInfo]:
        """
        Generate a semantic segmentation map from the spatial layout.

        Args:
            layout: SpatialLayout with bounding regions.
            scene_type: Scene type for room structure inference.

        Returns:
            Tuple of (segmentation_image, segmentation_info).
        """
        logger.info("=" * 50)
        logger.info("SEGMENTATION LAYOUT GENERATION")
        logger.info(f"  Scene type: {scene_type} | Objects: {len(layout.regions)}")
        logger.info("=" * 50)

        res = self.resolution
        seg_array = np.zeros((res, res, 3), dtype=np.uint8)
        class_counts: Dict[str, int] = {}

        # ── Room structure ──────────────────────────────────────
        is_indoor = scene_type in {
            "bedroom", "living_room", "kitchen", "bathroom",
            "hallway", "office", "basement", "warehouse", "garage",
        }

        if is_indoor:
            seg_array = self._draw_indoor_structure(seg_array, scene_type)
            class_counts["wall"] = int(np.sum(np.all(
                seg_array == SEMANTIC_PALETTE["wall"], axis=-1)))
            class_counts["floor"] = int(np.sum(np.all(
                seg_array == SEMANTIC_PALETTE["floor"], axis=-1)))
        else:
            seg_array = self._draw_outdoor_structure(seg_array, scene_type)

        # ── Object regions ──────────────────────────────────────
        # Draw farthest objects first (painter's algorithm)
        sorted_regions = sorted(layout.regions, key=lambda r: r.depth_order)

        for region in sorted_regions:
            sem_class = self._classify_object(region.name)
            color = SEMANTIC_PALETTE.get(sem_class, SEMANTIC_PALETTE["generic_object"])

            x1 = max(0, region.x1)
            y1 = max(0, region.y1)
            x2 = min(res, region.x2)
            y2 = min(res, region.y2)

            if x2 > x1 and y2 > y1:
                seg_array[y1:y2, x1:x2] = color
                pixel_count = (x2 - x1) * (y2 - y1)
                class_counts[sem_class] = class_counts.get(sem_class, 0) + pixel_count

            logger.info(f"  Region: {region.name} → class '{sem_class}' "
                        f"bbox=[{x1},{y1},{x2},{y2}]")

        # Create PIL image
        seg_image = Image.fromarray(seg_array, mode="RGB")

        info = SegmentationInfo(
            class_counts=class_counts,
            total_pixels=res * res,
            num_object_classes=len(set(
                self._classify_object(r.name) for r in layout.regions)),
            has_room_structure=is_indoor,
        )

        logger.info(f"  Generated segmentation: {len(class_counts)} classes, "
                     f"{info.num_object_classes} object types")

        return seg_image, info

    def _draw_indoor_structure(self, seg_array: np.ndarray,
                                scene_type: str) -> np.ndarray:
        """Draw indoor room structure: ceiling, walls, floor."""
        res = seg_array.shape[0]

        # Ceiling strip (top 5%)
        ceiling_end = int(res * 0.05)
        seg_array[0:ceiling_end, :] = SEMANTIC_PALETTE["ceiling"]

        # Back wall (5%–40%)
        wall_end = int(res * 0.40)
        seg_array[ceiling_end:wall_end, :] = SEMANTIC_PALETTE["wall"]

        # Side walls with perspective
        for y in range(ceiling_end, wall_end):
            frac = (y - ceiling_end) / max(wall_end - ceiling_end, 1)
            left_boundary = int(res * 0.02 + res * 0.08 * (1 - frac))
            right_boundary = int(res * 0.98 - res * 0.08 * (1 - frac))
            seg_array[y, 0:left_boundary] = SEMANTIC_PALETTE["wall"]
            seg_array[y, right_boundary:res] = SEMANTIC_PALETTE["wall"]

        # Floor (40%–100%)
        seg_array[wall_end:res, :] = SEMANTIC_PALETTE["floor"]

        # Scene-specific additions
        if scene_type == "bathroom":
            # Add mirror area on back wall
            mirror_y1 = ceiling_end + int((wall_end - ceiling_end) * 0.2)
            mirror_y2 = ceiling_end + int((wall_end - ceiling_end) * 0.6)
            mirror_x1 = int(res * 0.35)
            mirror_x2 = int(res * 0.65)
            seg_array[mirror_y1:mirror_y2, mirror_x1:mirror_x2] = SEMANTIC_PALETTE["mirror"]
        elif scene_type == "kitchen":
            # Counter area at wall-floor junction
            counter_y1 = wall_end - int(res * 0.05)
            counter_y2 = wall_end + int(res * 0.03)
            seg_array[counter_y1:counter_y2, :] = SEMANTIC_PALETTE["counter"]

        return seg_array

    def _draw_outdoor_structure(self, seg_array: np.ndarray,
                                 scene_type: str) -> np.ndarray:
        """Draw outdoor structure: sky, ground."""
        res = seg_array.shape[0]

        # Sky (top 40%)
        sky_end = int(res * 0.40)
        seg_array[0:sky_end, :] = (70, 130, 180)  # Sky blue

        # Ground (bottom 60%)
        if scene_type in {"street", "parking_lot"}:
            seg_array[sky_end:res, :] = (128, 128, 128)  # Asphalt gray
        elif scene_type == "alley":
            seg_array[sky_end:res, :] = (60, 60, 60)  # Dark ground
            # Alley walls on sides
            seg_array[0:res, 0:int(res * 0.15)] = SEMANTIC_PALETTE["wall"]
            seg_array[0:res, int(res * 0.85):res] = SEMANTIC_PALETTE["wall"]
        else:
            seg_array[sky_end:res, :] = (0, 128, 0)  # Green ground

        return seg_array

    def _classify_object(self, name: str) -> str:
        """Map an object name to a semantic class."""
        name_lower = name.lower().strip()

        # Direct match
        if name_lower in OBJECT_CLASS_MAP:
            return OBJECT_CLASS_MAP[name_lower]

        # Partial match
        for key, cls in OBJECT_CLASS_MAP.items():
            if key in name_lower or name_lower in key:
                return cls

        return "generic_object"

    def generate_composite_conditioning(
            self, layout: SpatialLayout,
            depth_map: Image.Image,
            scene_type: str = "unknown",
            depth_weight: float = 0.6,
            seg_weight: float = 0.4) -> Image.Image:
        """
        Generate a combined depth+segmentation conditioning image.

        Blends depth map with segmentation map using specified weights.
        This provides dual conditioning when used with ControlNet.

        Args:
            layout: Spatial layout.
            depth_map: Depth map image.
            scene_type: Scene type.
            depth_weight: Weight for depth map channel.
            seg_weight: Weight for segmentation channel.

        Returns:
            Blended conditioning image.
        """
        seg_image, _ = self.generate(layout, scene_type)

        # Ensure same size
        depth_resized = depth_map.resize((self.resolution, self.resolution)).convert("RGB")
        seg_resized = seg_image.resize((self.resolution, self.resolution))

        # Weighted blend
        depth_arr = np.array(depth_resized, dtype=np.float32)
        seg_arr = np.array(seg_resized, dtype=np.float32)

        blended = (depth_weight * depth_arr + seg_weight * seg_arr)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended, mode="RGB")
