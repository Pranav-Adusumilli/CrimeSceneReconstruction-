"""
STAGE 6 — Depth Map Generation
================================
Generates depth maps using MiDaS DPT Hybrid for ControlNet conditioning.

Two modes:
  1. From layout: Synthesize a depth map from spatial layout (no input image needed)
  2. From image: Estimate depth from an existing image using MiDaS

Output: Grayscale depth map image (512x512) for ControlNet.
"""

import logging
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from typing import Optional

from src.stages.stage5_spatial_layout import SpatialLayout
from src.utils.memory import flush_gpu_memory, log_memory

logger = logging.getLogger(__name__)


class DepthMapGenerator:
    """
    Generates depth maps for ControlNet conditioning.

    For the initial pipeline (no source image), we synthesize depth maps
    from the spatial layout. For refinement, MiDaS estimates depth from images.
    """

    def __init__(self, midas_model=None, midas_feature_extractor=None):
        """
        Args:
            midas_model: Pretrained DPTForDepthEstimation (can be None for layout-only mode).
            midas_feature_extractor: DPTFeatureExtractor.
        """
        self.midas_model = midas_model
        self.midas_feature_extractor = midas_feature_extractor

    def from_layout(self, layout: SpatialLayout) -> Image.Image:
        """
        Synthesize a room-aware depth map from the spatial layout.

        Creates a grayscale image where:
          - Darker = farther from camera
          - Lighter = closer to camera

        Generates proper room geometry: floor plane, back wall,
        side walls, and furniture-sized rectangular object regions.
        """
        logger.info("=" * 50)
        logger.info("STAGE 6: Depth Map Generation (from layout)")
        logger.info("=" * 50)

        res = layout.resolution
        depth_array = np.zeros((res, res), dtype=np.float32)

        # ── Room structure ──────────────────────────────────────
        # Back wall (top ~40% of image): uniform far depth
        wall_end = int(res * 0.40)
        depth_array[0:wall_end, :] = 0.15

        # Side wall gradients (perspective lines)
        for y in range(wall_end):
            # Left wall gradient
            left_boundary = int(res * 0.05 + (res * 0.15) * (1 - y / wall_end))
            for x in range(0, left_boundary):
                frac = x / max(left_boundary, 1)
                depth_array[y, x] = 0.12 + 0.10 * frac
            # Right wall gradient
            right_boundary = int(res * 0.95 - (res * 0.15) * (1 - y / wall_end))
            for x in range(right_boundary, res):
                frac = 1.0 - (x - right_boundary) / max(res - right_boundary, 1)
                depth_array[y, x] = 0.12 + 0.10 * frac

        # Floor plane (bottom ~60%): smooth depth gradient from far to near
        for y in range(wall_end, res):
            # Linear interpolation: top of floor is far (0.25), bottom is near (0.85)
            t = (y - wall_end) / max(res - wall_end - 1, 1)
            floor_depth = 0.25 + 0.60 * t
            depth_array[y, :] = floor_depth

        # ── Object regions ──────────────────────────────────────
        # Place objects as rectangular furniture-like regions
        for region in layout.regions:
            max_depth_order = max(r.depth_order for r in layout.regions) if layout.regions else 1
            if max_depth_order == 0:
                depth_val = 0.65
            else:
                depth_val = 0.35 + 0.50 * (region.depth_order / max_depth_order)

            # Rectangular region (furniture-like)
            half_w = max(region.width // 2, 8)
            half_h = max(region.height // 2, 8)

            x1 = max(0, region.x_center - half_w)
            x2 = min(res, region.x_center + half_w)
            y1 = max(0, region.y_center - half_h)
            y2 = min(res, region.y_center + half_h)

            # Edges slightly deeper (3D box effect)
            for py in range(y1, y2):
                for px in range(x1, x2):
                    # Distance from edge (normalized 0-1, 1 = center)
                    dx = min(px - x1, x2 - 1 - px) / max(half_w, 1)
                    dy = min(py - y1, y2 - 1 - py) / max(half_h, 1)
                    edge_factor = min(dx, dy)
                    edge_factor = min(edge_factor * 3.0, 1.0)  # Quick ramp

                    obj_depth = depth_val * (0.85 + 0.15 * edge_factor)
                    depth_array[py, px] = max(depth_array[py, px], obj_depth)

        # Normalize to [0, 255]
        depth_array = np.clip(depth_array, 0, 1)
        depth_uint8 = (depth_array * 255).astype(np.uint8)

        depth_image = Image.fromarray(depth_uint8, mode="L")

        # Apply Gaussian blur for smooth depth transitions
        depth_image = depth_image.filter(ImageFilter.GaussianBlur(radius=10))

        # Convert to RGB (ControlNet expects 3-channel)
        depth_rgb = depth_image.convert("RGB")

        logger.info(f"  Room-aware synthetic depth map generated: {res}x{res}")
        return depth_rgb

    def from_image(self, image: Image.Image, device: torch.device = None) -> Image.Image:
        """
        Estimate depth from an image using MiDaS DPT Hybrid.

        Args:
            image: Input PIL Image (RGB).
            device: Torch device for inference.

        Returns:
            Depth map as PIL Image (RGB, 512x512).
        """
        logger.info("STAGE 6: Depth Map Generation (from image via MiDaS)")

        if self.midas_model is None or self.midas_feature_extractor is None:
            logger.warning("MiDaS not loaded, falling back to uniform depth")
            return Image.new("RGB", (512, 512), color=(128, 128, 128))

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare input
        inputs = self.midas_feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Move model to device temporarily
        self.midas_model = self.midas_model.to(device)

        with torch.no_grad():
            outputs = self.midas_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to target size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(512, 512),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Normalize
        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_uint8 = (depth * 255).astype(np.uint8)

        # Move model back to CPU to save VRAM
        self.midas_model = self.midas_model.to("cpu")
        flush_gpu_memory()

        depth_image = Image.fromarray(depth_uint8, mode="L").convert("RGB")
        logger.info("  MiDaS depth estimation complete")
        return depth_image

    def shift_depth_map(self, depth_image: Image.Image,
                         shift_x: float = 0.0,
                         shift_y: float = 0.0,
                         scale: float = 1.0) -> Image.Image:
        """
        Apply a virtual camera shift to a depth map for multi-view generation.

        Args:
            depth_image: Source depth map (PIL RGB).
            shift_x: Horizontal shift [-1, 1].
            shift_y: Vertical shift [-1, 1].
            scale: Depth scaling factor.

        Returns:
            Modified depth map.
        """
        arr = np.array(depth_image).astype(np.float32)
        h, w = arr.shape[:2]

        # Apply shift via pixel translation
        shift_px_x = int(shift_x * w * 0.1)
        shift_px_y = int(shift_y * h * 0.1)

        shifted = np.zeros_like(arr)
        src_x1 = max(0, -shift_px_x)
        src_x2 = min(w, w - shift_px_x)
        src_y1 = max(0, -shift_px_y)
        src_y2 = min(h, h - shift_px_y)
        dst_x1 = max(0, shift_px_x)
        dst_x2 = min(w, w + shift_px_x)
        dst_y1 = max(0, shift_px_y)
        dst_y2 = min(h, h + shift_px_y)

        try:
            shifted[dst_y1:dst_y2, dst_x1:dst_x2] = arr[src_y1:src_y2, src_x1:src_x2]
        except ValueError:
            shifted = arr.copy()

        # Apply depth scale
        if scale != 1.0:
            shifted = np.clip(shifted * scale, 0, 255)

        return Image.fromarray(shifted.astype(np.uint8))
