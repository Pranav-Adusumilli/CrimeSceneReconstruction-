"""
Conditioning Package
=====================
Additional conditioning signals for the diffusion pipeline.

Components:
  - SegmentationLayoutGenerator: Semantic segmentation maps from layouts
"""

from src.conditioning.segmentation_layout import SegmentationLayoutGenerator

__all__ = ["SegmentationLayoutGenerator"]
