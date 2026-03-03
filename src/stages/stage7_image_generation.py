"""
STAGE 7 — Controlled Image Generation
=======================================
Generates photorealistic crime scene images using ControlNet + Stable Diffusion.

Pipeline: text prompt + depth map → ControlNet conditioning → SD generation.

Optimized for RTX 3060 6GB VRAM:
  - FP16 precision
  - Attention slicing
  - VAE slicing
  - Sequential CPU offload
  - 512x512 resolution
  - Batch size = 1
"""

import logging
import torch
from PIL import Image
from typing import Optional
from pathlib import Path

from src.stages.stage4_hypothesis_generation import SceneHypothesis
from src.utils.memory import flush_gpu_memory, log_memory

logger = logging.getLogger(__name__)


class ImageGenerator:
    """
    Generates images using Stable Diffusion with optional ControlNet depth conditioning.

    Supports two modes:
      1. ControlNet mode: depth-conditioned generation (primary pipeline)
      2. SD-only mode: text-to-image without depth conditioning (fallback/test)
    """

    def __init__(self, controlnet_pipeline=None, sd_pipeline=None,
                 num_inference_steps: int = 30,
                 guidance_scale: float = 7.5,
                 negative_prompt: str = "",
                 seed: int = 42):
        """
        Args:
            controlnet_pipeline: StableDiffusionControlNetPipeline (primary).
            sd_pipeline: StableDiffusionPipeline (fallback).
            num_inference_steps: Denoising steps.
            guidance_scale: CFG scale.
            negative_prompt: Negative prompt text.
            seed: Random seed for reproducibility.
        """
        self.controlnet_pipeline = controlnet_pipeline
        self.sd_pipeline = sd_pipeline
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.seed = seed

    def generate_with_controlnet(self, prompt: str, depth_image: Image.Image,
                                  output_path: str = None) -> Image.Image:
        """
        Generate an image conditioned on a depth map via ControlNet.

        Args:
            prompt: Text description of the scene.
            depth_image: Depth map (PIL Image, RGB, 512x512).
            output_path: Optional path to save the result.

        Returns:
            Generated PIL Image.
        """
        logger.info("=" * 50)
        logger.info("STAGE 7: Controlled Image Generation (ControlNet)")
        logger.info("=" * 50)
        logger.info(f"  Prompt: {prompt[:100]}...")
        logger.info(f"  Steps: {self.num_inference_steps} | CFG: {self.guidance_scale}")

        if self.controlnet_pipeline is None:
            raise RuntimeError("ControlNet pipeline not loaded. Call env.load_controlnet_pipeline() first.")

        # Ensure depth image is the right size
        depth_image = depth_image.resize((512, 512)).convert("RGB")

        generator = torch.Generator("cpu").manual_seed(self.seed)

        log_memory("Before ControlNet generation")

        with torch.inference_mode():
            result = self.controlnet_pipeline(
                prompt=prompt,
                image=depth_image,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                negative_prompt=self.negative_prompt,
                generator=generator,
            )

        image = result.images[0]

        log_memory("After ControlNet generation")

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            logger.info(f"  Image saved: {output_path}")

        return image

    def generate_text_to_image(self, prompt: str,
                                output_path: str = None) -> Image.Image:
        """
        Generate an image from text only (no depth conditioning).
        Used for testing and as a fallback.

        Args:
            prompt: Text description.
            output_path: Optional save path.

        Returns:
            Generated PIL Image.
        """
        logger.info("STAGE 7: Image Generation (text-to-image, no ControlNet)")
        logger.info(f"  Prompt: {prompt[:100]}...")

        if self.sd_pipeline is None:
            raise RuntimeError("SD pipeline not loaded. Call env.load_sd_pipeline() first.")

        generator = torch.Generator("cpu").manual_seed(self.seed)

        log_memory("Before SD generation")

        with torch.inference_mode():
            result = self.sd_pipeline(
                prompt=prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                negative_prompt=self.negative_prompt,
                generator=generator,
                height=512,
                width=512,
            )

        image = result.images[0]

        log_memory("After SD generation")

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            logger.info(f"  Image saved: {output_path}")

        return image

    def build_scene_prompt(self, scene_type: str, hypothesis: SceneHypothesis,
                            base_description: str = "") -> str:
        """
        Build a detailed, weighted generation prompt from hypothesis and scene type.

        Uses SD prompt weighting syntax (word:weight) to emphasize key objects.
        Stays within ~75 CLIP tokens to avoid truncation.
        """
        # Scene-type room templates (concise)
        ROOM_TEMPLATES = {
            "bedroom": "a bedroom",
            "living_room": "a living room",
            "kitchen": "a kitchen",
            "bathroom": "a bathroom",
            "hallway": "a hallway",
            "garage": "a garage",
            "basement": "a basement",
            "office": "an office",
            "alley": "a dark alley",
            "parking_lot": "a parking lot",
            "street": "a street at night",
            "warehouse": "a warehouse",
        }

        room_desc = ROOM_TEMPLATES.get(scene_type, f"a {scene_type}")

        # Build compact weighted object descriptions (no spatial — saves tokens)
        obj_descriptions = []
        for p in hypothesis.placements:
            name = p.name
            # Skip "bedroom" as object — it's already the scene type
            if name.lower() == scene_type.lower():
                continue
            attr_str = f"{' '.join(p.attributes)} " if p.attributes else ""
            obj_descriptions.append(f"({attr_str}{name}:1.3)")

        objects_text = ", ".join(obj_descriptions)

        # Compose prompt — front-load scene and objects, trim style to essentials
        prompt = (
            f"(crime scene photograph of {room_desc}:1.2), "
            f"{objects_text}, "
            f"(photorealistic:1.3), forensic photography, "
            f"indoor scene, harsh flash lighting, sharp focus, detailed, 8k"
        )

        logger.info(f"  Built prompt ({len(prompt.split())} words): {prompt[:200]}...")
        return prompt

    def two_pass_generate(self, prompt: str, depth_generator,
                           midas_device: torch.device = None,
                           output_dir: str = None) -> tuple:
        """
        Two-pass generation for higher quality:
          Pass 1: Text-to-image with SD (no depth conditioning)
          Pass 2: MiDaS depth from pass 1 → ControlNet refinement

        Returns:
            (final_image, midas_depth_map)
        """
        logger.info("="*50)
        logger.info("STAGE 7: Two-Pass Generation")
        logger.info("="*50)

        # Pass 1: Generate base image with SD text-to-image
        logger.info("  Pass 1: Text-to-image base generation...")
        if self.sd_pipeline is None and self.controlnet_pipeline is not None:
            # Use controlnet pipeline without depth for pass 1 by generating a flat depth
            flat_depth = Image.new("RGB", (512, 512), color=(128, 128, 128))
            generator = torch.Generator("cpu").manual_seed(self.seed)
            with torch.inference_mode():
                result = self.controlnet_pipeline(
                    prompt=prompt,
                    image=flat_depth,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    negative_prompt=self.negative_prompt,
                    generator=generator,
                    controlnet_conditioning_scale=0.3,  # Low influence for pass 1
                )
            base_image = result.images[0]
        elif self.sd_pipeline is not None:
            generator = torch.Generator("cpu").manual_seed(self.seed)
            with torch.inference_mode():
                result = self.sd_pipeline(
                    prompt=prompt,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    negative_prompt=self.negative_prompt,
                    generator=generator,
                    height=512, width=512,
                )
            base_image = result.images[0]
        else:
            raise RuntimeError("No pipeline loaded for pass 1")

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            base_image.save(str(Path(output_dir) / "pass1_base.png"))
            logger.info(f"  Pass 1 saved: {output_dir}/pass1_base.png")

        flush_gpu_memory()

        # Pass 2: Extract MiDaS depth from base image
        logger.info("  Pass 2: MiDaS depth estimation from base image...")
        device = midas_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas_depth = depth_generator.from_image(base_image, device=device)

        if output_dir:
            midas_depth.save(str(Path(output_dir) / "pass2_midas_depth.png"))
            logger.info(f"  MiDaS depth saved: {output_dir}/pass2_midas_depth.png")

        flush_gpu_memory()

        # Pass 3: ControlNet-conditioned refinement using MiDaS depth
        if self.controlnet_pipeline is not None:
            logger.info("  Pass 3: ControlNet refinement with MiDaS depth...")
            generator = torch.Generator("cpu").manual_seed(self.seed + 1)  # Slight variation
            with torch.inference_mode():
                result = self.controlnet_pipeline(
                    prompt=prompt,
                    image=midas_depth,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    negative_prompt=self.negative_prompt,
                    generator=generator,
                    controlnet_conditioning_scale=0.8,
                )
            final_image = result.images[0]
        else:
            logger.warning("  No ControlNet pipeline — using pass 1 result as final")
            final_image = base_image

        if output_dir:
            final_image.save(str(Path(output_dir) / "reconstruction_h1.png"))
            logger.info(f"  Final image saved: {output_dir}/reconstruction_h1.png")

        flush_gpu_memory()
        return final_image, midas_depth
