"""
STAGE 0 — Environment Initialization
=====================================
Loads all models, configures memory optimizations, and prepares runtime.

This module is the entry point for initializing the full pipeline state,
including Stable Diffusion, ControlNet, MiDaS, CLIP, alias dictionaries,
and spaCy NLP.
"""

import gc
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.config import Config, load_config, PROJECT_ROOT
from src.utils.memory import (
    flush_gpu_memory,
    optimize_pipeline_memory,
    get_gpu_memory_info,
    log_memory,
)

logger = logging.getLogger(__name__)


class RuntimeEnvironment:
    """
    Holds all loaded models and resources for the pipeline.
    Provides lazy loading to manage 6GB VRAM constraints.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.hardware.torch_device
        self.dtype = config.hardware.torch_dtype

        # Model slots (lazy loaded)
        self._sd_pipeline = None
        self._controlnet_pipeline = None
        self._midas_model = None
        self._midas_feature_extractor = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._nlp = None

        # Data slots
        self._object_aliases: Optional[Dict[str, str]] = None
        self._relationship_aliases: Optional[Dict[str, str]] = None

        logger.info(f"RuntimeEnvironment created | device={self.device} | dtype={self.dtype}")

    # ── Alias Dictionaries ──────────────────────────────────────

    def load_aliases(self):
        """Load Visual Genome alias files for vocabulary normalization."""
        obj_alias_path = self.config.resolve_path(self.config.object_alias_file)
        rel_alias_path = self.config.resolve_path(self.config.relationship_alias_file)

        self._object_aliases = _parse_alias_file(obj_alias_path)
        self._relationship_aliases = _parse_alias_file(rel_alias_path)

        logger.info(
            f"Loaded aliases: {len(self._object_aliases)} objects, "
            f"{len(self._relationship_aliases)} relationships"
        )

    @property
    def object_aliases(self) -> Dict[str, str]:
        if self._object_aliases is None:
            self.load_aliases()
        return self._object_aliases

    @property
    def relationship_aliases(self) -> Dict[str, str]:
        if self._relationship_aliases is None:
            self.load_aliases()
        return self._relationship_aliases

    # ── spaCy NLP ───────────────────────────────────────────────

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy en_core_web_sm loaded")
        return self._nlp

    # ── Stable Diffusion (vanilla, no ControlNet) ───────────────

    def load_sd_pipeline(self):
        """Load vanilla Stable Diffusion pipeline (for test generation)."""
        from diffusers import StableDiffusionPipeline

        flush_gpu_memory()
        logger.info(f"Loading SD model: {self.config.sd_model.model_id} (FP16)...")

        cache_dir = str(self.config.resolve_path(self.config.sd_model.cache_dir))

        # Load external VAE if configured (e.g., sd-vae-ft-mse for Realistic Vision)
        vae = None
        if self.config.vae_model and self.config.vae_model.model_id:
            from diffusers import AutoencoderKL
            vae_cache = str(self.config.resolve_path(self.config.vae_model.cache_dir))
            logger.info(f"  Loading external VAE: {self.config.vae_model.model_id}")
            vae = AutoencoderKL.from_pretrained(
                self.config.vae_model.model_id,
                torch_dtype=self.dtype,
                cache_dir=vae_cache,
            )

        self._sd_pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.sd_model.model_id,
            vae=vae,
            revision=self.config.sd_model.revision,
            torch_dtype=self.dtype,
            cache_dir=cache_dir,
            safety_checker=None,
            requires_safety_checker=False,
        )

        optimize_pipeline_memory(
            self._sd_pipeline,
            enable_cpu_offload=self.config.hardware.enable_cpu_offload,
            enable_attention_slicing=self.config.hardware.enable_attention_slicing,
            enable_vae_slicing=self.config.hardware.enable_vae_slicing,
        )

        log_memory("SD loaded")
        return self._sd_pipeline

    @property
    def sd_pipeline(self):
        if self._sd_pipeline is None:
            self.load_sd_pipeline()
        return self._sd_pipeline

    def unload_sd_pipeline(self):
        """Free SD pipeline from memory."""
        if self._sd_pipeline is not None:
            del self._sd_pipeline
            self._sd_pipeline = None
            flush_gpu_memory()
            logger.info("SD pipeline unloaded")

    # ── ControlNet + SD Pipeline ────────────────────────────────

    def load_controlnet_pipeline(self):
        """Load ControlNet + Stable Diffusion pipeline for depth-conditioned generation."""
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

        flush_gpu_memory()
        logger.info("Loading ControlNet Depth + SD pipeline...")

        cn_cache = str(self.config.resolve_path(self.config.controlnet_model.cache_dir))
        sd_cache = str(self.config.resolve_path(self.config.sd_model.cache_dir))

        controlnet = ControlNetModel.from_pretrained(
            self.config.controlnet_model.model_id,
            torch_dtype=self.dtype,
            cache_dir=cn_cache,
        )

        # Load external VAE if configured
        vae = None
        if self.config.vae_model and self.config.vae_model.model_id:
            from diffusers import AutoencoderKL
            vae_cache = str(self.config.resolve_path(self.config.vae_model.cache_dir))
            logger.info(f"  Loading external VAE: {self.config.vae_model.model_id}")
            vae = AutoencoderKL.from_pretrained(
                self.config.vae_model.model_id,
                torch_dtype=self.dtype,
                cache_dir=vae_cache,
            )

        self._controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.sd_model.model_id,
            controlnet=controlnet,
            vae=vae,
            revision=self.config.sd_model.revision,
            torch_dtype=self.dtype,
            cache_dir=sd_cache,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Use fast scheduler
        self._controlnet_pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self._controlnet_pipeline.scheduler.config
        )

        optimize_pipeline_memory(
            self._controlnet_pipeline,
            enable_cpu_offload=self.config.hardware.enable_cpu_offload,
            enable_attention_slicing=self.config.hardware.enable_attention_slicing,
            enable_vae_slicing=self.config.hardware.enable_vae_slicing,
        )

        log_memory("ControlNet pipeline loaded")
        return self._controlnet_pipeline

    @property
    def controlnet_pipeline(self):
        if self._controlnet_pipeline is None:
            self.load_controlnet_pipeline()
        return self._controlnet_pipeline

    def unload_controlnet_pipeline(self):
        if self._controlnet_pipeline is not None:
            del self._controlnet_pipeline
            self._controlnet_pipeline = None
            flush_gpu_memory()
            logger.info("ControlNet pipeline unloaded")

    # ── MiDaS Depth Estimator ──────────────────────────────────

    def load_midas(self):
        """Load MiDaS DPT Hybrid depth estimation model."""
        from transformers import DPTForDepthEstimation, DPTFeatureExtractor

        flush_gpu_memory()
        logger.info("Loading MiDaS DPT Hybrid...")

        cache_dir = str(self.config.resolve_path(self.config.midas_model.cache_dir))

        self._midas_feature_extractor = DPTFeatureExtractor.from_pretrained(
            self.config.midas_model.model_id,
            cache_dir=cache_dir,
        )
        self._midas_model = DPTForDepthEstimation.from_pretrained(
            self.config.midas_model.model_id,
            cache_dir=cache_dir,
        )
        self._midas_model.eval()
        # Keep MiDaS on CPU to save VRAM — move to GPU only during inference
        self._midas_model = self._midas_model.to("cpu")

        log_memory("MiDaS loaded (CPU)")

    @property
    def midas_model(self):
        if self._midas_model is None:
            self.load_midas()
        return self._midas_model

    @property
    def midas_feature_extractor(self):
        if self._midas_feature_extractor is None:
            self.load_midas()
        return self._midas_feature_extractor

    def unload_midas(self):
        if self._midas_model is not None:
            del self._midas_model
            del self._midas_feature_extractor
            self._midas_model = None
            self._midas_feature_extractor = None
            flush_gpu_memory()
            logger.info("MiDaS unloaded")

    # ── CLIP Model (for evaluation) ────────────────────────────

    def load_clip(self):
        """Load CLIP model for text-image similarity evaluation."""
        import open_clip

        flush_gpu_memory()
        logger.info("Loading CLIP model for evaluation...")

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.config.clip_model_name,
            pretrained=self.config.clip_pretrained,
            device="cpu",  # Stay on CPU, move during eval
        )
        tokenizer = open_clip.get_tokenizer(self.config.clip_model_name)

        self._clip_model = model.eval()
        self._clip_preprocess = preprocess
        self._clip_tokenizer = tokenizer

        log_memory("CLIP loaded (CPU)")

    @property
    def clip_model(self):
        if self._clip_model is None:
            self.load_clip()
        return self._clip_model

    @property
    def clip_preprocess(self):
        if self._clip_preprocess is None:
            self.load_clip()
        return self._clip_preprocess

    @property
    def clip_tokenizer(self):
        if self._clip_tokenizer is None:
            self.load_clip()
        return self._clip_tokenizer

    def unload_clip(self):
        if self._clip_model is not None:
            del self._clip_model, self._clip_preprocess, self._clip_tokenizer
            self._clip_model = None
            self._clip_preprocess = None
            self._clip_tokenizer = None
            flush_gpu_memory()
            logger.info("CLIP unloaded")

    # ── Full Initialization ────────────────────────────────────

    def initialize_full(self):
        """Load all non-heavy resources. Heavy models are lazy-loaded."""
        logger.info("=" * 50)
        logger.info("STAGE 0: Environment Initialization")
        logger.info("=" * 50)

        # Load lightweight resources immediately
        self.load_aliases()
        _ = self.nlp  # Warm up spaCy

        # Log GPU status
        info = get_gpu_memory_info()
        if info["available"]:
            logger.info(f"GPU: {info['device']} | {info['total_mb']:.0f}MB VRAM")
        else:
            logger.warning("No GPU available — falling back to CPU")

        logger.info("Stage 0 complete — heavy models will be loaded on first use")

    def cleanup_all(self):
        """Unload all models and free memory."""
        self.unload_sd_pipeline()
        self.unload_controlnet_pipeline()
        self.unload_midas()
        self.unload_clip()
        flush_gpu_memory()
        logger.info("All models unloaded, memory freed")


# ── Helper Functions ────────────────────────────────────────────


def _parse_alias_file(filepath: Path) -> Dict[str, str]:
    """
    Parse Visual Genome alias file.
    Format: canonical_name,alias1,alias2,...
    Returns: {alias -> canonical} mapping
    """
    aliases = {}
    if not filepath.exists():
        logger.warning(f"Alias file not found: {filepath}")
        return aliases

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 1:
                continue
            canonical = parts[0].lower()
            for part in parts:
                aliases[part.strip().lower()] = canonical

    return aliases


def initialize_environment(config_path: str = None) -> RuntimeEnvironment:
    """
    Top-level function to create and initialize the runtime environment.
    """
    config = load_config(config_path)
    env = RuntimeEnvironment(config)
    env.initialize_full()
    return env
