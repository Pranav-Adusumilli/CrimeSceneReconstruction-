"""
Configuration management for Crime Scene Reconstruction system.

Loads YAML config and provides typed access to all parameters.
"""

import os
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ── Project root (two levels up from this file) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class HardwareConfig:
    device: str = "cuda"
    dtype: str = "float16"
    vram_limit_gb: float = 6.0
    batch_size: int = 1
    resolution: int = 512
    enable_cpu_offload: bool = True
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float16 if self.dtype == "float16" else torch.float32

    @property
    def torch_device(self) -> torch.device:
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


@dataclass
class ModelConfig:
    model_id: str
    cache_dir: str
    torch_dtype: str = "float16"
    revision: Optional[str] = None


@dataclass
class GenerationConfig:
    num_inference_steps: int = 35
    guidance_scale: float = 9.0
    negative_prompt: str = ""
    seed: int = 42
    num_hypotheses: int = 3
    two_pass: bool = True
    controlnet_conditioning_scale: float = 0.8


@dataclass
class ViewConfig:
    name: str
    depth_shift: float
    prompt_suffix: str


@dataclass
class Config:
    """Master configuration for the entire pipeline."""

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    sd_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="SG161222/Realistic_Vision_V5.1_noVAE",
        cache_dir="models/stable_diffusion",
        revision=None,
    ))
    controlnet_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="lllyasviel/sd-controlnet-depth",
        cache_dir="models/controlnet",
    ))
    vae_model: Optional[ModelConfig] = field(default_factory=lambda: ModelConfig(
        model_id="stabilityai/sd-vae-ft-mse",
        cache_dir="models/vae",
    ))
    midas_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id="Intel/dpt-hybrid-midas",
        cache_dir="models/midas",
    ))
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    views: List[ViewConfig] = field(default_factory=list)
    scene_types: List[str] = field(default_factory=list)
    output_dir: str = "outputs"
    data_dir: str = "Data"
    object_alias_file: str = "Data/VisualGenome/object_alias.txt"
    relationship_alias_file: str = "Data/VisualGenome/relationship_alias.txt"
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"

    def resolve_path(self, relative: str) -> Path:
        """Resolve a relative path against the project root."""
        return PROJECT_ROOT / relative


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file, with sensible defaults."""

    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "default_config.yaml")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    # Hardware
    hw = raw.get("hardware", {})
    cfg.hardware = HardwareConfig(
        device=hw.get("device", "cuda"),
        dtype=hw.get("dtype", "float16"),
        vram_limit_gb=hw.get("vram_limit_gb", 6.0),
        batch_size=hw.get("batch_size", 1),
        resolution=hw.get("resolution", 512),
        enable_cpu_offload=hw.get("enable_cpu_offload", True),
        enable_attention_slicing=hw.get("enable_attention_slicing", True),
        enable_vae_slicing=hw.get("enable_vae_slicing", True),
    )

    # Models
    models = raw.get("models", {})
    sd = models.get("stable_diffusion", {})
    cfg.sd_model = ModelConfig(
        model_id=sd.get("model_id", "SG161222/Realistic_Vision_V5.1_noVAE"),
        cache_dir=sd.get("cache_dir", "models/stable_diffusion"),
        torch_dtype=sd.get("torch_dtype", "float16"),
        revision=sd.get("revision", None),
    )
    cn = models.get("controlnet", {})
    cfg.controlnet_model = ModelConfig(
        model_id=cn.get("model_id", "lllyasviel/sd-controlnet-depth"),
        cache_dir=cn.get("cache_dir", "models/controlnet"),
        torch_dtype=cn.get("torch_dtype", "float16"),
    )
    vae = models.get("vae", {})
    if vae.get("model_id"):
        cfg.vae_model = ModelConfig(
            model_id=vae.get("model_id", "stabilityai/sd-vae-ft-mse"),
            cache_dir=vae.get("cache_dir", "models/vae"),
            torch_dtype=vae.get("torch_dtype", "float16"),
        )
    else:
        cfg.vae_model = None

    mi = models.get("midas", {})
    cfg.midas_model = ModelConfig(
        model_id=mi.get("model_id", "Intel/dpt-hybrid-midas"),
        cache_dir=mi.get("cache_dir", "models/midas"),
    )
    clip_cfg = models.get("clip", {})
    cfg.clip_model_name = clip_cfg.get("model_name", "ViT-B-32")
    cfg.clip_pretrained = clip_cfg.get("pretrained", "openai")

    # Generation
    gen = raw.get("generation", {})
    cfg.generation = GenerationConfig(
        num_inference_steps=gen.get("num_inference_steps", 35),
        guidance_scale=gen.get("guidance_scale", 9.0),
        negative_prompt=gen.get("negative_prompt", ""),
        seed=gen.get("seed", 42),
        num_hypotheses=gen.get("num_hypotheses", 3),
        two_pass=gen.get("two_pass", True),
        controlnet_conditioning_scale=gen.get("controlnet_conditioning_scale", 0.8),
    )

    # Multi-view
    mv = raw.get("multiview", {})
    cfg.views = [
        ViewConfig(
            name=v["name"],
            depth_shift=v["depth_shift"],
            prompt_suffix=v["prompt_suffix"],
        )
        for v in mv.get("views", [])
    ]

    # NLP
    nlp = raw.get("nlp", {})
    cfg.scene_types = nlp.get("scene_types", [])

    # Data paths
    data = raw.get("data", {})
    cfg.output_dir = raw.get("project", {}).get("output_dir", "outputs")
    cfg.object_alias_file = data.get("object_alias_file", "Data/VisualGenome/object_alias.txt")
    cfg.relationship_alias_file = data.get("relationship_alias_file", "Data/VisualGenome/relationship_alias.txt")

    return cfg
