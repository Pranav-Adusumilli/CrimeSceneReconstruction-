"""
Model Download Script
=====================
Downloads all required pretrained models via Hugging Face.
Models: Stable Diffusion v1.4, ControlNet Depth, MiDaS DPT Hybrid.

Run: python scripts/download_models.py
"""

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def download_stable_diffusion():
    """Download Stable Diffusion v1.4 (FP16)."""
    from diffusers import StableDiffusionPipeline

    model_id = "CompVis/stable-diffusion-v1-4"
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "models", "stable_diffusion")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n[1/3] Downloading Stable Diffusion v1.4 ...")
    print(f"      Model ID : {model_id}")
    print(f"      Cache    : {cache_dir}")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False,
    )
    del pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("      [OK] Stable Diffusion v1.4 downloaded")


def download_controlnet():
    """Download ControlNet Depth model."""
    from diffusers import ControlNetModel

    model_id = "lllyasviel/sd-controlnet-depth"
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "models", "controlnet")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n[2/3] Downloading ControlNet Depth ...")
    print(f"      Model ID : {model_id}")
    print(f"      Cache    : {cache_dir}")

    controlnet = ControlNetModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    )
    del controlnet
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("      [OK] ControlNet Depth downloaded")


def download_midas():
    """Download MiDaS DPT Hybrid depth estimation model."""
    from transformers import DPTForDepthEstimation, DPTFeatureExtractor

    model_id = "Intel/dpt-hybrid-midas"
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "models", "midas")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n[3/3] Downloading MiDaS DPT Hybrid ...")
    print(f"      Model ID : {model_id}")
    print(f"      Cache    : {cache_dir}")

    model = DPTForDepthEstimation.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )
    feature_extractor = DPTFeatureExtractor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )
    del model, feature_extractor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("      [OK] MiDaS DPT Hybrid downloaded")


def main():
    print("=" * 60)
    print("  Pretrained Model Download")
    print("=" * 60)

    # Check HF token availability (optional for public models)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("  HF Token   : detected")
    else:
        print("  HF Token   : not set (OK for public models)")

    download_stable_diffusion()
    download_controlnet()
    download_midas()

    print("\n" + "=" * 60)
    print("  [DONE] All models downloaded successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
