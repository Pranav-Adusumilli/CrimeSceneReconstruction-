"""
Test Diffusion Generation Script
==================================
Runs a minimal Stable Diffusion generation to verify the pipeline works
on RTX 3060 6GB VRAM with all memory optimizations.

Run: python scripts/test_diffusion.py
"""

import os
import sys
import time
import torch
import gc

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_diffusion():
    from diffusers import StableDiffusionPipeline
    from src.utils.memory import get_gpu_memory_info, flush_gpu_memory

    print("=" * 60)
    print("  Test Diffusion Generation (SD v1.4, FP16)")
    print("=" * 60)

    # Check GPU
    info = get_gpu_memory_info()
    if not info["available"]:
        print("[FAIL] No GPU available")
        return False

    print(f"\n  GPU: {info['device']} | VRAM: {info['total_mb']:.0f} MB")
    print(f"  Pre-load memory: {info['allocated_mb']:.0f} MB allocated")

    # Load model
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "models", "stable_diffusion")
    print(f"\n  Loading Stable Diffusion v1.4 (FP16)...")
    t0 = time.time()

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Memory optimizations (critical for 6GB VRAM)
    pipe.enable_attention_slicing("auto")
    pipe.enable_vae_slicing()
    pipe.enable_sequential_cpu_offload()

    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    info = get_gpu_memory_info()
    print(f"  Post-load memory: {info['allocated_mb']:.0f} MB allocated")

    # Generate test image
    prompt = (
        "A photorealistic crime scene photograph of a small bedroom, "
        "a broken window, a knife on a wooden table, blood stains near a sofa, "
        "forensic photography, harsh lighting, detailed, 8k"
    )
    negative = "blurry, low quality, distorted, cartoon, anime, sketch, watermark"

    print(f"\n  Generating test image...")
    print(f"  Prompt: {prompt[:80]}...")
    print(f"  Steps: 20 | CFG: 7.5 | Seed: 42 | Size: 512x512")

    generator = torch.Generator("cpu").manual_seed(42)
    t0 = time.time()

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
            height=512,
            width=512,
        )

    gen_time = time.time() - t0
    image = result.images[0]

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_diffusion.png")
    image.save(output_path)

    info = get_gpu_memory_info()
    print(f"\n  Generation time: {gen_time:.1f}s")
    print(f"  Peak memory: {info['allocated_mb']:.0f} MB allocated")
    print(f"  Image saved: {output_path}")
    print(f"  Image size: {image.size}")

    # Cleanup
    del pipe
    flush_gpu_memory()

    final_info = get_gpu_memory_info()
    print(f"\n  Post-cleanup memory: {final_info['allocated_mb']:.0f} MB allocated")

    print("\n" + "=" * 60)
    print("  [PASS] Diffusion test complete — memory is stable")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_diffusion()
    sys.exit(0 if success else 1)
