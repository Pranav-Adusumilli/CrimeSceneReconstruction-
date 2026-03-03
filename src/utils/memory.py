"""
GPU memory management utilities for RTX 3060 6GB VRAM.

Provides memory tracking, cleanup, and optimization helpers.
"""

import gc
import torch
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> dict:
    """Return current GPU memory statistics in MB."""
    if not torch.cuda.is_available():
        return {"available": False}

    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    free = total - allocated

    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "total_mb": round(total, 1),
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "free_mb": round(free, 1),
        "utilization_pct": round(allocated / total * 100, 1),
    }


def flush_gpu_memory():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_memory(label: str = ""):
    """Log current GPU memory usage."""
    info = get_gpu_memory_info()
    if info["available"]:
        prefix = f"[{label}] " if label else ""
        logger.info(
            f"{prefix}GPU Memory: {info['allocated_mb']:.0f}MB allocated / "
            f"{info['total_mb']:.0f}MB total ({info['utilization_pct']:.1f}%)"
        )


@contextmanager
def gpu_memory_guard(label: str = ""):
    """Context manager that flushes GPU memory on entry and exit."""
    flush_gpu_memory()
    log_memory(f"{label} START")
    try:
        yield
    finally:
        flush_gpu_memory()
        log_memory(f"{label} END")


def optimize_pipeline_memory(pipe, enable_cpu_offload=True,
                              enable_attention_slicing=True,
                              enable_vae_slicing=True):
    """
    Apply all memory optimizations to a diffusers pipeline.
    Critical for RTX 3060 6GB VRAM.
    """
    if enable_attention_slicing:
        pipe.enable_attention_slicing(slice_size="auto")
        logger.info("Enabled attention slicing")

    if enable_vae_slicing:
        pipe.enable_vae_slicing()
        logger.info("Enabled VAE slicing")

    if enable_cpu_offload:
        pipe.enable_sequential_cpu_offload()
        logger.info("Enabled sequential CPU offload")

    return pipe


def safe_to_device(tensor_or_model, device, dtype=None):
    """Safely move tensor/model to device, handling OOM gracefully."""
    try:
        if dtype:
            return tensor_or_model.to(device=device, dtype=dtype)
        return tensor_or_model.to(device)
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM during device transfer, flushing and retrying...")
        flush_gpu_memory()
        if dtype:
            return tensor_or_model.to(device=device, dtype=dtype)
        return tensor_or_model.to(device)
