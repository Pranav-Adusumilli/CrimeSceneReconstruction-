"""
GPU Verification Script
=======================
Validates CUDA availability, GPU specs, and memory for RTX 3060 6GB.
Run: python scripts/verify_gpu.py
"""

import sys
import torch


def verify_gpu():
    print("=" * 60)
    print("  GPU & CUDA Verification")
    print("=" * 60)

    # Python
    print(f"\n  Python     : {sys.version.split()[0]}")
    print(f"  PyTorch    : {torch.__version__}")

    # CUDA
    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA avail : {cuda_ok}")

    if not cuda_ok:
        print("\n  [FAIL] CUDA is not available. Check your PyTorch installation.")
        return False

    print(f"  CUDA ver   : {torch.version.cuda}")
    print(f"  cuDNN ver  : {torch.backends.cudnn.version()}")
    print(f"  GPU count  : {torch.cuda.device_count()}")

    # Device info
    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory / 1024 ** 3
    print(f"\n  Device     : {props.name}")
    print(f"  VRAM       : {total_vram:.1f} GB")
    print(f"  Compute    : {props.major}.{props.minor}")
    print(f"  SMs        : {props.multi_processor_count}")

    # FP16 support
    fp16_ok = props.major >= 7 or (props.major == 6 and props.minor >= 0)
    print(f"  FP16 OK    : {fp16_ok}")

    # Memory test
    print("\n  Running memory allocation test...")
    try:
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        y = torch.matmul(x, x)
        del x, y
        torch.cuda.empty_cache()
        print("  [PASS] FP16 GEMM test succeeded")
    except Exception as e:
        print(f"  [FAIL] Memory test failed: {e}")
        return False

    # Memory stats
    alloc = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    print(f"\n  Mem alloc  : {alloc:.1f} MB")
    print(f"  Mem reserv : {reserved:.1f} MB")

    print("\n" + "=" * 60)
    print("  [PASS] GPU verification complete — system ready")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = verify_gpu()
    sys.exit(0 if success else 1)
