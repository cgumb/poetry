#!/usr/bin/env python3
"""
Quick GPU availability check for Poetry GP.

Tests if CuPy is available and GPU is accessible.
"""
from __future__ import annotations

import sys


def main() -> int:
    print("=" * 60)
    print("GPU Availability Check")
    print("=" * 60)
    print()

    # Check CuPy availability
    try:
        import cupy as cp
        print(f"✓ CuPy installed: version {cp.__version__}")
    except ImportError:
        print("✗ CuPy not installed")
        print()
        print("To enable GPU scoring, install CuPy:")
        print("  pip install cupy-cuda11x  # or cupy-cuda12x")
        print()
        return 1

    # Check GPU access
    try:
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        cuda_major = cuda_version // 1000
        cuda_minor = (cuda_version % 1000) // 10
        print(f"✓ CUDA runtime: {cuda_major}.{cuda_minor}")
    except Exception as e:
        print(f"✗ CUDA runtime error: {e}")
        return 1

    # Test GPU computation
    try:
        test_array = cp.array([1.0, 2.0, 3.0])
        result = test_array + test_array
        assert cp.allclose(result, cp.array([2.0, 4.0, 6.0]))
        print("✓ GPU computation working")
    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
        return 1

    # Get GPU info
    try:
        device = cp.cuda.Device(0)
        print()
        print("GPU Information:")
        print(f"  Device ID: {device.id}")
        print(f"  Name: {device.name}")
        mem_total = device.mem_info[1] / (1024**3)  # Convert to GB
        mem_free = device.mem_info[0] / (1024**3)
        print(f"  Memory: {mem_free:.1f} GB free / {mem_total:.1f} GB total")
        print(f"  Compute capability: {device.compute_capability}")
    except Exception as e:
        print(f"Warning: Could not get GPU info: {e}")

    # Check poetry_gp integration
    print()
    try:
        from poetry_gp.backends.gpu_scoring import is_gpu_available
        if is_gpu_available():
            print("✓ poetry_gp GPU backend available")
        else:
            print("✗ poetry_gp GPU backend not available (unexpected)")
            return 1
    except ImportError as e:
        print(f"✗ Could not import poetry_gp GPU backend: {e}")
        return 1

    print()
    print("=" * 60)
    print("SUCCESS: GPU is ready for scoring!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run GPU benchmark:")
    print("     sbatch scripts/gpu_scoring_bench.slurm")
    print()
    print("  2. Or quick test:")
    print("     python scripts/bench_scoring.py \\")
    print("       --m-rated 500 1000 2000 \\")
    print("       --n-candidates 10000 \\")
    print("       --output-csv results/gpu_quick_test.csv")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
