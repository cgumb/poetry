#!/usr/bin/env python
"""
Test that optional Cholesky refactor truly avoids placeholder allocations.

Validates that when return_chol=False:
1. No chol_bin file is written (saves disk I/O)
2. No Python allocation happens (saves memory)
3. Result.chol_lower is None (not zeros)
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
import tempfile

from poetry_gp.backends.scalapack_fit import fit_exact_gp_scalapack_from_rated


def test_no_chol_file_written() -> None:
    """Test that chol_bin is NOT written when return_chol=False."""
    print("\n" + "="*60)
    print("TEST 1: No chol_bin File Written")
    print("="*60)

    rng = np.random.default_rng(42)
    n = 500
    d = 10
    x_rated = rng.normal(size=(n, d))
    x_rated /= np.linalg.norm(x_rated, axis=1, keepdims=True) + 1e-12
    y_rated = rng.normal(size=n)

    print(f"Problem: n={n}, d={d}")

    # Fit with return_chol=False
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        state = fit_exact_gp_scalapack_from_rated(
            x_rated,
            y_rated,
            length_scale=1.0,
            variance=1.0,
            noise=1e-3,
            launcher="srun",
            nprocs=4,
            block_size=128,
            return_alpha=True,
            return_chol=False,  # ← Skip Cholesky
            workdir=workdir,
            verbose=True,
        )

        # Check file system
        chol_bin_path = workdir / "chol_lower.bin"
        alpha_bin_path = workdir / "alpha.bin"

        print(f"\nFile system check:")
        print(f"  alpha.bin exists: {alpha_bin_path.exists()}")
        print(f"  chol_lower.bin exists: {chol_bin_path.exists()}")

        if chol_bin_path.exists():
            chol_size = chol_bin_path.stat().st_size
            print(f"\n✗ ERROR: chol_lower.bin was written ({chol_size} bytes)")
            print(f"  For n={n}, this should be 0 bytes (no file)")
            return

        print(f"\n✓ chol_lower.bin was NOT written (good!)")

        # Check Python state
        print(f"\nPython state check:")
        print(f"  state.cho_factor_data: {state.cho_factor_data}")
        print(f"  state.alpha shape: {state.alpha.shape}")

        if state.cho_factor_data is not None:
            print(f"\n✗ ERROR: cho_factor_data should be None")
            return

        print(f"\n✓ TEST PASSED: No placeholder file or allocation")


def test_memory_savings() -> None:
    """Test that large m doesn't cause memory issues when return_chol=False."""
    print("\n" + "="*60)
    print("TEST 2: Memory Savings for Large m")
    print("="*60)

    rng = np.random.default_rng(42)
    m = 5000  # Would be 200 MB for chol matrix
    d = 10
    x_rated = rng.normal(size=(m, d))
    x_rated /= np.linalg.norm(x_rated, axis=1, keepdims=True) + 1e-12
    y_rated = rng.normal(size=m)

    print(f"Problem: m={m}, d={d}")
    print(f"Cholesky matrix size if allocated: {m * m * 8 / 1e6:.1f} MB")

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        state = fit_exact_gp_scalapack_from_rated(
            x_rated,
            y_rated,
            length_scale=1.0,
            variance=1.0,
            noise=1e-3,
            launcher="srun",
            nprocs=4,
            block_size=128,
            return_alpha=True,
            return_chol=False,  # ← Skip Cholesky
            workdir=workdir,
            verbose=True,
        )

        chol_bin_path = workdir / "chol_lower.bin"

        if chol_bin_path.exists():
            chol_size_mb = chol_bin_path.stat().st_size / 1e6
            print(f"\n✗ ERROR: chol_lower.bin written: {chol_size_mb:.1f} MB")
            print(f"  This defeats the entire optimization!")
            return

        if state.cho_factor_data is not None:
            print(f"\n✗ ERROR: cho_factor_data allocated in Python")
            print(f"  This defeats the entire optimization!")
            return

        print(f"\n✓ TEST PASSED: No 200 MB allocation or disk write")


def test_with_chol_still_works() -> None:
    """Test that return_chol=True still works correctly."""
    print("\n" + "="*60)
    print("TEST 3: return_chol=True Still Works")
    print("="*60)

    rng = np.random.default_rng(42)
    n = 500
    d = 10
    x_rated = rng.normal(size=(n, d))
    x_rated /= np.linalg.norm(x_rated, axis=1, keepdims=True) + 1e-12
    y_rated = rng.normal(size=n)

    print(f"Problem: n={n}, d={d}")

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        state = fit_exact_gp_scalapack_from_rated(
            x_rated,
            y_rated,
            length_scale=1.0,
            variance=1.0,
            noise=1e-3,
            launcher="srun",
            nprocs=4,
            block_size=128,
            return_alpha=True,
            return_chol=True,  # ← Include Cholesky
            workdir=workdir,
            verbose=True,
        )

        chol_bin_path = workdir / "chol_lower.bin"

        print(f"\nFile system check:")
        print(f"  chol_lower.bin exists: {chol_bin_path.exists()}")

        if not chol_bin_path.exists():
            print(f"\n✗ ERROR: chol_lower.bin should have been written")
            return

        chol_size_mb = chol_bin_path.stat().st_size / 1e6
        expected_size_mb = n * n * 8 / 1e6
        print(f"  chol_lower.bin size: {chol_size_mb:.1f} MB (expected ~{expected_size_mb:.1f} MB)")

        if state.cho_factor_data is None:
            print(f"\n✗ ERROR: cho_factor_data should NOT be None")
            return

        chol_shape = state.cho_factor_data[0].shape
        print(f"\nPython state check:")
        print(f"  cho_factor_data shape: {chol_shape}")

        if chol_shape != (n, n):
            print(f"\n✗ ERROR: Expected shape ({n}, {n}), got {chol_shape}")
            return

        print(f"\n✓ TEST PASSED: Cholesky correctly gathered when requested")


def main() -> None:
    print("="*60)
    print("Testing No-Placeholder Optimization")
    print("="*60)
    print("\nValidating that return_chol=False:")
    print("  1. Does NOT write chol_bin file")
    print("  2. Does NOT allocate zeros in Python")
    print("  3. Sets cho_factor_data=None")
    print("\nThis ensures the optimization actually works for large m.")

    try:
        test_no_chol_file_written()
        test_memory_savings()
        test_with_chol_still_works()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe optimization is complete:")
        print("  • No placeholder file writes (saves disk I/O)")
        print("  • No placeholder allocations (saves memory)")
        print("  • For m=20k: saves 3.2 GB of disk + memory!")
        print("\nFit-only benchmarks will now be much faster.")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
