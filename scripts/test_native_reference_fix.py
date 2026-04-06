#!/usr/bin/env python3
"""
Quick test to verify native_reference backend respects return_alpha/return_chol flags.

This was the root cause of all ScaLAPACK failures in the overhead_crossover benchmark.
"""

from __future__ import annotations

import numpy as np

from poetry_gp.backends.blocked import run_blocked_step
from poetry_gp.backends.backend_selection import get_backend_info


def test_native_reference_with_score_none():
    """Test that native_reference works with score_backend='none'."""
    print("=" * 80)
    print("TEST: native_reference backend with score_backend='none'")
    print("=" * 80)
    print()
    print("This was failing before the fix because:")
    print("  1. score_backend='none' sets return_chol=False")
    print("  2. run_mpi_reference() didn't accept return_alpha/return_chol params")
    print("  3. Native code never set has_alpha flag in meta JSON")
    print("  4. Python correctly returned alpha=None")
    print("  5. Error: 'Native fit did not return alpha (should always be gathered)'")
    print()
    print("After fix: run_mpi_reference() sets has_alpha=True when return_alpha=True")
    print()

    # Create small test problem
    m, n, d = 50, 100, 10
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(n, d))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    rated_indices = rng.choice(n, size=m, replace=False)
    ratings = rng.normal(size=m)

    print(f"Problem size: m={m}, n={n}, d={d}")
    print()

    try:
        result = run_blocked_step(
            embeddings,
            rated_indices,
            ratings,
            length_scale=1.0,
            variance=1.0,
            noise=1e-3,
            fit_backend="native_reference",
            score_backend="none",  # This triggers return_chol=False
            optimize_hyperparameters=False,
            scalapack_nprocs=4,
        )

        print("✓ Test PASSED!")
        print()
        print(f"Fit time: {result.profile.fit_seconds:.4f}s")
        print(f"GP state: alpha shape = {result.gp_state.alpha.shape}")
        print(f"          alpha is not None: {result.gp_state.alpha is not None}")
        print(f"          chol is not None: {result.gp_state.chol is not None}")
        print()
        print("Expected behavior:")
        print("  - alpha should be present (has_alpha=True in native code)")
        print("  - chol should be absent (has_chol=False because score_backend='none')")
        print()

        if result.gp_state.alpha is not None and result.gp_state.chol is None:
            print("✓ Correct: alpha present, chol absent (as expected)")
            return True
        else:
            print("✗ Unexpected: alpha or chol state is wrong")
            return False

    except RuntimeError as e:
        if "Native fit did not return alpha" in str(e):
            print("✗ Test FAILED!")
            print()
            print(f"Error: {e}")
            print()
            print("This means the fix didn't work. Check that:")
            print("  1. native/scalapack_gp_fit.cpp was rebuilt")
            print("  2. The updated binary is in PATH")
            print("  3. run_mpi_reference() signature has return_alpha/return_chol params")
            return False
        else:
            print(f"✗ Unexpected error: {e}")
            raise


def test_native_reference_with_score_python():
    """Test that native_reference works with score_backend='python' (return both)."""
    print("=" * 80)
    print("TEST: native_reference backend with score_backend='python'")
    print("=" * 80)
    print()
    print("This should request both alpha and chol (return_alpha=True, return_chol=True)")
    print()

    # Create small test problem
    m, n, d = 50, 100, 10
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(n, d))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    rated_indices = rng.choice(n, size=m, replace=False)
    ratings = rng.normal(size=m)

    print(f"Problem size: m={m}, n={n}, d={d}")
    print()

    try:
        result = run_blocked_step(
            embeddings,
            rated_indices,
            ratings,
            length_scale=1.0,
            variance=1.0,
            noise=1e-3,
            fit_backend="native_reference",
            score_backend="python",  # This triggers return_chol=True
            optimize_hyperparameters=False,
            scalapack_nprocs=4,
        )

        print("✓ Test PASSED!")
        print()
        print(f"Fit time: {result.profile.fit_seconds:.4f}s")
        print(f"Score time: {result.profile.score_seconds:.4f}s")
        print(f"GP state: alpha shape = {result.gp_state.alpha.shape}")
        print(f"          chol shape = {result.gp_state.chol.shape if result.gp_state.chol is not None else 'None'}")
        print()

        if result.gp_state.alpha is not None and result.gp_state.chol is not None:
            print("✓ Correct: both alpha and chol present (as expected)")
            return True
        else:
            print("✗ Unexpected: missing alpha or chol")
            return False

    except Exception as e:
        print(f"✗ Test FAILED: {e}")
        raise


def main():
    backend_info = get_backend_info()
    print("=" * 80)
    print("TESTING NATIVE_REFERENCE BACKEND FIX")
    print("=" * 80)
    print()
    print("Available backends:")
    for backend, available in backend_info.items():
        status = "✓" if available else "✗"
        print(f"  {status} {backend}")
    print()

    # native_reference is always available (subprocess-based)
    print("Testing native_reference backend (always available via subprocess)")
    print()

    test1_passed = test_native_reference_with_score_none()
    print()
    print()

    test2_passed = test_native_reference_with_score_python()
    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    status1 = "✓ PASSED" if test1_passed else "✗ FAILED"
    status2 = "✓ PASSED" if test2_passed else "✗ FAILED"
    print(f"Test 1 (score_backend='none'):   {status1}")
    print(f"Test 2 (score_backend='python'): {status2}")
    print()

    if test1_passed and test2_passed:
        print("All tests passed! The fix works correctly.")
        print()
        print("Next step: Run full benchmark suite")
        print("  sbatch scripts/pedagogical_benchmarks.slurm")
        return 0
    else:
        print("Some tests failed. Check build and deployment.")
        return 1


if __name__ == "__main__":
    exit(main())
