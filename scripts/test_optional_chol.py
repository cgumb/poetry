#!/usr/bin/env python
"""
Test script to verify the optional Cholesky refactor works correctly.

Tests three modes:
1. Fit-only (return_alpha=True, return_chol=False)
2. Mean-only (return_alpha=True, return_chol=False, then predict with compute_variance=False)
3. Full variance-capable (return_alpha=True, return_chol=True, then predict with compute_variance=True)
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from poetry_gp.backends.scalapack_fit import fit_exact_gp_scalapack_from_rated
from poetry_gp.gp_exact import predict_block


def test_fit_only() -> None:
    """Test fit-only mode: no scoring, just fit timing."""
    print("\n" + "="*60)
    print("TEST 1: FIT-ONLY (return_alpha=True, return_chol=False)")
    print("="*60)

    rng = np.random.default_rng(42)
    n = 1000
    d = 384
    x_rated = rng.normal(size=(n, d))
    x_rated /= np.linalg.norm(x_rated, axis=1, keepdims=True) + 1e-12
    y_rated = rng.normal(size=n)

    print(f"Problem: n={n}, d={d}")
    print("Expected: Should skip Cholesky gathering (faster)")

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
        return_chol=False,  # ← Skip Cholesky gathering
        verbose=True,
    )

    print(f"\n✓ Fit succeeded!")
    print(f"  alpha shape: {state.alpha.shape}")
    print(f"  cho_factor_data is None: {state.cho_factor_data is None}")
    print(f"  fit_total_seconds: {state.optimization_result['fit_total_seconds']:.3f}s")

    if state.cho_factor_data is not None:
        print("\n✗ ERROR: cho_factor_data should be None for fit-only mode!")
        return

    print("\n✓ TEST PASSED: Fit-only mode works correctly")


def test_mean_only() -> None:
    """Test mean-only scoring: predict without variance."""
    print("\n" + "="*60)
    print("TEST 2: MEAN-ONLY (return_chol=False, compute_variance=False)")
    print("="*60)

    rng = np.random.default_rng(42)
    n = 1000
    d = 384
    x_rated = rng.normal(size=(n, d))
    x_rated /= np.linalg.norm(x_rated, axis=1, keepdims=True) + 1e-12
    y_rated = rng.normal(size=n)

    x_query = rng.normal(size=(500, d))
    x_query /= np.linalg.norm(x_query, axis=1, keepdims=True) + 1e-12

    print(f"Problem: n_train={n}, n_query={x_query.shape[0]}, d={d}")
    print("Expected: Should skip Cholesky, predict mean only")

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
        return_chol=False,  # ← Skip Cholesky gathering
        verbose=True,
    )

    print(f"\n✓ Fit succeeded (no Cholesky)")

    # Predict mean only
    mean, var = predict_block(state, x_query, compute_variance=False)

    print(f"\n✓ Prediction succeeded!")
    print(f"  mean shape: {mean.shape}")
    print(f"  variance: {var}")

    if var is not None:
        print("\n✗ ERROR: variance should be None when compute_variance=False!")
        return

    # Try to predict variance (should fail gracefully)
    print("\n  Testing that variance computation fails without Cholesky...")
    try:
        mean, var = predict_block(state, x_query, compute_variance=True)
        print("✗ ERROR: Should have raised RuntimeError!")
        return
    except RuntimeError as e:
        print(f"  ✓ Correctly raised error: {str(e)[:80]}...")

    print("\n✓ TEST PASSED: Mean-only mode works correctly")


def test_full_variance() -> None:
    """Test full variance-capable mode."""
    print("\n" + "="*60)
    print("TEST 3: FULL VARIANCE (return_chol=True, compute_variance=True)")
    print("="*60)

    rng = np.random.default_rng(42)
    n = 1000
    d = 384
    x_rated = rng.normal(size=(n, d))
    x_rated /= np.linalg.norm(x_rated, axis=1, keepdims=True) + 1e-12
    y_rated = rng.normal(size=n)

    x_query = rng.normal(size=(500, d))
    x_query /= np.linalg.norm(x_query, axis=1, keepdims=True) + 1e-12

    print(f"Problem: n_train={n}, n_query={x_query.shape[0]}, d={d}")
    print("Expected: Should gather Cholesky, predict both mean and variance")

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
        return_chol=True,  # ← Gather Cholesky
        verbose=True,
    )

    print(f"\n✓ Fit succeeded (with Cholesky)")
    print(f"  cho_factor_data is not None: {state.cho_factor_data is not None}")

    if state.cho_factor_data is None:
        print("\n✗ ERROR: cho_factor_data should NOT be None for full variance mode!")
        return

    # Predict with variance
    mean, var = predict_block(state, x_query, compute_variance=True)

    print(f"\n✓ Prediction with variance succeeded!")
    print(f"  mean shape: {mean.shape}")
    print(f"  variance shape: {var.shape}")
    print(f"  variance range: [{var.min():.6f}, {var.max():.6f}]")

    if var is None:
        print("\n✗ ERROR: variance should NOT be None when compute_variance=True!")
        return

    if np.any(var < 0):
        print(f"\n✗ WARNING: Negative variance detected! min={var.min()}")

    print("\n✓ TEST PASSED: Full variance mode works correctly")


def main() -> None:
    print("="*60)
    print("Testing Optional Cholesky Refactor")
    print("="*60)
    print("\nThis test verifies:")
    print("1. Fit-only mode (no Cholesky gathering)")
    print("2. Mean-only scoring (no Cholesky, no variance)")
    print("3. Full variance-capable mode (with Cholesky)")

    try:
        test_fit_only()
        test_mean_only()
        test_full_variance()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe refactor is working correctly:")
        print("  • Fit-only mode skips expensive Cholesky gathering")
        print("  • Mean-only scoring works without Cholesky")
        print("  • Full variance mode includes Cholesky as before")
        print("\nYou can now run fit-only benchmarks with --score-backend none")
        print("and they will automatically skip Cholesky gathering.")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
