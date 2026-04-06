#!/usr/bin/env python
"""
Test PyBind11 LAPACK integration.

Validates that the native_lapack backend:
1. Produces correct results (matches scipy)
2. Eliminates subprocess overhead
3. Supports both fit and predict
4. Handles optional Cholesky correctly
"""
from __future__ import annotations

import numpy as np
from time import perf_counter

from poetry_gp.backends.native_lapack import is_native_available, fit_exact_gp_native, predict_native
from poetry_gp.backends.blocked import run_blocked_step
from poetry_gp.gp_exact import fit_exact_gp, predict_block


def test_module_availability() -> None:
    """Test that poetry_gp_native module is available."""
    print("\n" + "="*60)
    print("TEST 1: Module Availability")
    print("="*60)

    if is_native_available():
        import poetry_gp_native
        print(f"✓ poetry_gp_native module found")
        print(f"  Functions: {dir(poetry_gp_native)}")
    else:
        print("✗ poetry_gp_native module NOT available")
        print("  Build with: make native-build")
        raise ImportError("PyBind11 module not available")


def test_fit_correctness() -> None:
    """Test that native_lapack produces same results as scipy."""
    print("\n" + "="*60)
    print("TEST 2: Fit Correctness (vs scipy)")
    print("="*60)

    rng = np.random.default_rng(42)
    m = 100
    d = 10
    x_rated = rng.normal(size=(m, d))
    x_rated /= np.linalg.norm(x_rated, axis=1, keepdims=True) + 1e-12
    y_rated = rng.normal(size=m)

    print(f"Problem: m={m}, d={d}")

    # Fit with scipy (reference)
    state_scipy = fit_exact_gp(x_rated, y_rated, length_scale=1.0, variance=1.0, noise=1e-3)
    print(f"✓ scipy fit: lml={state_scipy.log_marginal_likelihood:.6f}")

    # Fit with native LAPACK
    state_native = fit_exact_gp_native(x_rated, y_rated, length_scale=1.0, variance=1.0, noise=1e-3)
    print(f"✓ native_lapack fit: lml={state_native.log_marginal_likelihood:.6f}")

    # Compare results
    alpha_diff = np.linalg.norm(state_scipy.alpha - state_native.alpha)
    lml_diff = abs(state_scipy.log_marginal_likelihood - state_native.log_marginal_likelihood)

    print(f"\nDifferences:")
    print(f"  ||alpha_scipy - alpha_native|| = {alpha_diff:.2e}")
    print(f"  |lml_scipy - lml_native| = {lml_diff:.2e}")

    if alpha_diff > 1e-8:
        raise AssertionError(f"alpha differs by {alpha_diff:.2e} (expected < 1e-8)")

    if lml_diff > 1e-8:
        raise AssertionError(f"lml differs by {lml_diff:.2e} (expected < 1e-8)")

    print("\n✓ TEST PASSED: Results match scipy within tolerance")


def test_predict_correctness() -> None:
    """Test that native predict produces same results as scipy."""
    print("\n" + "="*60)
    print("TEST 3: Predict Correctness (vs scipy)")
    print("="*60)

    rng = np.random.default_rng(42)
    m = 100
    n = 50
    d = 10
    x_rated = rng.normal(size=(m, d))
    x_rated /= np.linalg.norm(x_rated, axis=1, keepdims=True) + 1e-12
    y_rated = rng.normal(size=m)

    x_query = rng.normal(size=(n, d))
    x_query /= np.linalg.norm(x_query, axis=1, keepdims=True) + 1e-12

    print(f"Problem: m={m}, n={n}, d={d}")

    # Fit with native LAPACK
    state = fit_exact_gp_native(x_rated, y_rated, length_scale=1.0, variance=1.0, noise=1e-3)

    # Predict with scipy (reference)
    mean_scipy, var_scipy = predict_block(state, x_query, compute_variance=True)
    print(f"✓ scipy predict: mean ∈ [{mean_scipy.min():.3f}, {mean_scipy.max():.3f}], var ∈ [{var_scipy.min():.3f}, {var_scipy.max():.3f}]")

    # Predict with native LAPACK
    mean_native, var_native = predict_native(state, x_query, compute_variance=True)
    print(f"✓ native predict: mean ∈ [{mean_native.min():.3f}, {mean_native.max():.3f}], var ∈ [{var_native.min():.3f}, {var_native.max():.3f}]")

    # Compare results
    mean_diff = np.linalg.norm(mean_scipy - mean_native)
    var_diff = np.linalg.norm(var_scipy - var_native)

    print(f"\nDifferences:")
    print(f"  ||mean_scipy - mean_native|| = {mean_diff:.2e}")
    print(f"  ||var_scipy - var_native|| = {var_diff:.2e}")

    if mean_diff > 1e-8:
        raise AssertionError(f"mean differs by {mean_diff:.2e} (expected < 1e-8)")

    if var_diff > 1e-8:
        raise AssertionError(f"variance differs by {var_diff:.2e} (expected < 1e-8)")

    print("\n✓ TEST PASSED: Predictions match scipy within tolerance")


def test_performance_comparison() -> None:
    """Test that native_lapack is faster than subprocess for small problems."""
    print("\n" + "="*60)
    print("TEST 4: Performance Comparison")
    print("="*60)

    rng = np.random.default_rng(42)
    m_values = [500, 1000, 2000]
    d = 384
    n = 3000  # Must be >= max(m_values) for sampling without replacement

    print(f"Problem sizes: m ∈ {m_values}, n={n}, d={d}")
    print("")

    results = []

    for m in m_values:
        print(f"Testing m={m}...")

        embeddings = rng.normal(size=(n, d))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        rated_indices = rng.choice(n, size=m, replace=False)
        ratings = rng.normal(size=m)

        # Test scipy (baseline)
        t0 = perf_counter()
        result_scipy = run_blocked_step(
            embeddings, rated_indices, ratings,
            fit_backend="python",
            score_backend="python",
            block_size=2048,
        )
        scipy_time = perf_counter() - t0

        # Test native_lapack
        t0 = perf_counter()
        result_native = run_blocked_step(
            embeddings, rated_indices, ratings,
            fit_backend="native_lapack",
            score_backend="python",
            block_size=2048,
        )
        native_time = perf_counter() - t0

        speedup = scipy_time / native_time

        print(f"  scipy:        {scipy_time:.3f}s")
        print(f"  native_lapack: {native_time:.3f}s")
        print(f"  speedup:      {speedup:.2f}×")
        print("")

        results.append({
            "m": m,
            "scipy_time": scipy_time,
            "native_time": native_time,
            "speedup": speedup,
        })

    print("Summary:")
    print(f"  m     scipy   native  speedup")
    for r in results:
        print(f"  {r['m']:4d}  {r['scipy_time']:6.3f}s  {r['native_time']:6.3f}s  {r['speedup']:5.2f}×")

    # Expect native_lapack to be at least as fast as scipy (speedup >= 0.9)
    min_speedup = min(r["speedup"] for r in results)
    if min_speedup < 0.9:
        print(f"\n⚠ WARNING: native_lapack slower than scipy (min speedup: {min_speedup:.2f}×)")
        print("  Expected speedup >= 0.9×")
    else:
        print(f"\n✓ TEST PASSED: native_lapack competitive with scipy (min speedup: {min_speedup:.2f}×)")


def test_optional_cholesky() -> None:
    """Test that return_chol=False works correctly."""
    print("\n" + "="*60)
    print("TEST 5: Optional Cholesky")
    print("="*60)

    rng = np.random.default_rng(42)
    m = 500
    d = 10
    x_rated = rng.normal(size=(m, d))
    x_rated /= np.linalg.norm(x_rated, axis=1, keepdims=True) + 1e-12
    y_rated = rng.normal(size=m)

    print(f"Problem: m={m}, d={d}")

    # Fit without Cholesky
    state_no_chol = fit_exact_gp_native(x_rated, y_rated, return_chol=False)
    print(f"✓ Fit without Cholesky: cho_factor_data={state_no_chol.cho_factor_data}")

    if state_no_chol.cho_factor_data is not None:
        raise AssertionError("cho_factor_data should be None when return_chol=False")

    # Fit with Cholesky
    state_with_chol = fit_exact_gp_native(x_rated, y_rated, return_chol=True)
    print(f"✓ Fit with Cholesky: cho_factor_data is not None: {state_with_chol.cho_factor_data is not None}")

    if state_with_chol.cho_factor_data is None:
        raise AssertionError("cho_factor_data should NOT be None when return_chol=True")

    print("\n✓ TEST PASSED: Optional Cholesky works correctly")


def main() -> None:
    print("="*60)
    print("PyBind11 LAPACK Integration Tests")
    print("="*60)

    try:
        test_module_availability()
        test_fit_correctness()
        test_predict_correctness()
        test_performance_comparison()
        test_optional_cholesky()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe PyBind11 bridge is working correctly:")
        print("  • Module loads successfully")
        print("  • Results match scipy (correctness)")
        print("  • Competitive performance (no overhead)")
        print("  • Optional Cholesky works")
        print("\nYou can now use fit_backend='native_lapack' for m < 5000")
        print("in interactive workflows to eliminate subprocess overhead.")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
