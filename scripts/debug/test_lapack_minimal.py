#!/usr/bin/env python
"""
Minimal LAPACK wrapper test - bypasses GP entirely.

Tests fit_gp_lapack() as a pure linear algebra routine against SciPy.
Uses a hand-written 4x4 SPD matrix to isolate LAPACK interface bugs.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

try:
    import poetry_gp_native
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False
    print("✗ poetry_gp_native not available - build with: make native-build")


def test_minimal_lapack():
    """Test LAPACK wrapper with hand-written 4x4 SPD matrix."""
    print("=" * 60)
    print("Minimal LAPACK Test (4x4 SPD matrix)")
    print("=" * 60)

    if not NATIVE_AVAILABLE:
        raise ImportError("Native module not available")

    # Hand-written 4x4 symmetric positive definite matrix
    # Constructed as A = L @ L.T to ensure SPD
    L = np.array([
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 1.5, 0.0, 0.0],
        [0.5, 0.3, 1.2, 0.0],
        [0.2, 0.1, 0.4, 1.0],
    ])
    K = L @ L.T

    print(f"\nTest matrix K (4x4, constructed as L @ L.T):")
    print(K)
    print(f"\nK is C-contiguous: {K.flags['C_CONTIGUOUS']}")
    print(f"K is F-contiguous: {K.flags['F_CONTIGUOUS']}")

    y = np.array([1.0, 2.0, 3.0, 4.0])
    print(f"\nRHS vector y: {y}")

    # SciPy reference solution
    print("\n" + "-" * 60)
    print("SciPy solution:")
    print("-" * 60)

    cho_scipy, lower_scipy = cho_factor(K, lower=True)
    print(f"\ncho_factor returned lower={lower_scipy}")
    print(f"Cholesky factor (lower triangle):")
    # Extract lower triangle
    L_scipy = np.tril(cho_scipy)
    print(L_scipy)

    alpha_scipy = cho_solve((cho_scipy, lower_scipy), y)
    print(f"\nalpha (solution to K @ alpha = y):")
    print(alpha_scipy)

    # Compute logdet: 2 * sum(log(diag(L)))
    logdet_scipy = 2.0 * np.sum(np.log(np.diag(L_scipy)))
    print(f"\nlogdet(K) = 2 * sum(log(diag(L))): {logdet_scipy:.8f}")

    # Verify solution
    residual_scipy = K @ alpha_scipy - y
    print(f"\nResidual ||K @ alpha - y||: {np.linalg.norm(residual_scipy):.2e}")

    # Native LAPACK solution (C-order input)
    print("\n" + "-" * 60)
    print("Native LAPACK solution (C-order input):")
    print("-" * 60)

    K_c = np.ascontiguousarray(K)
    result_native_c = poetry_gp_native.fit_gp_lapack(K_c, y, return_chol=True)

    alpha_native_c = result_native_c["alpha"]
    chol_native_c = result_native_c["chol_lower"]
    logdet_native_c = result_native_c["logdet"]

    print(f"\nalpha:")
    print(alpha_native_c)
    print(f"\nCholesky factor (lower triangle):")
    print(chol_native_c)
    print(f"\nlogdet(K): {logdet_native_c:.8f}")

    residual_native_c = K @ alpha_native_c - y
    print(f"\nResidual ||K @ alpha - y||: {np.linalg.norm(residual_native_c):.2e}")

    # Native LAPACK solution (F-order input)
    print("\n" + "-" * 60)
    print("Native LAPACK solution (F-order input):")
    print("-" * 60)

    K_f = np.asfortranarray(K)
    print(f"\nK_f is C-contiguous: {K_f.flags['C_CONTIGUOUS']}")
    print(f"K_f is F-contiguous: {K_f.flags['F_CONTIGUOUS']}")

    result_native_f = poetry_gp_native.fit_gp_lapack(K_f, y, return_chol=True)

    alpha_native_f = result_native_f["alpha"]
    chol_native_f = result_native_f["chol_lower"]
    logdet_native_f = result_native_f["logdet"]

    print(f"\nalpha:")
    print(alpha_native_f)
    print(f"\nCholesky factor (lower triangle):")
    print(chol_native_f)
    print(f"\nlogdet(K): {logdet_native_f:.8f}")

    residual_native_f = K @ alpha_native_f - y
    print(f"\nResidual ||K @ alpha - y||: {np.linalg.norm(residual_native_f):.2e}")

    # Compare results
    print("\n" + "=" * 60)
    print("Comparison:")
    print("=" * 60)

    print("\nC-order input vs SciPy:")
    alpha_diff_c = np.linalg.norm(alpha_scipy - alpha_native_c)
    chol_diff_c = np.linalg.norm(L_scipy - chol_native_c)
    logdet_diff_c = abs(logdet_scipy - logdet_native_c)
    print(f"  ||alpha_scipy - alpha_native_c|| = {alpha_diff_c:.2e}")
    print(f"  ||L_scipy - L_native_c||         = {chol_diff_c:.2e}")
    print(f"  |logdet_scipy - logdet_native_c| = {logdet_diff_c:.2e}")

    print("\nF-order input vs SciPy:")
    alpha_diff_f = np.linalg.norm(alpha_scipy - alpha_native_f)
    chol_diff_f = np.linalg.norm(L_scipy - chol_native_f)
    logdet_diff_f = abs(logdet_scipy - logdet_native_f)
    print(f"  ||alpha_scipy - alpha_native_f|| = {alpha_diff_f:.2e}")
    print(f"  ||L_scipy - L_native_f||         = {chol_diff_f:.2e}")
    print(f"  |logdet_scipy - logdet_native_f| = {logdet_diff_f:.2e}")

    # Check correctness
    tol = 1e-10

    print("\n" + "=" * 60)
    if alpha_diff_c < tol and chol_diff_c < tol and logdet_diff_c < tol:
        print("✓ C-order input: PASS")
    else:
        print("✗ C-order input: FAIL")
        print(f"  Expected all differences < {tol:.2e}")

    if alpha_diff_f < tol and chol_diff_f < tol and logdet_diff_f < tol:
        print("✓ F-order input: PASS")
    else:
        print("✗ F-order input: FAIL")
        print(f"  Expected all differences < {tol:.2e}")

    print("=" * 60)

    # Raise if either failed
    if alpha_diff_c >= tol or alpha_diff_f >= tol:
        raise AssertionError(
            f"Native LAPACK does not match SciPy!\n"
            f"  C-order alpha diff: {alpha_diff_c:.2e}\n"
            f"  F-order alpha diff: {alpha_diff_f:.2e}"
        )


if __name__ == "__main__":
    test_minimal_lapack()
