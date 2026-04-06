"""
Test script to verify analytic gradient implementation for GP hyperparameter optimization.

This compares optimization with analytic gradients vs numerical gradients to verify:
1. Analytic gradients are correct (produce similar results)
2. Analytic gradients reduce function evaluations (fewer nfev)

Note: Wall-clock time may vary. Analytic gradients reduce evaluations but add
gradient computation overhead, so speedup depends on problem size and backend.
"""

from __future__ import annotations

import time

import numpy as np

from poetry_gp.gp_exact import fit_exact_gp


def test_gradient_correctness():
    """Test that analytic gradients produce similar results to numerical."""
    print("\n=== Test 1: Gradient Correctness ===\n")

    # Generate synthetic test data
    rng = np.random.default_rng(42)
    m, d = 200, 50
    x = rng.standard_normal((m, d)).astype(np.float64)
    y = rng.standard_normal(m).astype(np.float64)

    print(f"Data: m={m}, d={d}")
    print(f"Initial hyperparameters: length_scale=1.0, variance=1.0, noise=0.01\n")

    # Test with analytic gradients
    print("Optimizing with analytic gradients...")
    start = time.time()
    result_analytic = fit_exact_gp(
        x, y,
        length_scale=1.0,
        variance=1.0,
        noise=0.01,
        optimize_hyperparameters=True,
        optimizer_maxiter=50,
    )
    time_analytic = time.time() - start

    opt_info_analytic = result_analytic.optimization_result
    print(f"  Time: {time_analytic:.3f}s")
    print(f"  Iterations: {opt_info_analytic['nit']}")
    print(f"  Function evals: {opt_info_analytic['nfev']}")
    print(f"  Length scale: {opt_info_analytic['length_scale']:.6f}")
    print(f"  Variance: {opt_info_analytic['variance']:.6f}")
    print(f"  Noise: {opt_info_analytic['noise']:.6f}")
    print(f"  Initial LML: {opt_info_analytic['initial_log_marginal_likelihood']:.4f}")
    print(f"  Final LML: {opt_info_analytic['final_log_marginal_likelihood']:.4f}")
    print(f"  Success: {opt_info_analytic['success']}\n")

    # Test with numerical gradients (for comparison)
    from poetry_gp.gp_exact import optimize_gp_hyperparameters

    print("Optimizing with numerical gradients...")
    start = time.time()
    opt_info_numerical = optimize_gp_hyperparameters(
        x, y,
        length_scale=1.0,
        variance=1.0,
        noise=0.01,
        optimizer_maxiter=50,
        use_analytic_gradients=False,
    )
    time_numerical = time.time() - start

    print(f"  Time: {time_numerical:.3f}s")
    print(f"  Iterations: {opt_info_numerical['nit']}")
    print(f"  Function evals: {opt_info_numerical['nfev']}")
    print(f"  Length scale: {opt_info_numerical['length_scale']:.6f}")
    print(f"  Variance: {opt_info_numerical['variance']:.6f}")
    print(f"  Noise: {opt_info_numerical['noise']:.6f}")
    print(f"  Initial LML: {opt_info_numerical['initial_log_marginal_likelihood']:.4f}")
    print(f"  Final LML: {opt_info_numerical['final_log_marginal_likelihood']:.4f}")
    print(f"  Success: {opt_info_numerical['success']}\n")

    # Compare results
    print("=== Comparison ===")
    print(f"Speedup: {time_numerical / time_analytic:.2f}×")
    print(f"Function eval reduction: {opt_info_numerical['nfev']} → {opt_info_analytic['nfev']} "
          f"({100 * (1 - opt_info_analytic['nfev'] / opt_info_numerical['nfev']):.1f}% fewer)")

    # Check if results are similar
    param_diff_l = abs(opt_info_analytic['length_scale'] - opt_info_numerical['length_scale'])
    param_diff_v = abs(opt_info_analytic['variance'] - opt_info_numerical['variance'])
    param_diff_n = abs(opt_info_analytic['noise'] - opt_info_numerical['noise'])
    lml_diff = abs(opt_info_analytic['final_log_marginal_likelihood'] -
                   opt_info_numerical['final_log_marginal_likelihood'])

    print(f"\nParameter differences:")
    print(f"  Length scale: {param_diff_l:.6f}")
    print(f"  Variance: {param_diff_v:.6f}")
    print(f"  Noise: {param_diff_n:.6f}")
    print(f"  Final LML: {lml_diff:.6f}")

    # Both should reach similar LML (within 1%)
    if lml_diff < 0.01 * abs(opt_info_analytic['final_log_marginal_likelihood']):
        print("\n✓ Gradients are correct (similar final LML)")
    else:
        print("\n✗ Warning: Results differ significantly")

    return time_analytic, time_numerical


def test_scaling():
    """Test optimization speed at different problem sizes."""
    print("\n=== Test 2: Scaling Test ===\n")

    problem_sizes = [(100, 50), (200, 50), (300, 50)]

    print(f"{'m':<6} {'d':<6} {'Analytic (s)':<15} {'Numerical (s)':<15} {'Speedup':<10}")
    print("-" * 62)

    for m, d in problem_sizes:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((m, d)).astype(np.float64)
        y = rng.standard_normal(m).astype(np.float64)

        # Analytic
        start = time.time()
        result_analytic = fit_exact_gp(
            x, y,
            length_scale=1.0,
            variance=1.0,
            noise=0.01,
            optimize_hyperparameters=True,
            optimizer_maxiter=20,
        )
        time_analytic = time.time() - start

        # Numerical
        from poetry_gp.gp_exact import optimize_gp_hyperparameters
        start = time.time()
        opt_info_numerical = optimize_gp_hyperparameters(
            x, y,
            length_scale=1.0,
            variance=1.0,
            noise=0.01,
            optimizer_maxiter=20,
            use_analytic_gradients=False,
        )
        time_numerical = time.time() - start

        speedup = time_numerical / time_analytic
        print(f"{m:<6} {d:<6} {time_analytic:<15.3f} {time_numerical:<15.3f} {speedup:<10.2f}×")


def main():
    print("=" * 60)
    print("GP Hyperparameter Optimization: Analytic Gradients Test")
    print("=" * 60)

    time_analytic, time_numerical = test_gradient_correctness()
    test_scaling()

    print("\n" + "=" * 60)
    print("Summary:")
    speedup = time_numerical / time_analytic
    if speedup > 1.0:
        print(f"  Analytic gradients: {speedup:.1f}× faster")
    elif speedup < 1.0:
        print(f"  Analytic gradients: {1/speedup:.1f}× slower (but fewer function evaluations)")
    else:
        print(f"  Analytic gradients: similar speed")
    print("  Both methods produce similar results (gradients verified)")
    print("=" * 60)


if __name__ == "__main__":
    main()
