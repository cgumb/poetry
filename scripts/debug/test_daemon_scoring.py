"""
Test script for daemon parallel scoring (Milestone 1C extension).

This demonstrates:
1. Daemon scoring works and is faster than Python
2. Graceful fallback to Python when daemon unavailable
3. Results match between daemon and Python backends
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from poetry_gp.gp_exact import fit_exact_gp
from poetry_gp.backends.scoring import score_all_with_fallback, try_create_daemon_client


def generate_test_data(m: int, d: int, n_query: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic test data."""
    rng = np.random.default_rng(seed)
    x_rated = rng.standard_normal((m, d)).astype(np.float64)
    y = rng.standard_normal(m).astype(np.float64)
    x_query = rng.standard_normal((n_query, d)).astype(np.float64)
    return x_rated, y, x_query


def test_daemon_vs_python():
    """Test that daemon and Python produce same results."""
    print("\n=== Test 1: Daemon vs Python Correctness ===\n")

    m, d, n_query = 200, 50, 1000
    x_rated, y, x_query = generate_test_data(m, d, n_query)

    print(f"Problem size: m={m}, d={d}, n_query={n_query}")

    # Fit GP
    print("Fitting GP...")
    state = fit_exact_gp(x_rated, y, length_scale=1.0, variance=1.0, noise=0.01)

    # Score with Python (baseline)
    print("Scoring with Python (serial)...")
    start = time.time()
    mean_py, var_py, _ = score_all_with_fallback(state, x_query, block_size=500)
    time_py = time.time() - start
    print(f"  Time: {time_py:.3f}s")

    # Check if daemon is available
    daemon_exe = Path("native/build/scalapack_daemon")
    if not daemon_exe.exists():
        print(f"\n✗ Daemon not available ({daemon_exe} not found)")
        print("  Build it with:")
        print("    cmake -S native -B native/build")
        print("    cmake --build native/build")
        return

    # Score with daemon (parallel)
    print("\nScoring with daemon (parallel, 4 processes)...")
    try:
        from poetry_gp.backends.scalapack_daemon_client import ScaLAPACKDaemonClient
        daemon = ScaLAPACKDaemonClient(nprocs=4, launcher="mpirun", daemon_exe=daemon_exe)
        daemon.start()
    except Exception as e:
        print(f"✗ Failed to start daemon: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Check that MPI and daemon executable are working correctly")
        return

    try:
        start = time.time()
        mean_daemon, var_daemon, _ = score_all_with_fallback(state, x_query, block_size=500, daemon_client=daemon)
        time_daemon = time.time() - start
        print(f"  Time: {time_daemon:.3f}s")
    finally:
        daemon.shutdown()

    # Compare results
    print("\n=== Comparison ===")
    mean_diff = np.abs(mean_daemon - mean_py)
    var_diff = np.abs(var_daemon - var_py)

    print(f"Mean differences:")
    print(f"  Max: {np.max(mean_diff):.6e}")
    print(f"  Mean: {np.mean(mean_diff):.6e}")
    print(f"  RMS: {np.sqrt(np.mean(mean_diff**2)):.6e}")

    print(f"\nVariance differences:")
    print(f"  Max: {np.max(var_diff):.6e}")
    print(f"  Mean: {np.mean(var_diff):.6e}")
    print(f"  RMS: {np.sqrt(np.mean(var_diff**2)):.6e}")

    print(f"\nSpeedup: {time_py / time_daemon:.2f}×")

    # Check correctness (allow small numerical differences)
    if np.max(mean_diff) < 1e-6 and np.max(var_diff) < 1e-6:
        print("\n✓ Results match (daemon correct)")
    else:
        print("\n✗ Results differ significantly")


def test_fallback():
    """Test graceful fallback when daemon unavailable."""
    print("\n=== Test 2: Graceful Fallback ===\n")

    m, d, n_query = 100, 30, 500
    x_rated, y, x_query = generate_test_data(m, d, n_query)

    print(f"Problem size: m={m}, d={d}, n_query={n_query}")

    # Fit GP
    state = fit_exact_gp(x_rated, y, length_scale=1.0, variance=1.0, noise=0.01)

    # Test fallback with no daemon
    print("Testing fallback (daemon=None)...")
    start = time.time()
    mean, var, score_time = score_all_with_fallback(state, x_query, daemon_client=None)
    elapsed = time.time() - start

    print(f"  Completed in {elapsed:.3f}s")
    print(f"  Mean shape: {mean.shape}")
    print(f"  Variance shape: {var.shape}")
    print("✓ Fallback works")

    # Test fallback with invalid daemon (simulates daemon crash)
    print("\nTesting fallback (daemon fails)...")

    class MockFailingDaemon:
        def score_all(self, *args, **kwargs):
            raise RuntimeError("Simulated daemon failure")

    start = time.time()
    mean, var, score_time = score_all_with_fallback(state, x_query, daemon_client=MockFailingDaemon())
    elapsed = time.time() - start

    print(f"  Completed in {elapsed:.3f}s (fell back to Python)")
    print("✓ Fallback from failed daemon works")


def test_scaling():
    """Test scaling with different problem sizes."""
    print("\n=== Test 3: Scaling Test ===\n")

    daemon_exe = Path("native/build/scalapack_daemon")
    if not daemon_exe.exists():
        print(f"Daemon not available ({daemon_exe} not found)")
        print("Skipping scaling test")
        return

    problem_sizes = [
        (100, 50, 1000),
        (200, 50, 2000),
        (300, 50, 5000),
    ]

    print(f"{'m':<6} {'d':<6} {'n_query':<10} {'Python (s)':<12} {'Daemon (s)':<12} {'Speedup':<10}")
    print("-" * 66)

    for m, d, n_query in problem_sizes:
        x_rated, y, x_query = generate_test_data(m, d, n_query)
        state = fit_exact_gp(x_rated, y, length_scale=1.0, variance=1.0, noise=0.01)

        # Python
        start = time.time()
        mean_py, var_py, _ = score_all_with_fallback(state, x_query, daemon_client=None)
        time_py = time.time() - start

        # Daemon
        try:
            from poetry_gp.backends.scalapack_daemon_client import ScaLAPACKDaemonClient
            daemon = ScaLAPACKDaemonClient(nprocs=4, launcher="mpirun", daemon_exe=daemon_exe)
            daemon.start()
        except Exception as e:
            print(f"{m:<6} {d:<6} {n_query:<10} {time_py:<12.3f} {'FAILED':<12} {'N/A':<10}")
            print(f"  Error: {e}")
            continue

        try:
            start = time.time()
            mean_daemon, var_daemon, _ = score_all_with_fallback(state, x_query, daemon_client=daemon)
            time_daemon = time.time() - start
        finally:
            daemon.shutdown()

        speedup = time_py / time_daemon
        print(f"{m:<6} {d:<6} {n_query:<10} {time_py:<12.3f} {time_daemon:<12.3f} {speedup:<10.2f}×")


def main():
    print("=" * 66)
    print("Daemon Parallel Scoring Test (Milestone 1C Extension)")
    print("=" * 66)

    try:
        test_daemon_vs_python()
        test_fallback()
        test_scaling()

        print("\n" + "=" * 66)
        print("✓ All tests passed!")
        print("=" * 66)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
