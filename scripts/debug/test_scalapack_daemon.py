"""
Test script for persistent ScaLAPACK daemon (Milestone 1C).

This demonstrates the daemon eliminating subprocess overhead by keeping
MPI processes alive across multiple fit operations.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from poetry_gp.backends.scalapack_daemon_client import ScaLAPACKDaemonClient


def generate_test_data(m: int, d: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic test data for GP fitting."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((m, d)).astype(np.float64)
    y = rng.standard_normal(m).astype(np.float64)
    return x, y


def test_single_fit():
    """Test a single fit operation with the daemon."""
    print("\n=== Test 1: Single Fit ===")

    m, d = 500, 384
    x, y = generate_test_data(m, d)

    with ScaLAPACKDaemonClient(nprocs=4) as daemon:
        print(f"Testing fit with m={m}, d={d}")

        start = time.time()
        result = daemon.fit(
            x, y,
            length_scale=1.0,
            variance=1.0,
            noise=0.001,
            block_size=64,
        )
        elapsed = time.time() - start

        print(f"✓ Fit completed")
        print(f"  Wall time: {elapsed:.4f}s")
        print(f"  Daemon fit time: {result['fit_seconds']:.4f}s")
        print(f"  Daemon total time: {result['total_seconds']:.4f}s")
        print(f"  Log marginal likelihood: {result['log_marginal_likelihood']:.6f}")
        print(f"  Alpha shape: {result['alpha'].shape}")
        print(f"  L factor shape: {result['L_factor'].shape}")


def test_multiple_fits():
    """Test multiple fits to demonstrate subprocess overhead elimination."""
    print("\n=== Test 2: Multiple Fits (Daemon Reuse) ===")

    m, d = 500, 384
    n_fits = 5

    with ScaLAPACKDaemonClient(nprocs=4) as daemon:
        print(f"Running {n_fits} fits with same daemon")

        total_start = time.time()
        fit_times = []

        for i in range(n_fits):
            x, y = generate_test_data(m, d, seed=42 + i)

            start = time.time()
            result = daemon.fit(
                x, y,
                length_scale=1.0,
                variance=1.0,
                noise=0.001,
                block_size=64,
            )
            elapsed = time.time() - start
            fit_times.append(elapsed)

            print(f"  Fit {i+1}/{n_fits}: {elapsed:.4f}s (daemon: {result['fit_seconds']:.4f}s)")

        total_elapsed = time.time() - total_start

        print(f"\n✓ All fits completed")
        print(f"  Total time: {total_elapsed:.4f}s")
        print(f"  Average per fit: {np.mean(fit_times):.4f}s")
        print(f"  Min: {np.min(fit_times):.4f}s, Max: {np.max(fit_times):.4f}s")
        print(f"  Expected subprocess overhead per fit: ~0.16s")
        print(f"  Actual overhead (first fit may include daemon startup): ~{fit_times[0] - result['fit_seconds']:.4f}s")
        if n_fits > 1:
            print(f"  Subsequent fits overhead: ~{np.mean(fit_times[1:]) - result['fit_seconds']:.4f}s")


def test_different_problem_sizes():
    """Test daemon with different problem sizes."""
    print("\n=== Test 3: Different Problem Sizes ===")

    problem_sizes = [
        (200, 384),
        (500, 384),
        (1000, 384),
    ]

    with ScaLAPACKDaemonClient(nprocs=4) as daemon:
        for m, d in problem_sizes:
            x, y = generate_test_data(m, d)

            print(f"\nm={m}, d={d}:")

            start = time.time()
            result = daemon.fit(
                x, y,
                length_scale=1.0,
                variance=1.0,
                noise=0.001,
                block_size=64,
            )
            elapsed = time.time() - start

            print(f"  Wall time: {elapsed:.4f}s")
            print(f"  Daemon time: {result['fit_seconds']:.4f}s")
            print(f"  Overhead: {elapsed - result['fit_seconds']:.4f}s")


def main():
    """Run all tests."""
    print("="*60)
    print("ScaLAPACK Persistent Daemon Test (Milestone 1C)")
    print("="*60)

    # Check if daemon executable exists
    daemon_exe = Path("native/build/scalapack_daemon")
    if not daemon_exe.exists():
        print(f"\nError: Daemon executable not found at {daemon_exe}")
        print("Build it with:")
        print("  cmake -S native -B native/build")
        print("  cmake --build native/build")
        return

    print(f"\n✓ Daemon executable found: {daemon_exe}")

    try:
        test_single_fit()
        test_multiple_fits()
        test_different_problem_sizes()

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
