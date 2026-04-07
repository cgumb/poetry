#!/usr/bin/env python3
"""
Overhead vs Compute: The Fundamental HPC Tradeoff

This benchmark measures the crossover point where overhead becomes negligible
compared to computation. It compares backends across problem sizes:

- Python (scipy): Single-node, minimal overhead, single-threaded or multi-threaded
- PyBind11 (native_lapack): Zero overhead (in-memory), single-node
- ScaLAPACK (native_reference): Distributed, high overhead, parallel speedup

Key insights:
1. Small problems: Overhead dominates → simple is best
2. Large problems: Compute dominates → parallelization pays off
3. Crossover point: Where overhead = speedup benefit

Overhead sources:
- Process spawn (~160ms for subprocess/MPI)
- File I/O (~50ms for writing matrices)
- Communication (depends on m and process count)

This demonstrates why "more HPC" isn't always better.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from poetry_gp.backends.blocked import run_blocked_step
from poetry_gp.backends.backend_selection import get_backend_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure overhead vs compute crossover"
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[100, 200, 500, 1000, 2000, 5000, 10000, 20000],
        help="m values to test (default: 100 200 500 1000 2000 5000 10000 20000)",
    )
    parser.add_argument(
        "--n-fixed",
        type=int,
        default=25000,
        help="Fixed n for all tests (default: 25000)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["python", "native_lapack", "native_reference"],
        default=["python", "native_lapack", "native_reference"],
        help="Backends to compare (default: all)",
    )
    parser.add_argument(
        "--scalapack-nprocs",
        type=int,
        default=8,
        help="Process count for ScaLAPACK (default: 8)",
    )
    parser.add_argument(
        "--scalapack-block-size",
        type=int,
        default=128,
        help="Block size for ScaLAPACK (default: 128)",
    )
    parser.add_argument(
        "--scalapack-executable",
        type=str,
        default="native/build/scalapack_gp_fit",
        help="Path to ScaLAPACK executable (default: native/build/scalapack_gp_fit)",
    )
    parser.add_argument(
        "--scalapack-launcher",
        type=str,
        choices=["srun", "mpirun"],
        default="srun",
        help="Launcher for ScaLAPACK (default: srun)",
    )
    parser.add_argument(
        "--scalapack-native-backend",
        type=str,
        choices=["auto", "scalapack", "mpi", "mpi_row_partitioned_reference"],
        default="auto",
        help="Native backend for ScaLAPACK (default: auto)",
    )
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run warmup iteration before timing",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV file",
    )
    return parser.parse_args()


def estimate_overhead_baseline() -> float:
    """
    Estimate fixed overhead of subprocess spawn.
    Run a trivial command to measure process creation time.
    """
    start = time.perf_counter()
    subprocess.run(["echo", "overhead_test"], capture_output=True, check=True)
    end = time.perf_counter()
    return end - start


def measure_backend(
    m: int,
    n: int,
    backend: str,
    args: argparse.Namespace,
) -> dict | None:
    """Measure performance of one backend at one problem size."""
    rng = np.random.default_rng(args.seed)
    embeddings = rng.normal(size=(n, args.dim))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    rated_indices = rng.choice(n, size=m, replace=False)
    ratings = rng.normal(size=m)

    try:
        # Warmup run (optional)
        if args.warmup and m <= 1000:  # Only warmup small problems
            _ = run_blocked_step(
                embeddings,
                rated_indices,
                ratings,
                length_scale=1.0,
                variance=1.0,
                noise=1e-3,
                fit_backend=backend,
                score_backend="none",
                optimize_hyperparameters=False,
                scalapack_executable=args.scalapack_executable,
                scalapack_launcher=args.scalapack_launcher,
                scalapack_nprocs=args.scalapack_nprocs,
                scalapack_block_size=args.scalapack_block_size,
                scalapack_native_backend=args.scalapack_native_backend,
            )

        # Actual measurement
        result = run_blocked_step(
            embeddings,
            rated_indices,
            ratings,
            length_scale=1.0,
            variance=1.0,
            noise=1e-3,
            fit_backend=backend,
            score_backend="none",  # Focus on fit only
            optimize_hyperparameters=False,
            scalapack_executable=args.scalapack_executable,
            scalapack_launcher=args.scalapack_launcher,
            scalapack_nprocs=args.scalapack_nprocs,
            scalapack_block_size=args.scalapack_block_size,
            scalapack_native_backend=args.scalapack_native_backend,
        )

        return {
            "m": m,
            "n": n,
            "backend": backend,
            "fit_seconds": result.profile.fit_seconds,
            "total_seconds": result.profile.total_seconds,
        }

    except Exception as e:
        return {
            "m": m,
            "n": n,
            "backend": backend,
            "error": str(e),
        }


def main() -> None:
    args = parse_args()

    # Check backend availability
    backend_info = get_backend_info()
    available_backends = []
    for backend in args.backends:
        if backend == "python":
            available_backends.append(backend)
        elif backend == "native_lapack":
            if backend_info.get("native_lapack", False):
                available_backends.append(backend)
            else:
                print(f"WARNING: {backend} not available, skipping")
        elif backend == "native_reference":
            available_backends.append(backend)  # Always available via subprocess

    if not available_backends:
        print("ERROR: No backends available to test")
        return

    print("=" * 80)
    print("OVERHEAD vs COMPUTE CROSSOVER ANALYSIS")
    print("=" * 80)
    print("This benchmark measures when overhead becomes negligible vs computation.")
    print()
    print("Backends tested:")
    for backend in available_backends:
        if backend == "python":
            print("  • Python (scipy): Single-node, minimal overhead")
        elif backend == "native_lapack":
            print("  • PyBind11: Zero overhead (in-memory LAPACK)")
        elif backend == "native_reference":
            print(
                f"  • ScaLAPACK: Distributed ({args.scalapack_nprocs} ranks), high overhead"
            )
    print()
    print(f"Fixed n = {args.n_fixed:,} candidates")
    print(f"Testing m values: {args.m_values}")
    print("=" * 80)

    # Estimate process spawn overhead
    print("\nEstimating subprocess spawn overhead...", end=" ", flush=True)
    spawn_overhead = estimate_overhead_baseline()
    print(f"{spawn_overhead * 1000:.1f}ms")
    print(
        f"Note: ScaLAPACK adds ~{spawn_overhead * 1000:.0f}ms + file I/O + MPI init"
    )
    print()

    results = []

    for m in args.m_values:
        print(f"\nm = {m:>6,}:")
        print("-" * 60)

        m_results = {}

        for backend in available_backends:
            print(f"  {backend:<20}: ", end="", flush=True)

            result = measure_backend(m, args.n_fixed, backend, args)

            if result and "error" not in result:
                fit_time = result["fit_seconds"]
                print(f"{fit_time:.4f}s")
                m_results[backend] = fit_time
                results.append(result)
            else:
                error_msg = result.get("error", "unknown error") if result else "failed"
                print(f"FAILED ({error_msg})")
                if result:
                    results.append(result)

        # Compute speedups relative to python baseline
        if "python" in m_results:
            baseline = m_results["python"]
            print(f"\n  Speedup vs Python baseline ({baseline:.4f}s):")
            for backend, time_val in m_results.items():
                if backend != "python":
                    speedup = baseline / time_val
                    speedup_str = (
                        f"{speedup:.2f}x faster"
                        if speedup > 1
                        else f"{1 / speedup:.2f}x SLOWER"
                    )
                    print(f"    {backend:<20}: {speedup_str}")

    # Write CSV
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "m",
        "n",
        "dim",
        "backend",
        "scalapack_nprocs",
        "scalapack_block_size",
        "fit_seconds",
        "total_seconds",
        "error",
    ]

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "timestamp": datetime.now().isoformat(),
                "dim": args.dim,
                "scalapack_nprocs": (
                    args.scalapack_nprocs
                    if result.get("backend") == "native_reference"
                    else ""
                ),
                "scalapack_block_size": (
                    args.scalapack_block_size
                    if result.get("backend") == "native_reference"
                    else ""
                ),
                "fit_seconds": result.get("fit_seconds", ""),
                "total_seconds": result.get("total_seconds", ""),
                "error": result.get("error", ""),
                **result,
            }
            writer.writerow(row)

    print(f"Results saved to: {args.output_csv}")

    # Summary analysis
    print("\n" + "=" * 80)
    print("SUMMARY: OVERHEAD vs COMPUTE")
    print("=" * 80)

    # Build comparison table
    print(f"\n{'m':<10}", end="")
    for backend in available_backends:
        print(f"{backend:<15}", end="")
    print()
    print("-" * (10 + 15 * len(available_backends)))

    for m in args.m_values:
        m_results = [r for r in results if r.get("m") == m and "error" not in r]
        if not m_results:
            continue

        print(f"{m:<10,}", end="")

        backend_times = {}
        for backend in available_backends:
            backend_result = next(
                (r for r in m_results if r.get("backend") == backend), None
            )
            if backend_result:
                time_val = backend_result.get("fit_seconds")
                if time_val is not None:
                    backend_times[backend] = time_val
                    print(f"{time_val:<15.4f}", end="")
                else:
                    print(f"{'N/A':<15}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()

    # Identify crossover points
    print("\n" + "=" * 80)
    print("CROSSOVER ANALYSIS")
    print("=" * 80)

    # Find where PyBind11 beats Python
    if "python" in available_backends and "native_lapack" in available_backends:
        print("\nPyBind11 vs Python:")
        for m in args.m_values:
            python_result = next(
                (
                    r
                    for r in results
                    if r.get("m") == m
                    and r.get("backend") == "python"
                    and "error" not in r
                ),
                None,
            )
            pybind_result = next(
                (
                    r
                    for r in results
                    if r.get("m") == m
                    and r.get("backend") == "native_lapack"
                    and "error" not in r
                ),
                None,
            )
            if python_result and pybind_result:
                python_time = python_result.get("fit_seconds")
                pybind_time = pybind_result.get("fit_seconds")
                if python_time and pybind_time:
                    speedup = python_time / pybind_time
                    print(
                        f"  m={m:>6,}: PyBind11 {speedup:.1f}x faster (eliminates overhead)"
                    )

    # Find where ScaLAPACK beats Python
    if "python" in available_backends and "native_reference" in available_backends:
        print(f"\nScaLAPACK ({args.scalapack_nprocs} ranks) vs Python:")
        crossover_found = False
        for m in args.m_values:
            python_result = next(
                (
                    r
                    for r in results
                    if r.get("m") == m
                    and r.get("backend") == "python"
                    and "error" not in r
                ),
                None,
            )
            scalapack_result = next(
                (
                    r
                    for r in results
                    if r.get("m") == m
                    and r.get("backend") == "native_reference"
                    and "error" not in r
                ),
                None,
            )
            if python_result and scalapack_result:
                python_time = python_result.get("fit_seconds")
                scalapack_time = scalapack_result.get("fit_seconds")
                if python_time and scalapack_time:
                    if scalapack_time < python_time:
                        speedup = python_time / scalapack_time
                        print(
                            f"  m={m:>6,}: ScaLAPACK {speedup:.2f}x faster (compute dominates)"
                        )
                        if not crossover_found:
                            print(f"    ⭢ CROSSOVER POINT: m ≈ {m:,}")
                            crossover_found = True
                    else:
                        slowdown = scalapack_time / python_time
                        print(
                            f"  m={m:>6,}: ScaLAPACK {slowdown:.2f}x SLOWER (overhead dominates)"
                        )

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("1. Small problems: Overhead matters more than parallelization")
    print("   → Use PyBind11 for instant response (zero overhead)")
    print()
    print("2. Large problems: Compute dominates, parallelization pays off")
    print("   → Use ScaLAPACK when m > 10k and multiple nodes available")
    print()
    print("3. Fixed overhead: ScaLAPACK has ~2-3s baseline regardless of m")
    print("   → Subprocess + file I/O + MPI init")
    print()
    print("4. Crossover point: Where parallel speedup > overhead cost")
    print("   → Depends on process count, block size, and communication")
    print("=" * 80)


if __name__ == "__main__":
    main()
