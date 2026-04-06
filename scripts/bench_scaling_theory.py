#!/usr/bin/env python3
"""
Validate Theoretical Complexity of GP Operations

This benchmark measures the scaling behavior of GP fitting and scoring
to empirically validate theoretical complexity:

1. Fit time vs m: Should show O(m³) scaling (Cholesky factorization)
2. Score time vs m (fixed n): Should show O(m²) scaling (variance computation)
3. Score time vs n (fixed m): Should show O(n) scaling (linear in candidates)

The results can be plotted on log-log axes to verify slopes match theory:
- Slope ≈ 3 for fit vs m
- Slope ≈ 2 for score vs m
- Slope ≈ 1 for score vs n

This connects textbook complexity to real measurements.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from poetry_gp.backends.blocked import run_blocked_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate theoretical complexity with empirical measurements"
    )
    parser.add_argument(
        "--fit-backend",
        choices=["python", "native_lapack", "native_reference"],
        default="python",
        help="Backend for fitting (default: python)",
    )
    parser.add_argument(
        "--score-backend",
        choices=["python", "native_lapack", "gpu"],
        default="python",
        help="Backend for scoring (default: python)",
    )
    parser.add_argument(
        "--m-fit-sweep",
        type=int,
        nargs="+",
        default=[100, 200, 500, 1000, 2000, 5000],
        help="m values for fit scaling (default: 100 200 500 1000 2000 5000)",
    )
    parser.add_argument(
        "--m-score-sweep",
        type=int,
        nargs="+",
        default=[100, 200, 500, 1000, 2000, 5000],
        help="m values for score scaling (default: 100 200 500 1000 2000 5000)",
    )
    parser.add_argument(
        "--n-score-sweep",
        type=int,
        nargs="+",
        default=[5000, 10000, 20000, 50000, 100000],
        help="n values for score scaling (default: 5000 10000 20000 50000 100000)",
    )
    parser.add_argument(
        "--n-fixed", type=int, default=10000, help="Fixed n for score vs m test"
    )
    parser.add_argument(
        "--m-fixed", type=int, default=1000, help="Fixed m for score vs n test"
    )
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip-fit", action="store_true", help="Skip fit scaling test"
    )
    parser.add_argument(
        "--skip-score-vs-m",
        action="store_true",
        help="Skip score vs m scaling test",
    )
    parser.add_argument(
        "--skip-score-vs-n",
        action="store_true",
        help="Skip score vs n scaling test",
    )
    parser.add_argument(
        "--scalapack-nprocs",
        type=int,
        default=4,
        help="Process count for ScaLAPACK",
    )
    parser.add_argument(
        "--scalapack-block-size",
        type=int,
        default=128,
        help="Block size for ScaLAPACK",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV file",
    )
    return parser.parse_args()


def measure_fit_scaling(
    args: argparse.Namespace,
    results: list[dict],
) -> None:
    """Measure fit time vs m (expect O(m³))."""
    print("\n" + "=" * 70)
    print("TEST 1: Fit Time vs m (Expect O(m³) Scaling)")
    print("=" * 70)
    print(
        f"Testing m values: {args.m_fit_sweep}\n"
        f"Fit backend: {args.fit_backend}\n"
        f"Theory: Cholesky factorization is O(m³)"
    )
    print("-" * 70)

    for m in args.m_fit_sweep:
        print(f"\nm = {m:>6,}: ", end="", flush=True)

        # Generate synthetic data
        rng = np.random.default_rng(args.seed)
        embeddings = rng.normal(size=(m, args.dim))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        rated_indices = np.arange(m)
        ratings = rng.normal(size=m)

        try:
            result = run_blocked_step(
                embeddings,
                rated_indices,
                ratings,
                length_scale=1.0,
                variance=1.0,
                noise=1e-3,
                fit_backend=args.fit_backend,
                score_backend="none",  # Skip scoring
                optimize_hyperparameters=False,
                scalapack_launcher="mpirun",  # Use mpirun inside Slurm jobs
                scalapack_nprocs=args.scalapack_nprocs,
                scalapack_block_size=args.scalapack_block_size,
            )

            fit_time = result.profile.fit_seconds
            print(f"fit={fit_time:.4f}s", flush=True)

            results.append(
                {
                    "test_type": "fit_vs_m",
                    "m": m,
                    "n": None,
                    "fit_backend": args.fit_backend,
                    "score_backend": None,
                    "fit_seconds": fit_time,
                    "score_seconds": None,
                    "total_seconds": result.profile.total_seconds,
                }
            )

        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            results.append(
                {
                    "test_type": "fit_vs_m",
                    "m": m,
                    "n": None,
                    "fit_backend": args.fit_backend,
                    "score_backend": None,
                    "fit_seconds": None,
                    "score_seconds": None,
                    "total_seconds": None,
                    "error": str(e),
                }
            )

    print("-" * 70)
    print(
        "On a log-log plot, these should form a line with slope ≈ 3 (cubic scaling)"
    )


def measure_score_vs_m(
    args: argparse.Namespace,
    results: list[dict],
) -> None:
    """Measure score time vs m with fixed n (expect O(m²))."""
    print("\n" + "=" * 70)
    print(f"TEST 2: Score Time vs m (Fixed n={args.n_fixed:,}, Expect O(m²))")
    print("=" * 70)
    print(
        f"Testing m values: {args.m_score_sweep}\n"
        f"Fit backend: {args.fit_backend}\n"
        f"Score backend: {args.score_backend}\n"
        f"Theory: Variance computation is O(nm²), so O(m²) with fixed n"
    )
    print("-" * 70)

    for m in args.m_score_sweep:
        print(f"\nm = {m:>6,}: ", end="", flush=True)

        # Generate synthetic data
        rng = np.random.default_rng(args.seed)
        embeddings = rng.normal(size=(args.n_fixed, args.dim))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        rated_indices = rng.choice(args.n_fixed, size=m, replace=False)
        ratings = rng.normal(size=m)

        try:
            result = run_blocked_step(
                embeddings,
                rated_indices,
                ratings,
                length_scale=1.0,
                variance=1.0,
                noise=1e-3,
                fit_backend=args.fit_backend,
                score_backend=args.score_backend,
                optimize_hyperparameters=False,
                scalapack_launcher="mpirun",  # Use mpirun inside Slurm jobs
                scalapack_nprocs=args.scalapack_nprocs,
                scalapack_block_size=args.scalapack_block_size,
            )

            fit_time = result.profile.fit_seconds
            score_time = result.profile.score_seconds
            print(f"fit={fit_time:.4f}s, score={score_time:.4f}s", flush=True)

            results.append(
                {
                    "test_type": "score_vs_m",
                    "m": m,
                    "n": args.n_fixed,
                    "fit_backend": args.fit_backend,
                    "score_backend": args.score_backend,
                    "fit_seconds": fit_time,
                    "score_seconds": score_time,
                    "total_seconds": result.profile.total_seconds,
                }
            )

        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            results.append(
                {
                    "test_type": "score_vs_m",
                    "m": m,
                    "n": args.n_fixed,
                    "fit_backend": args.fit_backend,
                    "score_backend": args.score_backend,
                    "fit_seconds": None,
                    "score_seconds": None,
                    "total_seconds": None,
                    "error": str(e),
                }
            )

    print("-" * 70)
    print(
        "On a log-log plot, score_seconds vs m should have slope ≈ 2 (quadratic)"
    )


def measure_score_vs_n(
    args: argparse.Namespace,
    results: list[dict],
) -> None:
    """Measure score time vs n with fixed m (expect O(n))."""
    print("\n" + "=" * 70)
    print(f"TEST 3: Score Time vs n (Fixed m={args.m_fixed:,}, Expect O(n))")
    print("=" * 70)
    print(
        f"Testing n values: {args.n_score_sweep}\n"
        f"Fit backend: {args.fit_backend}\n"
        f"Score backend: {args.score_backend}\n"
        f"Theory: Variance computation is O(nm²), so O(n) with fixed m"
    )
    print("-" * 70)

    for n in args.n_score_sweep:
        print(f"\nn = {n:>6,}: ", end="", flush=True)

        # Generate synthetic data
        rng = np.random.default_rng(args.seed)
        embeddings = rng.normal(size=(n, args.dim))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        rated_indices = rng.choice(n, size=args.m_fixed, replace=False)
        ratings = rng.normal(size=args.m_fixed)

        try:
            result = run_blocked_step(
                embeddings,
                rated_indices,
                ratings,
                length_scale=1.0,
                variance=1.0,
                noise=1e-3,
                fit_backend=args.fit_backend,
                score_backend=args.score_backend,
                optimize_hyperparameters=False,
                scalapack_launcher="mpirun",  # Use mpirun inside Slurm jobs
                scalapack_nprocs=args.scalapack_nprocs,
                scalapack_block_size=args.scalapack_block_size,
            )

            fit_time = result.profile.fit_seconds
            score_time = result.profile.score_seconds
            print(f"fit={fit_time:.4f}s, score={score_time:.4f}s", flush=True)

            results.append(
                {
                    "test_type": "score_vs_n",
                    "m": args.m_fixed,
                    "n": n,
                    "fit_backend": args.fit_backend,
                    "score_backend": args.score_backend,
                    "fit_seconds": fit_time,
                    "score_seconds": score_time,
                    "total_seconds": result.profile.total_seconds,
                }
            )

        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            results.append(
                {
                    "test_type": "score_vs_n",
                    "m": args.m_fixed,
                    "n": n,
                    "fit_backend": args.fit_backend,
                    "score_backend": args.score_backend,
                    "fit_seconds": None,
                    "score_seconds": None,
                    "total_seconds": None,
                    "error": str(e),
                }
            )

    print("-" * 70)
    print(
        "On a log-log plot, score_seconds vs n should have slope ≈ 1 (linear)"
    )


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("THEORETICAL COMPLEXITY VALIDATION")
    print("=" * 70)
    print("This benchmark measures scaling behavior to validate theoretical complexity:")
    print("  • Fit vs m:    O(m³) - Cholesky factorization")
    print("  • Score vs m:  O(m²) - Variance computation (fixed n)")
    print("  • Score vs n:  O(n)  - Linear in candidates (fixed m)")
    print()
    print("Results can be plotted on log-log axes to verify slopes.")

    results = []

    # Test 1: Fit scaling
    if not args.skip_fit:
        measure_fit_scaling(args, results)

    # Test 2: Score vs m
    if not args.skip_score_vs_m:
        measure_score_vs_m(args, results)

    # Test 3: Score vs n
    if not args.skip_score_vs_n:
        measure_score_vs_n(args, results)

    # Write CSV
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "test_type",
        "m",
        "n",
        "dim",
        "fit_backend",
        "score_backend",
        "fit_seconds",
        "score_seconds",
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
                "error": result.get("error", ""),
                **result,
            }
            writer.writerow(row)

    print(f"Results saved to: {args.output_csv}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not args.skip_fit:
        print("\nFit Scaling (O(m³) theory):")
        print("-" * 50)
        print(f"{'m':<10} {'fit_seconds':<15} {'ratio':<10}")
        print("-" * 50)
        fit_results = [r for r in results if r["test_type"] == "fit_vs_m"]
        prev_m, prev_time = None, None
        for r in fit_results:
            if r.get("fit_seconds") is not None:
                m = r["m"]
                time = r["fit_seconds"]
                if prev_m and prev_time:
                    m_ratio = m / prev_m
                    time_ratio = time / prev_time
                    expected_ratio = m_ratio**3
                    ratio_str = f"{time_ratio:.2f} (expect {expected_ratio:.1f})"
                else:
                    ratio_str = "baseline"
                print(f"{m:<10,} {time:<15.4f} {ratio_str}")
                prev_m, prev_time = m, time

    if not args.skip_score_vs_m:
        print(f"\nScore vs m Scaling (O(m²) theory, n={args.n_fixed:,}):")
        print("-" * 50)
        print(f"{'m':<10} {'score_seconds':<15} {'ratio':<10}")
        print("-" * 50)
        score_m_results = [r for r in results if r["test_type"] == "score_vs_m"]
        prev_m, prev_time = None, None
        for r in score_m_results:
            if r.get("score_seconds") is not None:
                m = r["m"]
                time = r["score_seconds"]
                if prev_m and prev_time:
                    m_ratio = m / prev_m
                    time_ratio = time / prev_time
                    expected_ratio = m_ratio**2
                    ratio_str = f"{time_ratio:.2f} (expect {expected_ratio:.1f})"
                else:
                    ratio_str = "baseline"
                print(f"{m:<10,} {time:<15.4f} {ratio_str}")
                prev_m, prev_time = m, time

    if not args.skip_score_vs_n:
        print(f"\nScore vs n Scaling (O(n) theory, m={args.m_fixed:,}):")
        print("-" * 50)
        print(f"{'n':<10} {'score_seconds':<15} {'ratio':<10}")
        print("-" * 50)
        score_n_results = [r for r in results if r["test_type"] == "score_vs_n"]
        prev_n, prev_time = None, None
        for r in score_n_results:
            if r.get("score_seconds") is not None:
                n = r["n"]
                time = r["score_seconds"]
                if prev_n and prev_time:
                    n_ratio = n / prev_n
                    time_ratio = time / prev_time
                    expected_ratio = n_ratio
                    ratio_str = f"{time_ratio:.2f} (expect {expected_ratio:.1f})"
                else:
                    ratio_str = "baseline"
                print(f"{n:<10,} {time:<15.4f} {ratio_str}")
                prev_n, prev_time = n, time

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("To visualize these results on log-log plots:")
    print(f"  python scripts/visualize_scaling.py {args.output_csv}")
    print()
    print("Expected slopes on log-log plots:")
    print("  • Fit vs m:    slope ≈ 3")
    print("  • Score vs m:  slope ≈ 2")
    print("  • Score vs n:  slope ≈ 1")
    print("=" * 70)


if __name__ == "__main__":
    main()
