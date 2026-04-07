#!/usr/bin/env python3
"""
Time Breakdown Analysis: Where Does Time Go?

This benchmark provides detailed profiling of each phase in a GP iteration:
1. Kernel assembly: Computing K_rr from embeddings (O(m²d))
2. Cholesky factorization: Decomposing K_rr (O(m³))
3. Solve: Computing alpha = K^{-1} y (O(m²))
4. Mean computation: k*ᵀ alpha for all candidates (O(nmd))
5. Variance computation: Diagonal of K** - k*K^{-1}k*ᵀ (O(nm²))
6. Selection: Finding max mean or max variance (O(n))

The relative time spent in each phase depends on problem size:
- Small m: Cholesky dominates (O(m³))
- Large n: Variance computation dominates (O(nm²))
- Selection is always trivial (O(n))

This helps identify where optimization effort should focus.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from poetry_gp.backends.blocked import run_blocked_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile time breakdown in GP operations"
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 5000, 10000],
        help="m values to test (default: 100 500 1000 5000 10000)",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[10000, 50000, 100000],
        help="n values to test (default: 10000 50000 100000)",
    )
    parser.add_argument(
        "--fit-backend",
        choices=["python", "native_lapack", "native_reference"],
        default="python",
        help="Backend for fitting",
    )
    parser.add_argument(
        "--score-backend",
        choices=["python", "native_lapack", "gpu"],
        default="python",
        help="Backend for scoring",
    )
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
        "--scalapack-executable",
        type=str,
        default="native/build/scalapack_gp_fit",
        help="Path to ScaLAPACK executable (default: native/build/scalapack_gp_fit)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV file",
    )
    return parser.parse_args()


def measure_breakdown(
    m: int,
    n: int,
    args: argparse.Namespace,
) -> dict | None:
    """Measure detailed time breakdown for one (m, n) configuration."""
    rng = np.random.default_rng(args.seed)
    embeddings = rng.normal(size=(n, args.dim))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    rated_indices = rng.choice(n, size=m, replace=False)
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
            scalapack_executable=args.scalapack_executable,
            scalapack_nprocs=args.scalapack_nprocs,
            scalapack_block_size=args.scalapack_block_size,
        )

        fit_time = result.profile.fit_seconds
        score_time = result.profile.score_seconds
        select_time = result.profile.select_seconds
        total_time = result.profile.total_seconds

        # Estimate overhead (everything not accounted for)
        overhead = total_time - (fit_time + score_time + select_time)
        overhead = max(0.0, overhead)  # Can't be negative

        return {
            "m": m,
            "n": n,
            "fit_backend": args.fit_backend,
            "score_backend": args.score_backend,
            "fit_seconds": fit_time,
            "score_seconds": score_time,
            "select_seconds": select_time,
            "overhead_seconds": overhead,
            "total_seconds": total_time,
            "fit_pct": 100 * fit_time / total_time if total_time > 0 else 0,
            "score_pct": 100 * score_time / total_time if total_time > 0 else 0,
            "select_pct": 100 * select_time / total_time if total_time > 0 else 0,
            "overhead_pct": 100 * overhead / total_time if total_time > 0 else 0,
        }

    except Exception as e:
        print(f"    FAILED: {e}")
        return None


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("TIME BREAKDOWN ANALYSIS")
    print("=" * 80)
    print("This benchmark profiles where time is spent in GP operations:")
    print("  • Fit:      O(m³) Cholesky factorization + O(m²) solve")
    print("  • Score:    O(nm²) variance + O(nmd) mean computation")
    print("  • Select:   O(n) find max")
    print("  • Overhead: Data movement, kernel assembly, etc.")
    print()
    print(f"Fit backend:   {args.fit_backend}")
    print(f"Score backend: {args.score_backend}")
    print(f"Testing {len(args.m_values)} m × {len(args.n_values)} n = {len(args.m_values) * len(args.n_values)} configurations")
    print("=" * 80)

    results = []

    for m in args.m_values:
        for n in args.n_values:
            print(f"\nm={m:>6,}, n={n:>6,}: ", end="", flush=True)

            result = measure_breakdown(m, n, args)

            if result:
                print(
                    f"fit={result['fit_seconds']:.3f}s "
                    f"({result['fit_pct']:.1f}%), "
                    f"score={result['score_seconds']:.3f}s "
                    f"({result['score_pct']:.1f}%), "
                    f"total={result['total_seconds']:.3f}s"
                )
                results.append(result)
            else:
                # Still record the failed attempt
                results.append(
                    {
                        "m": m,
                        "n": n,
                        "fit_backend": args.fit_backend,
                        "score_backend": args.score_backend,
                        "error": "measurement_failed",
                    }
                )

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
        "fit_backend",
        "score_backend",
        "fit_seconds",
        "score_seconds",
        "select_seconds",
        "overhead_seconds",
        "total_seconds",
        "fit_pct",
        "score_pct",
        "select_pct",
        "overhead_pct",
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

    # Print summary tables
    print("\n" + "=" * 80)
    print("SUMMARY: WHERE DOES TIME GO?")
    print("=" * 80)

    # Group by n value
    for n in args.n_values:
        n_results = [r for r in results if r.get("n") == n and "error" not in r]
        if not n_results:
            continue

        print(f"\nn = {n:,} candidates")
        print("-" * 80)
        print(
            f"{'m':<10} {'Fit':<12} {'Score':<12} {'Select':<12} {'Overhead':<12} {'Total':<10}"
        )
        print("-" * 80)

        for r in n_results:
            m = r["m"]
            fit_str = f"{r['fit_seconds']:.3f}s ({r['fit_pct']:.0f}%)"
            score_str = f"{r['score_seconds']:.3f}s ({r['score_pct']:.0f}%)"
            select_str = f"{r['select_seconds']:.3f}s ({r['select_pct']:.0f}%)"
            overhead_str = (
                f"{r['overhead_seconds']:.3f}s ({r['overhead_pct']:.0f}%)"
            )
            total_str = f"{r['total_seconds']:.3f}s"

            print(
                f"{m:<10,} {fit_str:<12} {score_str:<12} {select_str:<12} {overhead_str:<12} {total_str:<10}"
            )

        # Identify bottleneck for this n
        if n_results:
            print()
            print("Bottleneck analysis:")
            for r in n_results:
                m = r["m"]
                phases = [
                    ("Fit", r["fit_pct"]),
                    ("Score", r["score_pct"]),
                    ("Select", r["select_pct"]),
                    ("Overhead", r["overhead_pct"]),
                ]
                dominant = max(phases, key=lambda x: x[1])
                print(f"  m={m:>6,}: {dominant[0]} dominates ({dominant[1]:.0f}%)")

    # Cross-problem-size analysis
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    # Find where scoring dominates
    score_dominant = [
        r for r in results if r.get("score_pct", 0) > 50 and "error" not in r
    ]
    if score_dominant:
        print("\nScore computation dominates (>50%) for:")
        for r in score_dominant:
            print(
                f"  m={r['m']:>6,}, n={r['n']:>6,}: "
                f"score={r['score_pct']:.0f}%, fit={r['fit_pct']:.0f}%"
            )
        print("  → Variance O(nm²) is the bottleneck for large n")

    # Find where fitting dominates
    fit_dominant = [
        r for r in results if r.get("fit_pct", 0) > 50 and "error" not in r
    ]
    if fit_dominant:
        print("\nFit computation dominates (>50%) for:")
        for r in fit_dominant:
            print(
                f"  m={r['m']:>6,}, n={r['n']:>6,}: "
                f"fit={r['fit_pct']:.0f}%, score={r['score_pct']:.0f}%"
            )
        print("  → Cholesky O(m³) is the bottleneck for small n or large m")

    # Find where overhead dominates
    overhead_dominant = [
        r for r in results if r.get("overhead_pct", 0) > 50 and "error" not in r
    ]
    if overhead_dominant:
        print("\nOverhead dominates (>50%) for:")
        for r in overhead_dominant:
            print(
                f"  m={r['m']:>6,}, n={r['n']:>6,}: "
                f"overhead={r['overhead_pct']:.0f}%, total={r['total_seconds']:.3f}s"
            )
        print("  → Problem too small; process spawn/communication dominates")

    print("\n" + "=" * 80)
    print("OPTIMIZATION IMPLICATIONS")
    print("=" * 80)
    print("Where to focus optimization effort:")
    print("  • Small m, small n:  Reduce overhead (use PyBind11)")
    print("  • Large m, small n:  Optimize Cholesky (use ScaLAPACK)")
    print("  • Small m, large n:  Optimize scoring (use GPU)")
    print("  • Large m, large n:  Both fit and score matter (use ScaLAPACK + GPU)")
    print()
    print("Selection (finding max) is always negligible (<1%).")
    print("=" * 80)


if __name__ == "__main__":
    main()
