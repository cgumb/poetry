#!/usr/bin/env python3
"""
Benchmark GP Scoring Performance: CPU vs GPU

Compares scoring (posterior prediction) performance across:
- Python (single-threaded BLAS)
- Python (multi-threaded BLAS)
- GPU (CuPy/CUDA)

Note: Scoring complexity is O(n_candidates × m × d) + O(n_candidates × m²),
where m is the number of rated points and n_candidates is the number of
points to score.

IMPORTANT: Thread count must be set BEFORE importing numpy/scipy!
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path


def set_num_threads(n: int | None) -> None:
    """Set number of BLAS threads BEFORE importing numpy."""
    if n is None:
        # Auto-detect from environment or use CPU count
        n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()

    n_str = str(n)
    os.environ["OMP_NUM_THREADS"] = n_str
    os.environ["OPENBLAS_NUM_THREADS"] = n_str
    os.environ["MKL_NUM_THREADS"] = n_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GP scoring performance")
    parser.add_argument("--m-rated", type=int, nargs="+", default=[100, 500, 1000, 2000, 5000],
                        help="Number of rated points to test")
    parser.add_argument("--n-candidates", type=int, default=25000,
                        help="Number of candidate points to score")
    parser.add_argument("--dim", type=int, default=384,
                        help="Embedding dimension")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="Number of threads for multi-threaded CPU (default: auto-detect)")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Output CSV file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # NOW import numpy and other heavy libraries (after thread env vars are set)
    import numpy as np
    from poetry_gp.backends.blocked import run_blocked_step
    from poetry_gp.backends.gpu_scoring import is_gpu_available
    from poetry_gp.backends.native_lapack import is_native_available

    # Check backend availability
    has_gpu = is_gpu_available()
    has_native = is_native_available()

    print("=" * 60)
    print("GP Scoring Performance Benchmark")
    print("=" * 60)
    print(f"n_candidates: {args.n_candidates:,}")
    print(f"dim: {args.dim}")
    print(f"m_rated values: {args.m_rated}")
    print(f"Native LAPACK available: {has_native}")
    print(f"GPU available: {has_gpu}")
    if args.cpu_threads:
        print(f"CPU threads (multi): {args.cpu_threads}")
    print("=" * 60)
    print()

    results = []

    # Test each m_rated value
    for m_rated in args.m_rated:
        print(f"Testing m_rated = {m_rated:,}")
        print("-" * 40)

        # Generate synthetic data once per m_rated
        rng = np.random.default_rng(args.seed)
        embeddings = rng.normal(size=(args.n_candidates, args.dim))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        rated_indices = rng.choice(args.n_candidates, size=m_rated, replace=False)
        ratings = rng.normal(size=m_rated)

        # 1. Python single-threaded
        print("  Python (1 thread)...", end=" ", flush=True)
        set_num_threads(1)
        # Force reimport of BLAS-using modules
        if 'scipy.linalg' in sys.modules:
            del sys.modules['scipy.linalg']

        result = run_blocked_step(
            embeddings,
            rated_indices,
            ratings,
            length_scale=1.0,
            variance=1.0,
            noise=1e-3,
            fit_backend="python",
            score_backend="python",
            block_size=2048,
        )

        results.append({
            "m_rated": m_rated,
            "n_candidates": args.n_candidates,
            "dim": args.dim,
            "score_backend": "python",
            "num_threads": 1,
            "fit_seconds": result.profile.fit_seconds,
            "score_seconds": result.profile.score_seconds,
            "total_seconds": result.profile.total_seconds,
        })
        print(f"score={result.profile.score_seconds:.3f}s")

        # 2. Python multi-threaded
        print(f"  Python ({args.cpu_threads or 'auto'} threads)...", end=" ", flush=True)
        set_num_threads(args.cpu_threads)
        # Force reimport of BLAS-using modules
        if 'scipy.linalg' in sys.modules:
            del sys.modules['scipy.linalg']

        result = run_blocked_step(
            embeddings,
            rated_indices,
            ratings,
            length_scale=1.0,
            variance=1.0,
            noise=1e-3,
            fit_backend="python",
            score_backend="python",
            block_size=2048,
        )

        results.append({
            "m_rated": m_rated,
            "n_candidates": args.n_candidates,
            "dim": args.dim,
            "score_backend": "python",
            "num_threads": args.cpu_threads if args.cpu_threads else "auto",
            "fit_seconds": result.profile.fit_seconds,
            "score_seconds": result.profile.score_seconds,
            "total_seconds": result.profile.total_seconds,
        })
        print(f"score={result.profile.score_seconds:.3f}s")

        # 3. Native LAPACK (if available)
        if has_native:
            print("  Native LAPACK (PyBind11)...", end=" ", flush=True)
            try:
                result = run_blocked_step(
                    embeddings,
                    rated_indices,
                    ratings,
                    length_scale=1.0,
                    variance=1.0,
                    noise=1e-3,
                    fit_backend="native_lapack",
                    score_backend="native_lapack",
                    block_size=2048,
                )

                results.append({
                    "m_rated": m_rated,
                    "n_candidates": args.n_candidates,
                    "dim": args.dim,
                    "score_backend": "native_lapack",
                    "num_threads": None,
                    "fit_seconds": result.profile.fit_seconds,
                    "score_seconds": result.profile.score_seconds,
                    "total_seconds": result.profile.total_seconds,
                })
                print(f"score={result.profile.score_seconds:.3f}s")
            except Exception as e:
                print(f"FAILED: {e}")

        # 4. GPU (if available)
        if has_gpu:
            print("  GPU (CuPy)...", end=" ", flush=True)
            try:
                result = run_blocked_step(
                    embeddings,
                    rated_indices,
                    ratings,
                    length_scale=1.0,
                    variance=1.0,
                    noise=1e-3,
                    fit_backend="python",
                    score_backend="gpu",
                    block_size=2048,
                )

                results.append({
                    "m_rated": m_rated,
                    "n_candidates": args.n_candidates,
                    "dim": args.dim,
                    "score_backend": "gpu",
                    "num_threads": None,
                    "fit_seconds": result.profile.fit_seconds,
                    "score_seconds": result.profile.score_seconds,
                    "total_seconds": result.profile.total_seconds,
                })
                print(f"score={result.profile.score_seconds:.3f}s")
            except Exception as e:
                print(f"FAILED: {e}")

        print()

    # Write CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_csv, 'w', newline='') as f:
        fieldnames = [
            "timestamp", "m_rated", "n_candidates", "dim",
            "score_backend", "num_threads",
            "fit_seconds", "score_seconds", "total_seconds"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {"timestamp": datetime.now().isoformat(), **result}
            writer.writerow(row)

    print("=" * 60)
    print(f"Results saved to: {args.output_csv}")
    print("=" * 60)

    # Print summary
    print("\nSummary (score_seconds only):")
    print("-" * 80)
    print(f"{'m_rated':<10} {'CPU (1t)':<12} {'CPU (Mt)':<12} {'Native':<12} {'GPU':<12} {'Best Speedup'}")
    print("-" * 80)

    for m in args.m_rated:
        m_results = [r for r in results if r['m_rated'] == m]

        cpu_1t = next((r['score_seconds'] for r in m_results if r['num_threads'] == 1), None)
        cpu_mt = next((r['score_seconds'] for r in m_results if r['num_threads'] != 1 and r['score_backend'] == 'python'), None)
        native = next((r['score_seconds'] for r in m_results if r['score_backend'] == 'native_lapack'), None)
        gpu = next((r['score_seconds'] for r in m_results if r['score_backend'] == 'gpu'), None)

        # Compute best speedup vs baseline (cpu_1t)
        speedup = ""
        if cpu_1t:
            best_time = min(filter(None, [cpu_mt, native, gpu]))
            if best_time:
                speedup = f"{cpu_1t/best_time:.1f}x"

        cpu_1t_str = f"{cpu_1t:.3f}s" if cpu_1t else "N/A"
        cpu_mt_str = f"{cpu_mt:.3f}s" if cpu_mt else "N/A"
        native_str = f"{native:.3f}s" if native else "N/A"
        gpu_str = f"{gpu:.3f}s" if gpu else "N/A"

        print(f"{m:<10,} {cpu_1t_str:<12} {cpu_mt_str:<12} {native_str:<12} {gpu_str:<12} {speedup}")

    print("-" * 80)


if __name__ == "__main__":
    main()
