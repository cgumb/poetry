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
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from poetry_gp.backends.blocked import run_blocked_step
from poetry_gp.backends.gpu_scoring import is_gpu_available


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


def set_num_threads(n: int | None) -> None:
    """Set number of BLAS threads."""
    if n is None:
        # Auto-detect from environment or use CPU count
        n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()

    n_str = str(n)
    os.environ["OMP_NUM_THREADS"] = n_str
    os.environ["OPENBLAS_NUM_THREADS"] = n_str
    os.environ["MKL_NUM_THREADS"] = n_str


def run_scoring_benchmark(
    m_rated: int,
    n_candidates: int,
    dim: int,
    seed: int,
    score_backend: str,
    num_threads: int | None = None,
) -> dict:
    """Run single scoring benchmark."""

    # Set threading for this run
    if score_backend == "python" and num_threads is not None:
        set_num_threads(num_threads)

    # Generate synthetic data
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n_candidates, dim))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12

    rated_indices = rng.choice(n_candidates, size=m_rated, replace=False)
    ratings = rng.normal(size=m_rated)

    # Run benchmark (fit + score)
    result = run_blocked_step(
        embeddings,
        rated_indices,
        ratings,
        length_scale=1.0,
        variance=1.0,
        noise=1e-3,
        fit_backend="python",  # Always use Python fit for fair comparison
        score_backend=score_backend,
        block_size=2048,
    )

    return {
        "m_rated": m_rated,
        "n_candidates": n_candidates,
        "dim": dim,
        "score_backend": score_backend,
        "num_threads": num_threads if score_backend == "python" else None,
        "fit_seconds": result.profile.fit_seconds,
        "score_seconds": result.profile.score_seconds,
        "total_seconds": result.profile.total_seconds,
    }


def main() -> None:
    args = parse_args()

    # Check GPU availability
    has_gpu = is_gpu_available()

    print("=" * 60)
    print("GP Scoring Performance Benchmark")
    print("=" * 60)
    print(f"n_candidates: {args.n_candidates:,}")
    print(f"dim: {args.dim}")
    print(f"m_rated values: {args.m_rated}")
    print(f"GPU available: {has_gpu}")
    if args.cpu_threads:
        print(f"CPU threads (multi): {args.cpu_threads}")
    print("=" * 60)
    print()

    results = []

    for m_rated in args.m_rated:
        print(f"Testing m_rated = {m_rated:,}")
        print("-" * 40)

        # 1. Python single-threaded
        print("  Python (1 thread)...", end=" ", flush=True)
        result = run_scoring_benchmark(
            m_rated, args.n_candidates, args.dim, args.seed,
            score_backend="python",
            num_threads=1,
        )
        results.append(result)
        print(f"score={result['score_seconds']:.3f}s")

        # 2. Python multi-threaded
        print(f"  Python ({args.cpu_threads or 'auto'} threads)...", end=" ", flush=True)
        result = run_scoring_benchmark(
            m_rated, args.n_candidates, args.dim, args.seed,
            score_backend="python",
            num_threads=args.cpu_threads,
        )
        results.append(result)
        print(f"score={result['score_seconds']:.3f}s")

        # 3. GPU (if available)
        if has_gpu:
            print("  GPU (CuPy)...", end=" ", flush=True)
            try:
                result = run_scoring_benchmark(
                    m_rated, args.n_candidates, args.dim, args.seed,
                    score_backend="gpu",
                )
                results.append(result)
                print(f"score={result['score_seconds']:.3f}s")
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
    print("-" * 60)
    print(f"{'m_rated':<10} {'CPU (1t)':<12} {'CPU (Mt)':<12} {'GPU':<12} {'Speedup'}")
    print("-" * 60)

    for m in args.m_rated:
        m_results = [r for r in results if r['m_rated'] == m]

        cpu_1t = next((r['score_seconds'] for r in m_results if r['num_threads'] == 1), None)
        cpu_mt = next((r['score_seconds'] for r in m_results if r['num_threads'] != 1 and r['score_backend'] == 'python'), None)
        gpu = next((r['score_seconds'] for r in m_results if r['score_backend'] == 'gpu'), None)

        speedup = ""
        if gpu and cpu_1t:
            speedup = f"{cpu_1t/gpu:.1f}x"

        cpu_1t_str = f"{cpu_1t:.3f}s" if cpu_1t else "N/A"
        cpu_mt_str = f"{cpu_mt:.3f}s" if cpu_mt else "N/A"
        gpu_str = f"{gpu:.3f}s" if gpu else "N/A"

        print(f"{m:<10,} {cpu_1t_str:<12} {cpu_mt_str:<12} {gpu_str:<12} {speedup}")

    print("-" * 60)


if __name__ == "__main__":
    main()
