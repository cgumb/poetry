#!/usr/bin/env python3
"""
Create synthetic score backend comparison data for demonstration.

This creates realistic-looking data based on expected performance characteristics:
- Python (1 thread): O(nm²) baseline, single-threaded BLAS
- Python (8 threads): ~4-6x speedup from multi-threading
- GPU (CuPy): ~10-20x speedup for large m, overhead for small m

This allows us to prepare the slide before running actual benchmarks.
"""

import csv
from datetime import datetime
from pathlib import Path
import numpy as np

def generate_sample_data(output_csv: Path):
    """Generate synthetic score backend comparison data."""

    # Parameters
    m_values = [100, 500, 1000, 2000, 5000]
    n_candidates = 85000
    dim = 384

    # Scaling constants (tuned to be realistic)
    # Score complexity: O(n*m*d) + O(n*m²)
    # For n=85k, m=5k, d=384: ~163M ops for kernel + 2.1B ops for variance

    base_time_per_m2 = 5e-9  # seconds per nm² operation (single-threaded)
    kernel_time_per_nmd = 1e-11  # seconds per nmd operation

    results = []

    for m in m_values:
        # Compute theoretical times
        kernel_ops = n_candidates * m * dim
        variance_ops = n_candidates * m * m

        # Python single-threaded (baseline)
        kernel_time = kernel_ops * kernel_time_per_nmd
        variance_time = variance_ops * base_time_per_m2
        python_1t_time = kernel_time + variance_time + np.random.normal(0, 0.05 * (kernel_time + variance_time))

        # Python multi-threaded (8 threads): ~5x speedup on variance, ~3x on kernel
        python_mt_time = (kernel_time / 3.0 + variance_time / 5.0) + np.random.normal(0, 0.05 * python_1t_time / 5)

        # GPU: Fast for large m, overhead for small m
        # Overhead: ~100ms for small problems, ~10ms for large
        # Compute: ~15x faster than single-threaded for variance-heavy workload
        gpu_overhead = 0.15 * np.exp(-m / 1000)  # Overhead decreases with problem size
        gpu_compute = (kernel_time + variance_time) / 15.0
        gpu_time = gpu_overhead + gpu_compute + np.random.normal(0, 0.02 * gpu_compute)

        # Ensure times are positive
        python_1t_time = max(0.01, python_1t_time)
        python_mt_time = max(0.01, python_mt_time)
        gpu_time = max(0.01, gpu_time)

        # Add results
        results.append({
            "timestamp": datetime.now().isoformat(),
            "m_rated": m,
            "n_candidates": n_candidates,
            "dim": dim,
            "score_backend": "python",
            "num_threads": 1,
            "fit_seconds": 0.05,  # Not important for this comparison
            "score_seconds": python_1t_time,
            "total_seconds": python_1t_time + 0.05,
        })

        results.append({
            "timestamp": datetime.now().isoformat(),
            "m_rated": m,
            "n_candidates": n_candidates,
            "dim": dim,
            "score_backend": "python",
            "num_threads": 8,
            "fit_seconds": 0.05,
            "score_seconds": python_mt_time,
            "total_seconds": python_mt_time + 0.05,
        })

        results.append({
            "timestamp": datetime.now().isoformat(),
            "m_rated": m,
            "n_candidates": n_candidates,
            "dim": dim,
            "score_backend": "gpu",
            "num_threads": None,
            "fit_seconds": 0.05,
            "score_seconds": gpu_time,
            "total_seconds": gpu_time + 0.05,
        })

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        fieldnames = [
            "timestamp", "m_rated", "n_candidates", "dim",
            "score_backend", "num_threads",
            "fit_seconds", "score_seconds", "total_seconds"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 60)
    print(f"Sample data created: {output_csv}")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"{'m_rated':<10} {'CPU (1t)':<12} {'CPU (8t)':<12} {'GPU':<12} {'Best Speedup'}")
    print("-" * 60)

    for m in m_values:
        m_results = [r for r in results if r['m_rated'] == m]

        cpu_1t = next(r['score_seconds'] for r in m_results if r['num_threads'] == 1)
        cpu_mt = next(r['score_seconds'] for r in m_results if r['num_threads'] == 8)
        gpu = next(r['score_seconds'] for r in m_results if r['score_backend'] == 'gpu')

        best_time = min(cpu_mt, gpu)
        speedup = cpu_1t / best_time

        print(f"{m:<10,} {cpu_1t:<12.3f} {cpu_mt:<12.3f} {gpu:<12.3f} {speedup:.1f}x")

    print("-" * 60)
    print()
    print("NOTE: This is SAMPLE data for demonstration.")
    print("Run 'sbatch scripts/bench_score_backends.slurm' for actual benchmarks.")

if __name__ == "__main__":
    output_csv = Path("../../results/score_backend_comparison.csv")
    generate_sample_data(output_csv)
