#!/usr/bin/env python3
"""
Demo Scaling Plots with Synthetic Data

This script generates example plots showing ideal theoretical scaling behavior.
Useful for understanding what to look for before running real benchmarks.

The plots show:
1. Perfect O(m³) scaling for fit time
2. Perfect O(m²) scaling for score time vs m
3. Perfect O(n) scaling for score time vs n

Real measurements will deviate from these ideals due to overhead, cache effects,
and communication costs. This demo helps you recognize the patterns.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demo scaling plots")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/demo"),
        help="Output directory (default: figures/demo)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format (default: png)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for raster formats")
    return parser.parse_args()


def generate_synthetic_data():
    """Generate synthetic data with ideal scaling behavior."""
    # Fit time: O(m³)
    m_fit = np.array([100, 200, 500, 1000, 2000, 5000, 10000])
    fit_time = 1e-9 * m_fit**3  # Cubic scaling

    # Score time vs m: O(m²) with fixed n=10000
    m_score = np.array([100, 200, 500, 1000, 2000, 5000])
    n_fixed = 10000
    score_time_vs_m = 1e-10 * n_fixed * m_score**2  # Quadratic in m

    # Score time vs n: O(n) with fixed m=1000
    n_score = np.array([5000, 10000, 20000, 50000, 100000])
    m_fixed = 1000
    score_time_vs_n = 1e-10 * n_score * m_fixed**2  # Linear in n

    return {
        "fit": (m_fit, fit_time),
        "score_vs_m": (m_score, score_time_vs_m, n_fixed),
        "score_vs_n": (n_score, score_time_vs_n, m_fixed),
    }


def plot_demo_scaling(data, args):
    """Create demo log-log plots showing ideal scaling."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Fit time vs m (O(m³))
    ax = axes[0]
    m_fit, fit_time = data["fit"]

    ax.loglog(m_fit, fit_time, "o-", label="Ideal O(m³)", linewidth=2, markersize=8)

    # Reference line
    m_ref = np.array([m_fit.min(), m_fit.max()])
    t_ref = fit_time[0] * (m_ref / m_fit[0]) ** 3
    ax.loglog(m_ref, t_ref, "--", color="gray", label="Slope = 3", alpha=0.7)

    ax.set_xlabel("m (rated points)", fontsize=11)
    ax.set_ylabel("Fit time (seconds)", fontsize=11)
    ax.set_title("Fit Scaling: Cholesky O(m³)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.text(
        0.05,
        0.95,
        "Perfect cubic scaling\n(no overhead)",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # Plot 2: Score time vs m (O(m²))
    ax = axes[1]
    m_score, score_time_vs_m, n_fixed = data["score_vs_m"]

    ax.loglog(
        m_score, score_time_vs_m, "o-", label="Ideal O(m²)", linewidth=2, markersize=8
    )

    # Reference line
    m_ref = np.array([m_score.min(), m_score.max()])
    t_ref = score_time_vs_m[0] * (m_ref / m_score[0]) ** 2
    ax.loglog(m_ref, t_ref, "--", color="gray", label="Slope = 2", alpha=0.7)

    ax.set_xlabel("m (rated points)", fontsize=11)
    ax.set_ylabel("Score time (seconds)", fontsize=11)
    ax.set_title(f"Score vs m (n={n_fixed:,}): O(m²)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.text(
        0.05,
        0.95,
        "Variance computation\n(quadratic in m)",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # Plot 3: Score time vs n (O(n))
    ax = axes[2]
    n_score, score_time_vs_n, m_fixed = data["score_vs_n"]

    ax.loglog(
        n_score, score_time_vs_n, "o-", label="Ideal O(n)", linewidth=2, markersize=8
    )

    # Reference line
    n_ref = np.array([n_score.min(), n_score.max()])
    t_ref = score_time_vs_n[0] * (n_ref / n_score[0])
    ax.loglog(n_ref, t_ref, "--", color="gray", label="Slope = 1", alpha=0.7)

    ax.set_xlabel("n (candidates)", fontsize=11)
    ax.set_ylabel("Score time (seconds)", fontsize=11)
    ax.set_title(f"Score vs n (m={m_fixed:,}): O(n)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.text(
        0.05,
        0.95,
        "Linear in candidates\n(embarrassingly parallel)",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    output_file = args.output_dir / f"demo_scaling_theory.{args.format}"
    plt.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_demo_breakdown(args):
    """Create demo stacked bar chart showing time breakdown."""
    fig, ax = plt.subplots(figsize=(10, 6))

    m_vals = [100, 500, 1000, 5000, 10000]
    n_fixed = 10000

    # Generate synthetic breakdown (shifts from score-dominated to fit-dominated)
    fit_times = np.array([0.001 * m**3 / 1e6 for m in m_vals])
    score_times = np.array([0.001 * n_fixed * m**2 / 1e7 for m in m_vals])
    select_times = np.array([0.0001] * len(m_vals))
    overhead = np.array([0.01] * len(m_vals))

    x = np.arange(len(m_vals))
    width = 0.6

    p1 = ax.bar(x, fit_times, width, label="Fit O(m³)", color="#1f77b4")
    p2 = ax.bar(
        x, score_times, width, bottom=fit_times, label="Score O(nm²)", color="#ff7f0e"
    )
    p3 = ax.bar(
        x,
        select_times,
        width,
        bottom=fit_times + score_times,
        label="Select O(n)",
        color="#2ca02c",
    )
    p4 = ax.bar(
        x,
        overhead,
        width,
        bottom=fit_times + score_times + select_times,
        label="Overhead",
        color="#d62728",
    )

    ax.set_xlabel("m (rated points)", fontsize=11)
    ax.set_ylabel("Time (seconds)", fontsize=11)
    ax.set_title(f"Time Breakdown (n={n_fixed:,})", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m:,}" for m in m_vals])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate shift in bottleneck
    ax.text(
        0.02,
        0.98,
        "Small m: Score dominates\nLarge m: Fit dominates",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    output_file = args.output_dir / f"demo_time_breakdown.{args.format}"
    plt.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_demo_crossover(args):
    """Create demo backend comparison showing overhead crossover."""
    fig, ax = plt.subplots(figsize=(10, 6))

    m_vals = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000])

    # Python: O(m³) with no overhead
    python_time = 1e-9 * m_vals**3

    # PyBind11: 10× faster (optimized LAPACK)
    pybind_time = python_time / 10

    # ScaLAPACK: 2.5s overhead + O(m³/8) computation (8 ranks)
    overhead = 2.5
    scalapack_time = overhead + (1e-9 * m_vals**3) / 8

    ax.loglog(m_vals, python_time, "o-", label="Python (scipy)", linewidth=2, markersize=8, color="#1f77b4")
    ax.loglog(m_vals, pybind_time, "s-", label="PyBind11 (zero overhead)", linewidth=2, markersize=8, color="#ff7f0e")
    ax.loglog(m_vals, scalapack_time, "^-", label="ScaLAPACK (8 ranks)", linewidth=2, markersize=8, color="#2ca02c")

    # Mark overhead floor
    ax.axhline(overhead, color="red", linestyle="--", alpha=0.5, linewidth=1.5)

    ax.set_xlabel("m (rated points)", fontsize=12)
    ax.set_ylabel("Fit time (seconds)", fontsize=12)
    ax.set_title("Overhead vs Compute Crossover", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Annotate crossover
    crossover_idx = np.where(scalapack_time < python_time)[0][0]
    crossover_m = m_vals[crossover_idx]
    ax.axvline(crossover_m, color="black", linestyle=":", alpha=0.5, linewidth=1.5)

    ax.text(
        0.05,
        0.95,
        f"ScaLAPACK overhead ≈{overhead}s\nCrossover ≈ m={crossover_m:,}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    output_file = args.output_dir / f"demo_overhead_crossover.{args.format}"
    plt.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING DEMO SCALING PLOTS")
    print("=" * 70)
    print("Creating synthetic data with ideal theoretical scaling...")
    print()
    print("These plots show what perfect O(m³), O(m²), O(n) look like.")
    print("Real measurements will deviate due to overhead and system effects.")
    print()
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)
    print()

    # Generate synthetic data
    data = generate_synthetic_data()

    # Create plots
    print("Creating plots:")
    print("  1. Scaling theory (log-log with ideal slopes)")
    plot_demo_scaling(data, args)

    print("  2. Time breakdown (stacked bars)")
    plot_demo_breakdown(args)

    print("  3. Overhead crossover (backend comparison)")
    plot_demo_crossover(args)

    print()
    print("=" * 70)
    print("DEMO PLOTS COMPLETE")
    print("=" * 70)
    print(f"Figures saved to: {args.output_dir}")
    print()
    print("These demonstrate:")
    print("  • What O(m³) looks like on a log-log plot (slope = 3)")
    print("  • How bottlenecks shift from score to fit as m grows")
    print("  • Why overhead matters for small problems")
    print()
    print("Now run real benchmarks to see how close reality matches theory!")
    print("  sbatch scripts/pedagogical_benchmarks.slurm")
    print("=" * 70)


if __name__ == "__main__":
    main()
