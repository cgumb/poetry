#!/usr/bin/env python3
"""
Visualize Scaling Behavior for Pedagogical Purposes

Creates publication-quality plots that connect theory to practice:

1. Log-log plots showing theoretical complexity slopes
2. Stacked bar charts showing time breakdown
3. Crossover analysis comparing backends

These visualizations are designed for teaching HPC concepts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize scaling benchmark results"
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="CSV files from benchmark scripts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures (default: figures/)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for raster formats (default: 150)",
    )
    return parser.parse_args()


def plot_scaling_theory(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """
    Create log-log plots showing theoretical complexity.
    Three subplots: fit vs m, score vs m, score vs n
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Fit time vs m (expect O(m³))
    ax = axes[0]
    fit_data = df[df["test_type"] == "fit_vs_m"].dropna(subset=["fit_seconds"])

    if not fit_data.empty:
        ax.loglog(
            fit_data["m"], fit_data["fit_seconds"], "o-", label="Measured", linewidth=2
        )

        # Reference line: O(m³)
        m_ref = np.array([fit_data["m"].min(), fit_data["m"].max()])
        # Fit to first point
        m0, t0 = fit_data.iloc[0]["m"], fit_data.iloc[0]["fit_seconds"]
        t_ref = t0 * (m_ref / m0) ** 3
        ax.loglog(m_ref, t_ref, "--", color="gray", label="O(m³) reference", alpha=0.7)

        ax.set_xlabel("m (rated points)", fontsize=11)
        ax.set_ylabel("Fit time (seconds)", fontsize=11)
        ax.set_title("Fit Scaling: Cholesky O(m³)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Annotate slope
        ax.text(
            0.05,
            0.95,
            "Expected slope = 3",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Plot 2: Score time vs m (expect O(m²))
    ax = axes[1]
    score_m_data = df[df["test_type"] == "score_vs_m"].dropna(subset=["score_seconds"])

    if not score_m_data.empty:
        ax.loglog(
            score_m_data["m"],
            score_m_data["score_seconds"],
            "o-",
            label="Measured",
            linewidth=2,
        )

        # Reference line: O(m²)
        m_ref = np.array([score_m_data["m"].min(), score_m_data["m"].max()])
        m0, t0 = score_m_data.iloc[0]["m"], score_m_data.iloc[0]["score_seconds"]
        t_ref = t0 * (m_ref / m0) ** 2
        ax.loglog(
            m_ref, t_ref, "--", color="gray", label="O(m²) reference", alpha=0.7
        )

        ax.set_xlabel("m (rated points)", fontsize=11)
        ax.set_ylabel("Score time (seconds)", fontsize=11)
        n_fixed = score_m_data.iloc[0]["n"]
        ax.set_title(
            f"Score vs m (n={n_fixed:,}): Variance O(m²)",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax.text(
            0.05,
            0.95,
            "Expected slope = 2",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Plot 3: Score time vs n (expect O(n))
    ax = axes[2]
    score_n_data = df[df["test_type"] == "score_vs_n"].dropna(subset=["score_seconds"])

    if not score_n_data.empty:
        ax.loglog(
            score_n_data["n"],
            score_n_data["score_seconds"],
            "o-",
            label="Measured",
            linewidth=2,
        )

        # Reference line: O(n)
        n_ref = np.array([score_n_data["n"].min(), score_n_data["n"].max()])
        n0, t0 = score_n_data.iloc[0]["n"], score_n_data.iloc[0]["score_seconds"]
        t_ref = t0 * (n_ref / n0)
        ax.loglog(n_ref, t_ref, "--", color="gray", label="O(n) reference", alpha=0.7)

        ax.set_xlabel("n (candidates)", fontsize=11)
        ax.set_ylabel("Score time (seconds)", fontsize=11)
        m_fixed = score_n_data.iloc[0]["m"]
        ax.set_title(
            f"Score vs n (m={m_fixed:,}): Linear", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax.text(
            0.05,
            0.95,
            "Expected slope = 1",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    output_file = args.output_dir / f"scaling_theory.{args.format}"
    plt.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_time_breakdown(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """
    Create stacked bar charts showing where time is spent.
    """
    # Filter valid data
    valid_data = df.dropna(
        subset=["fit_seconds", "score_seconds", "select_seconds", "total_seconds"]
    )

    if valid_data.empty:
        print("No time breakdown data to plot")
        return

    # Group by n value
    n_values = sorted(valid_data["n"].unique())

    n_plots = len(n_values)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for idx, n in enumerate(n_values):
        ax = axes[idx]
        n_data = valid_data[valid_data["n"] == n].sort_values("m")

        m_vals = n_data["m"].values
        fit_times = n_data["fit_seconds"].values
        score_times = n_data["score_seconds"].values
        select_times = n_data["select_seconds"].values
        overhead_times = n_data.get("overhead_seconds", np.zeros_like(fit_times)).values

        # Create stacked bars
        x = np.arange(len(m_vals))
        width = 0.6

        p1 = ax.bar(x, fit_times, width, label="Fit (O(m³))", color="#1f77b4")
        p2 = ax.bar(
            x, score_times, width, bottom=fit_times, label="Score (O(nm²))", color="#ff7f0e"
        )
        p3 = ax.bar(
            x,
            select_times,
            width,
            bottom=fit_times + score_times,
            label="Select (O(n))",
            color="#2ca02c",
        )
        p4 = ax.bar(
            x,
            overhead_times,
            width,
            bottom=fit_times + score_times + select_times,
            label="Overhead",
            color="#d62728",
        )

        ax.set_xlabel("m (rated points)", fontsize=11)
        ax.set_ylabel("Time (seconds)", fontsize=11)
        ax.set_title(f"Time Breakdown (n={n:,})", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m:,}" for m in m_vals], rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate dominant phase for each m
        for i, m in enumerate(m_vals):
            total = (
                fit_times[i] + score_times[i] + select_times[i] + overhead_times[i]
            )
            phases = [
                ("Fit", fit_times[i]),
                ("Score", score_times[i]),
                ("Overhead", overhead_times[i]),
            ]
            dominant = max(phases, key=lambda x: x[1])
            if dominant[1] / total > 0.5:  # Only annotate if >50%
                ax.text(
                    i,
                    total,
                    f"{dominant[0]}\n{100*dominant[1]/total:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    output_file = args.output_dir / f"time_breakdown.{args.format}"
    plt.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def plot_overhead_crossover(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """
    Plot backend comparison showing overhead vs compute crossover.
    """
    valid_data = df.dropna(subset=["fit_seconds"])

    if valid_data.empty:
        print("No overhead crossover data to plot")
        return

    backends = valid_data["backend"].unique()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"python": "#1f77b4", "native_lapack": "#ff7f0e", "native_reference": "#2ca02c"}
    markers = {"python": "o", "native_lapack": "s", "native_reference": "^"}
    labels = {
        "python": "Python (scipy)",
        "native_lapack": "PyBind11 (zero overhead)",
        "native_reference": "ScaLAPACK (distributed)",
    }

    for backend in backends:
        backend_data = valid_data[valid_data["backend"] == backend].sort_values("m")
        if not backend_data.empty:
            ax.loglog(
                backend_data["m"],
                backend_data["fit_seconds"],
                marker=markers.get(backend, "o"),
                label=labels.get(backend, backend),
                linewidth=2,
                markersize=8,
                color=colors.get(backend),
            )

    ax.set_xlabel("m (rated points)", fontsize=12)
    ax.set_ylabel("Fit time (seconds)", fontsize=12)
    ax.set_title(
        "Overhead vs Compute: Backend Crossover", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add annotation about overhead
    if "native_reference" in backends:
        scalapack_data = valid_data[valid_data["backend"] == "native_reference"]
        if not scalapack_data.empty:
            # Find overhead floor (minimum time)
            min_time = scalapack_data["fit_seconds"].min()
            ax.axhline(
                min_time,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"Overhead floor ≈{min_time:.2f}s",
            )
            ax.text(
                0.05,
                0.95,
                f"ScaLAPACK overhead ≈{min_time:.1f}s\n(subprocess + I/O + MPI)",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )

    plt.tight_layout()
    output_file = args.output_dir / f"overhead_crossover.{args.format}"
    plt.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def main() -> None:
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VISUALIZING BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Input files: {len(args.csv_files)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Format: {args.format}")
    print("=" * 70)

    # Load all CSV files
    dataframes = []
    for csv_file in args.csv_files:
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded: {csv_file} ({len(df)} rows)")
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    if not dataframes:
        print("No data loaded. Exiting.")
        return

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    print(f"\nTotal rows: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")

    # Determine which plots to create based on data
    has_scaling = "test_type" in combined_df.columns
    has_breakdown = "fit_seconds" in combined_df.columns and "score_seconds" in combined_df.columns
    has_overhead = "backend" in combined_df.columns

    print("\nGenerating plots:")

    if has_scaling:
        print("  • Scaling theory (log-log plots)")
        try:
            plot_scaling_theory(combined_df, args)
        except Exception as e:
            print(f"    Error: {e}")

    if has_breakdown:
        print("  • Time breakdown (stacked bars)")
        try:
            plot_time_breakdown(combined_df, args)
        except Exception as e:
            print(f"    Error: {e}")

    if has_overhead:
        print("  • Overhead crossover (backend comparison)")
        try:
            plot_overhead_crossover(combined_df, args)
        except Exception as e:
            print(f"    Error: {e}")

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"Figures saved to: {args.output_dir}")
    print("\nThese plots demonstrate:")
    print("  1. Theoretical complexity matches empirical measurements")
    print("  2. Bottlenecks shift with problem size")
    print("  3. Overhead vs compute tradeoffs are real and measurable")
    print("=" * 70)


if __name__ == "__main__":
    main()
