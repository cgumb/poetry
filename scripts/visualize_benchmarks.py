#!/usr/bin/env python3
"""
Visualize Performance Benchmark Results

This script creates publication-quality plots from benchmark CSV files to analyze:
- Python vs ScaLAPACK performance
- Scaling with problem size
- Scaling with process count
- Impact of block size on performance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize performance benchmark results")
    parser.add_argument("csv_file", type=Path, help="CSV file with benchmark results")
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"), help="Directory for output plots")
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="png", help="Output format")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for rasterized output")
    parser.add_argument("--style", default="seaborn-v0_8-darkgrid", help="Matplotlib style")
    return parser.parse_args()


def load_and_prepare_data(csv_file: Path) -> pd.DataFrame:
    """Load CSV and prepare for plotting."""
    df = pd.read_csv(csv_file)

    # Print column names for debugging
    print(f"CSV columns: {list(df.columns)}")
    print(f"Number of rows: {len(df)}")

    # Handle different possible column names and missing values
    # Check if we have fit_backend or backend column
    if "fit_backend" not in df.columns:
        if "backend" in df.columns:
            # Try to infer from backend and nprocs
            if "nprocs" in df.columns:
                def infer_backend(row):
                    nprocs = row.get("nprocs")
                    # Handle empty strings, NaN, and actual values
                    if pd.isna(nprocs) or nprocs == "" or nprocs is None:
                        return "python"
                    try:
                        nprocs_int = int(nprocs)
                        return "native_reference" if nprocs_int > 1 else "python"
                    except (ValueError, TypeError):
                        return "python"

                df["fit_backend"] = df.apply(infer_backend, axis=1)
            else:
                # Can't determine, assume Python
                df["fit_backend"] = "python"
        else:
            # No backend info at all
            df["fit_backend"] = "unknown"

    # Fill missing native_backend with empty string
    if "native_backend" not in df.columns:
        df["native_backend"] = ""
    else:
        df["native_backend"] = df["native_backend"].fillna("")

    # Fill missing nprocs/block_size with empty string
    # Also handle the case where the CSV has "nprocs" instead of "scalapack_nprocs"
    if "scalapack_nprocs" not in df.columns:
        if "nprocs" in df.columns:
            df["scalapack_nprocs"] = df["nprocs"]
        else:
            df["scalapack_nprocs"] = ""
    df["scalapack_nprocs"] = df["scalapack_nprocs"].fillna("")

    if "scalapack_block_size" not in df.columns:
        if "block_size" in df.columns:
            df["scalapack_block_size"] = df["block_size"]
        else:
            df["scalapack_block_size"] = ""
    df["scalapack_block_size"] = df["scalapack_block_size"].fillna("")

    # Create a combined backend label
    def make_label(row):
        if row.get("fit_backend") == "python":
            return "Python"
        else:
            native = row.get("native_backend", "")
            nprocs = row.get("scalapack_nprocs", "")
            bs = row.get("scalapack_block_size", "")
            # Filter out empty strings
            if native and str(nprocs).strip() and str(bs).strip():
                return f"ScaLAPACK-{native} (n={nprocs}, bs={bs})"
            else:
                return "ScaLAPACK"

    df["backend_label"] = df.apply(make_label, axis=1)

    # Simplified label for cleaner plots
    df["backend_simple"] = df.apply(
        lambda row: "Python" if row.get("fit_backend") == "python" else "ScaLAPACK",
        axis=1,
    )

    return df


def plot_python_vs_scalapack_by_size(df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> None:
    """Plot: Python vs ScaLAPACK performance across problem sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Group by m_rated and backend
    python_df = df[df["fit_backend"] == "python"]
    scalapack_df = df[df["fit_backend"] == "native_reference"]

    if python_df.empty:
        print("Warning: No Python data found")
        python_data = pd.DataFrame()
    else:
        python_data = python_df.groupby("m_rated").agg({
            "fit_seconds": "mean",
            "total_seconds": "mean",
        }).reset_index()

    if scalapack_df.empty:
        print("Warning: No ScaLAPACK data found")
        scalapack_best = pd.DataFrame()
        scalapack_avg = pd.DataFrame()
    else:
        # Best configuration (min time) for each problem size
        scalapack_best = scalapack_df.groupby("m_rated").agg({
            "fit_seconds": "min",
            "total_seconds": "min",
        }).reset_index()
        # Average across all configurations (for comparison)
        scalapack_avg = scalapack_df.groupby("m_rated").agg({
            "fit_seconds": "mean",
            "total_seconds": "mean",
        }).reset_index()

    # Plot 1: Fit time
    if not python_data.empty:
        ax1.plot(python_data["m_rated"], python_data["fit_seconds"], "o-", label="Python", linewidth=2, markersize=8)
    if not scalapack_best.empty:
        ax1.plot(scalapack_best["m_rated"], scalapack_best["fit_seconds"], "s-", label="ScaLAPACK (best)", linewidth=2, markersize=8, color="C2")
    if not scalapack_avg.empty:
        ax1.plot(scalapack_avg["m_rated"], scalapack_avg["fit_seconds"], "s--", label="ScaLAPACK (avg)", linewidth=1, markersize=4, alpha=0.5, color="C2")
    ax1.set_xlabel("Problem Size (m_rated)", fontsize=12)
    ax1.set_ylabel("Fit Time (seconds)", fontsize=12)
    ax1.set_title("GP Fit Performance vs Problem Size", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    # Add more x-axis ticks at actual data points
    if not python_data.empty or not scalapack_best.empty:
        all_m = set()
        if not python_data.empty:
            all_m.update(python_data["m_rated"].values)
        if not scalapack_best.empty:
            all_m.update(scalapack_best["m_rated"].values)
        ax1.set_xticks(sorted(all_m))
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Plot 2: Total time
    if not python_data.empty:
        ax2.plot(python_data["m_rated"], python_data["total_seconds"], "o-", label="Python", linewidth=2, markersize=8)
    if not scalapack_best.empty:
        ax2.plot(scalapack_best["m_rated"], scalapack_best["total_seconds"], "s-", label="ScaLAPACK (best)", linewidth=2, markersize=8, color="C2")
    if not scalapack_avg.empty:
        ax2.plot(scalapack_avg["m_rated"], scalapack_avg["total_seconds"], "s--", label="ScaLAPACK (avg)", linewidth=1, markersize=4, alpha=0.5, color="C2")
    ax2.set_xlabel("Problem Size (m_rated)", fontsize=12)
    ax2.set_ylabel("Total Time (seconds)", fontsize=12)
    ax2.set_title("Total Runtime vs Problem Size", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    # Add more x-axis ticks at actual data points
    if not python_data.empty or not scalapack_best.empty:
        all_m = set()
        if not python_data.empty:
            all_m.update(python_data["m_rated"].values)
        if not scalapack_best.empty:
            all_m.update(scalapack_best["m_rated"].values)
        ax2.set_xticks(sorted(all_m))
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()
    _save_figure(fig, output_dir / "performance_vs_size", fmt, dpi)
    plt.close()


def plot_scaling_analysis(df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> None:
    """Plot: Scaling with process count for ScaLAPACK."""
    scalapack_df = df[df["fit_backend"] == "native_reference"]

    if scalapack_df.empty:
        print("No ScaLAPACK data available for scaling analysis")
        return

    # Get unique m_rated values
    m_rated_values = sorted(scalapack_df["m_rated"].unique())

    fig, axes = plt.subplots(1, len(m_rated_values), figsize=(6 * len(m_rated_values), 5), squeeze=False)
    axes = axes.flatten()

    for idx, m_rated in enumerate(m_rated_values):
        ax = axes[idx]
        subset = scalapack_df[scalapack_df["m_rated"] == m_rated]

        # Group by nprocs and get best time for each
        scaling_data = subset.groupby("scalapack_nprocs")["fit_seconds"].min().reset_index()
        scaling_data = scaling_data.sort_values("scalapack_nprocs")

        nprocs = scaling_data["scalapack_nprocs"].values
        times = scaling_data["fit_seconds"].values

        # Compute speedup relative to single process
        if 1 in nprocs:
            baseline_time = times[np.where(nprocs == 1)[0][0]]
            speedup = baseline_time / times
            ideal_speedup = nprocs.astype(float)

            ax.plot(nprocs, speedup, "o-", label="Actual Speedup", linewidth=2, markersize=8)
            ax.plot(nprocs, ideal_speedup, "--", label="Ideal Speedup", linewidth=2, alpha=0.7)
            ax.set_ylabel("Speedup", fontsize=12)
            ax.set_title(f"Scaling: m_rated={m_rated}", fontsize=14, fontweight="bold")
        else:
            ax.plot(nprocs, times, "o-", label="Fit Time", linewidth=2, markersize=8)
            ax.set_ylabel("Fit Time (seconds)", fontsize=12)
            ax.set_title(f"Performance: m_rated={m_rated}", fontsize=14, fontweight="bold")

        ax.set_xlabel("Number of Processes", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)

    plt.tight_layout()
    _save_figure(fig, output_dir / "scaling_analysis", fmt, dpi)
    plt.close()


def plot_block_size_impact(df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> None:
    """Plot: Impact of block size on ScaLAPACK performance."""
    scalapack_df = df[df["fit_backend"] == "native_reference"]

    if scalapack_df.empty or "scalapack_block_size" not in scalapack_df.columns:
        print("No ScaLAPACK block size data available")
        return

    # Filter out empty block sizes
    scalapack_df = scalapack_df[scalapack_df["scalapack_block_size"].notna()]

    if scalapack_df.empty:
        return

    # Get unique m_rated and nprocs combinations
    configs = scalapack_df[["m_rated", "scalapack_nprocs"]].drop_duplicates().values

    num_configs = len(configs)
    if num_configs == 0:
        return

    cols = min(3, num_configs)
    rows = (num_configs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()

    for idx, (m_rated, nprocs) in enumerate(configs):
        if idx >= len(axes):
            break

        ax = axes[idx]
        subset = scalapack_df[
            (scalapack_df["m_rated"] == m_rated) &
            (scalapack_df["scalapack_nprocs"] == nprocs)
        ]

        if subset.empty:
            continue

        # Group by block size
        block_data = subset.groupby("scalapack_block_size")["fit_seconds"].min().reset_index()
        block_data = block_data.sort_values("scalapack_block_size")

        ax.plot(block_data["scalapack_block_size"], block_data["fit_seconds"], "o-", linewidth=2, markersize=8)
        ax.set_xlabel("Block Size", fontsize=12)
        ax.set_ylabel("Fit Time (seconds)", fontsize=12)
        ax.set_title(f"m={int(m_rated)}, nprocs={int(nprocs)}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(configs), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    _save_figure(fig, output_dir / "block_size_impact", fmt, dpi)
    plt.close()


def plot_detailed_breakdown(df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> None:
    """Plot: Detailed time breakdown (fit, score, select) with side-by-side comparison."""
    # Check if we have the required columns for detailed breakdown
    if "fit_seconds" not in df.columns:
        print(f"Skipping detailed breakdown - missing fit_seconds column")
        return

    # Check if scoring was actually performed (not score_backend=none)
    has_scoring = "score_seconds" in df.columns and df["score_seconds"].max() > 0.001
    if not has_scoring:
        print(f"Skipping detailed breakdown - no scoring data (likely score_backend=none)")
        return

    # Optional columns (not all CSVs have these)
    has_select = "select_seconds" in df.columns
    has_optimize = "optimize_seconds" in df.columns

    # Build aggregation dict based on available columns
    agg_dict_mean = {"fit_seconds": "mean", "score_seconds": "mean"}
    agg_dict_min = {"fit_seconds": "min", "score_seconds": "min"}
    if has_select:
        agg_dict_mean["select_seconds"] = "mean"
        agg_dict_min["select_seconds"] = "min"
    if has_optimize:
        agg_dict_mean["optimize_seconds"] = "mean"
        agg_dict_min["optimize_seconds"] = "min"

    # Get data for both backends
    python_df = df[df["fit_backend"] == "python"]
    scalapack_df = df[df["fit_backend"] == "native_reference"]

    if python_df.empty:
        print("No Python data for breakdown plot")
        return

    python_data = python_df.groupby("m_rated").agg(agg_dict_mean).reset_index()

    has_scalapack = False
    if not scalapack_df.empty:
        scalapack_data = scalapack_df.groupby("m_rated").agg(agg_dict_min).reset_index()
        has_scalapack = True

    # Get all problem sizes
    m_rated_values = sorted(python_data["m_rated"].unique())
    n_sizes = len(m_rated_values)

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(max(10, n_sizes * 1.5), 6))

    # Bar width and positions
    bar_width = 0.35
    x_pos = np.arange(n_sizes)

    # Define colors for each component (using matplotlib's default color cycle)
    colors = {
        'optimize': 'C3',  # Red
        'fit': 'C0',       # Blue
        'score': 'C2',     # Green
        'select': 'C1'     # Orange
    }

    # Plot Python bars (left side of each group)
    python_bottom = np.zeros(n_sizes)

    # Extract Python data aligned with m_rated_values
    python_dict = {row['m_rated']: row for _, row in python_data.iterrows()}

    # Track which components are actually plotted for legend
    has_optimize_data = False

    # Optimize time
    if has_optimize:
        opt_times = np.array([python_dict.get(m, {}).get('optimize_seconds', 0) for m in m_rated_values])
        if np.any(opt_times > 0.01):
            ax.bar(x_pos - bar_width/2, opt_times, bar_width, bottom=python_bottom,
                   label='Optimize HP (Python)', color=colors['optimize'], edgecolor='black', linewidth=0.5)
            python_bottom += opt_times
            has_optimize_data = True

    # Fit time
    fit_times = np.array([python_dict.get(m, {}).get('fit_seconds', 0) for m in m_rated_values])
    ax.bar(x_pos - bar_width/2, fit_times, bar_width, bottom=python_bottom,
           label='Fit (Python)', color=colors['fit'], edgecolor='black', linewidth=0.5)
    python_bottom += fit_times

    # Score time
    score_times = np.array([python_dict.get(m, {}).get('score_seconds', 0) for m in m_rated_values])
    ax.bar(x_pos - bar_width/2, score_times, bar_width, bottom=python_bottom,
           label='Score (Python)', color=colors['score'], edgecolor='black', linewidth=0.5)
    python_bottom += score_times

    # Select time
    if has_select:
        select_times = np.array([python_dict.get(m, {}).get('select_seconds', 0) for m in m_rated_values])
        if np.any(select_times > 0.001):
            ax.bar(x_pos - bar_width/2, select_times, bar_width, bottom=python_bottom,
                   label='Select (Python)', color=colors['select'], edgecolor='black', linewidth=0.5)

    # Plot ScaLAPACK bars (right side of each group) if available
    if has_scalapack:
        scalapack_bottom = np.zeros(n_sizes)
        scalapack_dict = {row['m_rated']: row for _, row in scalapack_data.iterrows()}

        # Optimize time
        if has_optimize:
            opt_times = np.array([scalapack_dict.get(m, {}).get('optimize_seconds', 0) for m in m_rated_values])
            if np.any(opt_times > 0.01):
                ax.bar(x_pos + bar_width/2, opt_times, bar_width, bottom=scalapack_bottom,
                       label='Optimize HP (ScaLAPACK)', color=colors['optimize'],
                       edgecolor='black', linewidth=0.5, hatch='///')
                scalapack_bottom += opt_times

        # Fit time
        fit_times = np.array([scalapack_dict.get(m, {}).get('fit_seconds', 0) for m in m_rated_values])
        ax.bar(x_pos + bar_width/2, fit_times, bar_width, bottom=scalapack_bottom,
               label='Fit (ScaLAPACK)', color=colors['fit'],
               edgecolor='black', linewidth=0.5, hatch='///')
        scalapack_bottom += fit_times

        # Score time
        score_times = np.array([scalapack_dict.get(m, {}).get('score_seconds', 0) for m in m_rated_values])
        ax.bar(x_pos + bar_width/2, score_times, bar_width, bottom=scalapack_bottom,
               label='Score (ScaLAPACK)', color=colors['score'],
               edgecolor='black', linewidth=0.5, hatch='///')
        scalapack_bottom += score_times

        # Select time
        if has_select:
            select_times = np.array([scalapack_dict.get(m, {}).get('select_seconds', 0) for m in m_rated_values])
            if np.any(select_times > 0.001):
                ax.bar(x_pos + bar_width/2, select_times, bar_width, bottom=scalapack_bottom,
                       label='Select (ScaLAPACK)', color=colors['select'],
                       edgecolor='black', linewidth=0.5, hatch='///')

    # Formatting
    ax.set_xlabel("Problem Size (m_rated)", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)

    title = "Time Breakdown: Python vs ScaLAPACK (Best)" if has_scalapack else "Time Breakdown: Python"
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{int(m):,}' for m in m_rated_values])

    # Add legend with better placement (solid = Python, striped = ScaLAPACK)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _save_figure(fig, output_dir / "time_breakdown", fmt, dpi)
    plt.close()


def _save_figure(fig: plt.Figure, path: Path, fmt: str, dpi: int) -> None:
    """Save figure in requested format(s)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt in ["png", "both"]:
        fig.savefig(f"{path}.png", dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}.png")

    if fmt in ["pdf", "both"]:
        fig.savefig(f"{path}.pdf", bbox_inches="tight")
        print(f"Saved: {path}.pdf")


def main() -> None:
    args = parse_args()

    if not args.csv_file.exists():
        print(f"Error: CSV file not found: {args.csv_file}")
        return

    # Set matplotlib style
    try:
        plt.style.use(args.style)
    except OSError:
        print(f"Warning: Style '{args.style}' not found, using default")

    # Load data
    print(f"Loading data from: {args.csv_file}")
    df = load_and_prepare_data(args.csv_file)
    print(f"Loaded {len(df)} benchmark results")

    # Create plots
    print("\nGenerating plots...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_python_vs_scalapack_by_size(df, args.output_dir, args.format, args.dpi)
    plot_scaling_analysis(df, args.output_dir, args.format, args.dpi)
    plot_block_size_impact(df, args.output_dir, args.format, args.dpi)
    plot_detailed_breakdown(df, args.output_dir, args.format, args.dpi)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
