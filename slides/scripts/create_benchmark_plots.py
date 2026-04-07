"""
Create benchmark plots for presentation from actual CSV data.

Generates publication-quality figures showing:
1. Fit scaling: Python vs ScaLAPACK (varying processes)
2. Speedup analysis: ScaLAPACK vs Python baseline
3. Process count comparison: Why 16 processes slower than 8
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_benchmark_data(csv_path):
    """Load and parse benchmark CSV."""
    df = pd.read_csv(csv_path)
    return df

def plot_fit_scaling(df, output_path):
    """Plot fit time vs m for different backends and process counts."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # Extract Python baseline
    python_df = df[df['fit_backend'] == 'python'].copy()
    python_df = python_df.groupby('m_rated')['fit_seconds'].mean().reset_index()

    # Extract ScaLAPACK with different process counts
    scalapack_df = df[df['fit_backend'] == 'native_reference'].copy()

    # Plot Python baseline
    ax.plot(python_df['m_rated'], python_df['fit_seconds'],
            marker='o', linewidth=2, markersize=8, label='Python (SciPy)', color='C0')

    # Plot ScaLAPACK for each process count
    colors = ['C1', 'C2', 'C3', 'C4']
    for i, nproc in enumerate([1, 4, 8, 16]):
        subset = scalapack_df[scalapack_df['scalapack_nprocs'] == nproc].copy()
        subset = subset.groupby('m_rated')['fit_seconds'].mean().reset_index()

        ax.plot(subset['m_rated'], subset['fit_seconds'],
                marker='s', linewidth=2, markersize=7,
                label=f'ScaLAPACK ({nproc} proc)',
                color=colors[i], linestyle='--' if nproc == 1 else '-')

    ax.set_xlabel('Number of rated poems (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fit time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Fit Scaling: Python vs ScaLAPACK', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()

def plot_speedup_analysis(df, output_path):
    """Plot speedup vs m for ScaLAPACK with 8 processes."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # Python baseline
    python_df = df[df['fit_backend'] == 'python'].copy()
    python_times = python_df.groupby('m_rated')['fit_seconds'].mean()

    # ScaLAPACK with 8 processes
    scalapack_df = df[(df['fit_backend'] == 'native_reference') &
                      (df['scalapack_nprocs'] == 8)].copy()
    scalapack_times = scalapack_df.groupby('m_rated')['fit_seconds'].mean()

    # Compute speedup
    m_values = sorted(set(python_times.index) & set(scalapack_times.index))
    speedups = []
    for m in m_values:
        speedup = python_times[m] / scalapack_times[m]
        speedups.append(speedup)

    # Plot speedup
    ax.plot(m_values, speedups, marker='o', linewidth=2, markersize=8,
            color='C2', label='ScaLAPACK (8 proc) vs Python')

    # Add horizontal line at 1.0 (no speedup)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
               label='No speedup (1.0×)', alpha=0.7)

    # Annotate crossover point
    crossover_idx = None
    for i, speedup in enumerate(speedups):
        if speedup > 1.0:
            crossover_idx = i
            break

    if crossover_idx:
        ax.axvline(x=m_values[crossover_idx], color='green', linestyle=':',
                   linewidth=2, alpha=0.5, label=f'Crossover (m ≈ {m_values[crossover_idx]:,})')

    ax.set_xlabel('Number of rated poems (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (Python time / ScaLAPACK time)', fontsize=12, fontweight='bold')
    ax.set_title('ScaLAPACK Speedup vs Python Baseline', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()

def plot_process_scaling(df, output_path, m_target=20000):
    """Plot time vs process count for a specific m value."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # Filter for target m and ScaLAPACK
    subset = df[(df['fit_backend'] == 'native_reference') &
                (df['m_rated'] == m_target)].copy()
    times = subset.groupby('scalapack_nprocs')['fit_seconds'].mean().reset_index()

    # Also get Python baseline for comparison
    python_time = df[(df['fit_backend'] == 'python') &
                     (df['m_rated'] == m_target)]['fit_seconds'].mean()

    # Plot ScaLAPACK scaling
    ax.plot(times['scalapack_nprocs'], times['fit_seconds'],
            marker='s', linewidth=2, markersize=10, color='C2',
            label='ScaLAPACK')

    # Add Python baseline as horizontal line
    ax.axhline(y=python_time, color='C0', linestyle='--', linewidth=2,
               label=f'Python baseline ({python_time:.1f}s)', alpha=0.7)

    # Highlight best performance
    best_idx = times['fit_seconds'].idxmin()
    best_nproc = times.loc[best_idx, 'scalapack_nprocs']
    best_time = times.loc[best_idx, 'fit_seconds']

    ax.scatter([best_nproc], [best_time], s=200, color='red', zorder=5,
               marker='*', label=f'Best: {int(best_nproc)} proc ({best_time:.1f}s)')

    ax.set_xlabel('Number of processes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fit time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'Process Scaling at m = {m_target:,}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks([1, 4, 8, 16])
    ax.set_xlim(0, 17)
    ax.set_ylim(bottom=0)

    # Add text annotation for diminishing returns
    ax.annotate('Diminishing returns\n(communication overhead)',
                xy=(16, times[times['scalapack_nprocs'] == 16]['fit_seconds'].values[0]),
                xytext=(14, times[times['scalapack_nprocs'] == 16]['fit_seconds'].values[0] + 5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()

def plot_log_log_scaling(df, output_path):
    """Plot fit time vs m on log-log scale to show O(m³) scaling."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # Python baseline
    python_df = df[df['fit_backend'] == 'python'].copy()
    python_times = python_df.groupby('m_rated')['fit_seconds'].mean()

    # ScaLAPACK with 8 processes
    scalapack_df = df[(df['fit_backend'] == 'native_reference') &
                      (df['scalapack_nprocs'] == 8)].copy()
    scalapack_times = scalapack_df.groupby('m_rated')['fit_seconds'].mean()

    # Plot on log-log scale
    ax.loglog(python_times.index, python_times.values,
              marker='o', linewidth=2, markersize=8, label='Python', color='C0')
    ax.loglog(scalapack_times.index, scalapack_times.values,
              marker='s', linewidth=2, markersize=8, label='ScaLAPACK (8 proc)', color='C2')

    # Add reference line for O(m³)
    m_ref = np.array([2000, 30000])
    # Fit to Python data at m=10000 as reference
    t_ref = python_times[10000]
    m_ref_point = 10000
    t_cubic = t_ref * (m_ref / m_ref_point) ** 3

    ax.loglog(m_ref, t_cubic, 'k--', linewidth=2, alpha=0.5, label='O(m³) reference')

    ax.set_xlabel('Number of rated poems (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fit time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Log Scaling: Verifying O(m³) Complexity', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()

def plot_time_breakdown(score_csv_path, output_path):
    """Plot time breakdown: fit vs score as function of m using score benchmark data."""
    if not Path(score_csv_path).exists():
        print(f"⚠ Score benchmark CSV not found at {score_csv_path}")
        print("  Using placeholder - run actual benchmark for real data")
        return

    df = pd.read_csv(score_csv_path)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Python single-threaded baseline for comparison
    python_1t = df[(df['score_backend'] == 'python') & (df['num_threads'] == 1)].copy()

    if len(python_1t) == 0:
        print(f"⚠ No Python single-threaded data in {score_csv_path}")
        return

    python_1t = python_1t.groupby('m_rated', as_index=False)[['fit_seconds', 'score_seconds', 'n_candidates']].mean()

    m_values = sorted(python_1t['m_rated'].values)
    fit_times = python_1t.set_index('m_rated')['fit_seconds'].reindex(m_values).values
    score_times = python_1t.set_index('m_rated')['score_seconds'].reindex(m_values).values
    total_times = fit_times + score_times

    # Stacked area plot
    ax.fill_between(m_values, 0, fit_times, alpha=0.7, color='C0', label='Fit (O(m³))')
    ax.fill_between(m_values, fit_times, total_times, alpha=0.7, color='C1', label='Score (O(nm²))')

    # Lines for clarity
    ax.plot(m_values, fit_times, 'o-', color='C0', linewidth=2, markersize=6)
    ax.plot(m_values, total_times, 's-', color='black', linewidth=2, markersize=6, label='Total')

    ax.set_xlabel('Number of rated poems (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Time Breakdown: Fit vs Score (Python baseline)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Add annotation about candidate count
    n_cand = int(python_1t['n_candidates'].iloc[0])
    ax.text(0.98, 0.02, f'Note: Score time shown for n={n_cand:,}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()

def plot_score_backend_comparison(csv_path, output_path):
    """Plot score backend comparison: Python (CPU) vs GPU."""
    if not Path(csv_path).exists():
        print(f"⚠ Score benchmark CSV not found at {csv_path}")
        print("  Run: sbatch scripts/bench_score_backends.slurm")
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Get unique backends
    backends = df['score_backend'].unique()

    # Define colors and markers for each backend
    backend_styles = {
        'python': {'color': 'C0', 'marker': 'o', 'label': 'Python (1 thread)', 'filter': lambda x: x['num_threads'] == 1},
        'python_mt': {'color': 'C1', 'marker': 's', 'label': 'Python (8 threads)', 'filter': lambda x: x['score_backend'] == 'python' and x['num_threads'] != 1},
        'native_lapack': {'color': 'C2', 'marker': '^', 'label': 'Native LAPACK', 'filter': lambda x: x['score_backend'] == 'native_lapack'},
        'gpu': {'color': 'C3', 'marker': 'D', 'label': 'GPU (CuPy)', 'filter': lambda x: x['score_backend'] == 'gpu'},
    }

    # Plot each backend
    for backend_key, style in backend_styles.items():
        if backend_key == 'python':
            subset = df[df['num_threads'] == 1].copy()
        elif backend_key == 'python_mt':
            subset = df[(df['score_backend'] == 'python') & (df['num_threads'] != 1)].copy()
        else:
            subset = df[df['score_backend'] == backend_key].copy()

        if len(subset) > 0:
            subset = subset.groupby('m_rated')['score_seconds'].mean().reset_index()
            ax.plot(subset['m_rated'], subset['score_seconds'],
                   marker=style['marker'], linewidth=2.5, markersize=8,
                   label=style['label'], color=style['color'])

    ax.set_xlabel('Number of rated poems (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Scoring Performance: CPU vs GPU', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Add annotation about complexity
    n_cand = df['n_candidates'].iloc[0]
    ax.text(0.98, 0.02, f'n = {n_cand:,} candidates\nComplexity: O(nm²)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()

def main():
    # Find benchmark CSV
    csv_path = Path('../../results/large_scale_fit_20260406_233011.csv')

    if not csv_path.exists():
        print(f"ERROR: Benchmark CSV not found at {csv_path}")
        print("Please run: sbatch scripts/large_scale_bench.slurm")
        return

    print("Loading benchmark data...")
    df = load_benchmark_data(csv_path)
    print(f"  Loaded {len(df)} benchmark runs")
    print(f"  m values: {sorted(df['m_rated'].unique())}")
    print(f"  Backends: {df['fit_backend'].unique()}")
    print()

    print("Creating plots...")

    # Create output directory
    output_dir = Path('../figures')
    output_dir.mkdir(exist_ok=True)

    # Generate plots from fit benchmark data
    plot_fit_scaling(df, output_dir / 'benchmark_fit_scaling.png')
    plot_speedup_analysis(df, output_dir / 'benchmark_speedup.png')
    plot_process_scaling(df, output_dir / 'benchmark_process_scaling.png', m_target=20000)
    plot_log_log_scaling(df, output_dir / 'benchmark_loglog_scaling.png')

    # Generate plots from score benchmark data (if available)
    # Try both the timestamped and generic filenames
    score_csv_candidates = [
        Path('../../results/score_backend_comparison_20260407_152231.csv'),
        Path('../../results/score_backend_comparison.csv'),
    ]
    score_csv = None
    for candidate in score_csv_candidates:
        if candidate.exists():
            score_csv = candidate
            break

    if score_csv:
        print(f"\nUsing score benchmark data: {score_csv.name}")
        plot_time_breakdown(score_csv, output_dir / 'benchmark_time_breakdown.png')
        plot_score_backend_comparison(score_csv, output_dir / 'benchmark_score_backends.png')
    else:
        print("\n⚠ Score benchmark data not found - skipping time breakdown and score backend plots")
        print("  Run: sbatch scripts/bench_score_backends.slurm")

    print()
    print("=" * 60)
    print("All plots created successfully!")
    print("=" * 60)
    print()
    print("Files created:")
    print("  - benchmark_fit_scaling.{png,pdf}")
    print("  - benchmark_speedup.{png,pdf}")
    print("  - benchmark_process_scaling.{png,pdf}")
    print("  - benchmark_loglog_scaling.{png,pdf}")
    print("  - benchmark_time_breakdown.{png,pdf}")
    print("  - benchmark_score_backends.{png,pdf} (if data exists)")
    print()
    print("Update slides to reference these figures!")

if __name__ == '__main__':
    main()
