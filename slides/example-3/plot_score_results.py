#!/usr/bin/env python3
"""
Plot score benchmark results.

Usage:
    python plot_score_results.py <csv_file>
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_score_results.py <csv_file>")
        sys.exit(1)

    csv_file = Path(sys.argv[1])
    if not csv_file.exists():
        print(f"Error: {csv_file} not found")
        sys.exit(1)

    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot each backend
    for backend_label, filter_func in [
        ('Python (1 thread)', lambda d: (d['score_backend'] == 'python') & (d['num_threads'] == 1)),
        ('Python (8 threads)', lambda d: (d['score_backend'] == 'python') & (d['num_threads'] == 8)),
        ('GPU (CuPy)', lambda d: d['score_backend'] == 'gpu'),
    ]:
        subset = df[filter_func(df)].copy()
        if len(subset) > 0:
            subset = subset.groupby('m_rated')['score_seconds'].mean().reset_index()
            ax.plot(subset['m_rated'], subset['score_seconds'],
                   marker='o', linewidth=2.5, markersize=8, label=backend_label)

    ax.set_xlabel('Number of rated poems (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Scoring Performance: CPU vs GPU', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Annotation
    n_cand = df['n_candidates'].iloc[0]
    ax.text(0.98, 0.02, f'n = {n_cand:,} candidates\nComplexity: O(nm²)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = Path(__file__).parent / 'score_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    print(f"✓ Saved: {output_path}")

if __name__ == '__main__':
    main()
