#!/usr/bin/env python3
"""Plot score benchmark results."""
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python plot_score_results.py <csv_file>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
fig, ax = plt.subplots(figsize=(8, 5))

# CPU (1 thread)
cpu = df[df['num_threads'] == 1].groupby('m_rated')['score_seconds'].mean()
ax.plot(cpu.index, cpu.values, 'o-', linewidth=2.5, markersize=8, label='Python (1 thread)')

# GPU
gpu = df[df['score_backend'] == 'gpu'].groupby('m_rated')['score_seconds'].mean()
ax.plot(gpu.index, gpu.values, 'D-', linewidth=2.5, markersize=8, label='GPU (CuPy)')

ax.set_xlabel('Number of rated poems (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Score time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Scoring Performance: CPU vs GPU', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
output = Path(__file__).parent / 'score_comparison.png'
plt.savefig(output, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output}")
