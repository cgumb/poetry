#!/usr/bin/env python3
"""Plot fit benchmark results."""
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python plot_fit_results.py <csv_file>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
fig, ax = plt.subplots(figsize=(8, 5))

# Python
py = df[df['fit_backend'] == 'python'].groupby('m_rated')['fit_seconds'].mean()
ax.plot(py.index, py.values, 'o-', linewidth=2.5, markersize=8, label='Python')

# ScaLAPACK
sc = df[df['fit_backend'] == 'native_reference'].groupby('m_rated')['fit_seconds'].mean()
ax.plot(sc.index, sc.values, 's-', linewidth=2.5, markersize=8, label='ScaLAPACK (4 proc)')

ax.set_xlabel('Number of rated poems (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Fit time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Fit Performance: Python vs ScaLAPACK', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
output = Path(__file__).parent / 'fit_comparison.png'
plt.savefig(output, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output}")
