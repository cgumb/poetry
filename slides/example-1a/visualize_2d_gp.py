#!/usr/bin/env python3
"""
Visualize a simple 2D Gaussian Process fit.

Creates a toy example showing:
- A few training points in 2D
- GP posterior mean (color)
- Uncertainty (shown via contours)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist

def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
    """RBF (squared exponential) kernel."""
    sq_dists = cdist(X1, X2, metric='sqeuclidean')
    return variance * np.exp(-sq_dists / (2 * length_scale**2))

def gp_predict(X_train, y_train, X_test, length_scale=1.0, variance=1.0, noise=0.01):
    """Simple GP prediction (mean only)."""
    K = rbf_kernel(X_train, X_train, length_scale, variance) + noise * np.eye(len(X_train))
    K_star = rbf_kernel(X_train, X_test, length_scale, variance)

    # Solve K * alpha = y
    alpha = np.linalg.solve(K, y_train)

    # Posterior mean
    mean = K_star.T @ alpha

    return mean

def main():
    print("Creating 2D GP visualization...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create training data (sparse points)
    n_train = 8
    X_train = np.random.uniform(-5, 5, size=(n_train, 2))

    # Target function: simple combination of sinusoids
    y_train = np.sin(X_train[:, 0]) + 0.5 * np.cos(X_train[:, 1])
    y_train += 0.1 * np.random.randn(n_train)  # Add noise

    # Create test grid
    x1 = np.linspace(-6, 6, 100)
    x2 = np.linspace(-6, 6, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_test = np.column_stack([X1.ravel(), X2.ravel()])

    # GP prediction
    mean = gp_predict(X_train, y_train, X_test, length_scale=2.0, variance=1.0)
    mean_grid = mean.reshape(X1.shape)

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot posterior mean as heatmap
    im = ax.contourf(X1, X2, mean_grid, levels=20, cmap='RdBu_r')

    # Add contour lines
    ax.contour(X1, X2, mean_grid, levels=10, colors='black', alpha=0.2, linewidths=0.5)

    # Plot training points
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                        s=200, cmap='RdBu_r', edgecolors='black',
                        linewidths=2, zorder=10, vmin=mean_grid.min(), vmax=mean_grid.max())

    ax.set_xlabel('x₁', fontsize=14, fontweight='bold')
    ax.set_ylabel('x₂', fontsize=14, fontweight='bold')
    ax.set_title('2D Gaussian Process Posterior Mean', fontsize=16, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Posterior Mean', fontsize=12, fontweight='bold')

    # Add grid
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent / 'gp_2d_example.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    print(f"✓ Saved: {output_path}")
    print()
    print("The plot shows:")
    print("  - Color: GP posterior mean (predicted function value)")
    print("  - Large dots: Training observations")
    print("  - Contours: Iso-lines of posterior mean")
    print()
    print("Notice how the GP interpolates between observations!")

if __name__ == '__main__':
    main()
