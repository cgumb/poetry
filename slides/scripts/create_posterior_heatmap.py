"""
Create a posterior mean heatmap example for presentation.

Shows GP posterior mean on 2D UMAP projection of poem embeddings.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_example_posterior_heatmap(output_path):
    """Create a synthetic posterior heatmap for demonstration."""
    np.random.seed(42)

    # Create grid
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    # Create synthetic posterior mean (multiple Gaussian bumps)
    Z = np.zeros_like(X)

    # Rated points (centers of high preference)
    rated_points = [
        (-5, 3, 0.8),   # (x, y, rating)
        (2, 6, 0.9),
        (4, -2, 0.3),
        (-3, -4, 0.5),
        (6, 1, 0.7),
    ]

    # Create smooth posterior surface
    for px, py, rating in rated_points:
        distance = np.sqrt((X - px)**2 + (Y - py)**2)
        Z += rating * np.exp(-distance**2 / 10.0)

    # Normalize to [0, 1]
    Z = (Z - Z.min()) / (Z.max() - Z.min())

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 7))

    # Create custom colormap (low to high preference)
    colors = ['#2c3e50', '#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('preference', colors, N=n_bins)

    # Plot heatmap
    im = ax.contourf(X, Y, Z, levels=20, cmap=cmap, alpha=0.9)

    # Overlay hexbin-style visualization
    hexbin = ax.hexbin(X.flatten(), Y.flatten(), C=Z.flatten(),
                       gridsize=30, cmap=cmap, alpha=0.6, edgecolors='none')

    # Plot rated points
    rated_x = [p[0] for p in rated_points]
    rated_y = [p[1] for p in rated_points]
    rated_vals = [p[2] for p in rated_points]

    scatter = ax.scatter(rated_x, rated_y, c=rated_vals,
                        s=200, cmap=cmap, edgecolors='white', linewidths=2,
                        marker='o', zorder=10, vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Posterior mean μ(x) (predicted rating)', fontsize=11, fontweight='bold')

    # Labels
    ax.set_xlabel('UMAP dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('GP Posterior Mean on Poem Embedding Space', fontsize=14, fontweight='bold')

    # Add legend
    legend_elements = [
        plt.scatter([], [], s=200, c='white', edgecolors='white', linewidths=2,
                   label='Rated poems')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()

def main():
    from pathlib import Path

    output_dir = Path('../figures')
    output_dir.mkdir(exist_ok=True)

    print("Creating posterior heatmap example...")
    create_example_posterior_heatmap(output_dir / 'posterior_heatmap.png')
    print("\n✓ Posterior heatmap created!")

if __name__ == '__main__':
    main()
