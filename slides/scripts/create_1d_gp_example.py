"""
Create a simple 1D GP example showing posterior mean and uncertainty.

This generates a figure for the presentation slides showing:
- Training points (dots)
- Posterior mean (solid line)
- Uncertainty band (shaded region)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
    """RBF kernel function."""
    dists = cdist(X1, X2, metric='sqeuclidean')
    return variance * np.exp(-dists / (2 * length_scale**2))

def gp_posterior(X_train, y_train, X_test, length_scale=1.0, variance=1.0, noise=0.1):
    """Compute GP posterior mean and variance."""
    # Kernel matrices
    K_train = rbf_kernel(X_train, X_train, length_scale, variance) + noise**2 * np.eye(len(X_train))
    K_test_train = rbf_kernel(X_test, X_train, length_scale, variance)
    K_test = rbf_kernel(X_test, X_test, length_scale, variance)

    # Posterior mean
    K_inv_y = np.linalg.solve(K_train, y_train)
    mean = K_test_train @ K_inv_y

    # Posterior variance
    v = np.linalg.solve(K_train, K_test_train.T)
    var = np.diag(K_test) - np.sum(v * K_test_train.T, axis=0)
    std = np.sqrt(np.maximum(var, 0))

    return mean, std

def main():
    np.random.seed(42)

    # True function (unknown to GP)
    def true_function(x):
        return np.sin(x) + 0.3 * np.sin(3*x)

    # Training data (sparse, noisy observations)
    X_train = np.array([[0.5], [1.5], [2.5], [4.0], [5.5]])
    y_train = true_function(X_train[:, 0]) + 0.1 * np.random.randn(len(X_train))

    # Test points (dense for visualization)
    X_test = np.linspace(0, 6, 200).reshape(-1, 1)

    # Compute posterior
    mean, std = gp_posterior(X_train, y_train, X_test, length_scale=0.8, variance=1.0, noise=0.1)

    # Plot
    plt.figure(figsize=(10, 5))

    # True function (for comparison)
    plt.plot(X_test, true_function(X_test[:, 0]), 'g--', alpha=0.3, label='True function (unknown)', linewidth=2)

    # Posterior mean
    plt.plot(X_test, mean, 'b-', label='GP posterior mean μ(x)', linewidth=2)

    # Uncertainty band (±2σ)
    plt.fill_between(X_test[:, 0], mean - 2*std, mean + 2*std, alpha=0.3, color='blue', label='Uncertainty ±2σ(x)')

    # Training points
    plt.scatter(X_train[:, 0], y_train, c='red', s=100, zorder=5, label='Rated points (observations)', edgecolors='black', linewidths=1.5)

    plt.xlabel('x (poem embedding space)', fontsize=12)
    plt.ylabel('y (preference rating)', fontsize=12)
    plt.title('1D Gaussian Process: Posterior Mean and Uncertainty', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save
    output_path = '../figures/gp_1d_example.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")

    # Also save PDF for slides
    output_pdf = '../figures/gp_1d_example.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Saved to {output_pdf}")

if __name__ == '__main__':
    main()
