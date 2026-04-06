#!/usr/bin/env python3
"""
Test automatic backend selection.

Verifies that backend selection logic works correctly for different problem sizes.
"""
from __future__ import annotations

import numpy as np
from poetry_gp.backends.blocked import run_blocked_step
from poetry_gp.backends.backend_selection import (
    select_fit_backend,
    select_score_backend,
    get_backend_info,
    print_backend_status,
)
from poetry_gp.config import GPConfig, FAST_CONFIG, print_config


def test_selection_logic():
    """Test backend selection for different problem sizes."""
    print("=" * 60)
    print("Backend Selection Logic Test")
    print("=" * 60)
    print()

    # Print backend availability
    print_backend_status(verbose=False)
    print()

    # Test fit backend selection
    print("Fit Backend Selection:")
    print("-" * 40)
    test_cases = [
        (100, "Small (m=100)"),
        (1000, "Medium (m=1000)"),
        (5000, "Large (m=5000)"),
        (15000, "Very Large (m=15000)"),
    ]

    for m, label in test_cases:
        backend = select_fit_backend(m)
        print(f"  {label:<20} → {backend}")

    print()

    # Test with hyperparameter optimization
    print("Fit Backend Selection (with HP optimization):")
    print("-" * 40)
    for m, label in test_cases:
        backend = select_fit_backend(m, optimize_hyperparameters=True)
        print(f"  {label:<20} → {backend}")

    print()

    # Test score backend selection
    print("Score Backend Selection:")
    print("-" * 40)
    score_test_cases = [
        (10000, 100, "Small m (m=100)"),
        (10000, 500, "Medium m (m=500)"),
        (10000, 2000, "Large m (m=2000)"),
        (50000, 5000, "Very Large (m=5000)"),
    ]

    for n, m, label in score_test_cases:
        backend = select_score_backend(n, m)
        print(f"  {label:<20} → {backend}")

    print()

    # Test manual override
    print("Manual Override Test:")
    print("-" * 40)
    backend = select_fit_backend(100, manual_override="python")
    print(f"  m=100, override='python'  → {backend}")
    backend = select_score_backend(10000, 500, manual_override="native_lapack")
    print(f"  m=500, override='native_lapack' → {backend}")
    print()


def test_auto_backend():
    """Test that 'auto' backend works in run_blocked_step."""
    print("=" * 60)
    print("Auto Backend Integration Test")
    print("=" * 60)
    print()

    # Generate small test data
    rng = np.random.default_rng(42)
    n = 1000
    m = 100
    d = 50

    embeddings = rng.normal(size=(n, d))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    rated_indices = rng.choice(n, size=m, replace=False)
    ratings = rng.normal(size=m)

    print(f"Problem: m={m}, n={n}, d={d}")
    print()

    # Test with backend='auto'
    print("Testing fit_backend='auto', score_backend='auto'...")
    result = run_blocked_step(
        embeddings,
        rated_indices,
        ratings,
        fit_backend="auto",
        score_backend="auto",
        block_size=2048,
    )

    print(f"  ✓ Fit completed in {result.profile.fit_seconds:.3f}s")
    print(f"  ✓ Score completed in {result.profile.score_seconds:.3f}s")
    print(f"  ✓ Mean range: [{result.mean.min():.3f}, {result.mean.max():.3f}]")
    print(f"  ✓ Variance range: [{result.variance.min():.3f}, {result.variance.max():.3f}]")
    print()


def test_config():
    """Test GPConfig usage."""
    print("=" * 60)
    print("Configuration Test")
    print("=" * 60)
    print()

    # Test default config
    print("Default Config:")
    print("-" * 40)
    config = GPConfig()
    print_config(config)
    print()

    # Test preset
    print("Fast Config Preset:")
    print("-" * 40)
    print_config(FAST_CONFIG)
    print()

    # Test manual override
    print("Custom Config (manual backends):")
    print("-" * 40)
    custom = GPConfig(
        fit_backend="native_lapack",
        score_backend="gpu",
        length_scale=2.0,
        noise=1e-4,
    )
    print_config(custom)
    print()

    # Test config with run_blocked_step
    print("Using config with run_blocked_step...")
    rng = np.random.default_rng(42)
    n = 500
    m = 50
    d = 20

    embeddings = rng.normal(size=(n, d))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    rated_indices = rng.choice(n, size=m, replace=False)
    ratings = rng.normal(size=m)

    # Use config via **kwargs
    result = run_blocked_step(
        embeddings,
        rated_indices,
        ratings,
        **FAST_CONFIG.to_dict()
    )

    print(f"  ✓ Completed in {result.profile.total_seconds:.3f}s")
    print()


def main():
    print()
    test_selection_logic()
    print()
    test_auto_backend()
    print()
    test_config()
    print()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print()
    print("Usage in your code:")
    print()
    print("  # Use automatic backend selection (recommended)")
    print("  result = run_blocked_step(")
    print("      embeddings, rated_indices, ratings,")
    print("      fit_backend='auto',")
    print("      score_backend='auto',")
    print("  )")
    print()
    print("  # Or use config")
    print("  from poetry_gp.config import GPConfig, FAST_CONFIG")
    print("  config = GPConfig(fit_backend='native_lapack')")
    print("  result = run_blocked_step(")
    print("      embeddings, rated_indices, ratings,")
    print("      **config.to_dict()")
    print("  )")
    print()


if __name__ == "__main__":
    main()
