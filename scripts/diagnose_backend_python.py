#!/usr/bin/env python3
"""
Diagnose which ScaLAPACK path is being used when called through Python.
"""

import numpy as np
from poetry_gp.backends.blocked import run_blocked_step

# Small test problem
m, n, d = 100, 500, 10
rng = np.random.default_rng(42)
embeddings = rng.normal(size=(n, d))
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
rated_indices = rng.choice(n, size=m, replace=False)
ratings = rng.normal(size=m)

print("=" * 80)
print("BACKEND PATH DIAGNOSTIC (Python → ScaLAPACK)")
print("=" * 80)
print(f"Problem: m={m}, n={n}, d={d}")
print()
print("This should use:")
print("  fit_exact_gp_scalapack_from_rated")
print("  → fit_exact_gp_scalapack_from_features (input_kind='features')")
print("  → Milestone 1B distributed kernel assembly")
print()
print("Check stderr for:")
print("  '[Milestone 1B] Using distributed kernel assembly' = GOOD (fast)")
print("  '[Legacy] Using centralized matrix scatter' = BAD (slow)")
print()
print("=" * 80)
print()

result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
    length_scale=1.0,
    variance=1.0,
    noise=1e-3,
    fit_backend="native_reference",
    score_backend="none",
    optimize_hyperparameters=False,
    scalapack_nprocs=4,
)

print()
print("=" * 80)
print("RESULT")
print("=" * 80)
print(f"Fit time: {result.profile.fit_seconds:.4f}s")
print(f"Alpha shape: {result.gp_state.alpha.shape}")
print()
print("Look at the stderr output above.")
print("If you see '[Milestone 1B]', distributed assembly is working (fast).")
print("If you see '[Legacy]', centralized scatter is being used (slow).")
