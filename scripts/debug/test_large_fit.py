"""
Test ScaLAPACK fit with the problem size that failed in the sweep.
This reproduces the m=7000 case that caused JSONDecodeError.
"""

from poetry_gp.backends.scalapack_fit import fit_exact_gp_scalapack_from_rated
import numpy as np

# Use the same parameters as the failing sweep
m_rated = 7000
n_poems = 25000
d = 384
seed = 42

print(f"Testing ScaLAPACK fit with m={m_rated}, n={n_poems}, d={d}...")
print("(This matches the failing benchmark sweep configuration)")
print()

rng = np.random.default_rng(seed)

# Generate random embeddings and sample rated indices (same as bench_step.py)
print("Generating random data...")
embeddings = rng.normal(size=(n_poems, d))
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
rated_indices = rng.choice(n_poems, size=m_rated, replace=False)
ratings = rng.normal(size=m_rated)

x_rated = embeddings[rated_indices]

print(f"  x_rated shape: {x_rated.shape}")
print(f"  ratings shape: {ratings.shape}")
print()

print("Running ScaLAPACK fit with 4 processes, block_size=32...")
try:
    result = fit_exact_gp_scalapack_from_rated(
        x_rated, ratings,
        length_scale=1.0,
        variance=1.0,
        noise=1e-3,
        launcher='mpirun',
        nprocs=4,
        block_size=32,
        native_backend='scalapack',
        verbose=True
    )
    print()
    print("✓ SUCCESS!")
    print(f"  alpha shape: {result.alpha.shape}")
    print(f"  log_marginal_likelihood: {result.log_marginal_likelihood:.6f}")
except Exception as e:
    print()
    print("✗ ERROR:", e)
    print()
    import traceback
    traceback.print_exc()
