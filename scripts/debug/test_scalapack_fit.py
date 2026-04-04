"""
Quick test script to diagnose ScaLAPACK fit errors.
"""

from poetry_gp.backends.scalapack_fit import fit_exact_gp_scalapack_from_rated
import numpy as np

rng = np.random.default_rng(42)
x = rng.normal(size=(100, 50))
y = rng.normal(size=100)

print("Testing ScaLAPACK fit with m=100, d=50...")
print()

try:
    result = fit_exact_gp_scalapack_from_rated(
        x, y,
        length_scale=1.0,
        variance=1.0,
        noise=0.01,
        launcher='mpirun',
        nprocs=4,
        block_size=64,
        verbose=True
    )
    print()
    print("✓ SUCCESS!")
    print(f"  alpha shape: {result.alpha.shape}")
    print(f"  chol shape: {result.cho_factor_data[0].shape}")
    print(f"  log_marginal_likelihood: {result.log_marginal_likelihood:.6f}")
    print(f"  length_scale: {result.length_scale:.4f}")
    print(f"  variance: {result.variance:.4f}")
    print(f"  noise: {result.noise:.6f}")
except Exception as e:
    print()
    print("✗ ERROR:", e)
    print()
    import traceback
    traceback.print_exc()
