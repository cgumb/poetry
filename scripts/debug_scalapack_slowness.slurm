#!/bin/bash
#SBATCH --job-name=debug_scalapack
#SBATCH --output=results/debug_scalapack_%j.out
#SBATCH --error=results/debug_scalapack_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=general

#
# Debug ScaLAPACK slowness with ONE focused test
#

set -e

echo "========================================================================"
echo "DEBUG: ScaLAPACK Performance Issue"
echo "========================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo ""

# Activate environment
source scripts/activate_env.sh

echo "========================================================================"
echo "TEST 1: Small problem (m=100) - should be ~2-3s"
echo "========================================================================"

python3 << 'EOF'
import numpy as np
from poetry_gp.backends.blocked import run_blocked_step
import time

# Small test
m, n, d = 100, 1000, 10
rng = np.random.default_rng(42)
embeddings = rng.normal(size=(n, d))
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
rated_indices = rng.choice(n, size=m, replace=False)
ratings = rng.normal(size=m)

print(f"Problem: m={m}, n={n}, d={d}")
print("Backend: native_reference")
print("Launcher: mpirun (should avoid nested srun)")
print()

start = time.time()
result = run_blocked_step(
    embeddings, rated_indices, ratings,
    length_scale=1.0, variance=1.0, noise=1e-3,
    fit_backend="native_reference",
    score_backend="none",
    optimize_hyperparameters=False,
    scalapack_launcher="mpirun",
    scalapack_nprocs=8,
    scalapack_block_size=128,
    scalapack_verbose=True,  # CRITICAL: Show MPI stderr
)
end = time.time()

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Total time: {end - start:.4f}s")
print(f"Fit time: {result.profile.fit_seconds:.4f}s")
print(f"Alpha shape: {result.state.alpha.shape}")
print()
print("Expected: ~2-3s (mostly overhead)")
if end - start > 5.0:
    print("⚠ WARNING: MUCH SLOWER THAN EXPECTED!")
else:
    print("✓ Reasonable performance")
EOF

echo ""
echo "========================================================================"
echo "TEST 2: Medium problem (m=1000) - should be ~3-4s"
echo "========================================================================"

python3 << 'EOF'
import numpy as np
from poetry_gp.backends.blocked import run_blocked_step
import time

m, n, d = 1000, 5000, 10
rng = np.random.default_rng(42)
embeddings = rng.normal(size=(n, d))
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
rated_indices = rng.choice(n, size=m, replace=False)
ratings = rng.normal(size=m)

print(f"Problem: m={m}, n={n}, d={d}")
print()

start = time.time()
result = run_blocked_step(
    embeddings, rated_indices, ratings,
    length_scale=1.0, variance=1.0, noise=1e-3,
    fit_backend="native_reference",
    score_backend="none",
    optimize_hyperparameters=False,
    scalapack_launcher="mpirun",
    scalapack_nprocs=8,
    scalapack_block_size=64,  # Better for m=1000
    scalapack_verbose=True,
)
end = time.time()

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Total time: {end - start:.4f}s")
print(f"Fit time: {result.profile.fit_seconds:.4f}s")
print()
print("Expected: ~3-4s")
if end - start > 10.0:
    print("⚠ WARNING: MUCH SLOWER THAN EXPECTED!")
    print("This suggests centralized assembly (Legacy path)")
else:
    print("✓ Reasonable performance")
EOF

echo ""
echo "========================================================================"
echo "TEST 3: Check workdir outputs"
echo "========================================================================"

# Find most recent workdir
WORKDIR=$(ls -td /tmp/scalapack_fit_* 2>/dev/null | head -1)
if [ -n "$WORKDIR" ]; then
    echo "Most recent workdir: $WORKDIR"
    echo ""
    echo "Files in workdir:"
    ls -lh "$WORKDIR"
    echo ""

    if [ -f "$WORKDIR/output_meta.json" ]; then
        echo "Output metadata:"
        cat "$WORKDIR/output_meta.json"
        echo ""

        echo "Backend info:"
        grep -E "backend|message|total_seconds|factor_seconds" "$WORKDIR/output_meta.json" || true
    else
        echo "⚠ WARNING: output_meta.json NOT FOUND!"
        echo "This means the MPI job failed or didn't complete"
    fi
else
    echo "⚠ WARNING: No scalapack_fit_* workdirs found in /tmp"
fi

echo ""
echo "========================================================================"
echo "DIAGNOSIS"
echo "========================================================================"
echo "If both tests took much longer than expected:"
echo "  → Problem is likely centralized assembly (Legacy path)"
echo "  → OR launcher issue (though we used mpirun)"
echo ""
echo "Check the stderr above for:"
echo "  [Milestone 1B] = GOOD (distributed assembly)"
echo "  [Legacy] = BAD (centralized assembly)"
echo ""
echo "If no messages appear:"
echo "  → MPI stderr not being captured (check scalapack_verbose setting)"
echo "========================================================================"
