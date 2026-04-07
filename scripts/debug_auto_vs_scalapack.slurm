#!/bin/bash
#SBATCH --job-name=debug_auto_vs_scalapack
#SBATCH --output=results/debug_auto_vs_scalapack_%j.out
#SBATCH --error=results/debug_auto_vs_scalapack_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=general

#
# DEBUG: Compare backend=auto (pedagogical) vs backend=scalapack (large_scale)
#
# This isolates whether the slowness is from:
# 1. C++ backend routing (auto vs scalapack)
# 2. Python-level parameter passing
# 3. Something else entirely
#

set -e

echo "========================================================================"
echo "DEBUG: auto vs scalapack backend comparison"
echo "========================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo ""

# Activate environment
source scripts/activate_env.sh

# Disable threading for fair comparison (match large_scale_bench.slurm)
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_mapping_policy=slot

# Use dedicated build directory for this job
# Skip PyBind11 build since we only need ScaLAPACK executable
BUILD_DIR="native/build_job_${SLURM_JOB_ID}"
EXECUTABLE="${BUILD_DIR}/scalapack_gp_fit"

# Rebuild native code (ensure latest routing fix)
echo "Rebuilding native code in ${BUILD_DIR}..."
cmake -S native -B "${BUILD_DIR}" -DPOETRY_ENABLE_PYBIND11=OFF
cmake --build "${BUILD_DIR}"
echo ""

# Check executable exists
if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: ${EXECUTABLE} not found after build!"
    exit 1
fi
echo "Built executable: ${EXECUTABLE}"
echo ""

# Test problem: m=2000, n=5000, d=10
M=2000
N=5000
D=10

echo "========================================================================"
echo "TEST 1: Python baseline (for comparison)"
echo "========================================================================"

python3 << EOF
import numpy as np
from poetry_gp.backends.blocked import run_blocked_step
import time

m, n, d = $M, $N, $D
rng = np.random.default_rng(42)
embeddings = rng.normal(size=(n, d))
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
rated_indices = rng.choice(n, size=m, replace=False)
ratings = rng.normal(size=m)

print(f"Problem: m={m}, n={n}, d={d}")
print("Backend: python")
print()

start = time.time()
result = run_blocked_step(
    embeddings, rated_indices, ratings,
    length_scale=1.0, variance=1.0, noise=1e-3,
    fit_backend="python",
    score_backend="none",
    optimize_hyperparameters=False,
)
end = time.time()

print(f"Python fit time: {result.profile.fit_seconds:.4f}s")
print(f"Python total time: {end - start:.4f}s")
print()
EOF

echo ""
echo "========================================================================"
echo "TEST 2: backend=auto (what pedagogical benchmarks use)"
echo "========================================================================"

python3 << EOF
import numpy as np
from poetry_gp.backends.blocked import run_blocked_step
import time

m, n, d = $M, $N, $D
rng = np.random.default_rng(42)
embeddings = rng.normal(size=(n, d))
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
rated_indices = rng.choice(n, size=m, replace=False)
ratings = rng.normal(size=m)

print(f"Problem: m={m}, n={n}, d={d}")
print("Backend: native_reference")
print("scalapack_native_backend: 'auto' (DEFAULT)")
print("scalapack_launcher: 'mpirun' (EXPLICIT - match large_scale)")
print()

start = time.time()
result = run_blocked_step(
    embeddings, rated_indices, ratings,
    length_scale=1.0, variance=1.0, noise=1e-3,
    fit_backend="native_reference",
    score_backend="none",
    optimize_hyperparameters=False,
    scalapack_executable="$EXECUTABLE",
    scalapack_launcher="mpirun",
    scalapack_nprocs=8,
    scalapack_block_size=128,
    # NOT PASSING scalapack_native_backend - defaults to "auto"
)
end = time.time()

print(f"native_reference (auto) fit time: {result.profile.fit_seconds:.4f}s")
print(f"native_reference (auto) total time: {end - start:.4f}s")
print()

# Check what backend was actually used
if result.state.optimization_result:
    print("Backend details from optimization_result:")
    print(f"  native_backend: {result.state.optimization_result.get('native_backend')}")
    print(f"  requested_native_backend: {result.state.optimization_result.get('requested_native_backend')}")
    print(f"  compiled_with_scalapack: {result.state.optimization_result.get('compiled_with_scalapack')}")
    print(f"  message: {result.state.optimization_result.get('message', '')[:100]}")
print()
EOF

echo ""
echo "========================================================================"
echo "TEST 3: backend=scalapack (what large_scale uses)"
echo "========================================================================"

python3 << EOF
import numpy as np
from poetry_gp.backends.blocked import run_blocked_step
import time

m, n, d = $M, $N, $D
rng = np.random.default_rng(42)
embeddings = rng.normal(size=(n, d))
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
rated_indices = rng.choice(n, size=m, replace=False)
ratings = rng.normal(size=m)

print(f"Problem: m={m}, n={n}, d={d}")
print("Backend: native_reference")
print("scalapack_native_backend: 'scalapack' (EXPLICIT)")
print("scalapack_launcher: 'mpirun' (EXPLICIT - match large_scale)")
print()

start = time.time()
result = run_blocked_step(
    embeddings, rated_indices, ratings,
    length_scale=1.0, variance=1.0, noise=1e-3,
    fit_backend="native_reference",
    score_backend="none",
    optimize_hyperparameters=False,
    scalapack_executable="$EXECUTABLE",
    scalapack_launcher="mpirun",
    scalapack_nprocs=8,
    scalapack_block_size=128,
    scalapack_native_backend="scalapack",  # EXPLICIT
)
end = time.time()

print(f"native_reference (scalapack) fit time: {result.profile.fit_seconds:.4f}s")
print(f"native_reference (scalapack) total time: {end - start:.4f}s")
print()

# Check what backend was actually used
if result.state.optimization_result:
    print("Backend details from optimization_result:")
    print(f"  native_backend: {result.state.optimization_result.get('native_backend')}")
    print(f"  requested_native_backend: {result.state.optimization_result.get('requested_native_backend')}")
    print(f"  compiled_with_scalapack: {result.state.optimization_result.get('compiled_with_scalapack')}")
    print(f"  message: {result.state.optimization_result.get('message', '')[:100]}")
print()
EOF

echo ""
echo "========================================================================"
echo "TEST 4: bench_step.py with backend=auto (mimic pedagogical)"
echo "========================================================================"

python scripts/bench_step.py \
    --backend blocked \
    --fit-backend native_reference \
    --score-backend none \
    --n-poems $N \
    --m-rated $M \
    --dim $D \
    --seed 42 \
    --scalapack-executable "${EXECUTABLE}" \
    --scalapack-launcher mpirun \
    --scalapack-nprocs 8 \
    --scalapack-block-size 128

echo ""
echo "========================================================================"
echo "TEST 5: bench_step.py with backend=scalapack (mimic large_scale)"
echo "========================================================================"

python scripts/bench_step.py \
    --backend blocked \
    --fit-backend native_reference \
    --score-backend none \
    --n-poems $N \
    --m-rated $M \
    --dim $D \
    --seed 42 \
    --scalapack-executable "${EXECUTABLE}" \
    --scalapack-launcher mpirun \
    --scalapack-nprocs 8 \
    --scalapack-block-size 128 \
    --scalapack-native-backend scalapack

echo ""
echo "========================================================================"
echo "CLEANUP"
echo "========================================================================"
echo "Removing build directory: ${BUILD_DIR}"
rm -rf "${BUILD_DIR}"
echo "Done."

echo ""
echo "========================================================================"
echo "DIAGNOSIS"
echo "========================================================================"
echo "If TEST 2 (auto) is slow but TEST 3 (scalapack) is fast:"
echo "  → C++ backend routing still broken for 'auto'"
echo "  → normalize_backend_name('auto') not returning 'scalapack'"
echo ""
echo "If both TEST 2 and TEST 3 are slow:"
echo "  → Problem is NOT backend routing"
echo "  → Problem is launcher, file I/O, or MPI overhead"
echo ""
echo "If TEST 4 is slow but TEST 5 is fast:"
echo "  → bench_step.py parameter handling issue"
echo ""
echo "If all tests are slow:"
echo "  → Native binary not using Milestone 1B distributed assembly"
echo "  → Check for [Legacy] vs [Milestone 1B] in workdir stderr"
echo "========================================================================"
