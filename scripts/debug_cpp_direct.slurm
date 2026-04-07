#!/bin/bash
#SBATCH --job-name=debug_cpp_direct
#SBATCH --output=results/debug_cpp_direct_%j.out
#SBATCH --error=results/debug_cpp_direct_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=general

#
# DEBUG: Call C++ executable directly with different --backend values
#
# This bypasses ALL Python code to test if the C++ routing is correct.
#

set -e

echo "========================================================================"
echo "DEBUG: Direct C++ executable test"
echo "========================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

# Disable threading for fair comparison (match large_scale_bench.slurm)
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_mapping_policy=slot
echo "Threading disabled: OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""

# Use dedicated build directory for this job
BUILD_DIR="native/build_job_${SLURM_JOB_ID}"
EXECUTABLE="${BUILD_DIR}/scalapack_gp_fit"

# Rebuild native code
echo "Rebuilding native code in ${BUILD_DIR}..."
cmake -S native -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}"
echo ""

# Check that executable exists
if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: ${EXECUTABLE} not found!"
    exit 1
fi

echo "Executable: $(ls -lh ${EXECUTABLE})"
echo ""

# Create test problem
WORKDIR=$(mktemp -d /tmp/cpp_debug_XXXXXX)
echo "Workdir: $WORKDIR"
echo ""

# Generate test data using Python
python3 << EOF
import numpy as np
import json

m, n, d = 2000, 5000, 10
rng = np.random.default_rng(42)
x = rng.normal(size=(m, d))
x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
y = rng.normal(size=m)

meta = {
    "input_kind": "features",
    "n": m,
    "d": d,
    "length_scale": 1.0,
    "variance": 1.0,
    "noise": 0.001,
    "rhs_cols": 1,
    "dtype": "float64",
    "matrix_layout": "row_major",
}

with open("$WORKDIR/input.json", "w") as f:
    json.dump(meta, f)

x.tofile("$WORKDIR/x.bin")
y.tofile("$WORKDIR/y.bin")

print(f"Created test problem: m={m}, n={n}, d={d}")
EOF

echo ""
echo "========================================================================"
echo "TEST 1: --backend auto"
echo "========================================================================"

srun -n 8 "${EXECUTABLE}" \
    --input-meta "$WORKDIR/input.json" \
    --matrix-bin "$WORKDIR/x.bin" \
    --rhs-bin "$WORKDIR/y.bin" \
    --output-meta "$WORKDIR/output_auto.json" \
    --alpha-bin "$WORKDIR/alpha_auto.bin" \
    --chol-bin "$WORKDIR/chol_auto.bin" \
    --backend auto \
    --block-size 128 \
    --return-alpha 1 \
    --return-chol 0

echo ""
if [ -f "$WORKDIR/output_auto.json" ]; then
    echo "Output from --backend auto:"
    cat "$WORKDIR/output_auto.json" | jq '{backend, requested_backend, compiled_with_scalapack, total_seconds, factor_seconds, message}'
else
    echo "ERROR: output_auto.json not created!"
fi

echo ""
echo "========================================================================"
echo "TEST 2: --backend scalapack"
echo "========================================================================"

srun -n 8 "${EXECUTABLE}" \
    --input-meta "$WORKDIR/input.json" \
    --matrix-bin "$WORKDIR/x.bin" \
    --rhs-bin "$WORKDIR/y.bin" \
    --output-meta "$WORKDIR/output_scalapack.json" \
    --alpha-bin "$WORKDIR/alpha_scalapack.bin" \
    --chol-bin "$WORKDIR/chol_scalapack.bin" \
    --backend scalapack \
    --block-size 128 \
    --return-alpha 1 \
    --return-chol 0

echo ""
if [ -f "$WORKDIR/output_scalapack.json" ]; then
    echo "Output from --backend scalapack:"
    cat "$WORKDIR/output_scalapack.json" | jq '{backend, requested_backend, compiled_with_scalapack, total_seconds, factor_seconds, message}'
else
    echo "ERROR: output_scalapack.json not created!"
fi

echo ""
echo "========================================================================"
echo "TEST 3: --backend mpi_row_partitioned_reference (should be slow)"
echo "========================================================================"

srun -n 8 "${EXECUTABLE}" \
    --input-meta "$WORKDIR/input.json" \
    --matrix-bin "$WORKDIR/x.bin" \
    --rhs-bin "$WORKDIR/y.bin" \
    --output-meta "$WORKDIR/output_mpi.json" \
    --alpha-bin "$WORKDIR/alpha_mpi.bin" \
    --chol-bin "$WORKDIR/chol_mpi.bin" \
    --backend mpi_row_partitioned_reference \
    --block-size 128 \
    --return-alpha 1 \
    --return-chol 0

echo ""
if [ -f "$WORKDIR/output_mpi.json" ]; then
    echo "Output from --backend mpi_row_partitioned_reference:"
    cat "$WORKDIR/output_mpi.json" | jq '{backend, requested_backend, compiled_with_scalapack, total_seconds, factor_seconds, message}'
else
    echo "ERROR: output_mpi.json not created!"
fi

echo ""
echo "========================================================================"
echo "COMPARISON"
echo "========================================================================"

python3 << EOF
import json

def load_result(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

auto = load_result("$WORKDIR/output_auto.json")
scalapack = load_result("$WORKDIR/output_scalapack.json")
mpi = load_result("$WORKDIR/output_mpi.json")

print("Resolved backends:")
if auto:
    print(f"  auto → {auto.get('backend')} ({auto.get('total_seconds', 0):.4f}s)")
if scalapack:
    print(f"  scalapack → {scalapack.get('backend')} ({scalapack.get('total_seconds', 0):.4f}s)")
if mpi:
    print(f"  mpi_row_partitioned_reference → {mpi.get('backend')} ({mpi.get('total_seconds', 0):.4f}s)")

print()
print("DIAGNOSIS:")
if auto and scalapack:
    auto_backend = auto.get('backend')
    scalapack_backend = scalapack.get('backend')
    auto_time = auto.get('total_seconds', 0)
    scalapack_time = scalapack.get('total_seconds', 0)

    if auto_backend != scalapack_backend:
        print(f"✗ PROBLEM: auto resolved to '{auto_backend}' instead of '{scalapack_backend}'")
        print("  → Backend routing is BROKEN in C++ code")
        print("  → Check normalize_backend_name() in scalapack_gp_fit.cpp")
    elif abs(auto_time - scalapack_time) > 1.0:
        print(f"✗ PROBLEM: auto and scalapack resolve to same backend but different times")
        print(f"  auto: {auto_time:.4f}s, scalapack: {scalapack_time:.4f}s")
        print("  → Something strange happening")
    else:
        print("✓ GOOD: auto and scalapack both resolve to same backend with similar times")
        print(f"  Both use: {auto_backend}, time: ~{auto_time:.4f}s")

        if mpi:
            mpi_time = mpi.get('total_seconds', 0)
            if auto_time > mpi_time * 5:
                print(f"  BUT: This is SLOWER than expected (mpi took {mpi_time:.4f}s)")
                print("  → Likely using Legacy scatter path instead of Milestone 1B")
EOF

echo ""
echo "Workdir preserved: $WORKDIR"
echo "You can inspect:"
echo "  - input.json: problem specification"
echo "  - output_*.json: results and timings"
echo ""
echo "Cleaning up build directory: ${BUILD_DIR}"
rm -rf "${BUILD_DIR}"
echo "Done."
echo "========================================================================"
