#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n16
#SBATCH -t 12:00:00
#SBATCH -J large-scale-bench
#SBATCH -o results/large-scale-bench-%j.out
#SBATCH -e results/large-scale-bench-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
SPACK_SETUP_SCRIPT="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050}"

cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "Large-Scale Fit Benchmark (Overnight)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "========================================"
echo ""

# Activate Python environment (auto-detects available setup)
source scripts/activate_env.sh --auto

# Disable threading for fair comparison
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_mapping_policy=slot

# Use home directory for temp files (compute nodes have limited /tmp)
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# Build native code in job-specific directory
# Skip PyBind11 build since we only need ScaLAPACK executable
BUILD_DIR="native/build_job_${SLURM_JOB_ID}"
EXECUTABLE="${BUILD_DIR}/scalapack_gp_fit"

echo "Building native code in ${BUILD_DIR}..."
cmake -S native -B "${BUILD_DIR}" -DPOETRY_ENABLE_PYBIND11=OFF
cmake --build "${BUILD_DIR}"
echo ""

if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: ${EXECUTABLE} not found after build!"
    exit 1
fi
echo "Built executable: ${EXECUTABLE}"
echo ""

# Output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="results/large_scale_fit_${TIMESTAMP}.csv"

echo "Running large-scale fit benchmark (overnight run)..."
echo "Output: $OUTPUT_CSV"
echo "Testing: m ∈ {2000, 5000, 7000, 10000, 15000, 20000, 25000, 30000}"
echo "         block_size ∈ {64, 128, 256}"
echo "         processes ∈ {1, 4, 8, 16}"
echo ""
echo "FIT-ONLY: Scoring is skipped (--score-backend none)"
echo ""

# Problem sizes: comprehensive sweep up to 30k
M_RATED_LIST="2000 5000 7000 10000 15000 20000 25000 30000"
BLOCK_SIZE_LIST="64 128 256"
NPROCS_LIST="1 4 8 16"
N_POEMS=35000  # Must be > max(M_RATED_LIST)

# Test each combination
for m_rated in $M_RATED_LIST; do
  echo "=== m_rated=$m_rated ==="

  # Python baseline (always 1 thread, once per m)
  echo "  Python (1 thread)..."
  python -u scripts/bench_step.py \
    --backend blocked \
    --fit-backend python \
    --score-backend none \
    --n-poems $N_POEMS \
    --m-rated $m_rated \
    --seed 42 \
    --scalapack-executable "${EXECUTABLE}" \
    --output-csv "$OUTPUT_CSV" \
    --append

  # ScaLAPACK with varying block sizes and process counts
  for nprocs in $NPROCS_LIST; do
    for block_size in $BLOCK_SIZE_LIST; do
      echo "  ScaLAPACK ($nprocs processes, block_size=$block_size)..."
      python -u scripts/bench_step.py \
        --backend blocked \
        --fit-backend native_reference \
        --score-backend none \
        --n-poems $N_POEMS \
        --m-rated $m_rated \
        --seed 42 \
        --scalapack-executable "${EXECUTABLE}" \
        --scalapack-launcher mpirun \
        --scalapack-nprocs $nprocs \
        --scalapack-block-size $block_size \
        --scalapack-native-backend scalapack \
        --output-csv "$OUTPUT_CSV" \
        --append
    done
  done

  echo ""
done

echo "========================================"
echo "Large-scale fit benchmark complete!"
echo "Output: $OUTPUT_CSV"
echo ""
echo "To visualize:"
echo "  python scripts/visualize_benchmarks.py $OUTPUT_CSV"
echo ""
echo "Expected results:"
echo "  m=2k-5k: Python likely faster (overhead dominant)"
echo "  m=7k-10k: ScaLAPACK competitive with 8-16 processes"
echo "  m=15k-20k: ScaLAPACK should beat Python with 8-16 processes"
echo "  m=25k-30k: ScaLAPACK dominates (Python O(m³) becomes prohibitive)"
echo ""
echo "Key insights to find:"
echo "  - Crossover point where ScaLAPACK beats Python"
echo "  - Optimal block size (64 vs 128 vs 256)"
echo "  - Scaling efficiency with process count"
echo "  - Whether ScaLAPACK enables problem sizes Python can't handle"
echo ""
echo "Cleaning up build directory: ${BUILD_DIR}"
rm -rf "${BUILD_DIR}"
echo "Done."
echo "========================================"
