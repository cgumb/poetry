#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n4
#SBATCH -t 00:15:00
#SBATCH -J quick-bench-test
#SBATCH -o results/quick-bench-test-%j.out
#SBATCH -e results/quick-bench-test-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
SPACK_SETUP_SCRIPT="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050}"

cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "Quick Benchmark Test"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"
echo ""

# Spack environment
if [[ -f "$SPACK_SETUP_SCRIPT" ]]; then
  source "$SPACK_SETUP_SCRIPT"
  spack env activate "$SPACK_ENV_NAME" || echo "Warning: Spack activation failed"
fi

# Python virtual environment
source .venv/bin/activate

# Disable threading for fair comparison
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_mapping_policy=slot

# Build native code
echo "Building native code..."
cmake -S native -B native/build
cmake --build native/build
echo ""

# Output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="results/quick_test_${TIMESTAMP}.csv"

echo "Running quick benchmark sweep..."
echo "Output: $OUTPUT_CSV"
echo ""

# Problem sizes (small for speed)
M_RATED_LIST="100 500 1000 2000"

# Test each problem size
for m_rated in $M_RATED_LIST; do
  echo "=== m_rated=$m_rated ==="

  # Python baseline
  echo "  Python (1 thread)..."
  python -u scripts/bench_step.py \
    --backend blocked \
    --fit-backend python \
    --n-poems 10000 \
    --m-rated $m_rated \
    --seed 42 \
    --output-csv "$OUTPUT_CSV" \
    --append

  # ScaLAPACK with 1 process (Milestone 1B: distributed assembly)
  echo "  ScaLAPACK (1 process, Milestone 1B)..."
  python -u scripts/bench_step.py \
    --backend blocked \
    --fit-backend native_reference \
    --n-poems 10000 \
    --m-rated $m_rated \
    --seed 42 \
    --scalapack-launcher mpirun \
    --scalapack-nprocs 1 \
    --scalapack-block-size 64 \
    --scalapack-native-backend scalapack \
    --output-csv "$OUTPUT_CSV" \
    --append

  # ScaLAPACK with 4 processes (Milestone 1B: parallel assembly!)
  echo "  ScaLAPACK (4 processes, Milestone 1B)..."
  python -u scripts/bench_step.py \
    --backend blocked \
    --fit-backend native_reference \
    --n-poems 10000 \
    --m-rated $m_rated \
    --seed 42 \
    --scalapack-launcher mpirun \
    --scalapack-nprocs 4 \
    --scalapack-block-size 64 \
    --scalapack-native-backend scalapack \
    --output-csv "$OUTPUT_CSV" \
    --append

  echo ""
done

echo "========================================"
echo "Quick benchmark complete!"
echo "Output: $OUTPUT_CSV"
echo ""
echo "To visualize (on login node is fine for viz):"
echo "  python scripts/visualize_benchmarks.py $OUTPUT_CSV"
echo "========================================"
