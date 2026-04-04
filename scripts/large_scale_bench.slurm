#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n16
#SBATCH -t 02:00:00
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
echo "Large-Scale Benchmark (Milestone 1B)"
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
OUTPUT_CSV="results/large_scale_${TIMESTAMP}.csv"

echo "Running large-scale benchmark sweep..."
echo "Output: $OUTPUT_CSV"
echo "Testing: m ∈ {1000, 2000, 5000, 10000}, processes ∈ {1, 4, 8, 16}"
echo ""

# Problem sizes: larger to see crossover point
M_RATED_LIST="1000 2000 5000 10000"
NPROCS_LIST="1 4 8 16"

# Test each combination
for m_rated in $M_RATED_LIST; do
  echo "=== m_rated=$m_rated ==="

  # Python baseline (always 1 thread)
  echo "  Python (1 thread)..."
  python -u scripts/bench_step.py \
    --backend blocked \
    --fit-backend python \
    --n-poems 10000 \
    --m-rated $m_rated \
    --seed 42 \
    --output-csv "$OUTPUT_CSV" \
    --append

  # ScaLAPACK with varying process counts
  for nprocs in $NPROCS_LIST; do
    echo "  ScaLAPACK ($nprocs processes, Milestone 1B)..."
    python -u scripts/bench_step.py \
      --backend blocked \
      --fit-backend native_reference \
      --n-poems 10000 \
      --m-rated $m_rated \
      --seed 42 \
      --scalapack-launcher mpirun \
      --scalapack-nprocs $nprocs \
      --scalapack-block-size 64 \
      --scalapack-native-backend scalapack \
      --output-csv "$OUTPUT_CSV" \
      --append
  done

  echo ""
done

echo "========================================"
echo "Large-scale benchmark complete!"
echo "Output: $OUTPUT_CSV"
echo ""
echo "To visualize:"
echo "  python scripts/visualize_benchmarks.py $OUTPUT_CSV"
echo ""
echo "Expected results:"
echo "  m=1000-2000: Python likely faster (overhead dominant)"
echo "  m=5000: ScaLAPACK competitive with 8-16 processes"
echo "  m=10000: ScaLAPACK should beat Python with 8-16 processes"
echo "========================================"
