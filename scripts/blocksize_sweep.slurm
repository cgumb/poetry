#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n16
#SBATCH -t 01:00:00
#SBATCH -J blocksize-sweep
#SBATCH -o results/blocksize-sweep-%j.out
#SBATCH -e results/blocksize-sweep-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
SPACK_SETUP_SCRIPT="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050}"

cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "Block Size Sweep"
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

# Use home directory for temp files (compute nodes have limited /tmp)
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# Build native code
echo "Building native code..."
cmake -S native -B native/build
cmake --build native/build
echo ""

# Output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="results/blocksize_sweep_${TIMESTAMP}.csv"

echo "Running block size sweep..."
echo "Output: $OUTPUT_CSV"
echo "Testing: m ∈ {7000, 10000, 15000, 20000}"
echo "         block_size ∈ {32, 64, 128, 256}"
echo "         processes ∈ {4, 8, 16}"
echo ""

# Problem sizes where ScaLAPACK is viable (m >= 7000)
M_RATED_LIST="7000 10000 15000 20000"
BLOCK_SIZE_LIST="32 64 128 256"
NPROCS_LIST="4 8 16"
N_POEMS=25000  # Must be > max(M_RATED_LIST) to allow sampling

# Test each combination
for m_rated in $M_RATED_LIST; do
  echo "=== m_rated=$m_rated ==="

  # Python baseline (once per m, for comparison)
  echo "  Python (1 thread)..."
  python -u scripts/bench_step.py \
    --backend blocked \
    --fit-backend python \
    --n-poems $N_POEMS \
    --m-rated $m_rated \
    --seed 42 \
    --output-csv "$OUTPUT_CSV" \
    --append

  # ScaLAPACK with varying block sizes and process counts
  for nprocs in $NPROCS_LIST; do
    for block_size in $BLOCK_SIZE_LIST; do
      echo "  ScaLAPACK ($nprocs processes, block_size=$block_size)..."
      python -u scripts/bench_step.py \
        --backend blocked \
        --fit-backend native_reference \
        --n-poems $N_POEMS \
        --m-rated $m_rated \
        --seed 42 \
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
echo "Block size sweep complete!"
echo "Output: $OUTPUT_CSV"
echo ""
echo "To visualize:"
echo "  python scripts/visualize_benchmarks.py $OUTPUT_CSV"
echo ""
echo "Expected insights:"
echo "  - Larger blocks may reduce communication overhead"
echo "  - Optimal block size likely between 64-256 for m=7k-20k"
echo "  - Too large blocks may hurt load balancing"
echo "  - Higher process counts may benefit from smaller blocks"
echo "========================================"
