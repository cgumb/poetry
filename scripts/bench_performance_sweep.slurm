#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n48
#SBATCH -t 02:00:00
#SBATCH -J perf-sweep
#SBATCH -o results/perf-sweep-%j.out
#SBATCH -e results/perf-sweep-%j.err

set -euo pipefail

# ==============================================================================
# Performance Benchmark Sweep
# ==============================================================================
# This script benchmarks Python vs ScaLAPACK GP fitting across different:
#   - Problem sizes (m_rated)
#   - Process counts (nprocs)
#   - Block sizes (for ScaLAPACK)
#
# Output: CSV files in results/ directory for easy visualization
# ==============================================================================

# Configuration (override with environment variables)
REPO_DIR="${REPO_DIR:-$HOME/poetry}"
SPACK_SETUP_SCRIPT="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050}"
SPACK_ACTIVATE_REQUIRED="${SPACK_ACTIVATE_REQUIRED:-0}"

# Benchmark parameters
N_POEMS="${N_POEMS:-80000}"
DIM="${DIM:-384}"
SEED="${SEED:-42}"

# Sweep ranges
M_RATED_LIST="${M_RATED_LIST:-100 500 1000 2000 5000 10000}"
NPROCS_LIST="${NPROCS_LIST:-1 2 4 8 16 24 32 48}"
BLOCK_SIZE_LIST="${BLOCK_SIZE_LIST:-64 128 256 512}"

# Native backend settings
NATIVE_BACKEND="${NATIVE_BACKEND:-scalapack}"
NATIVE_EXECUTABLE="${NATIVE_EXECUTABLE:-native/build/scalapack_gp_fit}"

# Output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="results/perf_sweep_${TIMESTAMP}.csv"

echo "========================================"
echo "Performance Benchmark Sweep"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Timestamp: $TIMESTAMP"
echo "Output: $OUTPUT_CSV"
echo "========================================"
echo ""

# ==============================================================================
# Environment Setup
# ==============================================================================

cd "$REPO_DIR"
mkdir -p results

echo "===== Environment Setup ====="

# Spack environment
if [[ -f "$SPACK_SETUP_SCRIPT" ]]; then
  # shellcheck disable=SC1090
  source "$SPACK_SETUP_SCRIPT"
  if ! spack env activate "$SPACK_ENV_NAME"; then
    if [[ "$SPACK_ACTIVATE_REQUIRED" == "1" ]]; then
      echo "ERROR: Failed to activate Spack environment: $SPACK_ENV_NAME" >&2
      exit 1
    fi
    echo "WARNING: Failed to activate Spack environment: $SPACK_ENV_NAME" >&2
  else
    echo "✓ Spack environment activated: $SPACK_ENV_NAME"
  fi
else
  if [[ "$SPACK_ACTIVATE_REQUIRED" == "1" ]]; then
    echo "ERROR: Spack setup script not found: $SPACK_SETUP_SCRIPT" >&2
    exit 1
  fi
  echo "WARNING: Spack setup script not found: $SPACK_SETUP_SCRIPT" >&2
fi

# Python virtual environment
if [[ ! -f .venv/bin/activate ]]; then
  echo "ERROR: Python virtual environment not found at .venv/" >&2
  exit 1
fi
source .venv/bin/activate
echo "✓ Python virtual environment activated"

# Disable threading to ensure fair single-thread comparison
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "✓ Threading disabled for BLAS libraries"

# MPI settings for Open MPI
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_mapping_policy=slot
echo "✓ MPI binding configured"

echo ""

# ==============================================================================
# Build Native Code
# ==============================================================================

echo "===== Building Native ScaLAPACK Code ====="
rm -rf native/build
cmake -S native -B native/build
cmake --build native/build

if [[ ! -f "$NATIVE_EXECUTABLE" ]]; then
  echo "ERROR: Native executable not found: $NATIVE_EXECUTABLE" >&2
  exit 1
fi
echo "✓ Native code built successfully"
echo ""

echo "===== Native Executable Info ====="
echo "Executable: $NATIVE_EXECUTABLE"
ldd "$NATIVE_EXECUTABLE" | grep -E 'scalapack|blas|lapack|openblas|mkl' || echo "(no BLAS/LAPACK libraries shown)"
echo ""

# ==============================================================================
# Run Benchmarks
# ==============================================================================

echo "===== Starting Benchmark Sweep ====="
echo "Problem sizes (m_rated): $M_RATED_LIST"
echo "Process counts: $NPROCS_LIST"
echo "Block sizes: $BLOCK_SIZE_LIST"
echo ""

# Create CSV header
echo "timestamp,m_rated,backend,nprocs,block_size,fit_seconds,score_seconds,total_seconds,log_marginal_likelihood,alpha_max_diff,mean_max_diff,variance_max_diff" > "$OUTPUT_CSV"

# Benchmark loop
total_runs=0
for m_rated in $M_RATED_LIST; do
  echo ""
  echo "========================================"
  echo "Problem Size: m_rated=$m_rated"
  echo "========================================"

  # Python baseline (single-threaded)
  echo ""
  echo "--- Python Baseline (single-threaded) ---"
  python -u scripts/bench_step.py \
    --n-poems "$N_POEMS" \
    --dim "$DIM" \
    --m-rated "$m_rated" \
    --seed "$SEED" \
    --backend blocked \
    --fit-backend python \
    --output-csv "$OUTPUT_CSV" \
    --append \
    || echo "WARNING: Python baseline failed for m_rated=$m_rated"
  total_runs=$((total_runs + 1))

  # ScaLAPACK with different nprocs and block sizes
  for nprocs in $NPROCS_LIST; do
    # Skip if requesting more processes than allocated
    if [[ $nprocs -gt $SLURM_NTASKS ]]; then
      echo "SKIP: nprocs=$nprocs exceeds allocated tasks ($SLURM_NTASKS)"
      continue
    fi

    for block_size in $BLOCK_SIZE_LIST; do
      echo ""
      echo "--- ScaLAPACK: nprocs=$nprocs block_size=$block_size ---"
      python -u scripts/bench_step.py \
        --n-poems "$N_POEMS" \
        --dim "$DIM" \
        --m-rated "$m_rated" \
        --seed "$SEED" \
        --backend blocked \
        --fit-backend native_reference \
        --scalapack-launcher mpirun \
        --scalapack-nprocs "$nprocs" \
        --scalapack-executable "$NATIVE_EXECUTABLE" \
        --scalapack-block-size "$block_size" \
        --scalapack-native-backend "$NATIVE_BACKEND" \
        --output-csv "$OUTPUT_CSV" \
        --append \
        || echo "WARNING: ScaLAPACK failed for m_rated=$m_rated nprocs=$nprocs block_size=$block_size"
      total_runs=$((total_runs + 1))
    done
  done
done

echo ""
echo "========================================"
echo "Benchmark Sweep Complete"
echo "========================================"
echo "Total benchmark runs: $total_runs"
echo "Output saved to: $OUTPUT_CSV"
echo ""
echo "Next steps:"
echo "  1. Download results: scp cluster:$REPO_DIR/$OUTPUT_CSV ."
echo "  2. Visualize: python scripts/visualize_benchmarks.py $OUTPUT_CSV"
echo "========================================"
