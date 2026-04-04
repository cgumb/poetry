#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n4
#SBATCH -t 00:10:00
#SBATCH -J test-daemon
#SBATCH -o results/test-daemon-%j.out
#SBATCH -e results/test-daemon-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
SPACK_SETUP_SCRIPT="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050}"

cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "ScaLAPACK Persistent Daemon Test"
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

# Disable threading
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_mapping_policy=slot

# Build daemon
echo "Building daemon..."
cmake -S native -B native/build
cmake --build native/build
echo ""

# Check if daemon executable exists
if [[ ! -f "native/build/scalapack_daemon" ]]; then
  echo "ERROR: Daemon executable not found!"
  exit 1
fi

echo "✓ Daemon executable found"
echo ""

# Run test
echo "Running daemon test..."
python -u scripts/debug/test_scalapack_daemon.py

echo ""
echo "========================================"
echo "Test complete!"
echo "========================================"
