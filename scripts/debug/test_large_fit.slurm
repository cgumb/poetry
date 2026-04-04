#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n8
#SBATCH -t 00:10:00
#SBATCH -J test-large-fit
#SBATCH -o results/test-large-fit-%j.out
#SBATCH -e results/test-large-fit-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
SPACK_SETUP_SCRIPT="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050}"

cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "Test Large ScaLAPACK Fit (m=7000)"
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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Running test..."
python -u scripts/debug/test_large_fit.py

echo ""
echo "========================================"
echo "Test complete!"
echo "========================================"
