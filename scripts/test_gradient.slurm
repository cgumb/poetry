#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n1
#SBATCH -t 00:10:00
#SBATCH -J test-gradient
#SBATCH -o results/test-gradient-%j.out
#SBATCH -e results/test-gradient-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
SPACK_SETUP_SCRIPT="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050}"

cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "GP Analytic Gradient Test"
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

# Run test
python -u scripts/debug/test_gradient_optimization.py

echo ""
echo "========================================"
echo "Test complete!"
echo "========================================"
