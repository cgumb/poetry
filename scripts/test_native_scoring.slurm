#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n1
#SBATCH -t 00:15:00
#SBATCH -J test-native-scoring
#SBATCH -o results/test-native-scoring-%j.out
#SBATCH -e results/test-native-scoring-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
SPACK_SETUP_SCRIPT="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050}"

cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "Native Scoring Integration Test"
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

# Disable threading for consistent benchmarks
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Build PyBind11 module if needed
if [[ ! -f "native/build/poetry_gp_native"*.so ]]; then
  echo "Building PyBind11 module..."
  make native-build
  echo ""
fi

echo "✓ PyBind11 module found"
echo ""

# Run test
python -u scripts/test_native_scoring.py

echo ""
echo "========================================"
echo "Test complete!"
echo "========================================"
