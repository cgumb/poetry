#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n4
#SBATCH -t 00:10:00
#SBATCH -J test-optional-chol
#SBATCH -o results/test-optional-chol-%j.out
#SBATCH -e results/test-optional-chol-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"

cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "Testing Optional Cholesky Refactor"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"
echo ""

# Activate environment (provides BLAS for build)
echo "Activating environment (Spack + Python)..."
source scripts/activate_env.sh --auto

# Rebuild native code with new flags (per-job build dir to avoid corruption)
echo ""
echo "Rebuilding native code..."
make native-build \
    BUILD_DIR="native/build-$SLURM_JOB_ID" \
    ENABLE_PYBIND11=OFF

# Set executable path for tests
export SCALAPACK_EXECUTABLE="native/build-$SLURM_JOB_ID/scalapack_gp_fit"

# Run test script
echo ""
echo "========================================"
echo "Running Tests"
echo "========================================"
python scripts/test_optional_chol.py

echo ""
echo "========================================"
echo "Test Complete!"
echo "========================================"
