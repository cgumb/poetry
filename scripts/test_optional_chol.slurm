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

# Rebuild native code with new flags
echo "Rebuilding native code..."
make native-build

# Activate Python environment
echo "Activating Python environment..."
source scripts/activate_env.sh --auto

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
