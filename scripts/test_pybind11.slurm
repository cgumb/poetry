#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n1
#SBATCH -t 00:15:00
#SBATCH -J test-pybind11
#SBATCH -o results/test-pybind11-%j.out
#SBATCH -e results/test-pybind11-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"

cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "Testing PyBind11 LAPACK Integration"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"
echo ""

# Activate environment (provides BLAS for build)
echo "Activating environment (Spack + Python)..."
source scripts/activate_env.sh --auto

# Rebuild native code with PyBind11
echo ""
echo "Rebuilding native code with PyBind11..."
make native-build

# Check if module loaded
echo ""
echo "Checking poetry_gp_native module..."
if python -c "import poetry_gp_native; print('✓ Module loaded:', poetry_gp_native.__doc__)" 2>/dev/null; then
    echo "✓ PyBind11 module is available!"
else
    echo "✗ PyBind11 module NOT available. Check build errors above."
    exit 1
fi

# Run test script
echo ""
echo "========================================"
echo "Running Tests"
echo "========================================"
python scripts/test_pybind11.py

echo ""
echo "========================================"
echo "Test Complete!"
echo "========================================"
