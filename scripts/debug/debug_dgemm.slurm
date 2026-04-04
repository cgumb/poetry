#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n1
#SBATCH -t 00:05:00
#SBATCH -J debug-dgemm
#SBATCH -o results/debug-dgemm-%j.out
#SBATCH -e results/debug-dgemm-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
SPACK_SETUP_SCRIPT="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050}"

cd "$REPO_DIR"

echo "========================================"
echo "Debug DGEMM Test"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Spack environment
if [[ -f "$SPACK_SETUP_SCRIPT" ]]; then
  source "$SPACK_SETUP_SCRIPT"
  spack env activate "$SPACK_ENV_NAME" || echo "Warning: Spack activation failed"
fi

# Run test
bash test_native_direct.sh
