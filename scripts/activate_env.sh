#!/usr/bin/env bash
# Helper script to activate the appropriate Python environment
# Usage: source scripts/activate_env.sh [--gpu]
#
# This script auto-detects which environment was created and activates it

# Check if we're being sourced (required for environment activation)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: This script must be sourced, not executed"
  echo "Usage: source $0 [--gpu]"
  exit 1
fi

# Parse arguments
PREFER_GPU=0
if [[ "$1" == "--gpu" ]]; then
  PREFER_GPU=1
fi

# Find repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Spack setup
SPACK_SETUP="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"

cd "$REPO_DIR"

#############################################################################
# Auto-detect which environment exists
#############################################################################

HAS_CONDA_ENV=0
HAS_VENV=0

# Check for conda environment
if command -v conda &> /dev/null; then
  if conda env list | grep -q "poetry-gpu"; then
    HAS_CONDA_ENV=1
  fi
fi

# Check for venv
if [[ -f ".venv/bin/activate" ]]; then
  HAS_VENV=1
fi

#############################################################################
# Activate appropriate environment
#############################################################################

if [[ $PREFER_GPU -eq 1 ]] || [[ $HAS_CONDA_ENV -eq 1 && $HAS_VENV -eq 0 ]]; then
  # GPU mode: Use conda environment
  if [[ $HAS_CONDA_ENV -eq 0 ]]; then
    echo "ERROR: GPU environment not found"
    echo "Create it with: bash scripts/bootstrap_env.sh --gpu"
    return 1
  fi

  echo "Activating GPU environment..."

  if [[ -f "$SPACK_SETUP" ]]; then
    source "$SPACK_SETUP"
    spack env activate CS-2050-mamba
  fi

  eval "$(conda shell.bash hook)"
  conda activate poetry-gpu

  echo "✓ GPU environment activated (mamba + conda)"
  echo "  Python: $(which python)"
  echo "  CuPy available: $(python -c 'import cupy; print("YES")' 2>/dev/null || echo 'NO')"

else
  # CPU mode: Use venv
  if [[ $HAS_VENV -eq 0 ]]; then
    echo "ERROR: CPU environment not found"
    echo "Create it with: bash scripts/bootstrap_env.sh"
    return 1
  fi

  echo "Activating CPU environment..."

  if [[ -f "$SPACK_SETUP" ]]; then
    source "$SPACK_SETUP"
    spack env activate CS-2050
  fi

  source .venv/bin/activate

  echo "✓ CPU environment activated (spack + venv)"
  echo "  Python: $(which python)"
fi
