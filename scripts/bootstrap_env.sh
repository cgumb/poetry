#!/usr/bin/env bash
# Bootstrap Python environment for poetry project
# Supports both CPU-only and GPU-enabled setups

set -euo pipefail

# Parse arguments
ENABLE_GPU=0
INSTALL_APP=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      ENABLE_GPU=1
      shift
      ;;
    --app)
      INSTALL_APP=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--gpu] [--app]"
      echo "  --gpu: Enable GPU support (uses mamba + conda env with cupy)"
      echo "  --app: Install app dependencies (Streamlit, etc.)"
      exit 1
      ;;
  esac
done

# Environment variables
SPACK_SETUP="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"

echo "========================================"
echo "Poetry Project Environment Bootstrap"
echo "========================================"
echo "GPU support: $([ $ENABLE_GPU -eq 1 ] && echo 'YES' || echo 'NO')"
echo "App dependencies: $([ $INSTALL_APP -eq 1 ] && echo 'YES' || echo 'NO')"
echo "========================================"
echo ""

cd "$REPO_DIR"

if [[ $ENABLE_GPU -eq 1 ]]; then
  #############################################################################
  # GPU MODE: Use CS-2050-mamba + conda environment
  #############################################################################
  echo "Setting up GPU-enabled environment with mamba..."
  echo ""

  # Activate Spack
  if [[ ! -f "$SPACK_SETUP" ]]; then
    echo "ERROR: Spack setup script not found: $SPACK_SETUP"
    echo "Set SPACK_SETUP_SCRIPT environment variable if in non-standard location"
    exit 1
  fi

  source "$SPACK_SETUP"
  spack env activate CS-2050-mamba || {
    echo "ERROR: Failed to activate CS-2050-mamba spack environment"
    echo "Available environments:"
    spack env list
    exit 1
  }

  # Check mamba is available
  if ! command -v mamba &> /dev/null; then
    echo "ERROR: mamba not found after activating CS-2050-mamba"
    echo "which python: $(which python)"
    exit 1
  fi

  echo "Mamba found: $(which mamba)"
  echo ""

  # Create conda environment
  ENV_NAME="poetry-gpu"
  ENV_FILE="$REPO_DIR/environment-gpu.yml"

  # Create environment spec
  cat > "$ENV_FILE" <<EOF
name: $ENV_NAME
channels:
  - conda-forge
  - nvidia
dependencies:
  - python>=3.11
  - numpy>=1.26
  - scipy>=1.11
  - pandas>=2.1
  - matplotlib>=3.8
  - cupy
  - cudatoolkit
  - pip
  - pip:
    - sentence-transformers>=3.0
    - huggingface_hub>=0.24
    - datasets>=2.20
    - mpi4py>=3.1
    - umap-learn>=0.5.6
    - rich>=13.7
EOF

  if [[ $INSTALL_APP -eq 1 ]]; then
    cat >> "$ENV_FILE" <<EOF
    - streamlit>=1.36
EOF
  fi

  echo "Creating conda environment '$ENV_NAME'..."
  echo "This may take several minutes..."
  echo ""

  # Remove existing env if present
  mamba env remove -n "$ENV_NAME" -y 2>/dev/null || true

  # Create new environment
  mamba env create -f "$ENV_FILE"

  # Activate the conda environment
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"

  # Install poetry_gp package in editable mode
  echo ""
  echo "Installing poetry_gp package..."
  pip install -e . --no-deps

  echo ""
  echo "========================================"
  echo "GPU environment created successfully!"
  echo "========================================"
  echo ""
  echo "To activate this environment:"
  echo "  source ~/161588/spack/share/spack/setup-env.sh"
  echo "  spack env activate CS-2050-mamba"
  echo "  conda activate $ENV_NAME"
  echo ""
  echo "Or use the helper script:"
  echo "  source scripts/activate_env.sh --gpu"
  echo ""
  echo "Test GPU support:"
  echo "  python -c 'import cupy; print(cupy.__version__)'"
  echo "========================================"

else
  #############################################################################
  # CPU MODE: Use CS-2050 + pip venv (existing approach)
  #############################################################################
  echo "Setting up CPU-only environment with venv..."
  echo ""

  # Activate Spack
  if [[ -f "$SPACK_SETUP" ]]; then
    source "$SPACK_SETUP"
    spack env activate CS-2050 || {
      echo "Warning: Failed to activate CS-2050 spack environment"
      echo "Continuing with system Python..."
    }
  else
    echo "Warning: Spack not found, using system Python"
  fi

  VENV_DIR=".venv"

  # Create venv
  python -m venv --system-site-packages "$VENV_DIR"
  source "$VENV_DIR/bin/activate"

  # Install dependencies
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r requirements-core.txt
  python -m pip install -e . --no-deps

  if [[ $INSTALL_APP -eq 1 ]]; then
    python -m pip install -r requirements-app.txt
  fi

  echo ""
  echo "========================================"
  echo "CPU environment created successfully!"
  echo "========================================"
  echo ""
  echo "To activate this environment:"
  echo "  source ~/161588/spack/share/spack/setup-env.sh"
  echo "  spack env activate CS-2050"
  echo "  source .venv/bin/activate"
  echo ""
  echo "Or use the helper script:"
  echo "  source scripts/activate_env.sh"
  echo ""
  echo "Verify installation:"
  echo "  python scripts/app/check_env.py"
  echo "========================================"
fi
