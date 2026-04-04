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

  # Check if mamba works (might fail due to CPU architecture mismatch)
  MAMBA_WORKS=0
  if command -v mamba &> /dev/null; then
    if mamba --version &> /dev/null; then
      MAMBA_WORKS=1
      echo "Mamba found: $(which mamba)"
      echo ""
    else
      echo "Warning: mamba found but crashes (likely CPU architecture mismatch)"
      echo "Will install miniconda in user space instead..."
      echo ""
    fi
  else
    echo "Warning: mamba not found in CS-2050-mamba environment"
    echo "Will install miniconda in user space instead..."
    echo ""
  fi

  # Install micromamba if system mamba doesn't work
  if [[ $MAMBA_WORKS -eq 0 ]]; then
    MICROMAMBA_BIN="$HOME/.local/bin/micromamba"
    MICROMAMBA_ROOT="$HOME/micromamba"

    if [[ ! -f "$MICROMAMBA_BIN" ]]; then
      echo "Installing micromamba (fast, lightweight conda alternative)..."
      echo "This is a one-time setup that will take ~30 seconds..."
      echo ""

      # Create directories
      mkdir -p "$(dirname "$MICROMAMBA_BIN")"
      mkdir -p "$MICROMAMBA_ROOT"

      # Download micromamba binary directly
      # Deactivate spack to use system tools only
      if [[ -n "${SPACK_ENV:-}" ]]; then
        spack env deactivate 2>/dev/null || true
      fi

      # Download tarball
      TARBALL="$TMPDIR/micromamba.tar.bz2"
      /usr/bin/curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$TARBALL"

      # Extract with system tar
      cd "$TMPDIR"
      /bin/tar -xjf "$TARBALL"
      mv bin/micromamba "$MICROMAMBA_BIN"
      rm -rf bin "$TARBALL"
      cd "$REPO_DIR"

      # Reactivate spack if it was active
      if [[ -f "$SPACK_SETUP" ]]; then
        source "$SPACK_SETUP"
        spack env activate CS-2050-mamba 2>/dev/null || true
      fi

      echo "Micromamba installed successfully"
      echo ""
    fi

    # Set up micromamba environment
    export MAMBA_ROOT_PREFIX="$MICROMAMBA_ROOT"
    eval "$("$MICROMAMBA_BIN" shell hook -s bash)"

    # Use micromamba for environment creation
    CONDA_CMD="$MICROMAMBA_BIN"
  else
    CONDA_CMD="mamba"
  fi

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

  echo "Creating conda environment '$ENV_NAME' using $CONDA_CMD..."
  echo "This may take several minutes..."
  echo ""

  # Remove existing env if present
  $CONDA_CMD env remove -n "$ENV_NAME" -y 2>/dev/null || true

  # Create new environment
  $CONDA_CMD env create -f "$ENV_FILE"

  # Activate the environment
  if [[ $MAMBA_WORKS -eq 0 ]]; then
    # Using micromamba
    micromamba activate "$ENV_NAME"
  else
    # Using system mamba
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
  fi

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
