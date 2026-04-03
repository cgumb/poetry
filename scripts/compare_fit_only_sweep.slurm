#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n48
#SBATCH -t 01:00:00
#SBATCH -J fit-only-sweep
#SBATCH -o fit-only-sweep-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
SPACK_SETUP_SCRIPT="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050}"
SPACK_ACTIVATE_REQUIRED="${SPACK_ACTIVATE_REQUIRED:-1}"
N_POEMS="${N_POEMS:-80000}"
DIM="${DIM:-384}"
M_RATED="${M_RATED:-10000}"
SEED="${SEED:-0}"
FIT_NPROCS_LIST="${FIT_NPROCS_LIST:-1 4 8 12 16 24 48}"
SCALAPACK_BLOCK_SIZE_LIST="${SCALAPACK_BLOCK_SIZE_LIST:-64 128 256 384 512}"
NATIVE_BACKEND="${NATIVE_BACKEND:-scalapack}"
NATIVE_EXECUTABLE="${NATIVE_EXECUTABLE:-native/build/scalapack_gp_fit}"

cd "$REPO_DIR"

if [[ -f "$SPACK_SETUP_SCRIPT" ]]; then
  # shellcheck disable=SC1090
  source "$SPACK_SETUP_SCRIPT"
  if ! spack env activate "$SPACK_ENV_NAME"; then
    if [[ "$SPACK_ACTIVATE_REQUIRED" == "1" ]]; then
      echo "Failed to activate Spack environment: $SPACK_ENV_NAME" >&2
      exit 1
    fi
    echo "Warning: failed to activate Spack environment: $SPACK_ENV_NAME" >&2
  fi
else
  if [[ "$SPACK_ACTIVATE_REQUIRED" == "1" ]]; then
    echo "Spack setup script not found: $SPACK_SETUP_SCRIPT" >&2
    exit 1
  fi
  echo "Warning: Spack setup script not found: $SPACK_SETUP_SCRIPT" >&2
fi

source .venv/bin/activate

unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_mapping_policy=slot

echo "JOB=$SLURM_JOB_ID STEP=${SLURM_STEP_ID:-unset} NTASKS=$SLURM_NTASKS STEP_TASKS=${SLURM_STEP_NUM_TASKS:-unset}"
echo

echo "===== clean rebuild ====="
rm -rf native/build
cmake -S native -B native/build
cmake --build native/build
echo

echo "===== ldd: native executable ====="
ldd "$NATIVE_EXECUTABLE" | egrep 'scalapack|blas|lapack|openblas' || true
echo

echo "===== fit-only sweep ====="
for nprocs in $FIT_NPROCS_LIST; do
  for block_size in $SCALAPACK_BLOCK_SIZE_LIST; do
    echo "===== nprocs=${nprocs} m_rated=${M_RATED} scalapack_block_size=${block_size} backend=${NATIVE_BACKEND} ====="
    python -u scripts/compare_fit_backends.py \
      --n-poems "$N_POEMS" \
      --dim "$DIM" \
      --m-rated "$M_RATED" \
      --seed "$SEED" \
      --scalapack-launcher mpirun \
      --scalapack-nprocs "$nprocs" \
      --scalapack-executable "$NATIVE_EXECUTABLE" \
      --scalapack-block-size "$block_size" \
      --scalapack-native-backend "$NATIVE_BACKEND" \
      --skip-score \
      --verbose
    echo
  done
done
