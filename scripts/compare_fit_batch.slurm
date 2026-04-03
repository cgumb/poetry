#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n4
#SBATCH -t 00:30:00
#SBATCH -J compare-fit
#SBATCH -o compare-fit-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
N_POEMS="${N_POEMS:-5000}"
M_RATED_LIST="${M_RATED_LIST:-128 512}"
NPROCS_LIST="${NPROCS_LIST:-1 4}"
NATIVE_BACKEND="${NATIVE_BACKEND:-auto}"
NATIVE_LAUNCHER="${NATIVE_LAUNCHER:-mpirun}"
NATIVE_EXECUTABLE="${NATIVE_EXECUTABLE:-native/build/scalapack_gp_fit}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
SCALAPACK_BLOCK_SIZE="${SCALAPACK_BLOCK_SIZE:-128}"
DIM="${DIM:-384}"
SEED="${SEED:-0}"

cd "$REPO_DIR"
source .venv/bin/activate
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cmake -S native -B native/build
cmake --build native/build

echo "JOB=$SLURM_JOB_ID STEP=${SLURM_STEP_ID:-unset} NTASKS=$SLURM_NTASKS STEP_TASKS=${SLURM_STEP_NUM_TASKS:-unset}"
if [[ "$NATIVE_LAUNCHER" == "mpirun" ]]; then
  mpirun -n "$(echo "$NPROCS_LIST" | awk '{print $NF}')" hostname
fi

for m_rated in $M_RATED_LIST; do
  for nprocs in $NPROCS_LIST; do
    echo "===== nprocs=$nprocs m_rated=$m_rated backend=$NATIVE_BACKEND ====="
    python -u scripts/compare_fit_backends.py \
      --n-poems "$N_POEMS" \
      --dim "$DIM" \
      --m-rated "$m_rated" \
      --seed "$SEED" \
      --block-size "$BLOCK_SIZE" \
      --scalapack-launcher "$NATIVE_LAUNCHER" \
      --scalapack-nprocs "$nprocs" \
      --scalapack-executable "$NATIVE_EXECUTABLE" \
      --scalapack-block-size "$SCALAPACK_BLOCK_SIZE" \
      --scalapack-native-backend "$NATIVE_BACKEND" \
      --verbose
  done
done
