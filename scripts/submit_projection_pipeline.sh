#!/usr/bin/env bash
set -euo pipefail

PARTITION=${PARTITION:-cpu}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
MEMORY=${MEMORY:-192G}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
REPO_DIR=${REPO_DIR:-$PWD}
VENV_PATH=${VENV_PATH:-$REPO_DIR/.venv}
PROJECT_ARGS=${PROJECT_ARGS:-"--n-jobs ${CPUS_PER_TASK} --pre-reduce-dims 50"}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  sbatch \
    --partition="$PARTITION" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --export=ALL,REPO_DIR="$REPO_DIR",VENV_PATH="$VENV_PATH",PROJECT_ARGS="$PROJECT_ARGS" \
    "$0"
  exit 0
fi

cd "$REPO_DIR"
source "$VENV_PATH/bin/activate"
python scripts/project_poems_2d.py $PROJECT_ARGS
python scripts/build_poet_centroids.py
python scripts/project_poet_centroids_2d.py
