#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <python-script> [script-args...]" >&2
  echo "Example: $0 scripts/bench_sweep.py --backends naive,blocked" >&2
  exit 1
fi

SCRIPT="$1"
shift

PARTITION="${SLURM_PARTITION_OVERRIDE:-shared}"
TIME_LIMIT="${SLURM_TIME_OVERRIDE:-00:30:00}"
CPUS="${SLURM_CPUS_OVERRIDE:-8}"
MEMORY="${SLURM_MEM_OVERRIDE:-16G}"
JOB_NAME="${SLURM_JOB_NAME_OVERRIDE:-poetry-job}"

srun \
  --partition="$PARTITION" \
  --time="$TIME_LIMIT" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --job-name="$JOB_NAME" \
  python "$SCRIPT" "$@"
