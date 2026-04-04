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
TIME_LIMIT="${SLURM_TIME_OVERRIDE:-01:00:00}"
CPUS="${SLURM_CPUS_OVERRIDE:-8}"
MEMORY="${SLURM_MEM_OVERRIDE:-16G}"
JOB_NAME="${SLURM_JOB_NAME_OVERRIDE:-poetry-batch}"
LOG_DIR="${SLURM_LOG_DIR_OVERRIDE:-slurm_logs}"
mkdir -p "$LOG_DIR"

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --time=$TIME_LIMIT
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEMORY
#SBATCH --output=$LOG_DIR/%x-%j.out
#SBATCH --error=$LOG_DIR/%x-%j.err

set -euo pipefail
python "$SCRIPT" "$@"
EOF
