#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <num-ranks> <python-script> [script-args...]" >&2
  echo "Example: $0 4 scripts/bench_mpi_sweep.py --n-poems 5000,10000" >&2
  exit 1
fi

RANKS="$1"
shift
SCRIPT="$1"
shift

PARTITION="${SLURM_PARTITION_OVERRIDE:-shared}"
TIME_LIMIT="${SLURM_TIME_OVERRIDE:-01:00:00}"
CPUS_PER_TASK="${SLURM_CPUS_OVERRIDE:-1}"
MEMORY="${SLURM_MEM_OVERRIDE:-16G}"
JOB_NAME="${SLURM_JOB_NAME_OVERRIDE:-poetry-mpi}"
LOG_DIR="${SLURM_LOG_DIR_OVERRIDE:-slurm_logs}"
mkdir -p "$LOG_DIR"

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --time=$TIME_LIMIT
#SBATCH --ntasks=$RANKS
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEMORY
#SBATCH --output=$LOG_DIR/%x-%j.out
#SBATCH --error=$LOG_DIR/%x-%j.err

set -euo pipefail
srun python "$SCRIPT" "$@"
EOF
