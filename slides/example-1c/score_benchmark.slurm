#!/bin/bash
#SBATCH -p gpu
#SBATCH -N1
#SBATCH -n8
#SBATCH --gres=gpu:1
#SBATCH -t 15:00
#SBATCH -J example1c-score
#SBATCH -o results/example1c_score_%j.out
#SBATCH -e results/example1c_score_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "Example 1c: Score Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

source scripts/activate_env.sh --auto

# Check GPU
nvidia-smi --query-gpu=name --format=csv,noheader
python -c "import cupy as cp; print(f'✓ CuPy {cp.__version__}')" || exit 1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="results/example1c_score_${TIMESTAMP}.csv"

echo "Problem sizes: m ∈ {100, 500, 1000, 2000}"
echo "Candidates: n = 10,000"
echo "Output: $OUTPUT_CSV"
echo ""

python scripts/bench_scoring.py \
  --m-rated 100 500 1000 2000 \
  --n-candidates 10000 \
  --dim 384 --seed 42 --cpu-threads 8 \
  --output-csv "$OUTPUT_CSV"

echo ""
echo "Summary:"
python -c "
import pandas as pd
df = pd.read_csv('$OUTPUT_CSV')
for m in sorted(df['m_rated'].unique()):
    m_data = df[df['m_rated'] == m]
    cpu = m_data[m_data['num_threads'] == 1]['score_seconds'].values[0]
    gpu = m_data[m_data['score_backend'] == 'gpu']['score_seconds'].values[0]
    print(f'm={m:4d}: CPU {cpu:.3f}s, GPU {gpu:.3f}s (speedup: {cpu/gpu:.1f}x)')
"
