#!/bin/bash
#SBATCH -p gpu
#SBATCH -N1
#SBATCH -n8
#SBATCH --gres=gpu:1
#SBATCH -t 15:00
#SBATCH -J example3-score
#SBATCH -o results/example3_score_bench_%j.out
#SBATCH -e results/example3_score_bench_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"

cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "Example 3: Score Backend Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "========================================"
echo ""

# Activate GPU environment
echo "Activating GPU environment..."
source scripts/activate_env.sh --auto

# Check GPU
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Verify CuPy
echo "Checking CuPy..."
if python -c "import cupy as cp; print(f'✓ CuPy {cp.__version__}')"; then
    echo "✓ GPU ready"
else
    echo "✗ CuPy not available - GPU benchmark will fail!"
    exit 1
fi
echo ""

# Output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="results/example3_score_bench_${TIMESTAMP}.csv"

echo "========================================"
echo "Running Score Benchmark"
echo "========================================"
echo "Output: $OUTPUT_CSV"
echo ""
echo "Backends: Python (1t), Python (8t), GPU"
echo "Problem sizes: m ∈ {100, 500, 1000, 2000}"
echo "Candidates: n = 10,000"
echo "========================================"
echo ""

# Run benchmark (smaller than production for demo)
python scripts/bench_scoring.py \
  --m-rated 100 500 1000 2000 \
  --n-candidates 10000 \
  --dim 384 \
  --seed 42 \
  --cpu-threads 8 \
  --output-csv "$OUTPUT_CSV"

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo "Results: $OUTPUT_CSV"
echo ""

# Summary
if grep -q "gpu" "$OUTPUT_CSV"; then
    echo "✓ GPU results found"
    echo ""
    echo "Summary:"
    python -c "
import pandas as pd
df = pd.read_csv('$OUTPUT_CSV')

print()
print('Score times (seconds):')
print(f'{'m':<6} {'CPU(1t)':<10} {'CPU(8t)':<10} {'GPU':<10} {'Speedup':<10}')
print('-' * 55)

for m in sorted(df['m_rated'].unique()):
    m_data = df[df['m_rated'] == m]
    cpu_1t = m_data[m_data['num_threads'] == 1]['score_seconds'].values[0]
    cpu_8t = m_data[(m_data['score_backend'] == 'python') & (m_data['num_threads'] == 8)]['score_seconds'].values[0]
    gpu = m_data[m_data['score_backend'] == 'gpu']['score_seconds'].values[0]
    speedup = cpu_1t / gpu
    print(f'{m:<6} {cpu_1t:<10.3f} {cpu_8t:<10.3f} {gpu:<10.3f} {speedup:<10.1f}x')
"
else
    echo "⚠ No GPU results - check errors above"
fi

echo ""
echo "Key insight:"
echo "  GPU overhead visible at m=100 (slower than CPU)"
echo "  GPU advantage grows with m (3-5× at m=1000+)"
echo "========================================"
