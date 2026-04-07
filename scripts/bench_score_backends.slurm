#!/bin/bash
#SBATCH -p gpu
#SBATCH -N1
#SBATCH -n8
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -J score-backend-bench
#SBATCH -o results/score-backend-bench-%j.out
#SBATCH -e results/score-backend-bench-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"

cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "Score Backend Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "========================================"
echo ""

# Activate Python environment (auto-detects GPU environment)
echo "Activating Python environment..."
source scripts/activate_env.sh --auto

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi || echo "Warning: nvidia-smi failed"
echo ""

# Check if CuPy is installed and working
echo "Checking CuPy installation and GPU access..."
if python -c "import cupy as cp; print(f'✓ CuPy version: {cp.__version__}'); print(f'✓ CUDA version: {cp.cuda.runtime.runtimeGetVersion()}'); cp.array([1.0])"; then
  echo "✓ GPU is available and working!"
else
  echo "✗ ERROR: CuPy not available or GPU not accessible!"
  echo "GPU scoring benchmark requires working CuPy on GPU node."
  echo "Did you bootstrap with --gpu flag?"
  exit 1
fi
echo ""

# Output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="results/score_backend_comparison_${TIMESTAMP}.csv"

echo "========================================"
echo "Running Score Backend Comparison"
echo "========================================"
echo "Output: $OUTPUT_CSV"
echo ""
echo "Backends to test:"
echo "  1. Python (1 thread)   - Baseline"
echo "  2. Python (8 threads)  - Multi-threaded BLAS"
echo "  3. GPU (CuPy/CUDA)     - CUDA acceleration"
echo ""
echo "Problem sizes: m ∈ {100, 500, 1k, 2k, 5k}"
echo "Candidates: n = 85,000 (full dataset)"
echo "Dimensions: d = 384"
echo ""
echo "This benchmark will test ALL backends for each m value."
echo "========================================"
echo ""

# Run benchmark
python scripts/bench_scoring.py \
  --m-rated 100 500 1000 2000 5000 \
  --n-candidates 85000 \
  --dim 384 \
  --seed 42 \
  --cpu-threads 8 \
  --output-csv "$OUTPUT_CSV"

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo "Output: $OUTPUT_CSV"
echo ""

# Verify GPU results are present
if grep -q "gpu" "$OUTPUT_CSV"; then
  GPU_COUNT=$(grep -c "gpu" "$OUTPUT_CSV")
  echo "✓ SUCCESS: GPU results found ($GPU_COUNT rows)"
  echo ""
  echo "Results summary:"
  python -c "
import pandas as pd
df = pd.read_csv('$OUTPUT_CSV')
print('Backend results:')
for backend in df['score_backend'].unique():
    count = len(df[df['score_backend'] == backend])
    print(f'  {backend}: {count} rows')
"
else
  echo "✗ WARNING: No GPU results found in output!"
  echo "GPU backend may have failed. Check errors above."
fi

echo ""
echo "To view results:"
echo "  cat $OUTPUT_CSV"
echo "  column -t -s, $OUTPUT_CSV | less -S"
echo ""
echo "Expected insights:"
echo "  m = 100:  GPU overhead, CPU may be competitive"
echo "  m = 1-2k: GPU starts showing 5-10× advantage"
echo "  m = 5k:   GPU should dominate with 15-20× speedup"
echo ""
echo "Complexity: O(nm²) for variance computation"
echo "            n = 85,000 candidates → GPU excels!"
echo "========================================"
