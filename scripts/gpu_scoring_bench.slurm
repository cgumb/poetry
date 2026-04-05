#!/bin/bash
#SBATCH -p gpu
#SBATCH -N1
#SBATCH -n8
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -J gpu-scoring-bench
#SBATCH -o results/gpu-scoring-bench-%j.out
#SBATCH -e results/gpu-scoring-bench-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"

cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "GPU Scoring Performance Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"
echo ""

# Activate Python environment (auto-detects based on partition)
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
OUTPUT_CSV="results/gpu_scoring_${TIMESTAMP}.csv"

echo "========================================"
echo "Running GPU vs CPU Scoring Comparison"
echo "========================================"
echo "Output: $OUTPUT_CSV"
echo ""
echo "Backends to test:"
echo "  1. CPU (1 thread)   - Baseline"
echo "  2. CPU (8 threads)  - Multi-threaded BLAS"
echo "  3. GPU (CuPy/CUDA)  - CUDA acceleration"
echo ""
echo "Problem sizes: m ∈ {100, 500, 1k, 2k, 5k, 10k, 15k, 20k}"
echo "Candidates: n = 25,000"
echo "Dimensions: d = 384"
echo ""
echo "This benchmark will test ALL THREE backends for each m value."
echo "========================================"
echo ""

# Run benchmark - bench_scoring.py automatically tests GPU if available
python scripts/bench_scoring.py \
  --m-rated 100 500 1000 2000 5000 10000 15000 20000 \
  --n-candidates 25000 \
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
  echo "  CPU (1t):  $(grep -c 'python,1' "$OUTPUT_CSV") rows"
  echo "  CPU (8t):  $(grep -c 'python,' "$OUTPUT_CSV" | awk '{print $1 - '$(grep -c 'python,1' "$OUTPUT_CSV")'}') rows (approx)"
  echo "  GPU:       $GPU_COUNT rows"
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
echo "  m < 1k:   GPU cold-start overhead, CPU may be faster"
echo "  m = 1-5k: GPU starts showing 2-5× advantage"
echo "  m > 10k:  GPU should dominate with 5-10× speedup"
echo ""
echo "Complexity reminder:"
echo "  Mean only:  O(n × m × d) + O(n × m)"
echo "  +Variance:  O(n × m²) ← Dominates for large m, GPU excels here!"
echo "========================================"
