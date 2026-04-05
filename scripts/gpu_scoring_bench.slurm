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

# Check if CuPy is installed
echo "Checking CuPy installation..."
python -c "import cupy as cp; print(f'CuPy version: {cp.__version__}'); print(f'CUDA version: {cp.cuda.runtime.runtimeGetVersion()}')" || echo "Warning: CuPy not available"
echo ""

# Output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="results/gpu_scoring_${TIMESTAMP}.csv"

echo "Running GPU vs CPU scoring benchmark..."
echo "Output: $OUTPUT_CSV"
echo "Testing: m ∈ {100, 500, 1k, 2k, 5k, 10k, 15k, 20k}"
echo "         n_candidates = 25k"
echo "         dim = 384"
echo ""

# Run benchmark with extended m values
python scripts/bench_scoring.py \
  --m-rated 100 500 1000 2000 5000 10000 15000 20000 \
  --n-candidates 25000 \
  --dim 384 \
  --seed 42 \
  --cpu-threads 8 \
  --output-csv "$OUTPUT_CSV"

echo ""
echo "========================================"
echo "GPU vs CPU scoring benchmark complete!"
echo "Output: $OUTPUT_CSV"
echo ""
echo "Expected results:"
echo "  m < 1k: GPU has cold-start overhead, CPU may be faster"
echo "  m = 1k-5k: GPU starts to show advantage (2-5x)"
echo "  m = 10k+: GPU should dominate (5-10x speedup)"
echo ""
echo "Key insights:"
echo "  - Where does GPU become worthwhile?"
echo "  - Threading benefit for CPU (1 vs 8 threads)"
echo "  - GPU speedup scaling with m"
echo "========================================"
