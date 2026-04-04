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
SPACK_SETUP_SCRIPT="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050}"

cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "GPU Scoring Performance Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"
echo ""

# Spack environment
if [[ -f "$SPACK_SETUP_SCRIPT" ]]; then
  source "$SPACK_SETUP_SCRIPT"
  spack env activate "$SPACK_ENV_NAME" || echo "Warning: Spack activation failed"
fi

# Python virtual environment
source .venv/bin/activate

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
OUTPUT_CSV="results/gpu_scoring_bench_${TIMESTAMP}.csv"

echo "Running GPU scoring benchmark..."
echo "Output: $OUTPUT_CSV"
echo ""

# Run benchmark
# Test m_rated from 100 to 10000 with 25k candidates
python scripts/bench_scoring.py \
  --m-rated 100 500 1000 2000 5000 10000 \
  --n-candidates 25000 \
  --dim 384 \
  --seed 42 \
  --cpu-threads 8 \
  --output-csv "$OUTPUT_CSV"

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "Output: $OUTPUT_CSV"
echo "========================================"
