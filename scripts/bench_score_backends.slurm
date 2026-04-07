#!/bin/bash
#SBATCH --job-name=score_bench
#SBATCH --output=logs/score_benchmark_%j.log
#SBATCH --error=logs/score_benchmark_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Benchmark scoring backends: Python (CPU) vs GPU (CuPy)
#
# This compares:
# - Python single-threaded (baseline)
# - Python multi-threaded (8 threads)
# - GPU (CuPy/CUDA)
#
# Tests scoring performance O(nm²) with varying m_rated

set -e

echo "=========================================="
echo "Score Backend Benchmark"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="
echo ""

# Setup environment
cd /home/chris/teaching/CS2050/2026/poetry
source scripts/activate_env.sh

# Create output directory
mkdir -p results
mkdir -p logs

# Run benchmark
# Testing m_rated from 100 to 5000 with n_candidates=85,000 (full dataset)
python scripts/bench_scoring.py \
    --m-rated 100 500 1000 2000 5000 \
    --n-candidates 85000 \
    --dim 384 \
    --cpu-threads 8 \
    --output-csv results/score_backend_comparison.csv

echo ""
echo "=========================================="
echo "Benchmark complete!"
echo "Results: results/score_backend_comparison.csv"
echo "=========================================="
