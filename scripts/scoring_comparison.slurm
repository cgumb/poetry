#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n16
#SBATCH -t 02:00:00
#SBATCH -J scoring-comparison
#SBATCH -o results/scoring-comparison-%j.out
#SBATCH -e results/scoring-comparison-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"

cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "Scoring Performance Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "========================================"
echo ""

# Activate Python environment (auto-detects available setup)
source scripts/activate_env.sh --auto

# Check Python version and numpy
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo ""

# Output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="results/scoring_comparison_${TIMESTAMP}.csv"

echo "Running scoring performance comparison..."
echo "Output: $OUTPUT_CSV"
echo ""
echo "Comparing:"
echo "  - Single-threaded CPU (baseline)"
echo "  - Multi-threaded CPU (8 threads)"
echo ""
echo "Testing: m ∈ {100, 500, 1k, 2k, 5k, 10k, 15k}"
echo "         n_candidates = 25k"
echo "         dim = 384"
echo ""

# Run CPU-only benchmarks (no GPU on general partition)
echo "Starting CPU scoring benchmarks..."
echo ""

# Use Python script directly - it will test both 1-thread and multi-thread
# The script sets threading environment vars BEFORE importing numpy
python scripts/bench_scoring.py \
  --m-rated 100 500 1000 2000 5000 10000 15000 \
  --n-candidates 25000 \
  --dim 384 \
  --seed 42 \
  --cpu-threads 8 \
  --output-csv "$OUTPUT_CSV"

echo ""
echo "========================================"
echo "Scoring comparison complete!"
echo "Output: $OUTPUT_CSV"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_CSV"
echo ""
echo "Expected insights:"
echo "  - Threading speedup for large m (should be ~2-4x)"
echo "  - Single-thread vs multi-thread crossover point"
echo "  - Scoring time scaling: O(n × m²) dominates for large m"
echo ""
echo "Complexity reminder:"
echo "  Kernel: O(n × m × d)"
echo "  Mean:   O(n × m)"
echo "  Var:    O(n × m²)  ← Dominates for large m!"
echo "========================================"
