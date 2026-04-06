#!/bin/bash
#SBATCH --job-name=test_bench
#SBATCH --output=results/test_benchmarks_%j.out
#SBATCH --error=results/test_benchmarks_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=general

#
# Quick smoke test for pedagogical benchmarks
# Runs with minimal problem sizes (should complete in ~1 minute)
#
# Usage: sbatch scripts/test_benchmarks.slurm
#

set -e

echo "========================================================================"
echo "BENCHMARK SMOKE TEST"
echo "========================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Testing with minimal problem sizes"
echo ""

# Create test output directory
TEST_DIR="results/benchmark_test_${SLURM_JOB_ID}"
mkdir -p "$TEST_DIR"
echo "Test results: $TEST_DIR"
echo ""

# Activate environment
echo "Activating environment..."
if [ -f "scripts/activate_env.sh" ]; then
    source scripts/activate_env.sh
else
    echo "ERROR: activate_env.sh not found"
    exit 1
fi
echo ""

#
# Test 1: Scaling Theory
#
echo "----------------------------------------------------------------------"
echo "TEST 1: bench_scaling_theory.py"
echo "----------------------------------------------------------------------"
python scripts/bench_scaling_theory.py \
  --fit-backend python \
  --score-backend python \
  --m-fit-sweep 10 20 50 \
  --m-score-sweep 10 20 50 \
  --n-score-sweep 100 200 500 \
  --n-fixed 100 \
  --m-fixed 10 \
  --output-csv "$TEST_DIR/scaling.csv" || {
    echo "❌ FAILED: bench_scaling_theory.py"
    exit 1
  }
echo "✓ bench_scaling_theory.py passed"
echo ""

#
# Test 2: Time Breakdown
#
echo "----------------------------------------------------------------------"
echo "TEST 2: bench_time_breakdown.py"
echo "----------------------------------------------------------------------"
python scripts/bench_time_breakdown.py \
  --m-values 10 20 50 \
  --n-values 100 200 \
  --fit-backend python \
  --score-backend python \
  --output-csv "$TEST_DIR/breakdown.csv" || {
    echo "❌ FAILED: bench_time_breakdown.py"
    exit 1
  }
echo "✓ bench_time_breakdown.py passed"
echo ""

#
# Test 3: Overhead Crossover (Python only for quick test)
#
echo "----------------------------------------------------------------------"
echo "TEST 3: bench_overhead_crossover.py"
echo "----------------------------------------------------------------------"
python scripts/bench_overhead_crossover.py \
  --m-values 10 20 50 \
  --n-fixed 100 \
  --backends python \
  --output-csv "$TEST_DIR/crossover.csv" || {
    echo "❌ FAILED: bench_overhead_crossover.py"
    exit 1
  }
echo "✓ bench_overhead_crossover.py passed"
echo ""

#
# Test 4: Visualization
#
echo "----------------------------------------------------------------------"
echo "TEST 4: visualize_scaling.py"
echo "----------------------------------------------------------------------"
python scripts/visualize_scaling.py \
  "$TEST_DIR"/*.csv \
  --output-dir "$TEST_DIR/figures" \
  --format png || {
    echo "❌ FAILED: visualize_scaling.py"
    exit 1
  }
echo "✓ visualize_scaling.py passed"
echo ""

#
# Test 5: Demo Plots
#
echo "----------------------------------------------------------------------"
echo "TEST 5: demo_scaling_plots.py"
echo "----------------------------------------------------------------------"
python scripts/demo_scaling_plots.py \
  --output-dir "$TEST_DIR/demo" \
  --format png || {
    echo "❌ FAILED: demo_scaling_plots.py"
    exit 1
  }
echo "✓ demo_scaling_plots.py passed"
echo ""

#
# Summary
#
echo "========================================================================"
echo "ALL TESTS PASSED ✓"
echo "========================================================================"
echo "End time: $(date)"
echo ""
echo "Generated files:"
ls -lh "$TEST_DIR"/*.csv
echo ""
echo "Generated figures:"
ls -lh "$TEST_DIR"/figures/*.png 2>/dev/null || echo "(none - check for errors)"
ls -lh "$TEST_DIR"/demo/*.png 2>/dev/null || echo "(none - check for errors)"
echo ""
echo "Output saved to:"
echo "  $TEST_DIR/"
echo ""
echo "Next steps:"
echo "  1. Check output: cat results/test_benchmarks_${SLURM_JOB_ID}.out"
echo "  2. View CSVs: cat $TEST_DIR/*.csv"
echo "  3. If all passed, run full suite: sbatch scripts/pedagogical_benchmarks.slurm"
echo "========================================================================"
