#!/bin/bash
#
# Quick smoke test for pedagogical benchmarks
# Run this on the cluster to verify everything works
#
# Usage: bash scripts/test_benchmarks.sh
#

set -e

echo "========================================================================"
echo "BENCHMARK SMOKE TEST"
echo "========================================================================"
echo "Testing with minimal problem sizes (should complete in ~30 seconds)"
echo ""

# Activate environment
source scripts/activate_env.sh

# Create test output directory
TEST_DIR="/tmp/benchmark_test_$$"
mkdir -p "$TEST_DIR"
echo "Test results: $TEST_DIR"
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
# Test 3: Overhead Crossover
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
echo ""
echo "Generated files:"
ls -lh "$TEST_DIR"/*.csv
echo ""
echo "Generated figures:"
ls -lh "$TEST_DIR"/figures/*.png 2>/dev/null || echo "(none)"
ls -lh "$TEST_DIR"/demo/*.png 2>/dev/null || echo "(none)"
echo ""
echo "To view results:"
echo "  cat $TEST_DIR/scaling.csv"
echo "  cat $TEST_DIR/breakdown.csv"
echo "  cat $TEST_DIR/crossover.csv"
echo ""
echo "To clean up:"
echo "  rm -rf $TEST_DIR"
echo "========================================================================"
