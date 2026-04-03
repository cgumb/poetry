#!/bin/bash
# Quick benchmark test for visualization sanity check
# Runs in ~5 minutes on a single node

set -euo pipefail

# Activate environment
source .venv/bin/activate

# Disable threading for fair comparison
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_mapping_policy=slot

# Build native code
echo "Building native code..."
cmake -S native -B native/build >/dev/null 2>&1
cmake --build native/build >/dev/null 2>&1

# Output file
OUTPUT_CSV="results/quick_test_$(date +%Y%m%d_%H%M%S).csv"
mkdir -p results

echo "Running quick benchmark sweep..."
echo "Output: $OUTPUT_CSV"
echo ""

# Problem sizes (small for speed)
M_RATED_LIST="100 500 1000 2000"

# Test each problem size
for m_rated in $M_RATED_LIST; do
  echo "=== m_rated=$m_rated ==="

  # Python baseline
  echo "  Python (1 thread)..."
  python scripts/bench_step.py \
    --backend blocked \
    --fit-backend python \
    --n-poems 10000 \
    --m-rated $m_rated \
    --seed 42 \
    --output-csv "$OUTPUT_CSV" \
    --append

  # ScaLAPACK with 1 process (serial)
  echo "  ScaLAPACK (1 process)..."
  python scripts/bench_step.py \
    --backend blocked \
    --fit-backend native_reference \
    --n-poems 10000 \
    --m-rated $m_rated \
    --seed 42 \
    --scalapack-launcher mpirun \
    --scalapack-nprocs 1 \
    --scalapack-block-size 64 \
    --scalapack-native-backend scalapack \
    --output-csv "$OUTPUT_CSV" \
    --append

  # ScaLAPACK with 4 processes
  echo "  ScaLAPACK (4 processes, bs=64)..."
  python scripts/bench_step.py \
    --backend blocked \
    --fit-backend native_reference \
    --n-poems 10000 \
    --m-rated $m_rated \
    --seed 42 \
    --scalapack-launcher mpirun \
    --scalapack-nprocs 4 \
    --scalapack-block-size 64 \
    --scalapack-native-backend scalapack \
    --output-csv "$OUTPUT_CSV" \
    --append

  # ScaLAPACK with 4 processes, larger block size
  echo "  ScaLAPACK (4 processes, bs=128)..."
  python scripts/bench_step.py \
    --backend blocked \
    --fit-backend native_reference \
    --n-poems 10000 \
    --m-rated $m_rated \
    --seed 42 \
    --scalapack-launcher mpirun \
    --scalapack-nprocs 4 \
    --scalapack-block-size 128 \
    --scalapack-native-backend scalapack \
    --output-csv "$OUTPUT_CSV" \
    --append

  echo ""
done

echo "========================================"
echo "Quick benchmark complete!"
echo "Output: $OUTPUT_CSV"
echo ""
echo "Visualize with:"
echo "  python scripts/visualize_benchmarks.py $OUTPUT_CSV"
echo "========================================"
