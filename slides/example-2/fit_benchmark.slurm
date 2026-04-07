#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n4
#SBATCH -t 10:00
#SBATCH -J example2-fit
#SBATCH -o results/example2_fit_bench_%j.out
#SBATCH -e results/example2_fit_bench_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "Example 2: Small Fit Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "========================================"
echo ""

# Activate environment
source scripts/activate_env.sh --auto

# Build native code if needed
if [ ! -f "native/build/release/scalapack_gp_fit" ]; then
    echo "Building native code..."
    make native-build
    echo ""
fi

# Disable threading for fair comparison
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="results/example2_fit_bench_${TIMESTAMP}.csv"

echo "Running small fit benchmark..."
echo "Output: $OUTPUT_CSV"
echo "Problem sizes: m ∈ {500, 1000, 2000}"
echo "Backends: Python, ScaLAPACK (4 processes)"
echo ""

# Small problem sizes for quick demo
M_RATED_LIST="500 1000 2000"
N_POEMS=5000
EXECUTABLE="native/build/release/scalapack_gp_fit"

for m_rated in $M_RATED_LIST; do
  echo "=== m_rated=$m_rated ==="

  # Python baseline
  echo "  Python..."
  python scripts/bench_step.py \
    --backend blocked \
    --fit-backend python \
    --score-backend none \
    --n-poems $N_POEMS \
    --m-rated $m_rated \
    --seed 42 \
    --output-csv "$OUTPUT_CSV" \
    --append

  # ScaLAPACK (4 processes only for speed)
  echo "  ScaLAPACK (4 processes)..."
  python scripts/bench_step.py \
    --backend blocked \
    --fit-backend native_reference \
    --score-backend none \
    --n-poems $N_POEMS \
    --m-rated $m_rated \
    --seed 42 \
    --scalapack-executable "$EXECUTABLE" \
    --scalapack-launcher mpirun \
    --scalapack-nprocs 4 \
    --scalapack-block-size 128 \
    --scalapack-native-backend scalapack \
    --output-csv "$OUTPUT_CSV" \
    --append

  echo ""
done

echo "========================================"
echo "Benchmark complete!"
echo "Results: $OUTPUT_CSV"
echo ""
echo "Summary:"
python -c "
import pandas as pd
df = pd.read_csv('$OUTPUT_CSV')
print()
print('Fit times by backend:')
for m in sorted(df['m_rated'].unique()):
    m_data = df[df['m_rated'] == m]
    python_time = m_data[m_data['fit_backend'] == 'python']['fit_seconds'].values[0]
    scalapack_time = m_data[m_data['fit_backend'] == 'native_reference']['fit_seconds'].values[0]
    speedup = python_time / scalapack_time
    print(f'  m={m:4d}: Python {python_time:.3f}s, ScaLAPACK {scalapack_time:.3f}s (speedup: {speedup:.2f}x)')
"
echo "========================================"
