#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n4
#SBATCH -t 10:00
#SBATCH -J example1c-fit
#SBATCH -o results/example1c_fit_%j.out
#SBATCH -e results/example1c_fit_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
cd "$REPO_DIR"
mkdir -p results

echo "========================================"
echo "Example 1c: Fit Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

source scripts/activate_env.sh --auto

# Build if needed
if [ ! -f "native/build/release/scalapack_gp_fit" ]; then
    make native-build
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="results/example1c_fit_${TIMESTAMP}.csv"

echo "Problem sizes: m ∈ {500, 1000, 2000}"
echo "Output: $OUTPUT_CSV"
echo ""

EXECUTABLE="native/build/release/scalapack_gp_fit"

for m_rated in 500 1000 2000; do
  echo "=== m=$m_rated ==="

  # Python
  python scripts/bench_step.py \
    --backend blocked --fit-backend python --score-backend none \
    --n-poems 5000 --m-rated $m_rated --seed 42 \
    --output-csv "$OUTPUT_CSV" --append

  # ScaLAPACK (4 proc)
  python scripts/bench_step.py \
    --backend blocked --fit-backend native_reference --score-backend none \
    --n-poems 5000 --m-rated $m_rated --seed 42 \
    --scalapack-executable "$EXECUTABLE" --scalapack-launcher mpirun \
    --scalapack-nprocs 4 --scalapack-block-size 128 \
    --scalapack-native-backend scalapack \
    --output-csv "$OUTPUT_CSV" --append
done

echo ""
echo "Summary:"
python -c "
import pandas as pd
df = pd.read_csv('$OUTPUT_CSV')
for m in sorted(df['m_rated'].unique()):
    m_data = df[df['m_rated'] == m]
    py = m_data[m_data['fit_backend'] == 'python']['fit_seconds'].values[0]
    sc = m_data[m_data['fit_backend'] == 'native_reference']['fit_seconds'].values[0]
    print(f'm={m:4d}: Python {py:.3f}s, ScaLAPACK {sc:.3f}s (speedup: {py/sc:.2f}x)')
"
