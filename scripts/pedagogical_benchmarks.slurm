#!/bin/bash
#SBATCH --job-name=pedagogy_bench
#SBATCH --output=results/pedagogical_benchmarks_%j.out
#SBATCH --error=results/pedagogical_benchmarks_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=general

#
# Pedagogical Benchmark Suite
#
# This runs the three high-priority benchmarks designed to teach HPC concepts:
# 1. Scaling Theory: Validate O(m³) and O(nm²) complexity
# 2. Time Breakdown: Reveal computational bottlenecks
# 3. Overhead Crossover: Show when overhead dominates vs compute
#
# Results are saved to results/ and can be visualized with visualize_scaling.py
#

set -e  # Exit on error

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/pedagogy_${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}"

echo "========================================================================"
echo "PEDAGOGICAL BENCHMARK SUITE"
echo "========================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Results directory: ${RESULTS_DIR}"
echo "========================================================================"
echo ""

# Activate environment using helper script
echo "Activating environment..."
if [ -f "scripts/activate_env.sh" ]; then
    source scripts/activate_env.sh
else
    echo "WARNING: activate_env.sh not found"
    echo "Run: bash scripts/bootstrap_env.sh"
    exit 1
fi
echo ""

# Disable threading for fair comparison (match large_scale_bench.slurm)
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_mapping_policy=slot
echo "Threading disabled: OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""

# Build native code (CRITICAL: ensures backend routing fix is compiled)
# Use job-specific build directory to avoid conflicts with concurrent jobs
BUILD_DIR="native/build_job_${SLURM_JOB_ID}"
EXECUTABLE="${BUILD_DIR}/scalapack_gp_fit"

echo "Building native code in ${BUILD_DIR}..."
cmake -S native -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}"
echo ""

if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: ${EXECUTABLE} not found after build!"
    exit 1
fi
echo "Built executable: ${EXECUTABLE}"
echo ""

#
# Benchmark 1: Scaling Theory Validation
#
echo "========================================================================"
echo "BENCHMARK 1: THEORETICAL COMPLEXITY VALIDATION"
echo "========================================================================"
echo "Validating O(m³) for fit, O(m²) for score vs m, O(n) for score vs n"
echo ""

python scripts/bench_scaling_theory.py \
    --fit-backend python \
    --score-backend python \
    --m-fit-sweep 100 200 500 1000 2000 5000 \
    --m-score-sweep 100 200 500 1000 2000 5000 \
    --n-score-sweep 5000 10000 20000 50000 100000 \
    --n-fixed 10000 \
    --m-fixed 1000 \
    --scalapack-executable "${EXECUTABLE}" \
    --output-csv "${RESULTS_DIR}/scaling_theory.csv"

echo ""
echo "Benchmark 1 complete."
echo ""

#
# Benchmark 2: Time Breakdown Analysis
#
echo "========================================================================"
echo "BENCHMARK 2: TIME BREAKDOWN ANALYSIS"
echo "========================================================================"
echo "Profiling where time is spent: fit, score, select, overhead"
echo ""

python scripts/bench_time_breakdown.py \
    --m-values 100 500 1000 5000 10000 \
    --n-values 10000 50000 100000 \
    --fit-backend python \
    --score-backend python \
    --scalapack-executable "${EXECUTABLE}" \
    --output-csv "${RESULTS_DIR}/time_breakdown.csv"

echo ""
echo "Benchmark 2 complete."
echo ""

#
# Benchmark 3: Overhead vs Compute Crossover
#
echo "========================================================================"
echo "BENCHMARK 3: OVERHEAD vs COMPUTE CROSSOVER"
echo "========================================================================"
echo "Comparing backends: Python, PyBind11, ScaLAPACK"
echo ""

python scripts/bench_overhead_crossover.py \
    --m-values 100 200 500 1000 2000 5000 10000 20000 \
    --n-fixed 25000 \
    --backends python native_lapack native_reference \
    --scalapack-executable "${EXECUTABLE}" \
    --scalapack-launcher mpirun \
    --scalapack-nprocs 8 \
    --scalapack-block-size 128 \
    --scalapack-native-backend scalapack \
    --output-csv "${RESULTS_DIR}/overhead_crossover.csv"

echo ""
echo "Benchmark 3 complete."
echo ""

#
# Generate Visualizations
#
echo "========================================================================"
echo "GENERATING VISUALIZATIONS"
echo "========================================================================"

python scripts/visualize_scaling.py \
    "${RESULTS_DIR}"/*.csv \
    --output-dir "${RESULTS_DIR}/figures" \
    --format png \
    --dpi 150

echo ""
echo "========================================================================"
echo "PEDAGOGICAL BENCHMARKS COMPLETE"
echo "========================================================================"
echo "End time: $(date)"
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Key figures for teaching:"
echo "  1. ${RESULTS_DIR}/figures/scaling_theory.png"
echo "     → Shows O(m³), O(m²), O(n) scaling on log-log plots"
echo ""
echo "  2. ${RESULTS_DIR}/figures/time_breakdown.png"
echo "     → Shows where time goes (fit vs score vs select)"
echo ""
echo "  3. ${RESULTS_DIR}/figures/overhead_crossover.png"
echo "     → Shows when overhead dominates vs compute"
echo ""
echo "These plots demonstrate fundamental HPC concepts:"
echo "  • Theoretical complexity matches practice"
echo "  • Bottlenecks shift with problem size"
echo "  • Overhead vs compute tradeoffs"
echo ""
echo "Cleaning up build directory: ${BUILD_DIR}"
rm -rf "${BUILD_DIR}"
echo "Done."
echo "========================================================================"
