# Running the Pedagogical Benchmarks

This guide explains how to run the benchmark suite designed to teach HPC concepts through empirical measurements.

## Setup

### First-Time Installation

If you haven't set up the environment yet:

```bash
cd ~/poetry

# Bootstrap the environment (one-time setup)
bash scripts/bootstrap_env.sh

# For GPU support (optional):
# bash scripts/bootstrap_env.sh --gpu
```

This creates a virtual environment, installs all dependencies, and sets up the package in editable mode.

### Activating the Environment

Before running benchmarks interactively:

```bash
cd ~/poetry
source scripts/activate_env.sh

# For GPU environment (if you bootstrapped with --gpu):
# source scripts/activate_env.sh --gpu
```

**Note**: The Slurm scripts handle activation automatically.

---

## Quick Start

### Run All Pedagogical Benchmarks (Recommended)

```bash
cd ~/poetry
sbatch scripts/pedagogical_benchmarks.slurm
```

This runs all three high-priority benchmarks and generates visualizations. Results are saved to `results/pedagogy_TIMESTAMP/`.

**Time**: ~1-2 hours
**Output**: CSV data files and PNG figures

---

## Individual Benchmarks

### 1. Scaling Theory Validation

**Purpose**: Validate that theoretical complexity (O(m³), O(m²), O(n)) matches empirical measurements.

**What it teaches**:
- Cholesky factorization is truly O(m³)
- Variance computation is O(nm²), which is O(m²) for fixed n and O(n) for fixed m
- Log-log plots reveal algorithmic complexity

**Run it**:
```bash
# Activate environment
source scripts/activate_env.sh

# Run benchmark
python scripts/bench_scaling_theory.py \
  --fit-backend python \
  --score-backend python \
  --m-fit-sweep 100 200 500 1000 2000 5000 \
  --m-score-sweep 100 200 500 1000 2000 5000 \
  --n-score-sweep 5000 10000 20000 50000 100000 \
  --output-csv results/scaling_theory.csv
```

**Expected results**:
```
Fit Scaling (O(m³) theory):
m          fit_seconds     ratio
100        0.0021          baseline
200        0.0144          6.86 (expect 8.0)
500        0.1967          13.66 (expect 15.6)
1000       1.4234          7.24 (expect 8.0)
```

The ratios should approximately match expected scaling factors (m₂/m₁)³.

---

### 2. Time Breakdown Analysis

**Purpose**: Profile where time is spent in each phase of GP operations.

**What it teaches**:
- Bottlenecks shift with problem size
- Small m: Cholesky dominates (O(m³))
- Large n: Variance dominates (O(nm²))
- Selection is always trivial (O(n))

**Run it**:
```bash
# Activate environment
source scripts/activate_env.sh

# Run benchmark
python scripts/bench_time_breakdown.py \
  --m-values 100 500 1000 5000 10000 \
  --n-values 10000 50000 100000 \
  --fit-backend python \
  --score-backend python \
  --output-csv results/time_breakdown.csv
```

**Expected insights**:
```
n = 10,000 candidates
m          Fit            Score          Total
100        0.002s (4%)    0.050s (96%)   0.052s
1000       0.100s (14%)   0.620s (86%)   0.720s
10000      10.0s (88%)    1.2s (12%)     11.2s
```

For n=10k:
- Small m (100): Score dominates (variance is O(nm²) = O(10k × 10⁴))
- Large m (10k): Fit dominates (Cholesky is O(m³) = O(10⁹))

---

### 3. Overhead vs Compute Crossover

**Purpose**: Measure when overhead becomes negligible compared to computation.

**What it teaches**:
- Small problems: Overhead matters more than parallelization
- Large problems: Compute dominates, parallelization pays off
- Crossover point: Where parallel speedup > overhead cost
- PyBind11 eliminates overhead for small problems

**Run it**:
```bash
# Activate environment
source scripts/activate_env.sh

# Run benchmark
python scripts/bench_overhead_crossover.py \
  --m-values 100 200 500 1000 2000 5000 10000 20000 \
  --n-fixed 25000 \
  --backends python native_lapack native_reference \
  --scalapack-nprocs 8 \
  --output-csv results/overhead_crossover.csv
```

**Expected pattern**:
```
m          Python      PyBind11    ScaLAPACK (8 ranks)
100        0.0100      0.0010      2.5000  (overhead dominates)
500        0.0500      0.0050      2.5100  (overhead still dominates)
1000       0.1000      0.0100      2.6000  (overhead > compute)
5000       2.5000      0.2500      3.5000  (overhead ≈ compute)
10000      10.000      1.0000      8.0000  (compute dominates, ScaLAPACK wins!)
20000      40.000      4.0000      20.000  (ScaLAPACK: 2× speedup)
```

**Key insight**: ScaLAPACK has ~2.5s fixed overhead (subprocess spawn + file I/O + MPI init). Only worth it when computation >> 2.5s.

---

## Visualizing Results

Generate publication-quality plots from benchmark CSV files:

```bash
# Activate environment
source scripts/activate_env.sh

# Generate plots
python scripts/visualize_scaling.py \
  results/*.csv \
  --output-dir figures/ \
  --format png \
  --dpi 150
```

**Output**:
1. `figures/scaling_theory.png` - Log-log plots with theoretical slopes
2. `figures/time_breakdown.png` - Stacked bar charts showing bottlenecks
3. `figures/overhead_crossover.png` - Backend comparison with crossover points

These plots are designed for teaching and can be used in lectures or reports.

---

## Interpreting Log-Log Plots

### How to Read Slopes

On a log-log plot, a power-law relationship T = c·m^k appears as a straight line with slope k.

**Fit time vs m** (expect O(m³)):
```
If doubling m multiplies time by 8, then slope ≈ 3
because 2³ = 8
```

**Score time vs m** (expect O(m²)):
```
If doubling m multiplies time by 4, then slope ≈ 2
because 2² = 4
```

**Score time vs n** (expect O(n)):
```
If doubling n multiplies time by 2, then slope ≈ 1
because 2¹ = 2
```

### Why Slopes Deviate

Real measurements may show slopes slightly different from theory:

- **Slope < expected**: BLAS optimization, cache effects, or overhead
- **Slope > expected**: Memory thrashing, cache misses, or communication costs
- **Slope matches at large sizes**: Asymptotic behavior dominates

---

## Customizing Benchmarks

### Test Specific Backend

```bash
# Activate environment first
source scripts/activate_env.sh

# PyBind11 only (fast for small m)
python scripts/bench_scaling_theory.py \
  --fit-backend native_lapack \
  --score-backend native_lapack \
  --output-csv results/scaling_pybind11.csv

# ScaLAPACK only (for large m)
python scripts/bench_scaling_theory.py \
  --fit-backend native_reference \
  --score-backend python \
  --scalapack-nprocs 16 \
  --output-csv results/scaling_scalapack.csv

# GPU scoring (requires GPU environment)
source scripts/activate_env.sh --gpu
python scripts/bench_time_breakdown.py \
  --fit-backend python \
  --score-backend gpu \
  --output-csv results/breakdown_gpu.csv
```

### Test Larger Problem Sizes

```bash
# Activate environment
source scripts/activate_env.sh

# Scaling up to m=50k (requires more time and memory)
python scripts/bench_scaling_theory.py \
  --m-fit-sweep 1000 5000 10000 20000 50000 \
  --skip-score-vs-m \
  --skip-score-vs-n \
  --output-csv results/scaling_large_m.csv
```

### Test ScaLAPACK Scaling

```bash
# Activate environment
source scripts/activate_env.sh

# Test different process counts
for nprocs in 4 8 16 32; do
  python scripts/bench_overhead_crossover.py \
    --m-values 5000 10000 20000 \
    --backends native_reference \
    --scalapack-nprocs $nprocs \
    --output-csv results/scalapack_${nprocs}procs.csv
done
```

---

## Common Issues

### ScaLAPACK Always Slower

**Symptom**: ScaLAPACK slower than Python for all m.

**Diagnosis**: Problem size too small, overhead dominates.

**Solution**: Test larger m (10k+) or accept that PyBind11 is better for your workload.

### GPU Slower Than CPU

**Symptom**: GPU consistently slower, even at large m.

**Diagnosis**: Transfer overhead dominates, or CPU is heavily multi-threaded.

**Solution**:
- Test larger m (>1000)
- Set `OMP_NUM_THREADS=1` before running to disable CPU threading
- Check that GPU is actually being used (not falling back to CPU)

### Unexpected Slopes

**Symptom**: Log-log slope is 2.5 instead of 3, or varies across range.

**Diagnosis**: Overhead, BLAS optimization, or cache effects.

**Interpretation**:
- Small m: Overhead flattens slope (appears sublinear)
- Large m: Asymptotic behavior emerges (matches theory)
- Cache effects: Can cause bumps or dips in scaling

**Action**: This is normal! Real systems aren't textbook perfect. Discuss the deviations and why they occur.

---

## Pedagogical Goals

These benchmarks are designed to teach:

1. **Empirical Validation**: Theory predicts O(m³), measurements confirm it
2. **Bottleneck Analysis**: Time profiling reveals where optimization matters
3. **Overhead Tradeoffs**: "More HPC" isn't always better
4. **Crossover Points**: Optimal solution depends on problem size
5. **Parallel Scaling**: Communication limits speedup (Amdahl's law)

Each benchmark includes:
- Clear problem statement
- Expected theoretical behavior
- Empirical measurements
- Interpretation guidance

Use these benchmarks to:
- Demonstrate HPC concepts in lectures
- Validate implementation correctness
- Motivate design decisions
- Show that theory connects to practice

---

## Next Steps

After running these benchmarks:

1. **Review the figures**: Do the slopes match theory?
2. **Identify bottlenecks**: Where does time go in your workload?
3. **Find crossover points**: When does each backend win?
4. **Discuss in class**: Why do deviations from theory occur?
5. **Experiment**: Try different configurations and problem sizes

The goal is not perfect scaling, but **understanding why systems behave the way they do**.
