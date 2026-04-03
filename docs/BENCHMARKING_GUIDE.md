# Benchmarking Guide

This guide explains how to run performance benchmarks comparing Python vs ScaLAPACK GP fitting and visualize the results.

## Quick Start

### On the Cluster

```bash
# 1. Navigate to the repository
cd ~/poetry

# 2. Submit the benchmark job
sbatch scripts/bench_performance_sweep.slurm

# 3. Monitor the job
squeue -u $USER
tail -f results/perf-sweep-JOBID.out

# 4. Download results when complete
scp cluster:~/poetry/results/perf_sweep_*.csv ./
```

### On Your Local Machine

```bash
# 1. Install visualization dependencies (if not already installed)
pip install matplotlib pandas

# 2. Generate plots
python scripts/visualize_benchmarks.py results/perf_sweep_TIMESTAMP.csv

# 3. View plots
open results/plots/  # macOS
xdg-open results/plots/  # Linux
```

---

## Detailed Instructions

### 1. Configuring the Benchmark

The Slurm script `bench_performance_sweep.slurm` can be customized via environment variables:

```bash
# Example: Custom configuration
sbatch \
  --export=ALL,M_RATED_LIST="1000 5000 10000 20000",NPROCS_LIST="1 4 8 16 32" \
  scripts/bench_performance_sweep.slurm
```

**Key Parameters:**

| Variable | Default | Description |
|----------|---------|-------------|
| `M_RATED_LIST` | `100 500 1000 2000 5000 10000` | Problem sizes to test |
| `NPROCS_LIST` | `1 2 4 8 16 24 32 48` | Process counts to test |
| `BLOCK_SIZE_LIST` | `64 128 256 512` | ScaLAPACK block sizes |
| `N_POEMS` | `80000` | Total corpus size |
| `DIM` | `384` | Embedding dimension |
| `NATIVE_BACKEND` | `scalapack` | Backend (`scalapack` or `mpi`) |

### 2. Understanding the Output

#### Console Output

The Slurm job produces:
- `results/perf-sweep-JOBID.out`: Standard output with progress
- `results/perf-sweep-JOBID.err`: Error messages (if any)

#### CSV Output

The benchmark produces a CSV file: `results/perf_sweep_TIMESTAMP.csv`

**Columns:**
- `timestamp`: When the benchmark ran
- `m_rated`: Problem size (number of rated points)
- `backend`: Backend used (`blocked`)
- `fit_backend`: Fit implementation (`python` or `native_reference`)
- `native_backend`: ScaLAPACK backend (`scalapack` or `mpi`)
- `nprocs`: Number of MPI processes
- `scalapack_block_size`: ScaLAPACK block size
- `fit_seconds`: Time for GP fitting (Cholesky factorization)
- `score_seconds`: Time for scoring all candidates
- `total_seconds`: Total runtime
- `log_marginal_likelihood`: Model evidence

### 3. Visualization

The `visualize_benchmarks.py` script creates four plots:

#### 1. Performance vs Problem Size
- Compares Python vs ScaLAPACK across different `m_rated`
- Shows both fit time and total time
- Log-log scale to see scaling trends

#### 2. Scaling Analysis
- Shows speedup vs number of processes
- Compares actual speedup to ideal linear speedup
- Separate plot for each problem size

#### 3. Block Size Impact
- Shows how ScaLAPACK block size affects performance
- Helps identify optimal block size for your hardware
- Separate plot for each (m_rated, nprocs) combination

#### 4. Time Breakdown
- Stacked bar chart showing fit/score/select time
- Helps identify bottlenecks
- Separate charts for Python and ScaLAPACK

**Customization:**

```bash
# Save as PDF instead of PNG
python scripts/visualize_benchmarks.py results/perf_sweep.csv --format pdf

# Save as both PNG and PDF
python scripts/visualize_benchmarks.py results/perf_sweep.csv --format both

# Higher resolution
python scripts/visualize_benchmarks.py results/perf_sweep.csv --dpi 300

# Custom output directory
python scripts/visualize_benchmarks.py results/perf_sweep.csv --output-dir my_plots/
```

---

## Running Individual Benchmarks

For quick tests or debugging, you can run individual benchmarks:

### Python Baseline

```bash
python scripts/bench_step.py \
  --backend blocked \
  --fit-backend python \
  --n-poems 5000 \
  --m-rated 100 \
  --output-csv results/test.csv
```

### ScaLAPACK with Different Configurations

```bash
# 4 processes, block size 128
python scripts/bench_step.py \
  --backend blocked \
  --fit-backend native_reference \
  --n-poems 5000 \
  --m-rated 1000 \
  --scalapack-launcher mpirun \
  --scalapack-nprocs 4 \
  --scalapack-block-size 128 \
  --scalapack-native-backend scalapack \
  --output-csv results/test.csv \
  --append

# 16 processes, block size 256
python scripts/bench_step.py \
  --backend blocked \
  --fit-backend native_reference \
  --n-poems 5000 \
  --m-rated 1000 \
  --scalapack-launcher mpirun \
  --scalapack-nprocs 16 \
  --scalapack-block-size 256 \
  --scalapack-native-backend scalapack \
  --output-csv results/test.csv \
  --append
```

---

## Interpreting Results

### What to Look For

1. **Crossover Point**: At what `m_rated` does ScaLAPACK become faster than Python?
   - **Expected**: m ≈ 5,000-10,000 (depends on overhead optimizations)
   - **Goal**: m ≈ 1,000-2,000 (after Milestone 1B)

2. **Scaling Efficiency**: How close is actual speedup to ideal?
   - **Good**: 80-90% of ideal speedup for 2-8 processes
   - **Fair**: 60-80% of ideal speedup
   - **Poor**: <60% indicates communication overhead dominates

3. **Optimal Block Size**: Which block size performs best?
   - **Typical**: 128-256 for m ≈ 5,000-10,000
   - **Larger m**: Prefer larger block sizes (256-512)
   - **Hardware-dependent**: Test on your specific cluster

### Known Bottlenecks (Current Implementation)

As identified in the analysis:

1. **Dense matrix materialization on root** (~0.5-1.0s for m=10k)
2. **Point-to-point distribution overhead** (~0.3-0.5s)
3. **Gather overhead for compatibility** (~0.8-1.0s)
4. **Subprocess launch overhead** (~0.1-0.2s)

**Total overhead**: ~2-3 seconds for m=10k

These will be addressed in Milestone 1B (distributed kernel assembly).

---

## Troubleshooting

### Build Errors

```bash
# Check if ScaLAPACK is available
ldd native/build/scalapack_gp_fit | grep scalapack

# Rebuild from scratch
rm -rf native/build
cmake -S native -B native/build
cmake --build native/build
```

### MPI Errors

```bash
# Check MPI is working
mpirun -n 4 hostname

# If using Open MPI, ensure binding is disabled
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_mapping_policy=slot
```

### Out of Memory

For large problems (m > 20,000), you may need more memory:

```bash
# Request more memory in Slurm
#SBATCH --mem=64G

# Or reduce problem size
export M_RATED_LIST="100 500 1000 2000 5000"
```

### Visualization Errors

```bash
# Install missing dependencies
pip install matplotlib pandas numpy

# If matplotlib style not found
python scripts/visualize_benchmarks.py results/perf_sweep.csv --style default
```

---

## Next Steps

After analyzing the baseline performance:

1. **Identify bottlenecks**: Which component dominates runtime?
2. **Implement optimizations**: See `docs/NATIVE_HPC_ROADMAP.md`
3. **Re-benchmark**: Compare before/after performance
4. **Scale up**: Test on larger problems (m = 50,000+)

For questions or issues, see the main `README.md` or open an issue on GitHub.
