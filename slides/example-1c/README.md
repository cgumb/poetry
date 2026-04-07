# Example 1c: Benchmarking (Fit & Score)

**Goal**: Benchmark fit (Python vs ScaLAPACK) and score (CPU vs GPU) performance.

## Prerequisites

- Completed Example 1a (environment set up)
- Access to CPU and GPU partitions

## Overview

This example runs two benchmarks:
1. **Fit benchmark**: Python vs ScaLAPACK (varying m)
2. **Score benchmark**: CPU vs GPU (varying m)

Both demonstrate HPC optimizations and scaling behavior.

## Important: Run from Login Node

**Exit any interactive sessions first**:
```bash
exit  # Return to login node
cd poetry
```

Slurm scripts must be submitted from the login node to avoid nested `srun` issues.

## Benchmark 1: Fit Performance

Compare Python vs ScaLAPACK for GP fitting (O(m³) Cholesky factorization).

```bash
sbatch slides/example-1c/fit_benchmark.slurm
```

**Monitor**:
```bash
squeue -u $USER
watch squeue -u $USER  # Auto-refresh
```

**Check results** (once complete):
```bash
cat results/example1c_fit*.out
```

**Expected insights**:
- Small m (500-1000): Python faster (overhead dominates)
- Large m (2000+): ScaLAPACK wins (parallelism pays off)
- Crossover point around m=1000-1500

## Benchmark 2: Score Performance

Compare Python (1t/8t) vs GPU for scoring (O(nm²) variance computation).

```bash
sbatch slides/example-1c/score_benchmark.slurm
```

**Check results**:
```bash
cat results/example1c_score*.out
```

**Expected insights**:
- m=100: GPU slower (overhead dominates)
- m=500+: GPU wins (3-5× speedup)
- GPU advantage grows with m (more parallel work)

## Visualize Results (Optional)

If you want to plot the benchmarks:

```bash
# Get compute node
srun --pty -p general -N1 -n4 -t 20 bash
cd poetry
source scripts/activate_env.sh

# Plot fit results
python slides/example-1c/plot_fit_results.py results/example1c_fit_<timestamp>.csv

# Plot score results
python slides/example-1c/plot_score_results.py results/example1c_score_<timestamp>.csv
```

Creates:
- `slides/example-1c/fit_comparison.png`
- `slides/example-1c/score_comparison.png`

## Understanding the Results

### Fit Benchmark

**Key factors**:
- **Overhead**: Process spawning, MPI init (~300ms)
- **Parallelism**: 4-way parallelism for Cholesky
- **Crossover**: Overhead < speedup when m is large enough

**Complexity**: O(m³) → cubic growth

### Score Benchmark

**Key factors**:
- **GPU overhead**: Data transfer, kernel launch (~100ms)
- **Parallelism**: Massive GPU parallelism for matrix ops
- **Scaling**: O(nm²) → GPU advantage grows quadratically

**Complexity**: O(nm²) where n=10,000 candidates

## Files Created

- `results/example1c_fit_bench_<jobid>.{csv,out,err}`
- `results/example1c_score_bench_<jobid>.{csv,out,err}`
- `slides/example-1c/{fit,score}_comparison.png` (if plotted)

## Key Takeaways

1. **Overhead matters**: HPC optimizations have fixed costs
2. **Problem size dictates**: Small problems → simple code wins
3. **Scaling behavior**: Match complexity to hardware (O(m³)→CPU, O(nm²)→GPU)
4. **Crossover points**: Empirical testing reveals when to switch backends

## Next Steps

✅ **All examples complete!**

You've now:
- Set up the environment
- Visualized a toy GP
- Used the interactive CLI
- Benchmarked HPC implementations

**Further exploration**:
- Try larger problem sizes
- Experiment with more processes (8, 16)
- Compare different block sizes (ScaLAPACK)
- Test on your own datasets
