# Example 3: Score Benchmark (CPU vs GPU)

**Goal**: Compare scoring performance between CPU and GPU backends.

## Prerequisites

- Completed Examples 1-2
- GPU node access (for meaningful comparison)

## Overview

This example benchmarks the scoring phase (computing posterior mean and variance) across:
- Python (single-threaded CPU)
- Python (multi-threaded CPU)
- GPU (CuPy/CUDA)

**Key insight**: Scoring is O(nm²), so GPU can provide significant speedup for large m.

## Important: Run from Login Node

Slurm scripts must be submitted from the **login node**. If you're in an interactive session:

```bash
exit  # Return to login node
cd poetry
```

## Step 1: Submit Score Benchmark

This benchmark tests scoring performance for m ∈ {100, 500, 1000, 2000} with n=10,000 candidates.

```bash
sbatch slides/example-3/score_benchmark.slurm
```

**Monitor job**:
```bash
squeue -u $USER
```

**Expected runtime**: ~5-10 minutes

## Step 2: Check Results

Once the job completes:

```bash
# View log
cat results/example3_score_bench*.out

# Check CSV
ls -lh results/example3_score_bench*.csv
```

## Understanding the Results

The output shows scoring times for each backend:

```
m=100:  Python(1t) 0.18s, Python(8t) 0.17s, GPU 0.44s
m=500:  Python(1t) 1.19s, Python(8t) 1.19s, GPU 0.25s
m=1000: Python(1t) 2.46s, Python(8t) 2.41s, GPU 0.62s
m=2000: Python(1t) 6.32s, Python(8t) 5.52s, GPU 1.68s
```

**Key observations**:
1. **m=100**: GPU is slower (overhead dominates)
2. **m=500+**: GPU starts winning (3-5× speedup)
3. **Multi-threading**: Modest CPU speedup (Python GIL limits benefit)

## Why GPU Wins for Scoring

Scoring requires computing:
```
variance = K_qq - diag(K_qr @ K^{-1} @ K_rq)
```

This is O(nm²) with:
- Large matrix multiplications → GPU parallelism helps!
- m² dominates for large m → GPU advantage grows

But there's overhead:
- Data transfer to GPU
- Kernel launch
- Small problems → overhead > speedup

## Visualize Results (Optional)

If you want to plot the results:

```bash
# Get on compute node
srun --pty -p general -N1 -n4 -t 20 bash
cd poetry
source scripts/activate_env.sh

# Run plotting script
python slides/example-3/plot_score_results.py results/example3_score_bench_<timestamp>.csv
```

This creates `slides/example-3/score_comparison.png`.

## Files Created

- `results/example3_score_bench_<jobid>.csv` - Benchmark data
- `results/example3_score_bench_<jobid>.out` - Job log
- `slides/example-3/score_comparison.png` - Plot (if created)

## Next Steps

Continue to **Example 4** to use the interactive CLI and create your own posterior visualization!
