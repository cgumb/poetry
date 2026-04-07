# Example 2: GP Fitting Benchmark & Visualization

**Goal**: Run a small fitting benchmark and visualize a 2D GP fit.

## Prerequisites

- Completed Example 1 (environment set up)
- Currently on a compute node with environment activated

## Overview

This example:
1. Runs a small fit benchmark comparing Python vs ScaLAPACK
2. Creates a 2D toy GP example and visualizes the posterior

## Important: Where to Run

**Slurm scripts must be submitted from the login node** (not from within an interactive node).

If you're currently in an interactive session from Example 1:
```bash
exit  # Return to login node
cd poetry
```

## Step 1: Run Small Fit Benchmark

We'll benchmark fitting for a few small problem sizes.

**Submit the benchmark job**:
```bash
sbatch slides/example-2/fit_benchmark.slurm
```

**Monitor the job**:
```bash
squeue -u $USER
```

**Check results** (once job completes):
```bash
cat results/example2_fit_bench*.out
ls -lh results/example2_fit_bench*.csv
```

This compares Python vs ScaLAPACK for m ∈ {500, 1000, 2000}.

## Step 2: Visualize 2D GP Fit

Now let's create a toy 2D example to see how GPs work.

**Get back on a compute node** (if you exited):
```bash
srun --pty -p general -N1 -n4 -t 30 bash
cd poetry
source scripts/activate_env.sh
```

**Run the visualization script**:
```bash
python slides/example-2/visualize_2d_gp.py
```

**View the output**:
```bash
ls -lh slides/example-2/gp_2d_example.png
```

This creates a 2D heatmap showing:
- Observed training points
- GP posterior mean
- Uncertainty regions

## Understanding the Results

### Benchmark Output

Look for lines like:
```
m=500:  Python 0.05s, ScaLAPACK(4 proc) 0.12s
m=2000: Python 0.20s, ScaLAPACK(4 proc) 0.15s
```

**Key insight**: For small m, Python is faster (overhead dominates). As m grows, ScaLAPACK becomes competitive.

### 2D Visualization

The plot shows:
- **Colors**: Posterior mean (predicted function value)
- **Dots**: Training observations
- **Shading intensity**: Uncertainty (darker = more certain)

Notice how uncertainty shrinks near observations!

## Files Created

- `results/example2_fit_bench_<jobid>.csv` - Benchmark data
- `results/example2_fit_bench_<jobid>.out` - Job log
- `slides/example-2/gp_2d_example.png` - 2D GP visualization

## Next Steps

Continue to **Example 3** for score backend benchmarking (CPU vs GPU).
