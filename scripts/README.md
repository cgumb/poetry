# Scripts Directory

## Active Benchmarking Scripts

### Core Scripts
- **`bench_step.py`** - Main benchmarking script for single problem configurations
- **`visualize_benchmarks.py`** - Visualization tool for benchmark CSV results

### Slurm Batch Scripts
- **`quick_bench_test.slurm`** - Quick test across multiple problem sizes (m=100-2000, 1-4 processes)
- **`bench_performance_sweep.slurm`** - Large performance sweep (various configurations)
- **`compare_fit_only_sweep.slurm`** - Compare fit-only performance
- **`compare_fit_batch.slurm`** - Batch comparison of fit backends

### Utilities
- **`bootstrap_venv.sh`** - Set up Python virtual environment

## Subdirectories

- **`debug/`** - Debug and test scripts for development
- **`archive/`** - Old/superseded scripts kept for reference
- **`app/`** - Application-specific scripts (corpus building, queries, demos)

## Usage

### Quick Test
```bash
sbatch scripts/quick_bench_test.slurm
```

### Visualize Results
```bash
python scripts/visualize_benchmarks.py results/quick_test_*.csv
```

### Custom Benchmark
```bash
python scripts/bench_step.py --backend blocked --fit-backend native_reference \
  --n-poems 10000 --m-rated 2000 --scalapack-nprocs 4
```
