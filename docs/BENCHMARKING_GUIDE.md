# Measuring GP Performance

## Why Benchmark?

Performance claims like "ScaLAPACK is faster for large problems" are meaningless without measurement. Benchmarking lets you:

1. **Find crossover points**: At what m does distributed beat single-node?
2. **Understand overhead**: Why is parallelization slower for small problems?
3. **Measure scaling**: Does 16 processes give 16× speedup?
4. **Validate decisions**: Is automatic backend selection choosing correctly?

**The goal**: Understand performance empirically, not theoretically.

---

## Quick Start: Pedagogical Benchmark Suite

**For teaching HPC concepts**, we provide a comprehensive benchmark suite that:
- Validates theoretical complexity (O(m³), O(m²), O(n))
- Reveals computational bottlenecks
- Demonstrates overhead vs compute tradeoffs

**Run the full suite**:
```bash
sbatch scripts/pedagogical_benchmarks.slurm
```

This runs three core benchmarks and generates visualizations automatically. Results are saved to `results/pedagogy_TIMESTAMP/`.

**See**: [RUNNING_BENCHMARKS.md](RUNNING_BENCHMARKS.md) for detailed instructions and interpretation.

---

## What to Measure

### Fit Performance (O(m³) Cholesky)

**Key questions**:
- How does time scale with m? (Should see O(m³) on log-log plot)
- When does overhead dominate? (Small m: subprocess spawn, communication)
- When does computation dominate? (Large m: actual FLOP count)
- When does distributed win? (Crossover point depends on process count)

**What to vary**:
- Problem size (m): 100, 500, 1k, 5k, 10k, 20k
- Backend: python, native_lapack, native_reference
- Process count (for ScaLAPACK): 1, 4, 8, 16
- Block size (for ScaLAPACK): 64, 128, 256

### Score Performance (O(nm²) variance)

**Key questions**:
- When does GPU overhead pay off? (Transfer vs compute)
- How does scoring scale with m and n?
- Is multi-threaded BLAS helping?

**What to vary**:
- Problem size (m): 100, 500, 1k, 5k, 10k
- Candidate count (n): 10k, 25k, 85k
- Backend: python, native_lapack, gpu
- Thread count: 1, 8 (for CPU backends)

---

## Running Benchmarks

### Quick Test (5 minutes)

Test on a few problem sizes to verify setup:

```bash
cd ~/poetry

# Fit benchmark
sbatch scripts/quick_bench_test.slurm

# Check output
tail -f results/quick_test_*.out

# When complete, visualize
python scripts/visualize_benchmarks.py results/quick_test_*.csv
```

This tests m = 100, 500, 1000, 2000 with 1-4 processes.

### Comprehensive Sweep (overnight)

Full parameter sweep for publication-quality results:

```bash
# Large-scale fit benchmark (m up to 30k)
sbatch scripts/large_scale_bench.slurm

# GPU vs CPU scoring comparison
sbatch scripts/gpu_scoring_bench.slurm

# Visualize when complete
python scripts/visualize_benchmarks.py results/large_scale_*.csv results/gpu_scoring_*.csv
```

### Custom Benchmark

For specific questions:

```python
python scripts/bench_step.py \
  --n-poems 10000 \
  --m-rated 2000 \
  --fit-backend native_reference \
  --scalapack-nprocs 16 \
  --scalapack-block-size 128 \
  --score-backend gpu
```

---

## Interpreting Results

### 1. Scaling Behavior (Log-Log Plots)

**What to look for**: Slope on log-log plot reveals complexity.

Fit time vs m (log-log):
```
Slope ≈ 3 → O(m³) scaling (Cholesky factorization)
Slope < 3 → Sublinear (communication overhead or BLAS optimization)
Slope > 3 → Superlinear (memory thrashing, cache effects)
```

Score time vs m (log-log):
```
Slope ≈ 2 → O(m²) scaling (variance computation dominates)
Slope ≈ 1 → O(m) scaling (mean-only computation)
```

**Example interpretation**:
```
m = 100:   Fit = 0.01s  (slope not visible yet)
m = 1000:  Fit = 0.10s  (10× increase for 10× m → linear?)
m = 10000: Fit = 10s    (100× increase for 10× m → cubic!)
```

The cubic scaling emerges at larger m.

### 2. Overhead vs Compute

**Small problems** (overhead dominates):
```
m = 500:
  Python:    0.01s (baseline)
  PyBind11:  0.001s (10× faster - eliminated overhead)
  ScaLAPACK: 2.5s (250× slower - overhead dominates!)
```

Overhead sources:
- Subprocess spawn: ~160ms
- File I/O: ~50ms
- MPI initialization: ~100ms
- Communication: Depends on m and process count

**Large problems** (compute dominates):
```
m = 20k:
  Python:    10s (single-node baseline)
  ScaLAPACK: 6s (16 ranks, 1.7× faster - parallelism wins)
```

**Crossover point**: m ≈ 7-10k (overhead = speedup)

### 3. Parallel Scaling

**Ideal scaling** (embarrassingly parallel):
```
P processes → P× speedup
```

**Realistic scaling** (with communication):
```
Speedup = P / (1 + α·(P-1))

α = communication fraction
α = 0   → ideal (embarrassingly parallel)
α = 0.1 → 90% parallel efficiency
α = 0.5 → Amdahl's law limiting
```

**What to look for**:
```
1 → 4 processes:  3.5× speedup (good!)
4 → 16 processes: 3.0× speedup (diminishing returns)
16 → 64 processes: 1.8× speedup (overhead dominates)
```

**Interpretation**: Communication overhead grows with process count.

### 4. Block Size Effects

ScaLAPACK block size trades off:
- **Large blocks**: Less communication, more sequential work per process
- **Small blocks**: More communication, better load balance

**Typical results**:
```
Block = 32:  Good for small m, high communication
Block = 128: Sweet spot for most problems
Block = 512: Good for large m, minimal communication
```

**What to look for**:
- Small m: Block size matters little (overhead dominates)
- Large m: Larger blocks often win (reduce communication)

### 5. GPU Crossover

**Cold-start overhead**:
```
m = 100:
  CPU: 0.05s
  GPU: 0.27s (5× slower - transfer dominates)
```

**Compute payoff**:
```
m = 500:
  CPU: 0.35s
  GPU: 0.08s (4.6× faster - compute dominates)
```

**What determines crossover**:
- Transfer overhead: Fixed (~150ms)
- Compute time: Grows as O(nm²)
- Crossover: When compute > transfer

---

## Common Patterns

### Pattern 1: Overhead Wall

```
All ScaLAPACK times ≈ 2.5s for m < 5k
```

**Interpretation**: Overhead (subprocess + MPI) is ~2.5s regardless of m.
**Conclusion**: Don't use ScaLAPACK for small problems.

### Pattern 2: Cubic Scaling

```
Log-log plot: Slope ≈ 3
```

**Interpretation**: Cholesky factorization is O(m³) as expected.
**Conclusion**: Problem gets expensive fast - need parallelization for large m.

### Pattern 3: Scaling Plateau

```
1 → 4 → 16 → 64 processes
Speedup: 1 → 3.5 → 10 → 18 (sublinear)
```

**Interpretation**: Communication overhead increasing with process count.
**Conclusion**: Diminishing returns beyond ~16 processes for this problem size.

### Pattern 4: GPU Dominance

```
m ≥ 500: GPU consistently 3-4× faster
```

**Interpretation**: Parallel triangular solves overcome transfer overhead.
**Conclusion**: Use GPU for scoring when m ≥ 500.

---

## Visualization Outputs

### 1. Performance vs Problem Size

**X-axis**: m (rated points)
**Y-axis**: Time (seconds)
**Lines**: Different backends (Python, PyBind11, ScaLAPACK)

**What to look for**:
- Cubic scaling (straight line on log-log, slope ≈ 3)
- Crossover point (where lines intersect)
- Overhead floor (ScaLAPACK flat at small m)

### 2. Scaling Analysis

**X-axis**: Number of processes
**Y-axis**: Speedup vs single-process

**What to look for**:
- Compare to ideal (dotted line)
- Parallel efficiency (actual / ideal)
- Diminishing returns (plateau)

### 3. Block Size Comparison

**X-axis**: m (rated points)
**Y-axis**: Time (seconds)
**Lines**: Different block sizes (64, 128, 256, 512)

**What to look for**:
- Optimal block size for each m
- Convergence at large m (block size matters less)

### 4. GPU vs CPU

**X-axis**: m (rated points)
**Y-axis**: Time (seconds)
**Lines**: CPU (1 thread), CPU (8 threads), GPU

**What to look for**:
- Crossover point (GPU becomes faster)
- Threading benefit (8 threads vs 1 thread)
- GPU speedup (CPU time / GPU time)

---

## Benchmarking Workflow

### 1. Establish Baseline

```bash
# Python reference (always available)
python scripts/bench_step.py --fit-backend python --m-rated 100 500 1000
```

### 2. Test PyBind11

```bash
# Zero-overhead single-node
python scripts/bench_step.py --fit-backend native_lapack --m-rated 100 500 1000
```

**Expected**: 10× faster than Python (eliminates overhead).

### 3. Test ScaLAPACK

```bash
# Distributed memory
sbatch scripts/quick_bench_test.slurm
```

**Expected**: Slower than Python for m < 5k (overhead dominates).

### 4. Find Crossover

```bash
# Test around expected crossover (m ≈ 7-10k)
sbatch scripts/large_scale_bench.slurm
```

**Expected**: ScaLAPACK wins for m > 10k with 8-16 processes.

### 5. Test GPU

```bash
# GPU scoring
sbatch scripts/gpu_scoring_bench.slurm
```

**Expected**: GPU wins for m ≥ 500 (3-4× speedup).

---

## Profiling Individual Runs

For detailed timing breakdown:

```python
result = run_blocked_step(
    embeddings, rated_indices, ratings,
    fit_backend="native_reference",
    score_backend="gpu",
)

# Examine profile
print(f"Fit:     {result.profile.fit_seconds:.3f}s")
print(f"Score:   {result.profile.score_seconds:.3f}s")
print(f"Select:  {result.profile.select_seconds:.3f}s")
print(f"Total:   {result.profile.total_seconds:.3f}s")
```

**What to look for**:
- Which phase dominates? (Fit vs Score vs Select)
- Does it match expectations? (O(m³) vs O(nm²))
- Where should optimization focus?

---

## Common Issues

### Issue: ScaLAPACK Not Faster

**Symptoms**: ScaLAPACK slower than Python for all m.

**Possible causes**:
1. Problem too small (m < 10k): Overhead dominates
2. Too few processes (P < 8): Not enough parallelism
3. Wrong block size: Try 128 or 256
4. Communication overhead: Check network latency

**Debug**:
```bash
# Test with different parameters
python scripts/bench_step.py \
  --m-rated 20000 \
  --scalapack-nprocs 16 \
  --scalapack-block-size 128
```

### Issue: GPU Slower Than CPU

**Symptoms**: GPU always slower.

**Possible causes**:
1. Problem too small (m < 500): Transfer overhead dominates
2. CPU heavily multi-threaded: BLAS using all cores
3. Old GPU: Compute units insufficient

**Debug**:
```bash
# Test with larger m
python scripts/bench_scoring.py --m-rated 1000 2000 5000
```

### Issue: Scaling Plateau

**Symptoms**: Speedup stops improving beyond P=8.

**Possible causes**:
1. Communication overhead: Amdahl's law
2. Load imbalance: Some processes idle
3. Problem too small: Not enough work per process

**Interpretation**: Normal behavior - parallel efficiency drops with process count.

---

## Key Takeaways

1. **Measure, don't guess**: Theory says O(m³), but overhead changes everything
2. **Overhead matters**: Small problems need simple solutions
3. **Crossover points are real**: No single backend wins everywhere
4. **Scaling is limited**: Communication and Amdahl's law limit speedup
5. **Visualize results**: Log-log plots reveal scaling behavior

**For teaching**:
- Benchmarking connects theory (O(m³)) to practice (actual timings)
- Overhead vs compute is a fundamental HPC tradeoff
- Parallel scaling limits are measurable, not theoretical

---

## Scripts Reference

| Script | Purpose | Time |
|--------|---------|------|
| `quick_bench_test.slurm` | Quick validation (m < 2k) | 5 min |
| `large_scale_bench.slurm` | Comprehensive sweep (m up to 30k) | 12 hrs |
| `gpu_scoring_bench.slurm` | GPU vs CPU comparison | 30 min |
| `bench_step.py` | Custom benchmarks | Varies |
| `visualize_benchmarks.py` | Generate plots | 1 min |

**Typical workflow**:
1. Quick test to verify setup
2. Overnight comprehensive sweep
3. Visualize and analyze results
4. Custom benchmarks for specific questions
